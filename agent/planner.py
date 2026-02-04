"""
Planner LLM - The brain of the AI Agent.
Uses OpenAI's vision-capable model to analyze screenshots and decide actions.
"""

import json
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from PIL import Image

from openai import OpenAI

from .config import config
from .prompt_loader import load_prompt, format_prompt
from .actions import (
    PlannerResponse,
    AgentAction,
    parse_action_from_dict,
    OCRClickAction,
    CoordinateClickAction,
    ChainAction,
    ChainedStep,
)
from .screenshot import get_screen_capture


@dataclass
class AgentState:
    """Current state of the agent for context."""
    task: str
    goal: str
    step_number: int = 0
    last_action: Optional[str] = None
    last_action_result: Optional[str] = None
    last_action_success: Optional[bool] = None
    active_window_title: Optional[str] = None
    ocr_summary: Optional[str] = None
    history: List[str] = field(default_factory=list)
    system_language: str = "English"
    operator_updates: List[str] = field(default_factory=list)
    consecutive_failures: int = 0
    last_failed_action: Optional[str] = None
    failure_counts: Dict[str, int] = field(default_factory=dict)
    
    def add_to_history(self, entry: str):
        self.history.append(f"Step {self.step_number}: {entry}")
        # Keep only last 10 entries
        if len(self.history) > 10:
            self.history = self.history[-10:]
    
    def to_context_string(self) -> str:
        """Convert state to a context string for the LLM."""
        parts = [f"Final goal: {self.goal}"]
        parts.append(f"Task: {self.task}")
        parts.append(f"Current step: {self.step_number}")
        parts.append(f"System language: {self.system_language}")
        
        if self.active_window_title:
            parts.append(f"Active window: {self.active_window_title}")
        
        if self.last_action:
            result_str = "succeeded" if self.last_action_success else "failed"
            parts.append(f"Last action: {self.last_action} ({result_str})")
            if self.last_action_result:
                parts.append(f"Result: {self.last_action_result}")

        if self.consecutive_failures:
            parts.append(f"Consecutive failures: {self.consecutive_failures}")
            if self.last_failed_action:
                parts.append(f"Last failed action: {self.last_failed_action}")
            repeated = [f"{k} (x{v})" for k, v in self.failure_counts.items() if v > 1]
            if repeated:
                parts.append(f"Repeated failures: {', '.join(repeated)}")

        if self.operator_updates:
            parts.append("\nHuman updates (most recent last):")
            for update in self.operator_updates[-5:]:
                parts.append(f"  - {update}")
        
        if self.ocr_summary:
            parts.append(f"Visible text (partial): {self.ocr_summary[:500]}")
        
        if self.history:
            parts.append("\nRecent history:")
            for entry in self.history[-5:]:
                parts.append(f"  - {entry}")
        
        return "\n".join(parts)


SYSTEM_PROMPT = load_prompt("planner/system_prompt.txt")
USER_MESSAGE_TEMPLATE = load_prompt("planner/user_message.txt")
VERIFY_ACTION_TEMPLATE = load_prompt("planner/verify_action_user.txt")


class PlannerLLM:
    """The main planning brain that decides what actions to take."""
    
    def __init__(self):
        config.validate()
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_model
        self.screen = get_screen_capture()

    def _token_kwargs(self, tokens: int) -> dict:
        """Return the appropriate max token argument for the current model."""
        if self.model and self.model.startswith("gpt-5"):
            return {"max_completion_tokens": tokens}
        return {"max_tokens": tokens}
    
    def plan_next_action(
        self,
        screenshot: Image.Image,
        state: AgentState
    ) -> PlannerResponse:
        """
        Analyze the screenshot and state, then decide the next action.
        
        Args:
            screenshot: Current screenshot
            state: Current agent state with context
        
        Returns:
            PlannerResponse with the action to take
        """
        # Convert screenshot to base64
        screenshot_b64 = self.screen.to_base64(screenshot, format="PNG")
        
        # Build the user message with state context
        user_message = format_prompt(
            USER_MESSAGE_TEMPLATE,
            state_context=state.to_context_string(),
            system_language=state.system_language,
        )
        
        # Call the vision API
        response = self.client.chat.completions.create(
            model=self.model,
            **self._token_kwargs(config.openai_max_tokens),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        )
        
        # Parse the response
        response_text = response.choices[0].message.content.strip()
        parsed = self._parse_response(response_text)
        return self._validate_response(parsed, state)

    def _should_avoid_ocr(self, target_text: Optional[str], state: AgentState) -> bool:
        task_lower = (state.task or "").lower()
        if any(word in task_lower for word in ("logo", "icon", "symbol", "avatar", "badge")):
            return True
        ocr_summary = (state.ocr_summary or "").strip()
        if not ocr_summary:
            return True
        if target_text:
            if target_text.lower() not in ocr_summary.lower():
                return True
        return False

    def _ocr_to_coordinate_click(self, target_text: Optional[str], state: AgentState) -> CoordinateClickAction:
        task_target = (state.task or "").strip()
        if task_target:
            target = task_target
        elif target_text:
            target = f"{target_text} icon or label"
        else:
            target = "the requested target"
        return CoordinateClickAction(
            target=target,
            prefer_primary=True,
            button="left",
            clicks=1,
            description=f"Click {target}",
        )

    def _validate_response(self, response: PlannerResponse, state: AgentState) -> PlannerResponse:
        action = response.action
        if isinstance(action, OCRClickAction):
            if self._should_avoid_ocr(action.text, state):
                response.action = self._ocr_to_coordinate_click(action.text, state)
            return response

        if isinstance(action, ChainAction):
            updated = False
            new_steps = []
            for step in action.steps:
                if step.action_type == "ocr_click":
                    if self._should_avoid_ocr(step.text, state):
                        target = (state.task or "").strip() or (step.text or "the requested target")
                        new_steps.append(
                            ChainedStep(
                                action_type="coordinate_click",
                                text=target,
                                target=target,
                                button=step.button or "left",
                                clicks=step.clicks or 1,
                                delay_after=step.delay_after,
                            )
                        )
                        updated = True
                        continue
                new_steps.append(step)
            if updated:
                action.steps = new_steps
                response.action = action
        return response
    
    def _parse_response(self, response_text: str) -> PlannerResponse:
        """Parse the LLM response into a PlannerResponse."""
        try:
            # Try to extract JSON from the response
            # Handle markdown code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to parse the whole thing as JSON
                json_str = response_text
            
            data = json.loads(json_str)
            
            # Parse the action
            action_data = data.get("action", {})
            action = parse_action_from_dict(action_data)
            
            return PlannerResponse(
                thought=data.get("thought", ""),
                action=action,
                success_criteria=data.get("success_criteria")
            )
        except json.JSONDecodeError as e:
            # If we can't parse JSON, create a fail action
            from .actions import FailAction
            return PlannerResponse(
                thought=f"Failed to parse LLM response: {response_text[:200]}",
                action=FailAction(reason=f"Invalid response format: {str(e)}"),
                success_criteria=None
            )
        except Exception as e:
            from .actions import FailAction
            return PlannerResponse(
                thought=f"Error: {str(e)}",
                action=FailAction(reason=str(e)),
                success_criteria=None
            )
    
    def verify_action(
        self,
        screenshot: Image.Image,
        criteria: str,
        state: AgentState
    ) -> bool:
        """
        Verify if an action succeeded based on the criteria.
        
        Args:
            screenshot: Screenshot after action
            criteria: Success criteria to check
            state: Current state
        
        Returns:
            True if criteria is met, False otherwise
        """
        if not criteria:
            return True  # No criteria means assume success
        
        screenshot_b64 = self.screen.to_base64(screenshot, format="PNG")
        
        prompt = format_prompt(
            VERIFY_ACTION_TEMPLATE,
            task=state.task,
            last_action=state.last_action or "",
            criteria=criteria,
            system_language=state.system_language,
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            **self._token_kwargs(10),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_b64}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ]
        )
        
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")


# Singleton instance
_planner: Optional[PlannerLLM] = None


def get_planner() -> PlannerLLM:
    """Get or create the planner singleton."""
    global _planner
    if _planner is None:
        _planner = PlannerLLM()

    return _planner
