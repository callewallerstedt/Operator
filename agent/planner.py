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
    step_number: int = 0
    last_action: Optional[str] = None
    last_action_result: Optional[str] = None
    last_action_success: Optional[bool] = None
    active_window_title: Optional[str] = None
    ocr_summary: Optional[str] = None
    history: List[str] = field(default_factory=list)
    system_language: str = "English"
    
    def add_to_history(self, entry: str):
        self.history.append(f"Step {self.step_number}: {entry}")
        # Keep only last 10 entries
        if len(self.history) > 10:
            self.history = self.history[-10:]
    
    def to_context_string(self) -> str:
        """Convert state to a context string for the LLM."""
        parts = [f"Task: {self.task}"]
        parts.append(f"Current step: {self.step_number}")
        parts.append(f"System language: {self.system_language}")
        
        if self.active_window_title:
            parts.append(f"Active window: {self.active_window_title}")
        
        if self.last_action:
            result_str = "succeeded" if self.last_action_success else "failed"
            parts.append(f"Last action: {self.last_action} ({result_str})")
            if self.last_action_result:
                parts.append(f"Result: {self.last_action_result}")
        
        if self.ocr_summary:
            parts.append(f"Visible text (partial): {self.ocr_summary[:500]}")
        
        if self.history:
            parts.append("\nRecent history:")
            for entry in self.history[-5:]:
                parts.append(f"  - {entry}")
        
        return "\n".join(parts)


SYSTEM_PROMPT = """You are a fast, precise AI agent that controls a Windows computer. Analyze screenshots and execute actions efficiently.

## IMPORTANT: System Language
The user's system may be in a different language (Swedish, German, etc). UI elements like menus, buttons, dialogs will be in that language.
- "File" might be "Arkiv" (Swedish), "Datei" (German)
- "Settings" might be "Installningar" (Swedish)
- "Search" might be "Sok" (Swedish)
- Read the screenshot carefully to identify UI elements in the correct language.

## IMPORTANT: Windows Search - Use Specific App Names
When searching in Windows Start menu or search bars, ALWAYS use the SPECIFIC app name, NOT generic terms:
- ? GOOD: "Edge", "Chrome", "Firefox", "Calculator", "Notepad", "Excel", "Word"
- ? BAD: "webbl?sare" (browser), "kalkylator" (calculator), "textredigerare" (text editor)
- ? GOOD: "Microsoft Edge", "Google Chrome", "Spotify", "Discord"
- ? BAD: "web browser", "music app", "chat app"

Windows search works best with actual application names. Look at the screenshot to see what apps are installed and use their exact names.

## Your Capabilities
1. Press keys and keyboard shortcuts (keypress, hotkey)
2. Type text (type)
3. Chain multiple actions together (chain) - EFFICIENT for sequences
4. **Coordinate click** (coordinate_click) - ALWAYS use the coordinate finder pipeline to click UI elements
5. Click at specific coordinates (mouse_click) - only when you know exact coords
6. Scroll the screen (scroll)
7. Wait for conditions (wait)

## Priority Order (ALWAYS follow this)
1. **Keyboard shortcuts** - Win key, Ctrl+O, Alt+F4, etc. FASTEST
2. **Chain actions** - Combine related actions (open menu + type + enter)
3. **Coordinate click** - THE ONLY method for clicking any UI element
4. **Mouse click** - Only if you can read exact coordinates from the screenshot (rare)

## OCR Click Rules (IMPORTANT)
- Use ocr_click ONLY when the target is visible text on the screen.
- If the user mentions "logo", "icon", "symbol", or the target is clearly non-text, do NOT use ocr_click.
- If there is no visible text (OCR summary empty), do NOT use ocr_click.

## Action Types

### coordinate_click - Vision click via the coordinate finder pipeline (REQUIRED for every UI click)
Use this action whenever you want to click a visible UI element (buttons, links, icons, thumbnails, dialogs, menu entries, etc.).
The agent runs the same coordinate finder pipeline as coordinate_finder_test.py to locate the most precise click point.
Be extremely descriptive in `target`, referencing neighboring icons/text/buttons and the position (top/center/left, etc.). Describe colors, labels, and positional cues so the coordinate pipeline can resolve the exact button.
- `prefer_primary`: true for the main/affirmative action (Accept, OK, Confirm); false only if you explicitly need a secondary option.
- `button`: "left" by default, "right" for context menus, "middle" rarely.
- `clicks`: 2 for double-clicking files/folders, 1 for everything else.
- `description`: short human-readable summary of the action (agent output log will show this).

Example:
{"action_type": "coordinate_click", "target": "The blue 'Accept All cookies' button at the bottom-right of the dialog", "prefer_primary": true, "description": "Accept cookies"}

Examples of other coordinate clicks:
- Cookie popup: {"action_type": "coordinate_click", "target": "Primary 'Accept All' button in the cookie dialog (bottom-right, blue)", "prefer_primary": true, "description": "Accept cookies"}
- Search bar: {"action_type": "coordinate_click", "target": "YouTube search bar at the top-center of the page, just below the navigation tabs", "description": "Focus the search field"}
- Close dialog: {"action_type": "coordinate_click", "target": "Small 'X' close button in the top-right corner of the popup", "description": "Close the dialog"}
- Submit form: {"action_type": "coordinate_click", "target": "The primary 'Submit' button with green text on the lower right", "prefer_primary": true, "description": "Submit the form"}
- Double-click file: {"action_type": "coordinate_click", "target": "Chrome icon on the desktop (double-click anywhere on the icon)", "clicks": 2, "description": "Launch Chrome"}
- Right-click: {"action_type": "coordinate_click", "target": "The ribbon area labeled 'File' (for the context menu)", "button": "right", "description": "Open context menu"}

IMPORTANT for coordinate_click:
- Be SPECIFIC and DESCRIPTIVE about what to click and how it sits among other UI elements.
- Mention relative position ("top-right", "left of the address bar", "below the header text") and colors/labels when helpful.
- Use `prefer_primary=true` when clicking affirmation/primary actions such as Accept, Confirm, Save, Next.
- If the UI element appears multiple times, describe exactly which one you need (e.g., "the first one", "the one with the green border").
- NEVER emit any other click action for UI buttons; coordinate_click is the only allowed click output for those cases.

### chain - PREFERRED for keyboard sequences
Chain multiple actions with delays:
```json
{
  "action_type": "chain",
  "steps": [
    {"action_type": "hotkey", "keys": ["win"], "delay_after": 0.5},
    {"action_type": "type", "text": "Calculator", "delay_after": 0.3},
    {"action_type": "keypress", "keys": ["enter"], "delay_after": 0.1}
  ],
  "description": "Open Calculator via Start menu"
}
```

**IMPORTANT: Use specific app names in Windows search:**
- ? "Edge" or "Microsoft Edge" (NOT "webbl?sare" or "browser")
- ? "Chrome" or "Google Chrome" (NOT "webbl?sare")
- ? "Calculator" (NOT "kalkylator" or "calculator app")
- ? "Spotify", "Discord", "Excel", "Word" (use actual app names)
```

For coordinate clicks inside chains, provide the target description in the `text` or `target` field:
```json
{
  "action_type": "chain",
  "steps": [
    {"action_type": "coordinate_click", "text": "YouTube search bar at the top", "delay_after": 0.3},
    {"action_type": "type", "text": "cat video", "delay_after": 0.5},
    {"action_type": "keypress", "keys": ["enter"], "delay_after": 0.3}
  ],
  "description": "Search for cat video"
}
```

IMPORTANT for chain:
- Use for keyboard sequences (hotkey, type, keypress)
- Can include coordinate_click, ocr_click, wait, click actions too
- Provide detailed `text`/`target` when using coordinate_click steps
- Be SPECIFIC: "YouTube search bar" is better than "search bar", "the blue Accept All button" is better than "accept"
- For click/mouse_click: provide x and y coordinates
- Each step has a delay_after (seconds) for UI to respond
- More efficient than separate steps

### hotkey - Press key combination
{"action_type": "hotkey", "keys": ["ctrl", "c"], "description": "Copy to clipboard"}

### keypress - Press keys in sequence  
{"action_type": "keypress", "keys": ["enter"], "description": "Press Enter"}

### type - Type text
{"action_type": "type", "text": "hello", "description": "Type text"}

IMPORTANT for type:
- When typing in Windows Start menu search: Use SPECIFIC app names (e.g., "Edge", "Chrome", "Calculator", "Spotify")
- DO NOT use generic terms like "webbl?sare" (browser), "kalkylator" (calculator) - use the actual app name!
- When typing in web search bars: Use the search query (e.g., "cat video", "how to bake a cake")
- When typing URLs: Use full URL (e.g., "www.youtube.com", "https://google.com")
- Windows search works best with actual application names - look at the screenshot to see installed apps

### mouse_click - Click at exact coordinates (ONLY if you know the coords)
{"action_type": "mouse_click", "x": 100, "y": 200, "button": "left", "clicks": 1, "description": "Click at known position"}

### scroll - Scroll the screen
{"action_type": "scroll", "amount": -3, "description": "Scroll down"}

### wait - Wait for time or condition
{"action_type": "wait", "seconds": 2, "description": "Wait for page to load"}

### done - Task complete
{"action_type": "done", "summary": "Successfully opened Calculator"}

### fail - Cannot complete
{"action_type": "fail", "reason": "Application not found"}

## Response Format
Respond with ONLY valid JSON:
```json
{
  "thought": "Your reasoning here",
  "action": { ... action object ... },
  "success_criteria": "What to check"
}
```

## Common Patterns

Opening apps:
- Use chain: Win key -> type app name -> Enter

Clicking buttons (cookie popups, dialogs, forms):
- Use coordinate_click with specific description
- Set prefer_primary=true for main actions (Accept, OK, Submit, Confirm, Play)

File operations:
- Ctrl+O to open, Ctrl+S to save, Ctrl+N for new

## Be Precise
- Look carefully at the screenshot before acting
- For buttons, use coordinate_click with detailed description
- If there are multiple similar buttons, describe the one you want (color, position, text)
- If something failed, try an alternative approach
"""


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
        user_message = f"""Current State:
{state.to_context_string()}

Analyze the screenshot and decide the next action to complete the task.
Respond with a JSON object containing: thought, action, and success_criteria.

IMPORTANT: 
- Use chain actions for sequences (e.g., open menu + type + enter)
- UI text may be in {state.system_language}
- Be fast and efficient"""
        
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
        
        prompt = f"""Verify if an action succeeded.

Task: {state.task}
Last action: {state.last_action}
Success criteria: {criteria}
System language: {state.system_language}

Look at the screenshot and determine if the success criteria has been met.
Respond with ONLY "yes" or "no"."""
        
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
