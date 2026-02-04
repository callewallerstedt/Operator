"""
Main Agent Loop - Orchestrates the entire agent workflow.
Captures screenshots, calls planner, executes actions, and verifies results.
"""

import time
import json
import re
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from .config import config
from .screenshot import get_screen_capture
from .screenshot_viewer import get_viewer
from .ocr import get_ocr_engine
from .executors import (
    get_keyboard, get_mouse, get_ocr_click,
    ActionResult, MouseButton
)
from .planner import get_planner, AgentState
from .click_pipeline import get_click_pipeline
from .actions import (
    ActionType, KeypressAction, TypeAction, HotkeyAction,
    OCRClickAction, CoordinateClickAction, CoordinateDoubleClickAction, CoordinateRightClickAction,
    MouseClickAction, MouseMoveAction,
    ScrollAction, DragAction, WaitAction, VisionClickAction,
    ChainAction, ChainedStep,
    DoneAction, FailAction, AgentAction
)
from .utils import get_active_window_info


console = Console()


class AgentStatus(Enum):
    """Status of the agent."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class AgentResult:
    """Final result of an agent run."""
    status: AgentStatus
    task: str
    steps_taken: int
    duration: float
    final_message: str
    history: list = field(default_factory=list)


class AgentLoop:
    """
    Main agent loop that orchestrates task completion.
    
    The loop:
    1. Captures a screenshot
    2. Builds state context (OCR, window info)
    3. Sends to planner LLM
    4. Executes the planned action
    5. Verifies the outcome
    6. Repeats until done or failed
    """
    
    def __init__(
        self,
        show_viewer: bool = True,
        monitor_offset: Optional[Tuple[int, int]] = None,
        message_log_path: Optional[str] = None,
        session_id: str = "",
    ):
        self.screen = get_screen_capture()
        self.ocr = get_ocr_engine()
        self.keyboard = get_keyboard()
        self.mouse = get_mouse()
        self.ocr_click = get_ocr_click()
        self.planner = get_planner()
        self.click_pipeline = None
        self.viewer = get_viewer() if show_viewer else None
        
        # Monitor offset for coordinate conversion
        # When capturing a single monitor, screenshot coords are relative to monitor (0,0 at monitor top-left)
        # This offset converts them to absolute screen coordinates
        self.monitor_offset = monitor_offset or (0, 0)
        
        self._stop_requested = False
        self._on_step_callback: Optional[Callable] = None
        self._on_image_callback: Optional[Callable] = None  # Real-time image callback for coordinate_click
        self._on_status_callback: Optional[Callable] = None  # Real-time status callback
        self._on_plan_callback: Optional[Callable] = None  # Callback after planning, before action
        self._on_screenshot_callback: Optional[Callable] = None  # Callback right after screenshot capture
        self._on_assist_callback: Optional[Callable] = None  # Callback when operator help is needed

        self._message_log_path = Path(message_log_path) if message_log_path else None
        self._message_log_pos = 0
        self._session_id = session_id or ""
        self._last_assist_step = 0
        
        # Start the viewer if enabled
        if self.viewer and show_viewer:
            self.viewer.start()
            console.print("[dim]Screenshot viewer window opened - you can move it to another monitor[/dim]")
    
    def run(
        self,
        task: str,
        on_step: Optional[Callable[[int, AgentAction, ActionResult], None]] = None,
        system_language: str = "English"
    ) -> AgentResult:
        """
        Run the agent loop to complete a task.
        
        Args:
            task: The task description
            on_step: Optional callback called after each step
        
        Returns:
            AgentResult with the final status
        """
        self._stop_requested = False
        self._on_step_callback = on_step
        
        start_time = time.time()
        state = AgentState(task=task, goal=task, system_language=system_language)
        
        console.print(Panel(
            f"[bold cyan]Starting task:[/bold cyan] {task}",
            title="AI AGENT",
            border_style="cyan"
        ))
        
        while state.step_number < config.max_steps and not self._stop_requested:
            state.step_number += 1
            
            try:
                result = self._execute_step(state)
                
                if result.status in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.STOPPED):
                    duration = time.time() - start_time
                    return AgentResult(
                        status=result.status,
                        task=task,
                        steps_taken=state.step_number,
                        duration=duration,
                        final_message=result.final_message,
                        history=state.history
                    )
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Agent stopped by user[/yellow]")
                return AgentResult(
                    status=AgentStatus.STOPPED,
                    task=task,
                    steps_taken=state.step_number,
                    duration=time.time() - start_time,
                    final_message="Stopped by user",
                    history=state.history
                )
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                state.last_action_success = False
                state.last_action_result = str(e)
            
            # Small delay between iterations
            time.sleep(config.loop_delay)
        
        if self._stop_requested:
            duration = time.time() - start_time
            return AgentResult(
                status=AgentStatus.STOPPED,
                task=task,
                steps_taken=state.step_number,
                duration=duration,
                final_message="Stopped by operator",
                history=state.history
            )

        # Max steps reached
        duration = time.time() - start_time
        return AgentResult(
            status=AgentStatus.FAILED,
            task=task,
            steps_taken=state.step_number,
            duration=duration,
            final_message=f"Max steps ({config.max_steps}) reached without completion",
            history=state.history
        )
    
    def stop(self):
        """Request the agent to stop."""
        self._stop_requested = True
    
    def _execute_step(self, state: AgentState) -> AgentResult:
        """Execute a single step of the agent loop."""
        # Status update
        if self._on_status_callback:
            self._on_status_callback(f"Step {state.step_number}: Capturing screenshot...")
        
        # 1. Capture screenshot (full screen)
        console.print(f"\n[dim]Step {state.step_number}[/dim]")
        screenshot = self.screen.capture_full()
        if self._on_screenshot_callback:
            try:
                self._on_screenshot_callback(state.step_number, screenshot)
            except Exception:
                pass
        
        # Update viewer with screenshot
        if self.viewer:
            info = f"Step {state.step_number} - {state.task}"
            self.viewer.update_screenshot(screenshot, info)
        
        # 2. Update state with context
        window_title, _ = get_active_window_info()
        state.active_window_title = window_title
        
        # Get OCR summary for context
        try:
            ocr_result = self.ocr.process(screenshot)
            state.ocr_summary = ocr_result.raw_text[:500]
        except Exception:
            state.ocr_summary = None

        # Consume operator messages (e.g., from Discord) before planning
        operator_updates, stop_requested, operator_clicked = self._read_operator_messages()
        if stop_requested:
            self._stop_requested = True
            return AgentResult(
                status=AgentStatus.STOPPED,
                task=state.task,
                steps_taken=state.step_number,
                duration=0,
                final_message="Stopped by operator",
                history=state.history,
            )
        if operator_updates:
            state.operator_updates.extend(operator_updates)
            # Keep only the last 10 updates to limit context size
            if len(state.operator_updates) > 10:
                state.operator_updates = state.operator_updates[-10:]
        if operator_clicked:
            # Refresh context after operator click.
            screenshot = self.screen.capture_full()
            if self.viewer:
                info = f"Step {state.step_number} - Operator click"
                self.viewer.update_screenshot(screenshot, info)
            window_title, _ = get_active_window_info()
            state.active_window_title = window_title
            try:
                ocr_result = self.ocr.process(screenshot)
                state.ocr_summary = ocr_result.raw_text[:500]
            except Exception:
                state.ocr_summary = None

        # 3. Call planner
        console.print("[dim]Planning...[/dim]")
        if self._on_status_callback:
            self._on_status_callback(
                f"Step {state.step_number}: Planning next action ({config.openai_model})..."
            )
        plan = self.planner.plan_next_action(screenshot, state)
        if self._on_plan_callback:
            try:
                self._on_plan_callback(state.step_number, plan)
            except Exception:
                pass
        
        # Display thought
        console.print(Panel(
            plan.thought,
            title="THINKING",
            border_style="blue"
        ))
        
        action = plan.action
        
        # 4. Check for terminal actions
        if isinstance(action, DoneAction):
            console.print(Panel(
                f"[green]TASK COMPLETE: {action.summary}[/green]",
                title="Task Complete",
                border_style="green"
            ))
            return AgentResult(
                status=AgentStatus.COMPLETED,
                task=state.task,
                steps_taken=state.step_number,
                duration=0,
                final_message=action.summary,
                history=state.history
            )
        
        if isinstance(action, FailAction):
            console.print(Panel(
                f"[red]TASK FAILED: {action.reason}[/red]",
                title="Task Failed",
                border_style="red"
            ))
            return AgentResult(
                status=AgentStatus.FAILED,
                task=state.task,
                steps_taken=state.step_number,
                duration=0,
                final_message=action.reason,
                history=state.history
            )
        
        # 5. Execute action
        if self._on_status_callback:
            if isinstance(action, CoordinateClickAction):
                self._on_status_callback(f"Step {state.step_number}: Executing {action.action_type} - '{action.target}'")
            else:
                self._on_status_callback(f"Step {state.step_number}: Executing {action.action_type}...")
        console.print(f"[yellow]EXECUTING: {action.description}[/yellow]")
        result = self._execute_action(action, screenshot)
        
        # Wait for UI to respond after action
        time.sleep(config.post_action_delay)

        # Additional wait after app launches or heavy transitions
        extra_wait = self._extra_wait_after_action(action, result)
        if extra_wait > 0:
            if self._on_status_callback:
                self._on_status_callback(f"Waiting {extra_wait:.1f}s for app to load...")
            time.sleep(extra_wait)
        
        # 6. Update state
        state.last_action = action.description
        state.last_action_success = result.success
        state.last_action_result = result.message
        state.add_to_history(f"{action.description} -> {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            state.last_failed_action = action.description
            state.failure_counts[action.description] = state.failure_counts.get(action.description, 0) + 1
        
        if result.success:
            console.print(f"[green]SUCCESS: {result.message}[/green]")
        else:
            console.print(f"[red]FAILED: {result.message}[/red]")

        self._maybe_request_assist(state, screenshot, action, result)
        
        # Note: No explicit verification step - the planner will see the next screenshot
        # and can determine from context if the action succeeded or not
        
        # Callback with full AI output
        if self._on_step_callback:
            # Pass thought and plan details to callback
            # Also pass debug images if available (from coordinate_click)
            debug_image = None
            if hasattr(result, 'data') and result.data and 'debug_images' in result.data:
                debug_images = result.data['debug_images']
                if debug_images:
                    debug_image = debug_images[-1]  # Pass the last/most relevant image
            
            try:
                self._on_step_callback(state.step_number, action, result, plan.thought, plan, debug_image)
            except TypeError:
                # Callback doesn't accept debug_image parameter, use old signature
                self._on_step_callback(state.step_number, action, result, plan.thought, plan)
        
        # Return running status (continue loop)
        return AgentResult(
            status=AgentStatus.RUNNING,
            task=state.task,
            steps_taken=state.step_number,
            duration=0,
            final_message="",
            history=state.history
        )

    def _read_operator_messages(self) -> Tuple[List[str], bool, bool]:
        if not self._message_log_path:
            return [], False, False
        path = self._message_log_path
        if not path.exists():
            return [], False, False
        try:
            with path.open("r", encoding="utf-8") as f:
                f.seek(self._message_log_pos)
                lines = f.readlines()
                self._message_log_pos = f.tell()
        except Exception:
            return [], False, False
        updates: List[str] = []
        stop_requested = False
        operator_clicked = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if self._session_id and payload.get("session_id") and payload.get("session_id") != self._session_id:
                continue
            if payload.get("type") == "operator_command":
                command = (payload.get("command") or "").strip()
                if command == "stop":
                    stop_requested = True
                    continue
                click = self._parse_click_command(command)
                if click:
                    x, y = click
                    if self._on_status_callback:
                        self._on_status_callback(f"Operator click received: ({x}, {y})")
                    result = self.mouse.click(x, y, button=MouseButton.LEFT)
                    operator_clicked = True
                    updates.append(f"Human override: clicked at ({x}, {y}) -> {result.message}")
                continue
            if payload.get("type") != "operator_message":
                continue
            author = payload.get("author") or "human"
            content = (payload.get("content") or "").strip()
            if not content:
                continue
            timestamp = payload.get("timestamp")
            source = payload.get("source") or "discord"
            prefix = f"Human update ({source}"
            if timestamp:
                prefix += f", {timestamp}"
            prefix += f", {author})"
            updates.append(f"{prefix}: {content}")
        return updates, stop_requested, operator_clicked

    def _parse_click_command(self, command: str) -> Optional[Tuple[int, int]]:
        if not command:
            return None
        raw = command.strip().lower()
        if raw.startswith("click"):
            raw = raw[5:].strip()
        if raw.startswith(":"):
            raw = raw[1:].strip()
        if not raw:
            return None
        # Accept "x y" or "x,y"
        match = re.match(r"(-?\d+)\s*[,\s]\s*(-?\d+)", raw)
        if not match:
            return None
        try:
            x = int(match.group(1))
            y = int(match.group(2))
        except ValueError:
            return None
        return x, y

    def _maybe_request_assist(self, state: AgentState, screenshot: Image.Image, action: AgentAction, result: ActionResult) -> None:
        if not config.assist_enabled:
            return
        if result.success:
            return
        if state.consecutive_failures < config.assist_after_failures:
            return
        if state.step_number - self._last_assist_step < config.assist_cooldown_steps:
            return
        self._last_assist_step = state.step_number
        if self._on_assist_callback:
            try:
                reason = result.message or "Action failed"
                action_desc = getattr(action, "description", str(action))
                self._on_assist_callback(state.step_number, screenshot, reason, action_desc)
            except Exception:
                pass

    def _extra_wait_after_action(self, action: AgentAction, result: ActionResult) -> float:
        if not result.success:
            return 0.0
        # Don't stack extra waits after explicit wait actions
        if isinstance(action, WaitAction):
            return 0.0
        wait_seconds = getattr(config, "app_launch_wait_seconds", 0.0) or 0.0
        if wait_seconds <= 0:
            return 0.0
        if isinstance(action, CoordinateDoubleClickAction):
            return wait_seconds
        if isinstance(action, CoordinateClickAction) and getattr(action, "clicks", 1) >= 2:
            return wait_seconds
        if isinstance(action, MouseClickAction) and getattr(action, "clicks", 1) >= 2:
            return wait_seconds
        if isinstance(action, ChainAction):
            if any(step.action_type == "wait" and (step.seconds or 0) >= 2 for step in action.steps):
                return 0.0
            if self._chain_looks_like_app_launch(action.steps):
                return wait_seconds
        desc = (getattr(action, "description", "") or "").lower()
        if any(word in desc for word in ("open", "launch", "start", "run", "load app", "open app", "open application")):
            return wait_seconds
        return 0.0

    def _chain_looks_like_app_launch(self, steps: List[ChainedStep]) -> bool:
        has_win = False
        has_type = False
        has_enter = False
        for step in steps:
            if step.action_type == "hotkey" and step.keys:
                if any(k.lower() in ("win", "windows") for k in step.keys):
                    has_win = True
            if step.action_type == "type" and (step.text or "").strip():
                has_type = True
            if step.action_type in ("keypress", "hotkey") and step.keys:
                if any(k.lower() in ("enter", "return") for k in step.keys):
                    has_enter = True
        if has_win and has_type and has_enter:
            return True
        for step in steps:
            if step.action_type == "coordinate_double_click":
                return True
            if step.action_type == "coordinate_click" and (step.clicks or 1) >= 2:
                return True
        return False
    
    def _text_needs_clipboard(self, text: str) -> bool:
        return any(ord(ch) > 127 for ch in text)

    def _type_text(self, text: str, use_clipboard: bool = False) -> ActionResult:
        if use_clipboard or self._text_needs_clipboard(text):
            return self.keyboard.type_text_safe(text)
        return self.keyboard.type_text(text)

    def _resolve_scroll_amount(self, action: ScrollAction, screenshot: Image.Image) -> int:
        """Return the scroll amount, converting percent-based requests into absolute units."""
        amount = action.amount
        if action.percent is not None and screenshot is not None:
            height = max(1, screenshot.height or 1)
            percent = float(action.percent)
            amount = int(round(height * (percent / 100.0)))
        return amount

    def _execute_action(
        self,
        action: AgentAction,
        screenshot: Image.Image
    ) -> ActionResult:
        """Execute a specific action."""
        
        if isinstance(action, KeypressAction):
            return self.keyboard.press_keys(action.keys)
        
        elif isinstance(action, TypeAction):
            return self._type_text(action.text, action.use_clipboard)
        
        elif isinstance(action, HotkeyAction):
            return self.keyboard.hotkey(*action.keys)
        
        elif isinstance(action, OCRClickAction):
            button = MouseButton(action.button)
            return self.ocr_click.click_text(
                text=action.text,
                occurrence=action.occurrence,
                region=action.region,
                exact=action.exact,
                button=button
            )
        
        elif isinstance(action, MouseClickAction):
            button = MouseButton(action.button)
            return self.mouse.click(
                action.x,
                action.y,
                button=button,
                clicks=action.clicks
            )
        
        elif isinstance(action, MouseMoveAction):
            return self.mouse.move_to(action.x, action.y)
        
        elif isinstance(action, ScrollAction):
            amount = self._resolve_scroll_amount(action, screenshot)
            return self.mouse.scroll(amount, action.x, action.y)
        
        elif isinstance(action, DragAction):
            return self.mouse.drag(
                action.start_x,
                action.start_y,
                action.end_x,
                action.end_y
            )
        
        elif isinstance(action, WaitAction):
            time.sleep(action.seconds)
            if action.condition:
                # Check condition via OCR
                found, _, _ = self.ocr_click.find_text(action.condition)
                if found:
                    return ActionResult(True, f"Waited {action.seconds}s, condition met")
                return ActionResult(False, f"Waited {action.seconds}s, condition not met")
            return ActionResult(True, f"Waited {action.seconds} seconds")
        
        elif isinstance(action, (CoordinateClickAction, CoordinateDoubleClickAction, CoordinateRightClickAction)):
            if self._on_status_callback:
                self._on_status_callback(f"Coordinate click: Looking for '{action.target}'...")

            pipeline_point = None
            try:
                pipeline = self.click_pipeline or get_click_pipeline()
                self.click_pipeline = pipeline
                pipeline._screen_size = screenshot.size

                def _pipeline_log(msg, tag):
                    if tag in ("error", "success") and self._on_status_callback:
                        self._on_status_callback(f"Click pipeline: {msg}")

                pipeline.set_log_callback(_pipeline_log)
                pipeline_point = pipeline.find_click(screenshot, action.target or "")
            except Exception as exc:
                if self._on_status_callback:
                    self._on_status_callback(f"Click pipeline error: {exc}")
                pipeline_point = None

            clicks = getattr(action, "clicks", 1)
            button_value = getattr(action, "button", "left")
            if isinstance(action, CoordinateDoubleClickAction):
                clicks = 2
            elif isinstance(action, CoordinateRightClickAction):
                button_value = "right"

            if pipeline_point:
                abs_x = pipeline_point[0] + self.monitor_offset[0]
                abs_y = pipeline_point[1] + self.monitor_offset[1]
                click_type = "Double-clicking" if clicks == 2 else ("Right-clicking" if button_value == "right" else "Clicking")
                if self._on_status_callback:
                    self._on_status_callback(f"{click_type} at ({abs_x}, {abs_y})...")
                button = MouseButton(button_value)
                click_result = self.mouse.click(abs_x, abs_y, button=button, clicks=clicks)
                if click_result.success:
                    click_desc = "Double-clicked" if clicks == 2 else ("Right-clicked" if button_value == "right" else "Clicked")
                    return ActionResult(
                        True,
                        f"{click_desc} on '{action.target}' at ({abs_x}, {abs_y})",
                        screenshot=screenshot,
                    )
                return click_result

            return ActionResult(
                False,
                f"Coordinate click failed: coordinate-finder pipeline could not find '{action.target}'",
                screenshot=screenshot,
            )
        
        elif isinstance(action, VisionClickAction):
            # Route legacy vision click through coordinate-finder pipeline.
            pipeline_point = None
            try:
                pipeline = self.click_pipeline or get_click_pipeline()
                self.click_pipeline = pipeline
                pipeline._screen_size = screenshot.size
                pipeline_point = pipeline.find_click(screenshot, action.goal or "")
            except Exception as exc:
                if self._on_status_callback:
                    self._on_status_callback(f"Click pipeline error: {exc}")
                pipeline_point = None

            if pipeline_point:
                abs_x = pipeline_point[0] + self.monitor_offset[0]
                abs_y = pipeline_point[1] + self.monitor_offset[1]
                button = MouseButton(action.button)
                return self.mouse.click(abs_x, abs_y, button=button)
            return ActionResult(False, f"Coordinate-finder pipeline could not find: {action.goal}")
        
        elif isinstance(action, ChainAction):
            # Execute chain of actions with delays
            return self._execute_chain(action)
        
        return ActionResult(False, f"Unknown action type: {type(action)}")
    
    def _execute_chain(self, chain: ChainAction) -> ActionResult:
        """Execute a chain of actions with delays between them."""
        results = []
        
        # Capture screenshot for OCR operations
        screenshot = self.screen.capture_full()
        
        for i, step in enumerate(chain.steps):
            try:
                step_result = None
                
                if step.action_type == "hotkey" and step.keys:
                    step_result = self.keyboard.hotkey(*step.keys)
                elif step.action_type == "keypress" and step.keys:
                    step_result = self.keyboard.press_keys(step.keys)
                elif step.action_type == "type" and step.text is not None:
                    step_result = self._type_text(step.text)
                elif step.action_type == "ocr_click" and step.text is not None:
                    # Support OCR click in chains
                    from .executors import MouseButton
                    button = MouseButton.LEFT
                    result = self.ocr_click.click_text(
                        text=step.text,
                        occurrence=1,
                        region="full",
                        exact=False,
                        button=button
                    )
                    step_result = result
                elif step.action_type in ("coordinate_click", "coordinate_double_click", "coordinate_right_click"):
                    # Support coordinate click in chains
                    # Use target field if available, otherwise fall back to text
                    target_description = None
                    if hasattr(step, 'target') and step.target:
                        target_description = step.target
                    elif hasattr(step, 'text') and step.text:
                        target_description = step.text
                    
                    if not target_description:
                        step_result = ActionResult(False, "Coordinate click requires 'target' or 'text' field with description")
                    else:
                        # Refresh screenshot for coordinate click (page may have changed)
                        screenshot = self.screen.capture_full()

                        # Use coordinate-finder pipeline exclusively.
                        pipeline_point = None
                        try:
                            pipeline = self.click_pipeline or get_click_pipeline()
                            self.click_pipeline = pipeline
                            pipeline._screen_size = screenshot.size
                            pipeline_point = pipeline.find_click(screenshot, target_description)
                        except Exception as exc:
                            if self._on_status_callback:
                                self._on_status_callback(f"Click pipeline error: {exc}")
                            pipeline_point = None

                        if pipeline_point:
                            abs_x = pipeline_point[0] + self.monitor_offset[0]
                            abs_y = pipeline_point[1] + self.monitor_offset[1]
                            button_str = getattr(step, 'button', 'left') or 'left'
                            num_clicks = getattr(step, 'clicks', 1) or 1
                            if step.action_type == "coordinate_double_click":
                                num_clicks = 2
                            elif step.action_type == "coordinate_right_click":
                                button_str = "right"
                            click_type = "Double-clicking" if num_clicks == 2 else ("Right-clicking" if button_str == "right" else "Clicking")
                            if self._on_status_callback:
                                self._on_status_callback(f"{click_type} at ({abs_x}, {abs_y})...")
                            from .executors import MouseButton
                            button = MouseButton(button_str)
                            click_result = self.mouse.click(abs_x, abs_y, button=button, clicks=num_clicks)
                            step_result = click_result
                        else:
                            step_result = ActionResult(
                                False,
                                f"Coordinate click failed: coordinate-finder pipeline could not find '{target_description}'"
                            )
                elif step.action_type == "click" or step.action_type == "mouse_click":
                    # Support regular mouse click in chains
                    if step.x is not None and step.y is not None:
                        from .executors import MouseButton
                        click_result = self.mouse.click(step.x, step.y, button=MouseButton.LEFT)
                        step_result = click_result
                    else:
                        step_result = ActionResult(False, "Mouse click requires x and y coordinates")
                elif step.action_type == "wait":
                    wait_time = step.seconds or 1.0
                    time.sleep(wait_time)
                    step_result = ActionResult(True, f"Waited {wait_time}s")
                else:
                    step_result = ActionResult(False, f"Unknown chain step type: {step.action_type}")
                
                if step_result:
                    results.append(step_result)
                    if not step_result.success:
                        return ActionResult(
                            False,
                            f"Chain failed at step {i+1}: {step_result.message}"
                        )
                
                # Delay after step
                if step.delay_after > 0:
                    time.sleep(step.delay_after)
                
                # Update screenshot after each step for next OCR operations
                if i < len(chain.steps) - 1:  # Not the last step
                    screenshot = self.screen.capture_full()
                    
            except Exception as e:
                return ActionResult(False, f"Chain error at step {i+1}: {str(e)}")
        
        return ActionResult(True, f"Chain completed: {len(chain.steps)} steps")


def run_agent(task: str) -> AgentResult:
    """Convenience function to run the agent with a task."""
    agent = AgentLoop()
    return agent.run(task)
