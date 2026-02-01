"""
Main Agent Loop - Orchestrates the entire agent workflow.
Captures screenshots, calls planner, executes actions, and verifies results.
"""

import time
from typing import Optional, Callable, Tuple
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
from .vision_click import get_vision_click
from .actions import (
    ActionType, KeypressAction, TypeAction, HotkeyAction,
    OCRClickAction, SmartClickAction, MouseClickAction, MouseMoveAction,
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
    
    def __init__(self, show_viewer: bool = True, monitor_offset: Optional[Tuple[int, int]] = None):
        self.screen = get_screen_capture()
        self.ocr = get_ocr_engine()
        self.keyboard = get_keyboard()
        self.mouse = get_mouse()
        self.ocr_click = get_ocr_click()
        self.planner = get_planner()
        self.vision_click = get_vision_click()
        self.viewer = get_viewer() if show_viewer else None
        
        # Monitor offset for coordinate conversion
        # When capturing a single monitor, screenshot coords are relative to monitor (0,0 at monitor top-left)
        # This offset converts them to absolute screen coordinates
        self.monitor_offset = monitor_offset or (0, 0)
        
        self._stop_requested = False
        self._on_step_callback: Optional[Callable] = None
        self._on_image_callback: Optional[Callable] = None  # Real-time image callback for smart_click
        self._on_status_callback: Optional[Callable] = None  # Real-time status callback
        
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
        state = AgentState(task=task, system_language=system_language)
        
        console.print(Panel(
            f"[bold cyan]Starting task:[/bold cyan] {task}",
            title="AI AGENT",
            border_style="cyan"
        ))
        
        while state.step_number < config.max_steps and not self._stop_requested:
            state.step_number += 1
            
            try:
                result = self._execute_step(state)
                
                if result.status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
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
        
        # 3. Call planner
        console.print("[dim]Planning...[/dim]")
        if self._on_status_callback:
            self._on_status_callback(f"Step {state.step_number}: Planning next action (GPT-4o)...")
        plan = self.planner.plan_next_action(screenshot, state)
        
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
            # Include target description for smart_click
            if isinstance(action, SmartClickAction):
                self._on_status_callback(f"Step {state.step_number}: Executing {action.action_type} - '{action.target}'")
            else:
                self._on_status_callback(f"Step {state.step_number}: Executing {action.action_type}...")
        console.print(f"[yellow]EXECUTING: {action.description}[/yellow]")
        result = self._execute_action(action, screenshot)
        
        # Wait for UI to respond after action
        time.sleep(config.post_action_delay)
        
        # 6. Update state
        state.last_action = action.description
        state.last_action_success = result.success
        state.last_action_result = result.message
        state.add_to_history(f"{action.description} -> {'SUCCESS' if result.success else 'FAILED'}")
        
        if result.success:
            console.print(f"[green]SUCCESS: {result.message}[/green]")
        else:
            console.print(f"[red]FAILED: {result.message}[/red]")
        
        # Note: No explicit verification step - the planner will see the next screenshot
        # and can determine from context if the action succeeded or not
        
        # Callback with full AI output
        if self._on_step_callback:
            # Pass thought and plan details to callback
            # Also pass debug images if available (from smart_click)
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
    
    def _execute_action(
        self,
        action: AgentAction,
        screenshot: Image.Image
    ) -> ActionResult:
        """Execute a specific action."""
        
        if isinstance(action, KeypressAction):
            return self.keyboard.press_keys(action.keys)
        
        elif isinstance(action, TypeAction):
            if action.use_clipboard:
                return self.keyboard.type_text_safe(action.text)
            return self.keyboard.type_text(action.text)
        
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
            return self.mouse.scroll(action.amount, action.x, action.y)
        
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
        
        elif isinstance(action, SmartClickAction):
            # Use smart vision-based clicking with two-step zoom method
            # AI will click immediately if confident, otherwise zoom then click
            if self._on_status_callback:
                self._on_status_callback(f"Smart Click: Looking for '{action.target}'...")
            
            result = self.vision_click.find_element(
                screenshot,
                description=action.target,
                context=f"Task: {self._current_task}" if hasattr(self, '_current_task') else "",
                prefer_primary=action.prefer_primary,
                return_debug_images=True,  # Always return debug images to show in GUI
                image_callback=self._on_image_callback,  # Real-time image updates
                status_callback=self._on_status_callback  # Real-time status updates
            )
            target, debug_images = result
            
            if target:
                # Convert relative coordinates to absolute screen coordinates
                abs_x = target.x + self.monitor_offset[0]
                abs_y = target.y + self.monitor_offset[1]
                
                # Determine click type for status message
                click_type = "Double-clicking" if action.clicks == 2 else ("Right-clicking" if action.button == "right" else "Clicking")
                if self._on_status_callback:
                    self._on_status_callback(f"{click_type} at ({abs_x}, {abs_y})...")
                
                button = MouseButton(action.button)
                clicks = getattr(action, 'clicks', 1)
                click_result = self.mouse.click(abs_x, abs_y, button=button, clicks=clicks)
                if click_result.success:
                    # Include debug images in result data so GUI can display them
                    result_data = {"debug_images": debug_images} if debug_images else None
                    # Build descriptive message
                    click_desc = "Double-clicked" if clicks == 2 else ("Right-clicked" if action.button == "right" else "Clicked")
                    return ActionResult(
                        True,
                        f"{click_desc} on '{target.description}' at ({abs_x}, {abs_y})",
                        screenshot=screenshot,
                        data=result_data
                    )
                return click_result
            return ActionResult(
                False, 
                f"Smart click failed: could not find '{action.target}'",
                screenshot=screenshot,
                data={"debug_images": debug_images} if debug_images else None
            )
        
        elif isinstance(action, VisionClickAction):
            # Legacy vision click - redirect to smart click
            success, target, message, _ = self.vision_click.find_and_click(
                screenshot,
                description=action.goal,
                prefer_primary=True,
                return_debug_images=False
            )
            if success and target:
                button = MouseButton(action.button)
                return self.mouse.click(target.x, target.y, button=button)
            return ActionResult(False, f"Vision click could not find: {action.goal}")
        
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
                    step_result = self.keyboard.type_text(step.text)
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
                elif step.action_type == "smart_click":
                    # Support smart click in chains
                    # Use target field if available, otherwise fall back to text
                    target_description = None
                    if hasattr(step, 'target') and step.target:
                        target_description = step.target
                    elif hasattr(step, 'text') and step.text:
                        target_description = step.text
                    
                    if not target_description:
                        step_result = ActionResult(False, "Smart click requires 'target' or 'text' field with description")
                    else:
                        # Refresh screenshot for smart_click (page may have changed)
                        screenshot = self.screen.capture_full()
                        
                        # Use the new find_element method with two-step zoom and callbacks
                        result = self.vision_click.find_element(
                            screenshot,
                            description=target_description,
                            context=f"Task: {self._current_task}" if hasattr(self, '_current_task') else "",
                            prefer_primary=True,
                            return_debug_images=True,
                            image_callback=self._on_image_callback,  # Real-time image updates
                            status_callback=self._on_status_callback  # Real-time status updates
                        )
                        target, _ = result
                        
                        if target:
                            # Convert relative coordinates to absolute screen coordinates
                            abs_x = target.x + self.monitor_offset[0]
                            abs_y = target.y + self.monitor_offset[1]
                            
                            # Get click parameters from step
                            button_str = getattr(step, 'button', 'left') or 'left'
                            num_clicks = getattr(step, 'clicks', 1) or 1
                            
                            click_type = "Double-clicking" if num_clicks == 2 else ("Right-clicking" if button_str == "right" else "Clicking")
                            if self._on_status_callback:
                                self._on_status_callback(f"{click_type} at ({abs_x}, {abs_y})...")
                            
                            from .executors import MouseButton
                            button = MouseButton(button_str)
                            click_result = self.mouse.click(abs_x, abs_y, button=button, clicks=num_clicks)
                            step_result = click_result
                        else:
                            step_result = ActionResult(False, f"Smart click failed: could not find '{target_description}'")
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
