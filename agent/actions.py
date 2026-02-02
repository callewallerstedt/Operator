"""
Action schemas and types for the AI Agent.
Defines all possible actions the agent can take, including chained actions.
"""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum


class ActionType(str, Enum):
    """Types of actions the agent can perform."""
    KEYPRESS = "keypress"
    TYPE = "type"
    HOTKEY = "hotkey"
    OCR_CLICK = "ocr_click"
    COORDINATE_CLICK = "coordinate_click"
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_SCROLL = "scroll"
    MOUSE_DRAG = "drag"
    WAIT = "wait"
    VISION_CLICK = "vision_click"
    CHAIN = "chain"
    DONE = "done"
    FAIL = "fail"


class KeypressAction(BaseModel):
    """Press one or more keys in sequence."""
    action_type: Literal["keypress"] = "keypress"
    keys: List[str] = Field(description="List of keys to press in sequence")
    description: str = Field(description="Human-readable description of what this does")


class TypeAction(BaseModel):
    """Type a string of text."""
    action_type: Literal["type"] = "type"
    text: str = Field(description="Text to type")
    use_clipboard: bool = Field(
        default=False,
        description="Use clipboard paste for special characters"
    )
    description: str = Field(description="Human-readable description")


class HotkeyAction(BaseModel):
    """Press a keyboard shortcut combination."""
    action_type: Literal["hotkey"] = "hotkey"
    keys: List[str] = Field(
        description="Keys to press simultaneously (e.g., ['ctrl', 'c'])"
    )
    description: str = Field(description="Human-readable description")


class OCRClickAction(BaseModel):
    """Find text on screen using OCR and click on it."""
    action_type: Literal["ocr_click"] = "ocr_click"
    text: str = Field(description="Text to find and click")
    occurrence: int = Field(
        default=1,
        description="Which occurrence to click (1-based)"
    )
    region: Literal["full", "active_window"] = Field(
        default="full",
        description="Screen region to search"
    )
    exact: bool = Field(
        default=False,
        description="Require exact text match"
    )
    button: Literal["left", "right", "middle"] = Field(
        default="left",
        description="Mouse button to use"
    )
    description: str = Field(description="Human-readable description")


class MouseClickAction(BaseModel):
    """Click at specific coordinates."""
    action_type: Literal["mouse_click"] = "mouse_click"
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")
    button: Literal["left", "right", "middle"] = Field(default="left")
    clicks: int = Field(default=1, description="Number of clicks (1 or 2)")
    description: str = Field(description="Human-readable description")


class MouseMoveAction(BaseModel):
    """Move mouse to coordinates."""
    action_type: Literal["mouse_move"] = "mouse_move"
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")
    description: str = Field(description="Human-readable description")


class ScrollAction(BaseModel):
    """Scroll the mouse wheel."""
    action_type: Literal["scroll"] = "scroll"
    amount: int = Field(description="Scroll amount (positive=up, negative=down)")
    x: Optional[int] = Field(default=None, description="X coordinate to scroll at")
    y: Optional[int] = Field(default=None, description="Y coordinate to scroll at")
    description: str = Field(description="Human-readable description")


class DragAction(BaseModel):
    """Drag from one position to another."""
    action_type: Literal["drag"] = "drag"
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    description: str = Field(description="Human-readable description")


class WaitAction(BaseModel):
    """Wait for a condition or time."""
    action_type: Literal["wait"] = "wait"
    seconds: float = Field(default=1.0, description="Seconds to wait")
    condition: Optional[str] = Field(
        default=None,
        description="Optional condition to wait for (OCR text to appear)"
    )
    description: str = Field(description="Human-readable description")


class CoordinateClickAction(BaseModel):
    """
    Uses the coordinate-finder pipeline to locate and click UI elements.
    This is the ONLY click action the agent should emit.
    
    Supports:
    - Single click (default)
    - Double click (clicks=2) for opening files/folders
    - Right click (button="right") for context menus
    """
    action_type: Literal["coordinate_click"] = "coordinate_click"
    target: str = Field(
        description="Detailed description of what to click (e.g., 'the blue Accept All button')"
    )
    prefer_primary: bool = Field(
        default=True,
        description="Prefer the main/primary action when multiple candidates exist"
    )
    button: Literal["left", "right", "middle"] = Field(default="left")
    clicks: int = Field(
        default=1,
        description="Number of clicks: 1 for a single click, 2 for a double click"
    )
    description: str = Field(description="Human-readable description")


class VisionClickAction(BaseModel):
    """
    Legacy vision click - use coordinate_click instead.
    """
    action_type: Literal["vision_click"] = "vision_click"
    goal: str = Field(
        description="Description of what to click (e.g., 'the red X button')"
    )
    button: Literal["left", "right", "middle"] = Field(default="left")
    description: str = Field(description="Human-readable description")


class DoneAction(BaseModel):
    """Signal that the task is complete."""
    action_type: Literal["done"] = "done"
    summary: str = Field(description="Summary of what was accomplished")


class FailAction(BaseModel):
    """Signal that the task cannot be completed."""
    action_type: Literal["fail"] = "fail"
    reason: str = Field(description="Reason for failure")


# Simple action types for chaining (without nested model complexity)
class ChainedStep(BaseModel):
    """A single step in a chain of actions."""
    action_type: str = Field(description="Type of action: hotkey, type, keypress, wait, coordinate_click, click, ocr_click")
    keys: Optional[List[str]] = Field(default=None, description="Keys for hotkey/keypress")
    text: Optional[str] = Field(default=None, description="Text for type/ocr_click action, or target description for coordinate_click")
    target: Optional[str] = Field(default=None, description="Target description for coordinate_click (alternative to text)")
    x: Optional[int] = Field(default=None, description="X coordinate for click/mouse actions")
    y: Optional[int] = Field(default=None, description="Y coordinate for click/mouse actions")
    button: Optional[str] = Field(default="left", description="Mouse button: left, right, middle")
    clicks: Optional[int] = Field(default=1, description="Number of clicks: 1 for single, 2 for double click")
    seconds: Optional[float] = Field(default=None, description="Seconds for wait")
    delay_after: float = Field(default=0.3, description="Delay after this step in seconds")


class ChainAction(BaseModel):
    """
    Execute multiple actions in sequence with delays between them.
    This is more efficient than separate steps for common sequences like:
    - Open start menu and type app name
    - Ctrl+O then type file path
    """
    action_type: Literal["chain"] = "chain"
    steps: List[ChainedStep] = Field(
        description="List of steps to execute in sequence"
    )
    description: str = Field(description="Human-readable description of the full chain")


# Union type for all possible actions
AgentAction = Union[
    KeypressAction,
    TypeAction,
    HotkeyAction,
    OCRClickAction,
    CoordinateClickAction,
    MouseClickAction,
    MouseMoveAction,
    ScrollAction,
    DragAction,
    WaitAction,
    VisionClickAction,
    ChainAction,
    DoneAction,
    FailAction,
]


class PlannerResponse(BaseModel):
    """Response from the planner LLM."""
    thought: str = Field(
        description="Step-by-step reasoning about the current state and what to do"
    )
    action: AgentAction = Field(description="The action to take")
    success_criteria: Optional[str] = Field(
        default=None,
        description="How to verify this action succeeded (OCR text, window title, etc.)"
    )


def parse_action_from_dict(data: dict) -> AgentAction:
    """Parse an action from a dictionary."""
    action_type = data.get("action_type")
    
    action_classes = {
        "keypress": KeypressAction,
        "type": TypeAction,
        "hotkey": HotkeyAction,
        "ocr_click": OCRClickAction,
        "coordinate_click": CoordinateClickAction,
        "mouse_click": MouseClickAction,
        "mouse_move": MouseMoveAction,
        "scroll": ScrollAction,
        "drag": DragAction,
        "wait": WaitAction,
        "vision_click": VisionClickAction,
        "chain": ChainAction,
        "done": DoneAction,
        "fail": FailAction,
    }
    
    action_class = action_classes.get(action_type)
    if action_class is None:
        raise ValueError(f"Unknown action type: {action_type}")
    
    return action_class(**data)
