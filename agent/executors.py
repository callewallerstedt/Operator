"""
Deterministic executors for keyboard and mouse actions.
These execute actions directly without LLM involvement.
"""

import time
import ctypes
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import pyautogui
from PIL import Image

from .config import config
from .screenshot import ScreenRegion, get_screen_capture
from .ocr import get_ocr_engine, OCRMatch


# Disable pyautogui failsafe (we handle safety ourselves)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    message: str
    screenshot: Optional[Image.Image] = None
    data: Optional[dict] = None


class KeyboardExecutor:
    """Handles keyboard input actions."""
    
    # Common key mappings
    SPECIAL_KEYS = {
        "enter": "enter",
        "return": "enter",
        "tab": "tab",
        "escape": "escape",
        "esc": "escape",
        "backspace": "backspace",
        "delete": "delete",
        "space": "space",
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "home": "home",
        "end": "end",
        "pageup": "pageup",
        "pagedown": "pagedown",
        "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
        "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
        "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
        "ctrl": "ctrl",
        "alt": "alt",
        "shift": "shift",
        "win": "win",
        "windows": "win",
        "command": "command",
        "cmd": "command",
    }
    
    def press_key(self, key: str) -> ActionResult:
        """Press a single key."""
        try:
            key_lower = key.lower()
            actual_key = self.SPECIAL_KEYS.get(key_lower, key)
            pyautogui.press(actual_key)
            time.sleep(config.click_delay)
            return ActionResult(True, f"Pressed key: {key}")
        except Exception as e:
            return ActionResult(False, f"Failed to press key {key}: {str(e)}")
    
    def press_keys(self, keys: List[str]) -> ActionResult:
        """Press multiple keys in sequence."""
        try:
            for key in keys:
                result = self.press_key(key)
                if not result.success:
                    return result
            return ActionResult(True, f"Pressed keys: {', '.join(keys)}")
        except Exception as e:
            return ActionResult(False, f"Failed to press keys: {str(e)}")
    
    def hotkey(self, *keys: str) -> ActionResult:
        """Press a hotkey combination (e.g., Ctrl+C)."""
        try:
            mapped_keys = [self.SPECIAL_KEYS.get(k.lower(), k) for k in keys]
            pyautogui.hotkey(*mapped_keys)
            # Longer delay for hotkeys that open menus (like Win key)
            if any(k.lower() in ('win', 'windows') for k in keys):
                time.sleep(0.5)  # Extra delay for start menu
            else:
                time.sleep(config.click_delay)
            return ActionResult(True, f"Pressed hotkey: {'+'.join(keys)}")
        except Exception as e:
            return ActionResult(False, f"Failed hotkey {'+'.join(keys)}: {str(e)}")
    
    def type_text(self, text: str, interval: float = None) -> ActionResult:
        """Type a string of text."""
        try:
            interval = interval or config.type_delay
            pyautogui.write(text, interval=interval)
            time.sleep(config.click_delay)
            return ActionResult(True, f"Typed text: {text[:50]}...")
        except Exception as e:
            return ActionResult(False, f"Failed to type text: {str(e)}")
    
    def type_text_safe(self, text: str) -> ActionResult:
        """
        Type text using clipboard (handles special characters better).
        Uses Ctrl+V to paste from clipboard.
        """
        try:
            import pyperclip
            # Save current clipboard
            try:
                old_clipboard = pyperclip.paste()
            except Exception:
                old_clipboard = ""
            
            # Copy text to clipboard and paste
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(config.click_delay)
            
            # Restore clipboard
            try:
                pyperclip.copy(old_clipboard)
            except Exception:
                pass
            
            return ActionResult(True, f"Typed text via clipboard: {text[:50]}...")
        except ImportError:
            # Fallback to regular typing if pyperclip not available
            return self.type_text(text)
        except Exception as e:
            return ActionResult(False, f"Failed to type text: {str(e)}")


class MouseExecutor:
    """Handles mouse input actions."""
    
    def move_to(self, x: int, y: int, duration: float = 0.1) -> ActionResult:
        """Move mouse to absolute position."""
        try:
            pyautogui.moveTo(x, y, duration=duration)
            return ActionResult(True, f"Moved mouse to ({x}, {y})")
        except Exception as e:
            return ActionResult(False, f"Failed to move mouse: {str(e)}")
    
    def click(
        self,
        x: int,
        y: int,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1
    ) -> ActionResult:
        """Click at absolute position."""
        try:
            pyautogui.click(x, y, button=button.value, clicks=clicks)
            time.sleep(config.click_delay)
            return ActionResult(True, f"Clicked at ({x}, {y}) with {button.value} button")
        except Exception as e:
            return ActionResult(False, f"Failed to click: {str(e)}")
    
    def double_click(self, x: int, y: int) -> ActionResult:
        """Double-click at position."""
        return self.click(x, y, clicks=2)
    
    def right_click(self, x: int, y: int) -> ActionResult:
        """Right-click at position."""
        return self.click(x, y, button=MouseButton.RIGHT)
    
    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5
    ) -> ActionResult:
        """Drag from start to end position."""
        try:
            pyautogui.moveTo(start_x, start_y)
            pyautogui.drag(
                end_x - start_x,
                end_y - start_y,
                duration=duration,
                button="left"
            )
            time.sleep(config.click_delay)
            return ActionResult(
                True,
                f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            )
        except Exception as e:
            return ActionResult(False, f"Failed to drag: {str(e)}")
    
    def scroll(self, amount: int, x: int = None, y: int = None) -> ActionResult:
        """Scroll at position (positive = up, negative = down)."""
        try:
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
            pyautogui.scroll(amount)
            time.sleep(config.click_delay)
            direction = "up" if amount > 0 else "down"
            return ActionResult(True, f"Scrolled {direction} by {abs(amount)}")
        except Exception as e:
            return ActionResult(False, f"Failed to scroll: {str(e)}")


class OCRClickExecutor:
    """
    Finds text on screen using OCR and clicks on it.
    This is a deterministic executor that uses OCR output directly.
    """
    
    def __init__(self):
        self.ocr = get_ocr_engine()
        self.screen = get_screen_capture()
        self.mouse = MouseExecutor()
    
    def click_text(
        self,
        text: str,
        occurrence: int = 1,
        region: str = "full",
        exact: bool = False,
        button: MouseButton = MouseButton.LEFT
    ) -> ActionResult:
        """
        Find text on screen and click on it.
        
        Args:
            text: Text to find and click
            occurrence: Which occurrence to click (1-based)
            region: "full" for full screen, "active_window" for active window only
            exact: If True, require exact text match
            button: Mouse button to use
        
        Returns:
            ActionResult with success status and message
        """
        try:
            # Capture screenshot
            if region == "active_window":
                result = self.screen.capture_active_window()
                if result:
                    image, screen_region = result
                    offset = (screen_region.left, screen_region.top)
                else:
                    image = self.screen.capture_full()
                    offset = (0, 0)
            else:
                image = self.screen.capture_full()
                offset = (0, 0)
            
            # Try OCR with normal processing first
            ocr_result = self.ocr.process(image, offset)
            match = ocr_result.find_text(text, exact=exact, occurrence=occurrence, min_confidence=50.0)  # Lower confidence
            
            # If not found, try with preprocessing
            if not match:
                ocr_result = self.ocr.process_with_preprocessing(image, offset)
                match = ocr_result.find_text(text, exact=exact, occurrence=occurrence, min_confidence=40.0)  # Even lower for preprocessed
            
            if match:
                x, y = match.center
                click_result = self.mouse.click(x, y, button=button)
                if click_result.success:
                    return ActionResult(
                        True,
                        f"Clicked on '{text}' at ({x}, {y})",
                        data={"x": x, "y": y, "matched_text": match.text}
                    )
                return click_result
            else:
                # Return what we found for debugging - show more matches
                all_texts = [m.text for m in ocr_result.matches if m.confidence > 30]
                # Show unique texts
                unique_texts = list(dict.fromkeys(all_texts))[:20]  # First 20 unique
                return ActionResult(
                    False,
                    f"Could not find text '{text}' on screen. Found: {unique_texts}",
                    data={"found_texts": unique_texts, "total_matches": len(ocr_result.matches)}
                )
        except Exception as e:
            return ActionResult(False, f"OCR click failed: {str(e)}")
    
    def find_text(
        self,
        text: str,
        region: str = "full",
        exact: bool = False
    ) -> Tuple[bool, Optional[OCRMatch], List[OCRMatch]]:
        """
        Find text on screen without clicking.
        Returns (found, best_match, all_matches)
        """
        try:
            if region == "active_window":
                result = self.screen.capture_active_window()
                if result:
                    image, screen_region = result
                    offset = (screen_region.left, screen_region.top)
                else:
                    image = self.screen.capture_full()
                    offset = (0, 0)
            else:
                image = self.screen.capture_full()
                offset = (0, 0)
            
            ocr_result = self.ocr.process(image, offset)
            match = ocr_result.find_text(text, exact=exact)
            all_matches = ocr_result.find_all(text, exact=exact)
            
            return (match is not None, match, all_matches)
        except Exception:
            return (False, None, [])


# Singleton instances
_keyboard: Optional[KeyboardExecutor] = None
_mouse: Optional[MouseExecutor] = None
_ocr_click: Optional[OCRClickExecutor] = None


def get_keyboard() -> KeyboardExecutor:
    global _keyboard
    if _keyboard is None:
        _keyboard = KeyboardExecutor()
    return _keyboard


def get_mouse() -> MouseExecutor:
    global _mouse
    if _mouse is None:
        _mouse = MouseExecutor()
    return _mouse


def get_ocr_click() -> OCRClickExecutor:
    global _ocr_click
    if _ocr_click is None:
        _ocr_click = OCRClickExecutor()
    return _ocr_click
