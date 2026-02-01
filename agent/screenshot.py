"""
Screenshot capture module with DPI awareness and region support.
"""

import io
import base64
import ctypes
from typing import Optional, Tuple
from dataclasses import dataclass

import mss
import mss.tools
from PIL import Image
import numpy as np

from .config import config


@dataclass
class ScreenRegion:
    """Represents a region of the screen."""
    left: int
    top: int
    width: int
    height: int
    
    @property
    def right(self) -> int:
        return self.left + self.width
    
    @property
    def bottom(self) -> int:
        return self.top + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)
    
    def contains_point(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom


class ScreenCapture:
    """Handles screenshot capture with DPI awareness."""
    
    def __init__(self):
        self.sct = mss.mss()
        self._setup_dpi_awareness()
    
    def _setup_dpi_awareness(self):
        """Set DPI awareness on Windows for accurate coordinates."""
        try:
            # Set DPI awareness to per-monitor aware
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                # Fallback to system DPI aware
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass  # Not on Windows or already set
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get the primary monitor size."""
        monitor = self.sct.monitors[1]  # Primary monitor
        return monitor["width"], monitor["height"]
    
    def capture_full(self) -> Image.Image:
        """Capture the entire screen (all monitors combined)."""
        # monitors[0] is all monitors combined, monitors[1] is primary only
        # Use all monitors to see everything
        monitor = self.sct.monitors[0]  # All monitors
        screenshot = self.sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    
    def capture_region(self, region: ScreenRegion) -> Image.Image:
        """Capture a specific region of the screen."""
        monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height
        }
        screenshot = self.sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    
    def capture_active_window(self) -> Optional[Tuple[Image.Image, ScreenRegion]]:
        """Capture the currently active window."""
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                rect = win32gui.GetWindowRect(hwnd)
                region = ScreenRegion(
                    left=rect[0],
                    top=rect[1],
                    width=rect[2] - rect[0],
                    height=rect[3] - rect[1]
                )
                return self.capture_region(region), region
        except ImportError:
            pass
        except Exception:
            pass
        return None
    
    def to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert image to base64 string for API calls with high quality."""
        buffer = io.BytesIO()
        if format.upper() == "JPEG":
            # Convert to RGB if necessary (JPEG doesn't support alpha)
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
            # Use high quality for JPEG
            image.save(buffer, format="JPEG", quality=95, optimize=False)
        else:
            # PNG - use compress_level=1 for good quality and reasonable size
            # Level 1 is fastest compression but still high quality
            image.save(buffer, format="PNG", compress_level=1)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array."""
        return np.array(image)
    
    def close(self):
        """Clean up resources."""
        self.sct.close()


# Singleton instance
_capture_instance: Optional[ScreenCapture] = None


def get_screen_capture() -> ScreenCapture:
    """Get or create the screen capture singleton."""
    global _capture_instance
    if _capture_instance is None:
        _capture_instance = ScreenCapture()
    return _capture_instance
