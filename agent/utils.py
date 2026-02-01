"""
Utility functions for the AI Agent.
"""

import ctypes
from typing import Optional, Tuple


def get_active_window_info() -> Tuple[Optional[str], Optional[str]]:
    """
    Get information about the currently active window.
    
    Returns:
        (window_title, process_name) or (None, None) if unavailable
    """
    try:
        import win32gui
        import win32process
        import psutil
        
        hwnd = win32gui.GetForegroundWindow()
        if hwnd:
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                process = psutil.Process(pid)
                process_name = process.name()
            except Exception:
                process_name = None
            return (title, process_name)
    except ImportError:
        pass
    except Exception:
        pass
    
    return (None, None)


def get_screen_resolution() -> Tuple[int, int]:
    """Get the primary screen resolution."""
    try:
        user32 = ctypes.windll.user32
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    except Exception:
        return (1920, 1080)  # Default fallback


def is_admin() -> bool:
    """Check if the script is running with admin privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
