"""
Screenshot Viewer Window
Displays what the agent sees in real-time in a separate window.
Can be moved to another monitor for monitoring.
"""

import threading
from typing import Optional
from PIL import Image, ImageTk
import tkinter as tk


class ScreenshotViewer:
    """A window that displays screenshots in real-time."""
    
    def __init__(self, title: str = "Agent Vision"):
        self.title = title
        self.root: Optional[tk.Tk] = None
        self.label: Optional[tk.Label] = None
        self.current_image: Optional[Image.Image] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
    
    def start(self):
        """Start the viewer window in a separate thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_window, daemon=True)
        self.thread.start()
        # Give it a moment to initialize
        import time
        time.sleep(0.5)
    
    def _run_window(self):
        """Run the tkinter window in this thread."""
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("1280x720")  # Default size, can be resized
        
        # Position window on second monitor if available, or offset from primary
        try:
            # Try to position on second monitor (right side)
            screen_width = self.root.winfo_screenwidth()
            self.root.geometry(f"1280x720+{screen_width}+0")
        except Exception:
            pass
        
        # Make window resizable and always on top
        self.root.resizable(True, True)
        self.root.attributes("-topmost", True)  # Keep on top so you can see it
        
        # Create label for image
        self.label = tk.Label(self.root, text="Waiting for screenshot...")
        self.label.pack(fill=tk.BOTH, expand=True)
        
        # Add info label
        info_label = tk.Label(
            self.root,
            text="Agent Vision Monitor - This shows what the agent sees",
            bg="black",
            fg="white",
            font=("Arial", 10)
        )
        info_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update loop
        self._update_display()
        
        # Start the tkinter main loop
        self.root.mainloop()
        self.is_running = False
    
    def update_screenshot(self, image: Image.Image, info: str = ""):
        """Update the displayed screenshot."""
        with self.lock:
            self.current_image = image.copy()
            self.current_info = info
    
    def _update_display(self):
        """Update the display with current screenshot."""
        if not self.root or not self.label:
            return
        
        try:
            with self.lock:
                if self.current_image:
                    # Get window size (with fallback)
                    try:
                        self.root.update_idletasks()  # Update geometry
                        window_width = max(self.root.winfo_width(), 800)
                        window_height = max(self.root.winfo_height(), 600)
                    except Exception:
                        window_width, window_height = 1280, 720
                    
                    # Calculate scale to fit
                    img_width, img_height = self.current_image.size
                    scale_w = window_width / img_width
                    scale_h = (window_height - 50) / img_height  # Leave space for info
                    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
                    
                    if scale < 1.0:
                        new_width = int(img_width * scale)
                        new_height = int(img_height * scale)
                        display_image = self.current_image.resize(
                            (new_width, new_height),
                            Image.Resampling.LANCZOS
                        )
                    else:
                        # If image is smaller than window, just use it as-is
                        display_image = self.current_image
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(display_image)
                    self.label.config(image=photo, text="")
                    self.label.image = photo  # Keep a reference
                else:
                    self.label.config(text="Waiting for screenshot...", image="")
        except Exception as e:
            try:
                self.label.config(text=f"Error: {str(e)[:50]}", image="")
            except Exception:
                pass
        
        # Schedule next update
        if self.is_running and self.root:
            try:
                self.root.after(200, self._update_display)  # Update every 200ms
            except Exception:
                pass
    
    def stop(self):
        """Stop the viewer window."""
        self.is_running = False
        if self.root:
            try:
                self.root.quit()
            except Exception:
                pass


# Global instance
_viewer_instance: Optional[ScreenshotViewer] = None


def get_viewer() -> ScreenshotViewer:
    """Get or create the screenshot viewer singleton."""
    global _viewer_instance
    if _viewer_instance is None:
        _viewer_instance = ScreenshotViewer("Agent Vision Monitor")
    return _viewer_instance
