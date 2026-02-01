"""
AI Computer Agent - GUI Application
A full graphical interface for controlling and monitoring the AI agent.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import time
import locale
import ctypes
from typing import Optional, List, Tuple
from dataclasses import dataclass
from PIL import Image, ImageTk
import mss

from .config import config
from .screenshot import ScreenCapture, get_screen_capture
from .loop import AgentLoop, AgentStatus, AgentResult
from .planner import AgentState


@dataclass
class MonitorInfo:
    """Information about a monitor."""
    index: int
    name: str
    left: int
    top: int
    width: int
    height: int
    is_primary: bool
    
    def __str__(self):
        primary = " (Primary)" if self.is_primary else ""
        return f"Monitor {self.index}: {self.width}x{self.height}{primary}"


class AgentGUI:
    """Main GUI application for the AI Agent."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Computer Control Agent")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        # Set dark theme colors
        self.colors = {
            "bg": "#1e1e1e",
            "fg": "#e0e0e0",
            "accent": "#0078d4",
            "accent_hover": "#1a8cff",
            "panel_bg": "#252526",
            "input_bg": "#3c3c3c",
            "success": "#4ec9b0",
            "error": "#f14c4c",
            "warning": "#cca700",
            "border": "#3c3c3c"
        }
        
        self.root.configure(bg=self.colors["bg"])
        
        # Configure styles
        self._setup_styles()
        
        # State
        self.agent: Optional[AgentLoop] = None
        self.agent_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.message_queue = queue.Queue()
        self.current_screenshot: Optional[Image.Image] = None
        self.monitors: List[MonitorInfo] = []
        self.screenshot_capture = None
        
        # Detect system language
        self.system_language = self._detect_system_language()
        
        # Initialize debug mode state (before any methods that use it)
        self.debug_mode_enabled = False
        self.showing_debug_image = False  # Flag to prevent live preview from overwriting debug images
        
        # Build UI
        self._create_widgets()
        # Defer monitor loading until needed (lazy loading)
        
        # Start message processing
        self._process_messages()

        # Delay live preview startup until UI is fully initialized (start after 1 second)
        self.root.after(1000, self._start_live_preview)
    
    def _detect_system_language(self) -> str:
        """Detect the system's display language."""
        try:
            # Windows-specific language detection
            windll = ctypes.windll.kernel32
            lang_id = windll.GetUserDefaultUILanguage()
            
            # Common language codes
            lang_map = {
                0x041D: "Swedish",
                0x0409: "English",
                0x0407: "German",
                0x040C: "French",
                0x0410: "Italian",
                0x0413: "Dutch",
                0x0414: "Norwegian",
                0x0406: "Danish",
                0x040B: "Finnish",
                0x0C0A: "Spanish",
                0x0416: "Portuguese",
                0x0415: "Polish",
            }
            
            return lang_map.get(lang_id, "English")
        except Exception:
            # Fallback to locale
            try:
                lang = locale.getdefaultlocale()[0]
                if lang:
                    if lang.startswith("sv"):
                        return "Swedish"
                    elif lang.startswith("en"):
                        return "English"
                    elif lang.startswith("de"):
                        return "German"
            except Exception:
                pass
        return "English"
    
    def _setup_styles(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure colors
        style.configure(".", 
            background=self.colors["bg"],
            foreground=self.colors["fg"],
            fieldbackground=self.colors["input_bg"]
        )
        
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["fg"])
        style.configure("TLabelframe", background=self.colors["panel_bg"])
        style.configure("TLabelframe.Label", background=self.colors["panel_bg"], foreground=self.colors["fg"])
        
        style.configure("TButton",
            background=self.colors["accent"],
            foreground="white",
            padding=(10, 5)
        )
        style.map("TButton",
            background=[("active", self.colors["accent_hover"])]
        )
        
        style.configure("TEntry",
            fieldbackground=self.colors["input_bg"],
            foreground=self.colors["fg"]
        )
        
        style.configure("TCombobox",
            fieldbackground=self.colors["input_bg"],
            foreground=self.colors["fg"],
            background=self.colors["input_bg"]
        )
        
        style.configure("TScale",
            background=self.colors["bg"],
            troughcolor=self.colors["input_bg"]
        )
        
        style.configure("TCheckbutton",
            background=self.colors["bg"],
            foreground=self.colors["fg"]
        )
    
    def _create_widgets(self):
        """Create all UI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls and Settings
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self._create_task_input(left_panel)
        self._create_settings_panel(left_panel)
        self._create_debug_panel(left_panel)
        self._create_log_panel(left_panel)
        
        # Right panel - Screenshot viewer
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self._create_screenshot_panel(right_panel)
    
    def _create_task_input(self, parent):
        """Create the task input section."""
        frame = ttk.LabelFrame(parent, text=" Task Input ", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Task entry
        ttk.Label(frame, text="Enter your task:").pack(anchor=tk.W)
        
        self.task_entry = tk.Text(frame, height=4, bg=self.colors["input_bg"],
                                   fg=self.colors["fg"], insertbackground=self.colors["fg"],
                                   wrap=tk.WORD, font=("Segoe UI", 10))
        self.task_entry.pack(fill=tk.X, pady=(5, 10))
        self.task_entry.insert("1.0", "open youtube and start a cat video")
        self.task_entry.bind("<Control-Return>", lambda e: self._start_task())
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Task", command=self._start_task)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_task, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(frame, textvariable=self.status_var)
        self.status_label.pack(anchor=tk.W, pady=(10, 0))
        
        # Current action status (more detailed)
        self.action_status_var = tk.StringVar(value="")
        self.action_status_label = ttk.Label(frame, textvariable=self.action_status_var, 
                                              font=("Segoe UI", 9, "italic"),
                                              foreground="#00aaff")
        self.action_status_label.pack(anchor=tk.W, pady=(5, 0))
    
    def _create_settings_panel(self, parent):
        """Create the settings panel."""
        frame = ttk.LabelFrame(parent, text=" Settings ", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Monitor selection
        ttk.Label(frame, text="Capture Monitor:").pack(anchor=tk.W)
        self.monitor_var = tk.StringVar()
        self.monitor_combo = ttk.Combobox(frame, textvariable=self.monitor_var, state="readonly")
        self.monitor_combo.pack(fill=tk.X, pady=(2, 10))
        self.monitor_combo.bind("<<ComboboxSelected>>", self._on_monitor_change)
        # Load monitors when combo is first clicked (lazy loading)
        self.monitor_combo.bind("<Button-1>", lambda e: self._ensure_monitors_loaded())
        
        # Language
        ttk.Label(frame, text=f"System Language: {self.system_language}").pack(anchor=tk.W, pady=(0, 10))
        
        # Action delay
        delay_frame = ttk.Frame(frame)
        delay_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(delay_frame, text="Action Delay (ms):").pack(side=tk.LEFT)
        self.delay_var = tk.IntVar(value=int(config.post_action_delay * 1000))
        self.delay_spinbox = ttk.Spinbox(delay_frame, from_=100, to=3000, increment=100,
                                          textvariable=self.delay_var, width=8)
        self.delay_spinbox.pack(side=tk.RIGHT)
        
        # Click delay
        click_frame = ttk.Frame(frame)
        click_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(click_frame, text="Click Delay (ms):").pack(side=tk.LEFT)
        self.click_delay_var = tk.IntVar(value=int(config.click_delay * 1000))
        self.click_delay_spinbox = ttk.Spinbox(click_frame, from_=50, to=1000, increment=50,
                                                textvariable=self.click_delay_var, width=8)
        self.click_delay_spinbox.pack(side=tk.RIGHT)
        
        # Max steps
        steps_frame = ttk.Frame(frame)
        steps_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(steps_frame, text="Max Steps:").pack(side=tk.LEFT)
        self.max_steps_var = tk.IntVar(value=config.max_steps)
        self.max_steps_spinbox = ttk.Spinbox(steps_frame, from_=5, to=200, increment=5,
                                              textvariable=self.max_steps_var, width=8)
        self.max_steps_spinbox.pack(side=tk.RIGHT)
        
        # Chain commands option
        self.chain_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Allow command chaining", variable=self.chain_var).pack(anchor=tk.W, pady=(5, 0))
        
        # Verify actions option
        self.verify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Verify actions", variable=self.verify_var).pack(anchor=tk.W)
        
        # Live preview option
        self.preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Live preview (updates every 500ms)", variable=self.preview_var).pack(anchor=tk.W)

        # AI screenshots only option
        self.ai_screenshots_only_var = tk.BooleanVar(value=False)
        ai_screenshot_cb = ttk.Checkbutton(frame, text="Show only AI screenshots (exact vision)", variable=self.ai_screenshots_only_var)
        ai_screenshot_cb.pack(anchor=tk.W)

        # Tooltip-like info
        ttk.Label(frame, text="When enabled: See exactly what GPT-4 Vision analyzes", font=("Segoe UI", 8), foreground=self.colors["fg"]).pack(anchor=tk.W, padx=(20, 0))
        
        # Apply button
        ttk.Button(frame, text="Apply Settings", command=self._apply_settings).pack(anchor=tk.W, pady=(10, 0))
    
    def _create_log_panel(self, parent):
        """Create the log output panel."""
        frame = ttk.LabelFrame(parent, text=" Activity Log ", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(
            frame, height=12, bg=self.colors["panel_bg"],
            fg=self.colors["fg"], insertbackground=self.colors["fg"],
            wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for colored output
        self.log_text.tag_configure("info", foreground=self.colors["fg"])
        self.log_text.tag_configure("success", foreground=self.colors["success"])
        self.log_text.tag_configure("error", foreground=self.colors["error"])
        self.log_text.tag_configure("warning", foreground=self.colors["warning"])
        self.log_text.tag_configure("action", foreground=self.colors["accent"])
        
        # Clear button
        ttk.Button(frame, text="Clear Log", command=self._clear_log).pack(anchor=tk.W, pady=(5, 0))
    
    def _create_screenshot_panel(self, parent):
        """Create the screenshot viewer panel with terminal below."""
        # Main container for screenshot and terminal
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Screenshot panel (top part)
        screenshot_frame = ttk.LabelFrame(main_container, text=" Agent Vision (What the AI sees) ", padding=10)
        screenshot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Info bar
        info_frame = ttk.Frame(screenshot_frame)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.screenshot_info_var = tk.StringVar(value="No screenshot yet")
        ttk.Label(info_frame, textvariable=self.screenshot_info_var).pack(side=tk.LEFT)
        
        # Mouse coordinates display
        self.mouse_coords_var = tk.StringVar(value="Mouse: (0, 0)")
        ttk.Label(info_frame, textvariable=self.mouse_coords_var, font=("Consolas", 9)).pack(side=tk.LEFT, padx=(20, 0))
        
        self.step_info_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.step_info_var).pack(side=tk.RIGHT)
        
        # Track mouse movement
        self._start_mouse_tracking()
        
        # Screenshot canvas
        self.canvas_frame = ttk.Frame(screenshot_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.screenshot_canvas = tk.Canvas(
            self.canvas_frame, bg=self.colors["panel_bg"],
            highlightthickness=1, highlightbackground=self.colors["border"]
        )
        self.screenshot_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Keep reference to photo
        self.current_photo = None
        
        # Terminal output panel (bottom part)
        terminal_frame = ttk.LabelFrame(main_container, text=" AI Terminal Output (Full Commands & Thoughts) ", padding=10)
        terminal_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 0))
        terminal_frame.configure(height=200)  # Fixed height for terminal
        
        # Terminal text widget
        self.terminal_text = scrolledtext.ScrolledText(
            terminal_frame, height=8, bg="#000000",  # Black terminal background
            fg="#00ff00",  # Green text like a terminal
            insertbackground="#00ff00",
            wrap=tk.WORD, font=("Consolas", 9),
            state=tk.DISABLED
        )
        self.terminal_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure terminal text tags
        self.terminal_text.tag_configure("thought", foreground="#00ffff")  # Cyan for thoughts
        self.terminal_text.tag_configure("action", foreground="#ffff00")  # Yellow for actions
        self.terminal_text.tag_configure("command", foreground="#00ff00")  # Green for commands
        self.terminal_text.tag_configure("error", foreground="#ff0000")  # Red for errors
        self.terminal_text.tag_configure("info", foreground="#ffffff")  # White for info
        
        # Terminal clear button
        ttk.Button(terminal_frame, text="Clear Terminal", command=self._clear_terminal).pack(anchor=tk.W, pady=(5, 0))
    
    def _create_debug_panel(self, parent):
        """Create the debug/testing panel with OCR and Smart Click."""
        frame = ttk.LabelFrame(parent, text=" Debug & Testing ", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Debug mode toggle
        self.debug_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable Debug Mode", variable=self.debug_mode_var,
                       command=self._toggle_debug_mode).pack(anchor=tk.W, pady=(0, 10))
        
        # Smart Click test (NEW - RECOMMENDED)
        smart_frame = ttk.Frame(frame)
        smart_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(smart_frame, text="SMART CLICK (AI Vision - Recommended):").pack(anchor=tk.W)
        
        smart_input_frame = ttk.Frame(smart_frame)
        smart_input_frame.pack(fill=tk.X, pady=(5, 5))
        
        ttk.Label(smart_input_frame, text="Target:").pack(side=tk.LEFT, padx=(0, 5))
        self.smart_click_text = tk.Entry(smart_input_frame, bg=self.colors["input_bg"], fg=self.colors["fg"], width=25)
        self.smart_click_text.pack(side=tk.LEFT, padx=(0, 5))
        self.smart_click_text.insert(0, "godkänn alla")
        self.smart_click_text.bind("<Return>", lambda e: self._test_smart_click())
        
        ttk.Button(smart_input_frame, text="Smart Click", command=self._test_smart_click).pack(side=tk.LEFT)
        
        # Spacing controls
        spacing_frame = ttk.Frame(smart_frame)
        spacing_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(spacing_frame, text="Initial Spacing:").pack(side=tk.LEFT, padx=(0, 5))
        self.initial_spacing_var = tk.IntVar(value=40)
        spacing_entry = tk.Entry(spacing_frame, textvariable=self.initial_spacing_var, width=5,
                                bg=self.colors["input_bg"], fg=self.colors["fg"])
        spacing_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(spacing_frame, text="Zoom Spacing:").pack(side=tk.LEFT, padx=(0, 5))
        self.zoom_spacing_var = tk.IntVar(value=60)
        zoom_spacing_entry = tk.Entry(spacing_frame, textvariable=self.zoom_spacing_var, width=5,
                                      bg=self.colors["input_bg"], fg=self.colors["fg"])
        zoom_spacing_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Button to regenerate overlay with new spacing
        ttk.Button(spacing_frame, text="Update Preview", command=self._update_smart_click_preview).pack(side=tk.LEFT)
        
        self.smart_click_result_var = tk.StringVar(value="Describe UI element to click")
        ttk.Label(smart_frame, textvariable=self.smart_click_result_var, 
                 font=("Consolas", 9), foreground=self.colors["success"]).pack(anchor=tk.W, pady=(5, 0))
        
        # Separator
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # OCR test section (fallback)
        test_frame = ttk.Frame(frame)
        test_frame.pack(fill=tk.X)
        
        ttk.Label(test_frame, text="OCR Click (Fallback):").pack(anchor=tk.W)
        
        # Text input for OCR test
        input_frame = ttk.Frame(test_frame)
        input_frame.pack(fill=tk.X, pady=(5, 5))
        
        ttk.Label(input_frame, text="Text:").pack(side=tk.LEFT, padx=(0, 5))
        self.ocr_test_text = tk.Entry(input_frame, bg=self.colors["input_bg"], fg=self.colors["fg"], width=15)
        self.ocr_test_text.pack(side=tk.LEFT, padx=(0, 5))
        self.ocr_test_text.bind("<Return>", lambda e: self._test_ocr_click())
        
        # Test button
        ttk.Button(input_frame, text="Find", command=self._test_ocr_find).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(input_frame, text="Click", command=self._test_ocr_click).pack(side=tk.LEFT)
        
        # Results display
        self.ocr_test_result_var = tk.StringVar(value="")
        ttk.Label(test_frame, textvariable=self.ocr_test_result_var, 
                 font=("Consolas", 9), foreground=self.colors["accent"]).pack(anchor=tk.W, pady=(5, 0))
        
        # Manual coordinate test
        coord_frame = ttk.Frame(frame)
        coord_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(coord_frame, text="Manual Click (X, Y):").pack(anchor=tk.W)
        
        coord_input_frame = ttk.Frame(coord_frame)
        coord_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(coord_input_frame, text="X:").pack(side=tk.LEFT, padx=(0, 5))
        self.manual_x_var = tk.StringVar(value="")
        tk.Entry(coord_input_frame, textvariable=self.manual_x_var, width=6,
                 bg=self.colors["input_bg"], fg=self.colors["fg"]).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(coord_input_frame, text="Y:").pack(side=tk.LEFT, padx=(0, 5))
        self.manual_y_var = tk.StringVar(value="")
        tk.Entry(coord_input_frame, textvariable=self.manual_y_var, width=6,
                 bg=self.colors["input_bg"], fg=self.colors["fg"]).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(coord_input_frame, text="Click", command=self._test_manual_click).pack(side=tk.LEFT)
        
        self.manual_click_result_var = tk.StringVar(value="")
        ttk.Label(coord_frame, textvariable=self.manual_click_result_var,
                 font=("Consolas", 9), foreground=self.colors["success"]).pack(anchor=tk.W, pady=(5, 0))
    
    def _load_monitors(self):
        """Load available monitors."""
        try:
            # Lazy initialization of screenshot capture
            if not self.screenshot_capture:
                from .screenshot import ScreenCapture
                self.screenshot_capture = ScreenCapture()
            sct = self.screenshot_capture.sct
            
            self.monitors = []
            # Skip monitor 0 (combined) and add individual monitors
            for i, m in enumerate(sct.monitors[1:], 1):
                is_primary = (i == 1)  # First monitor is usually primary
                info = MonitorInfo(
                    index=i,
                    name=f"Monitor {i}",
                    left=m["left"],
                    top=m["top"],
                    width=m["width"],
                    height=m["height"],
                    is_primary=is_primary
                )
                self.monitors.append(info)
            
            # Add "All Monitors" option
            all_mon = sct.monitors[0]
            self.monitors.insert(0, MonitorInfo(
                index=0,
                name="All Monitors",
                left=all_mon["left"],
                top=all_mon["top"],
                width=all_mon["width"],
                height=all_mon["height"],
                is_primary=False
            ))
            
            # Update combobox
            monitor_names = [str(m) for m in self.monitors]
            self.monitor_combo["values"] = monitor_names
            if monitor_names:
                # Set monitor 3 as default if available, otherwise monitor 0
                default_monitor_index = 3 if len(self.monitors) > 3 else 0
                self.monitor_combo.current(default_monitor_index)
            
            self.log("Loaded monitors: " + ", ".join(monitor_names), "info")
            
        except Exception as e:
            self.log(f"Error loading monitors: {e}", "error")
    
    def _start_live_preview(self):
        """Start the live preview after UI initialization."""
        self._update_live_preview()

    def _ensure_monitors_loaded(self):
        """Ensure monitors are loaded when needed."""
        if not self.monitors:
            self._load_monitors()

    def _on_monitor_change(self, event=None):
        """Handle monitor selection change."""
        self._update_live_preview()
    
    def _apply_settings(self):
        """Apply settings from the UI."""
        config.post_action_delay = self.delay_var.get() / 1000.0
        config.click_delay = self.click_delay_var.get() / 1000.0
        config.max_steps = self.max_steps_var.get()
        
        self.log(f"Settings applied: delay={config.post_action_delay}s, "
                 f"click_delay={config.click_delay}s, max_steps={config.max_steps}", "info")
    
    def _get_selected_monitor(self) -> MonitorInfo:
        """Get the currently selected monitor."""
        # Lazy load monitors if not loaded yet
        if not self.monitors:
            self._load_monitors()

        idx = self.monitor_combo.current()
        if 0 <= idx < len(self.monitors):
            return self.monitors[idx]
        return self.monitors[0] if self.monitors else None
    
    def _capture_screenshot(self) -> Optional[Image.Image]:
        """Capture screenshot from selected monitor."""
        try:
            monitor = self._get_selected_monitor()
            if not monitor:
                return None

            # Lazy initialization of screenshot capture
            if not self.screenshot_capture:
                from .screenshot import ScreenCapture
                self.screenshot_capture = ScreenCapture()
            
            mon_dict = {
                "left": monitor.left,
                "top": monitor.top,
                "width": monitor.width,
                "height": monitor.height
            }
            
            screenshot = self.screenshot_capture.sct.grab(mon_dict)
            return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        except Exception as e:
            self.log(f"Screenshot error: {e}", "error")
            return None
    
    def _update_live_preview(self):
        """Update the live preview of the screen."""
        # Don't update if we're showing a debug image
        if self.showing_debug_image:
            # Schedule next update but don't change the display
            self.root.after(500, self._update_live_preview)
            return
        
        # Only show live preview if enabled and not in "AI screenshots only" mode
        if self.preview_var.get() and not self.ai_screenshots_only_var.get() and not self.is_running:
            img = self._capture_screenshot()
            if img:
                self._display_screenshot(img, "Live Preview")

        # Schedule next update
        self.root.after(500, self._update_live_preview)
    
    def _display_screenshot(self, image: Image.Image, info: str = ""):
        """Display a screenshot in the canvas."""
        try:
            # Get canvas size
            self.screenshot_canvas.update_idletasks()
            canvas_width = self.screenshot_canvas.winfo_width()
            canvas_height = self.screenshot_canvas.winfo_height()
            
            if canvas_width < 10 or canvas_height < 10:
                return
            
            # Calculate scale to fit
            img_width, img_height = image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h, 1.0)
            
            # Resize image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            if new_width > 0 and new_height > 0:
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.current_photo = ImageTk.PhotoImage(resized)
                
                # Center on canvas
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                
                self.screenshot_canvas.delete("all")
                self.screenshot_canvas.create_image(x, y, anchor=tk.NW, image=self.current_photo)
                
                # Update info
                self.screenshot_info_var.set(f"{info} | {img_width}x{img_height}")
        except Exception as e:
            pass  # Ignore display errors
    
    def _start_task(self):
        """Start executing the task."""
        task = self.task_entry.get("1.0", tk.END).strip()
        if not task:
            messagebox.showwarning("No Task", "Please enter a task to execute.")
            return
        
        if self.is_running:
            return
        
        # Apply settings
        self._apply_settings()
        
        # Update UI state
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Running...")

        # Clear screenshot if showing only AI screenshots
        if self.ai_screenshots_only_var.get():
            self.screenshot_canvas.delete("all")
            self.screenshot_info_var.set("AI Vision Mode - Waiting for analysis...")
            self.step_info_var.set("AI Processing")

        self.log(f"Starting task: {task}", "action")
        self.log(f"System language: {self.system_language}", "info")
        
        # Start agent in background thread
        self.agent_thread = threading.Thread(target=self._run_agent, args=(task,), daemon=True)
        self.agent_thread.start()
    
    def _stop_task(self):
        """Stop the current task and all agent processes."""
        self.log("Stop requested - terminating all agent processes...", "warning")
        self.terminal_output(">>> STOP COMMAND RECEIVED - TERMINATING AGENT", "error")
        
        # Set stop flag
        self.is_running = False
        
        # Stop the agent if it exists
        if self.agent:
            try:
                self.agent.stop()
                self.terminal_output("Agent stop() called", "info")
            except Exception as e:
                self.terminal_output(f"Error stopping agent: {e}", "error")
        
        # Force stop the thread if it's still running
        if self.agent_thread and self.agent_thread.is_alive():
            self.terminal_output("Agent thread still alive, forcing termination...", "warning")
            # Note: We can't forcefully kill threads in Python, but we've set the flag
        
        # Clear agent reference
        self.agent = None
        self.agent_thread = None
        
        self.status_var.set("Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.log("Agent stopped", "warning")
        self.terminal_output(">>> AGENT TERMINATED", "error")
    
    def _run_agent(self, task: str):
        """Run the agent in a background thread."""
        try:
            # Get selected monitor for coordinate conversion
            monitor = self._get_selected_monitor()
            monitor_offset = (0, 0)  # Default: all monitors (coordinates already absolute)
            
            if monitor and monitor.index != 0:
                # Single monitor selected - coordinates will be relative to this monitor
                # Need to add monitor offset to convert to absolute screen coordinates
                monitor_offset = (monitor.left, monitor.top)
            
            # Create agent with GUI monitoring and monitor offset
            from .loop import AgentLoop
            self.agent = AgentLoop(show_viewer=False, monitor_offset=monitor_offset)
            
            # Override screenshot capture to use selected monitor
            original_capture = self.agent.screen.capture_full
            
            def capture_selected_monitor():
                monitor = self._get_selected_monitor()
                if monitor and monitor.index != 0:
                    # Capture specific monitor
                    mon_dict = {
                        "left": monitor.left,
                        "top": monitor.top,
                        "width": monitor.width,
                        "height": monitor.height
                    }
                    screenshot = self.agent.screen.sct.grab(mon_dict)
                    return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                else:
                    # All monitors - use original capture
                    return original_capture()
            
            self.agent.screen.capture_full = capture_selected_monitor
            
            # Custom step callback with full AI output and debug images
            def on_step(step_num, action, result, thought=None, plan=None, debug_image=None):
                self.message_queue.put(("step", step_num, action, result, thought, plan))
                # Send debug image if available (from smart_click)
                if debug_image:
                    self.message_queue.put(("debug_image", debug_image, f"Step {step_num} - Smart Click Overlay"))
                # Also send screenshot
                img = capture_selected_monitor()
                if img:
                    self.message_queue.put(("screenshot", img, f"Step {step_num}"))
            
            # Real-time image callback for smart_click overlays
            def on_image(image, info_text):
                self.message_queue.put(("smart_click_image", image, info_text))
            
            # Real-time status callback
            def on_status(status_text):
                self.message_queue.put(("status", status_text))
            
            # Set callbacks on agent
            self.agent._on_image_callback = on_image
            self.agent._on_status_callback = on_status
            
            # Set system language in planner state
            from .planner import AgentState as PlannerState
            
            # Run with language context
            enhanced_task = f"{task}\n\n[System language: {self.system_language}. UI elements may be in {self.system_language}.]"
            
            self.terminal_output(f">>> STARTING TASK: {task}", "info")
            self.terminal_output(f">>> SYSTEM LANGUAGE: {self.system_language}", "info")
            self.terminal_output(">>> AI AGENT INITIALIZED\n", "info")
            
            result = self.agent.run(enhanced_task, on_step=on_step, system_language=self.system_language)
            
            self.terminal_output(f"\n>>> TASK COMPLETED: {result.status.value}", "info")
            
            # Send result
            self.message_queue.put(("complete", result))
            
        except Exception as e:
            self.message_queue.put(("error", str(e)))
            self.terminal_output(f">>> ERROR: {str(e)}", "error")
        finally:
            # Ensure agent is stopped
            if self.agent:
                try:
                    self.agent.stop()
                except Exception:
                    pass
            self.agent = None
            self.terminal_output(">>> AGENT CLEANUP COMPLETE", "info")
    
    def _process_messages(self):
        """Process messages from the agent thread."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                msg_type = msg[0]
                
                if msg_type == "step":
                    # Handle both old format (3 items) and new format (5 items)
                    if len(msg) >= 5:
                        _, step_num, action, result, thought, plan = msg
                    else:
                        _, step_num, action, result = msg
                        thought = None
                        plan = None
                    
                    self.step_info_var.set(f"Step {step_num}")
                    self.log(f"Step {step_num}: {action.description}", "action")
                    if result.success:
                        self.log(f"  Result: {result.message}", "success")
                    else:
                        self.log(f"  Result: {result.message}", "error")
                    
                    # Terminal output - AI thoughts
                    if thought:
                        self.terminal_output(f"\n{'='*60}", "info")
                        self.terminal_output(f"STEP {step_num} - AI THOUGHT:", "thought")
                        self.terminal_output(f"{thought}", "thought")
                        self.terminal_output(f"{'='*60}\n", "info")
                    
                    # Terminal output - Full action details
                    self.terminal_output(f"ACTION TYPE: {action.action_type}", "action")
                    self.terminal_output(f"DESCRIPTION: {action.description}", "action")
                    
                    # Show full action JSON
                    import json
                    try:
                        action_dict = action.model_dump() if hasattr(action, 'model_dump') else action.dict()
                        action_json = json.dumps(action_dict, indent=2)
                        self.terminal_output(f"FULL ACTION COMMAND:\n{action_json}", "command")
                    except Exception:
                        self.terminal_output(f"ACTION: {str(action)}", "command")
                    
                    # Success criteria
                    if plan and hasattr(plan, 'success_criteria') and plan.success_criteria:
                        self.terminal_output(f"SUCCESS CRITERIA: {plan.success_criteria}", "info")
                    
                    # Result
                    status_color = "command" if result.success else "error"
                    self.terminal_output(f"RESULT: {result.message}", status_color)
                    self.terminal_output("", "info")  # Blank line
                    
                    # Update screenshot (or show debug image if available from smart_click)
                    # Check if result has debug images
                    if hasattr(result, 'data') and result.data and isinstance(result.data, dict) and 'debug_images' in result.data:
                        debug_images = result.data['debug_images']
                        if debug_images:
                            # Show all debug images (overlay and zoom if applicable)
                            for i, debug_img in enumerate(debug_images):
                                self.showing_debug_image = True
                                if i == len(debug_images) - 1:
                                    # Last image is most relevant
                                    self._display_screenshot(debug_img, f"Step {step_num} - Smart Click (Final)")
                                else:
                                    self._display_screenshot(debug_img, f"Step {step_num} - Smart Click Overlay {i+1}")
                    else:
                        img = self._capture_screenshot()
                        if img:
                            self._display_screenshot(img, f"Step {step_num}")
                
                elif msg_type == "debug_image":
                    # Handle debug image message separately
                    _, debug_img, info = msg
                    self.showing_debug_image = True
                    self._display_screenshot(debug_img, info)
                
                elif msg_type == "complete":
                    result = msg[1]
                    if result.status == AgentStatus.COMPLETED:
                        self.log(f"Task completed: {result.final_message}", "success")
                        self.status_var.set("Completed")
                    elif result.status == AgentStatus.FAILED:
                        self.log(f"Task failed: {result.final_message}", "error")
                        self.status_var.set("Failed")
                    else:
                        self.log(f"Task stopped: {result.final_message}", "warning")
                        self.status_var.set("Stopped")
                    
                    self._task_finished()
                
                elif msg_type == "error":
                    self.log(f"Error: {msg[1]}", "error")
                    self.status_var.set("Error")
                    self._task_finished()
                
                elif msg_type == "screenshot":
                    img = msg[1]
                    info = msg[2] if len(msg) > 2 else ""
                    self._display_screenshot(img, info)
                
                elif msg_type == "smart_click_image":
                    # Real-time smart_click overlay image
                    img = msg[1]
                    info = msg[2] if len(msg) > 2 else "Smart Click"
                    self.showing_debug_image = True  # Prevent live preview from overwriting
                    self._display_screenshot(img, info)
                    # Also log it
                    self.terminal_output(f"[SMART CLICK] {info}", "action")
                
                elif msg_type == "status":
                    # Real-time status update
                    status_text = msg[1]
                    self.action_status_var.set(status_text)
                    # Also log it
                    self.terminal_output(f"[STATUS] {status_text}", "info")
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._process_messages)
    
    def _task_finished(self):
        """Clean up after task finishes."""
        self.is_running = False
        self.showing_debug_image = False  # Re-enable live preview
        self.action_status_var.set("")  # Clear action status
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def log(self, message: str, level: str = "info"):
        """Add a message to the log."""
        self.log_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] ", "info")
        self.log_text.insert(tk.END, f"{message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _clear_log(self):
        """Clear the log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def terminal_output(self, message: str, tag: str = "info"):
        """Add output to the terminal window."""
        self.terminal_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.terminal_text.insert(tk.END, f"[{timestamp}] ", "info")
        self.terminal_text.insert(tk.END, f"{message}\n", tag)
        self.terminal_text.see(tk.END)
        self.terminal_text.config(state=tk.DISABLED)
    
    def _clear_terminal(self):
        """Clear the terminal output."""
        self.terminal_text.config(state=tk.NORMAL)
        self.terminal_text.delete("1.0", tk.END)
        self.terminal_text.config(state=tk.DISABLED)
    
    def _start_mouse_tracking(self):
        """Start tracking mouse coordinates."""
        def update_mouse_coords():
            try:
                import pyautogui
                x, y = pyautogui.position()
                self.mouse_coords_var.set(f"Mouse: ({x}, {y})")
            except Exception:
                pass
            # Update every 100ms
            self.root.after(100, update_mouse_coords)
        
        update_mouse_coords()
    
    def _toggle_debug_mode(self):
        """Toggle debug mode on/off."""
        self.debug_mode_enabled = self.debug_mode_var.get()
        if self.debug_mode_enabled:
            self.log("Debug mode enabled", "info")
        else:
            self.log("Debug mode disabled", "info")
    
    def _test_ocr_find(self):
        """Test OCR finding without clicking."""
        text = self.ocr_test_text.get().strip()
        if not text:
            self.ocr_test_result_var.set("Please enter text to find")
            return
        
        try:
            from .executors import get_ocr_click
            ocr_click = get_ocr_click()
            
            # Capture screenshot
            img = self._capture_screenshot()
            if not img:
                self.ocr_test_result_var.set("ERROR: Could not capture screenshot")
                return
            
            # Run OCR
            from .ocr import get_ocr_engine
            ocr = get_ocr_engine()
            ocr_result = ocr.process(img)
            
            # Try to find text
            match = ocr_result.find_text(text, exact=False, occurrence=1, min_confidence=40.0)
            
            if match:
                x, y = match.center
                self.ocr_test_result_var.set(
                    f"FOUND: '{match.text}' at ({x}, {y}) | Confidence: {match.confidence:.1f}%"
                )
                self.log(f"OCR Test: Found '{text}' -> '{match.text}' at ({x}, {y})", "success")
                
                # Show all matches
                all_matches = ocr_result.find_all(text, exact=False, min_confidence=40.0)
                if len(all_matches) > 1:
                    self.log(f"  Total matches found: {len(all_matches)}", "info")
            else:
                # Show what was found
                all_texts = [m.text for m in ocr_result.matches if m.confidence > 30][:15]
                self.ocr_test_result_var.set(
                    f"NOT FOUND. Sample detected text: {all_texts[:5]}"
                )
                self.log(f"OCR Test: Could not find '{text}'. Found: {all_texts[:10]}", "error")
        
        except Exception as e:
            self.ocr_test_result_var.set(f"ERROR: {str(e)}")
            self.log(f"OCR Test Error: {e}", "error")
    
    def _test_smart_click(self):
        """Test Smart Click (AI Vision) functionality - Step-by-step debug mode."""
        target = self.smart_click_text.get().strip()
        if not target:
            self.smart_click_result_var.set("Please enter target description")
            return
        
        try:
            from .vision_click import get_vision_click, ClickTarget
            from .executors import get_mouse, MouseButton
            from .screenshot import get_screen_capture
            
            # Initialize step tracking if not exists
            if not hasattr(self, '_smart_click_step'):
                self._smart_click_step = -1
            
            # Get selected monitor
            monitor = self._get_selected_monitor()
            if not monitor:
                self.smart_click_result_var.set("ERROR: No monitor selected")
                return
            
            # Capture screenshot if first time or need fresh capture
            if self._smart_click_step == -1:
                screen = get_screen_capture()
                if monitor.index == 0:
                    screenshot = screen.capture_full()
                    monitor_offset = (0, 0)
                else:
                    mon_dict = {
                        "left": monitor.left,
                        "top": monitor.top,
                        "width": monitor.width,
                        "height": monitor.height
                    }
                    screenshot = screen.sct.grab(mon_dict)
                    screenshot = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    monitor_offset = (monitor.left, monitor.top)
                
                self._current_screenshot = screenshot
                self._current_monitor = monitor
                self._current_monitor_offset = monitor_offset
                self._smart_click_step = 0
            
            vision = get_vision_click()
            screenshot = self._current_screenshot
            
            # Step 0: Show numbered overlay immediately
            if self._smart_click_step == 0:
                spacing = self.initial_spacing_var.get()
                zoom_spacing = self.zoom_spacing_var.get()
                result, overlay, number_map = vision.find_element_step_by_step(
                    screenshot, target, "", True, step=0, spacing=spacing, zoom_spacing=zoom_spacing
                )
                self.showing_debug_image = True
                self._display_screenshot(overlay, f"Numbered Overlay (spacing={spacing}) - Click again to analyze or adjust spacing")
                self.smart_click_result_var.set(f"Numbers shown (spacing={spacing}) - Adjust spacing and click 'Update Preview' or click 'Smart Click' to send to AI")
                self._smart_click_step = 1
                self._current_number_map = number_map
                self._current_spacing = spacing
                self._current_zoom_spacing = zoom_spacing
                return
            
            # Step 1: Send to AI, show result
            elif self._smart_click_step == 1:
                spacing = getattr(self, '_current_spacing', self.initial_spacing_var.get())
                zoom_spacing = getattr(self, '_current_zoom_spacing', self.zoom_spacing_var.get())
                result, overlay, number_map = vision.find_element_step_by_step(
                    screenshot, target, "", True, step=1, spacing=spacing, zoom_spacing=zoom_spacing
                )
                self.showing_debug_image = True
                
                # Always show AI response in terminal
                ai_response = getattr(vision, '_last_ai_response', 'No response captured')
                ai_data = getattr(vision, '_last_ai_data', None)
                ai_error = getattr(vision, '_last_ai_error', None)
                
                self.terminal_output(f"[AI RESPONSE - Step 1]\n{ai_response}\n", "info")
                if ai_data:
                    self.terminal_output(
                        f"[AI PARSED DATA]\n"
                        f"  Action: {ai_data.get('action', 'N/A')}\n"
                        f"  Number: {ai_data.get('number', 'N/A')}\n"
                        f"  Confidence: {ai_data.get('confidence', 'N/A')}\n"
                        f"  Reasoning: {ai_data.get('reasoning', 'N/A')}\n",
                        "command"
                    )
                if ai_error:
                    self.terminal_output(f"[AI ERROR]\n  {ai_error}\n", "error")
                
                if isinstance(result, ClickTarget):
                    # Click result - green crosshair
                    self._display_screenshot(overlay, "AI Result: CLICK (Green Crosshair)")
                    self.smart_click_result_var.set(f"AI chose to CLICK at number -> ({result.x}, {result.y}) - Click again to move mouse")
                    self._current_click_target = result
                    self._smart_click_step = 10  # Final step
                elif isinstance(result, tuple) and result[0] == "zoom":
                    # Zoom result - red crosshair
                    zoom_x, zoom_y = result[1]
                    self._display_screenshot(overlay, "AI Result: ZOOM (Red Crosshair)")
                    self.smart_click_result_var.set(f"AI chose to ZOOM at number -> ({zoom_x}, {zoom_y}) - Click again to show zoomed view")
                    self._current_zoom_point = (zoom_x, zoom_y)
                    self._smart_click_step = 2
                else:
                    self.smart_click_result_var.set("AI could not find the element - check terminal for AI response")
                    self.terminal_output(f"[RESULT] AI returned: {result}\n", "error")
                    self.showing_debug_image = False
                    self._smart_click_step = -1
                return
            
            # Step 2: Show zoomed overlay
            elif self._smart_click_step == 2:
                zoom_point = getattr(self, '_current_zoom_point', None)
                if zoom_point:
                    spacing = getattr(self, '_current_spacing', self.initial_spacing_var.get())
                    zoom_spacing = getattr(self, '_current_zoom_spacing', self.zoom_spacing_var.get())
                    result, overlay, number_map = vision.find_element_step_by_step(
                        screenshot, target, "", True, step=2, zoom_point=zoom_point, spacing=spacing, zoom_spacing=zoom_spacing
                    )
                    self.showing_debug_image = True
                    self._display_screenshot(overlay, f"Zoomed View with Numbers (spacing={zoom_spacing}) - Click again to analyze or adjust spacing")
                    self.smart_click_result_var.set(f"Zoomed overlay shown (spacing={zoom_spacing}) - Adjust spacing and click 'Update Preview' or click 'Smart Click' to send to AI")
                    self._smart_click_step = 3
                    self._current_zoom_number_map = number_map
                return
            
            # Step 3: Analyze zoomed overlay
            elif self._smart_click_step == 3:
                zoom_point = getattr(self, '_current_zoom_point', None)
                if zoom_point:
                    spacing = getattr(self, '_current_spacing', self.initial_spacing_var.get())
                    zoom_spacing = getattr(self, '_current_zoom_spacing', self.zoom_spacing_var.get())
                    result, overlay, number_map = vision.find_element_step_by_step(
                        screenshot, target, "", True, step=3, zoom_point=zoom_point, spacing=spacing, zoom_spacing=zoom_spacing
                    )
                    self.showing_debug_image = True
                    
                    # Always show AI response in terminal
                    ai_response = getattr(vision, '_last_ai_response', 'No response captured')
                    ai_data = getattr(vision, '_last_ai_data', None)
                    ai_error = getattr(vision, '_last_ai_error', None)
                    
                    self.terminal_output(f"[AI RESPONSE - Step 3 (Zoomed)]\n{ai_response}\n", "info")
                    if ai_data:
                        self.terminal_output(
                            f"[AI PARSED DATA - Zoomed]\n"
                            f"  Action: {ai_data.get('action', 'N/A')}\n"
                            f"  Number: {ai_data.get('number', 'N/A')}\n"
                            f"  Confidence: {ai_data.get('confidence', 'N/A')}\n"
                            f"  Reasoning: {ai_data.get('reasoning', 'N/A')}\n",
                            "command"
                        )
                    if ai_error:
                        self.terminal_output(f"[AI ERROR - Zoomed]\n  {ai_error}\n", "error")
                    
                    if isinstance(result, ClickTarget):
                        # Final click result
                        self._display_screenshot(overlay, "AI Result: CLICK (Green Crosshair)")
                        self.smart_click_result_var.set(f"AI chose to CLICK at ({result.x}, {result.y}) - Click again to move mouse")
                        self._current_click_target = result
                        self._smart_click_step = 10  # Final step
                    else:
                        self.smart_click_result_var.set("AI could not find the element in zoomed view - check terminal for AI response")
                        self.terminal_output(f"[RESULT - Zoomed] AI returned: {result}\n", "error")
                        self.showing_debug_image = False
                        self._smart_click_step = -1
                return
            
            # Step 10: Move mouse to final target
            elif self._smart_click_step == 10:
                click_target = getattr(self, '_current_click_target', None)
                if click_target:
                    # Convert to absolute coordinates
                    abs_x = click_target.x + self._current_monitor_offset[0]
                    abs_y = click_target.y + self._current_monitor_offset[1]
                    
                    mouse = get_mouse()
                    move_result = mouse.move_to(abs_x, abs_y, duration=0.3)
                    
                    if move_result.success:
                        self.smart_click_result_var.set(f"SUCCESS: Moved mouse to ({abs_x}, {abs_y})")
                        self.log(f"Smart Click: Moved mouse to ({abs_x}, {abs_y})", "success")
                        self.terminal_output(
                            f"[SMART CLICK] Final position: ({abs_x}, {abs_y})\n"
                            f"  Description: {click_target.description}\n",
                            "command"
                        )
                    else:
                        self.smart_click_result_var.set(f"MOVE FAILED: {move_result.message}")
                    
                    # Reset
                    self.showing_debug_image = False
                    self._smart_click_step = -1
                    if hasattr(self, '_current_click_target'):
                        delattr(self, '_current_click_target')
                    if hasattr(self, '_current_zoom_point'):
                        delattr(self, '_current_zoom_point')
                return
            
            # Check if we're continuing from a previous debug step (old code path - remove later)
            if hasattr(self, '_current_debug_images') and self._current_debug_images:
                # Continue showing debug images
                if hasattr(self, '_current_debug_step'):
                    self._current_debug_step += 1
                else:
                    self._current_debug_step = 1
                
                if self._current_debug_step < len(self._current_debug_images):
                    # Show next debug image
                    self.showing_debug_image = True  # Keep preventing live preview
                    debug_img = self._current_debug_images[self._current_debug_step]
                    if self._current_debug_step == 1:
                        self._display_screenshot(debug_img, f"Refined Region with Crosshair - Final Result")
                        self.smart_click_result_var.set("Refined region shown - click 'Smart Click' again to move mouse")
                    else:
                        self._display_screenshot(debug_img, f"Debug Step {self._current_debug_step + 1}")
                    self.root.update()
                    return
                else:
                    # All debug images shown, proceed with click
                    if hasattr(self, '_current_click_target') and self._current_click_target:
                        click_target = self._current_click_target
                        monitor_offset = getattr(self, '_current_monitor_offset', (0, 0))
                        monitor = getattr(self, '_current_monitor', None)
                        screenshot = getattr(self, '_current_screenshot', None)
                        success = True
                        message = "Continuing from debug view"
                    else:
                        self.smart_click_result_var.set("No target found - please run Smart Click again")
                        return
            else:
                # First run - capture and analyze
                self.log(f"Smart Click: Looking for '{target}'", "action")
                self.smart_click_result_var.set("Analyzing screenshot with AI...")
                self.root.update()
                
                # Get selected monitor
                monitor = self._get_selected_monitor()
                if not monitor:
                    self.smart_click_result_var.set("ERROR: No monitor selected")
                    return
                
                # Capture screenshot from selected monitor
                screen = get_screen_capture()
                if monitor.index == 0:
                    # All monitors - use full capture
                    screenshot = screen.capture_full()
                    monitor_offset = (0, 0)  # Coordinates already absolute
                else:
                    # Single monitor - capture that specific monitor
                    mon_dict = {
                        "left": monitor.left,
                        "top": monitor.top,
                        "width": monitor.width,
                        "height": monitor.height
                    }
                    screenshot = screen.sct.grab(mon_dict)
                    screenshot = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    # Coordinates from screenshot are relative to monitor, need to add offset
                    monitor_offset = (monitor.left, monitor.top)
                
                # Store for later use
                self._current_monitor = monitor
                self._current_monitor_offset = monitor_offset
                self._current_screenshot = screenshot
                
                self.terminal_output(
                    f"[SMART CLICK] Using monitor: {monitor.name}\n"
                    f"  Monitor bounds: ({monitor.left}, {monitor.top}) to ({monitor.left + monitor.width}, {monitor.top + monitor.height})\n"
                    f"  Screenshot size: {screenshot.width}x{screenshot.height}\n"
                    f"  Monitor offset: {monitor_offset}\n",
                    "info"
                )
                
                # Use vision to find element (with debug images)
                vision = get_vision_click()
                success, click_target, message, debug_images = vision.find_and_click(
                    screenshot,
                    description=target,
                    prefer_primary=True,
                    return_debug_images=True  # Get grid overlay images
                )
                
                # Store debug images and target for later
                self._current_debug_images = debug_images if debug_images else []
                self._current_click_target = click_target if success else None
                self._current_debug_step = -1
                
                # Display first debug image (grid overlay) - keep it visible
                if debug_images and len(debug_images) > 0:
                    self.showing_debug_image = True  # Prevent live preview from overwriting
                    self._display_screenshot(debug_images[0], f"Grid Overlay - Click 'Smart Click' again to see next step")
                    self.smart_click_result_var.set("Grid overlay shown - click 'Smart Click' again to continue")
                    self.root.update()
                    return  # Exit early, user can click again to see next step
            
            if success and click_target:
                # Convert relative coordinates to absolute screen coordinates
                # Screenshot coordinates are relative to the monitor (0,0 at monitor top-left)
                # Need to add monitor offset to get absolute screen coordinates
                abs_x = click_target.x + monitor_offset[0]
                abs_y = click_target.y + monitor_offset[1]
                
                # Validate coordinates are within monitor bounds
                if click_target.x < 0 or click_target.x >= screenshot.width or \
                   click_target.y < 0 or click_target.y >= screenshot.height:
                    self.smart_click_result_var.set(
                        f"ERROR: Relative coordinates out of bounds! ({click_target.x}, {click_target.y}) vs screenshot ({screenshot.width}, {screenshot.height})"
                    )
                    self.log(f"Smart Click: Invalid relative coordinates!", "error")
                    self.terminal_output(
                        f"[SMART CLICK ERROR] Relative coordinates out of bounds!\n"
                        f"  Relative: ({click_target.x}, {click_target.y})\n"
                        f"  Screenshot: ({screenshot.width}, {screenshot.height})\n"
                        f"  Monitor offset: {monitor_offset}\n",
                        "error"
                    )
                    self.showing_debug_image = False
                    return
                
                # Validate absolute coordinates are within monitor bounds
                if abs_x < monitor.left or abs_x >= monitor.left + monitor.width or \
                   abs_y < monitor.top or abs_y >= monitor.top + monitor.height:
                    self.smart_click_result_var.set(
                        f"ERROR: Absolute coordinates out of monitor bounds! ({abs_x}, {abs_y}) vs monitor ({monitor.left}, {monitor.top}) to ({monitor.left + monitor.width}, {monitor.top + monitor.height})"
                    )
                    self.log(f"Smart Click: Invalid absolute coordinates!", "error")
                    self.terminal_output(
                        f"[SMART CLICK ERROR] Absolute coordinates out of bounds!\n"
                        f"  Absolute: ({abs_x}, {abs_y})\n"
                        f"  Monitor: ({monitor.left}, {monitor.top}) to ({monitor.left + monitor.width}, {monitor.top + monitor.height})\n",
                        "error"
                    )
                    self.showing_debug_image = False
                    return
                
                # Just move mouse (don't click) in debug mode
                mouse = get_mouse()
                move_result = mouse.move_to(abs_x, abs_y, duration=0.3)
                
                if move_result.success:
                    self.smart_click_result_var.set(
                        f"SUCCESS: Moved mouse to '{click_target.description}' at ({abs_x}, {abs_y})"
                    )
                    self.log(f"Smart Click: Found and moved mouse to ({abs_x}, {abs_y})", "success")
                    self.terminal_output(
                        f"[SMART CLICK] Target: '{target}'\n"
                        f"  Found: '{click_target.description}'\n"
                        f"  Relative position: ({click_target.x}, {click_target.y})\n"
                        f"  Absolute position: ({abs_x}, {abs_y})\n"
                        f"  Monitor offset: {monitor_offset}\n"
                        f"  Confidence: {click_target.confidence:.2f}\n"
                        f"  Screenshot size: {screenshot.width}x{screenshot.height}\n"
                        f"  Monitor: {monitor.name} ({monitor.width}x{monitor.height})\n",
                        "command"
                    )
                    self.showing_debug_image = False  # Re-enable live preview after completion
                else:
                    self.smart_click_result_var.set(f"MOVE FAILED: {move_result.message}")
                    self.log(f"Smart Click: Move failed - {move_result.message}", "error")
                    self.showing_debug_image = False
            else:
                self.smart_click_result_var.set(f"NOT FOUND: {message}")
                self.log(f"Smart Click: Could not find '{target}'", "error")
                self.terminal_output(f"[SMART CLICK FAILED] Target: '{target}'\n  {message}\n", "error")
                self.showing_debug_image = False
        
        except Exception as e:
            import traceback
            self.smart_click_result_var.set(f"ERROR: {str(e)}")
            self.log(f"Smart Click Error: {e}", "error")
            self.terminal_output(f"[SMART CLICK ERROR] {str(e)}\n{traceback.format_exc()}\n", "error")
    
    def _update_smart_click_preview(self):
        """Update the smart click preview with new spacing values."""
        if not hasattr(self, '_current_screenshot') or self._smart_click_step not in [0, 2]:
            self.smart_click_result_var.set("Please start Smart Click first (step 0 or 2)")
            return
        
        try:
            from .vision_click import get_vision_click
            vision = get_vision_click()
            screenshot = self._current_screenshot
            target = self.smart_click_text.get().strip()
            
            if not target:
                self.smart_click_result_var.set("Please enter target description")
                return
            
            spacing = self.initial_spacing_var.get()
            zoom_spacing = self.zoom_spacing_var.get()
            
            if self._smart_click_step == 0:
                # Update initial overlay
                result, overlay, number_map = vision.find_element_step_by_step(
                    screenshot, target, "", True, step=0, spacing=spacing, zoom_spacing=zoom_spacing
                )
                self.showing_debug_image = True
                self._display_screenshot(overlay, f"Numbered Overlay (spacing={spacing}) - Updated")
                self.smart_click_result_var.set(f"Preview updated with spacing={spacing}")
                self._current_number_map = number_map
                self._current_spacing = spacing
                self._current_zoom_spacing = zoom_spacing
            elif self._smart_click_step == 2:
                # Update zoomed overlay
                zoom_point = getattr(self, '_current_zoom_point', None)
                if zoom_point:
                    result, overlay, number_map = vision.find_element_step_by_step(
                        screenshot, target, "", True, step=2, zoom_point=zoom_point, spacing=spacing, zoom_spacing=zoom_spacing
                    )
                    self.showing_debug_image = True
                    self._display_screenshot(overlay, f"Zoomed View (spacing={zoom_spacing}) - Updated")
                    self.smart_click_result_var.set(f"Preview updated with zoom spacing={zoom_spacing}")
                    self._current_zoom_number_map = number_map
                    self._current_zoom_spacing = zoom_spacing
        except Exception as e:
            self.smart_click_result_var.set(f"Error updating preview: {str(e)}")
            self.log(f"Update preview error: {e}", "error")
    
    def _test_ocr_click(self):
        """Test OCR click functionality (fallback method)."""
        text = self.ocr_test_text.get().strip()
        if not text:
            self.ocr_test_result_var.set("Please enter text to find")
            return
        
        try:
            from .executors import get_ocr_click, MouseButton
            ocr_click = get_ocr_click()
            
            self.log(f"OCR Test: Attempting to click '{text}'", "action")
            result = ocr_click.click_text(
                text=text,
                occurrence=1,
                region="full",
                exact=False,
                button=MouseButton.LEFT
            )
            
            if result.success:
                coords = result.data.get("x"), result.data.get("y") if result.data else (None, None)
                matched = result.data.get("matched_text", "") if result.data else ""
                self.ocr_test_result_var.set(
                    f"SUCCESS: Clicked at ({coords[0]}, {coords[1]}) | Matched: '{matched}'"
                )
                self.log(f"OCR Test: Successfully clicked '{text}' at {coords}", "success")
            else:
                self.ocr_test_result_var.set(f"FAILED: {result.message}")
                self.log(f"OCR Test: Failed - {result.message}", "error")
        
        except Exception as e:
            self.ocr_test_result_var.set(f"ERROR: {str(e)}")
            self.log(f"OCR Test Error: {e}", "error")
    
    def _test_manual_click(self):
        """Test manual coordinate click."""
        try:
            x_str = self.manual_x_var.get().strip()
            y_str = self.manual_y_var.get().strip()
            
            if not x_str or not y_str:
                self.manual_click_result_var.set("Please enter X and Y coordinates")
                return
            
            x = int(x_str)
            y = int(y_str)
            
            from .executors import get_mouse, MouseButton
            mouse = get_mouse()
            
            self.log(f"Manual Click Test: Clicking at ({x}, {y})", "action")
            result = mouse.click(x, y, button=MouseButton.LEFT)
            
            if result.success:
                self.manual_click_result_var.set(f"SUCCESS: Clicked at ({x}, {y})")
                self.log(f"Manual Click: Successfully clicked at ({x}, {y})", "success")
            else:
                self.manual_click_result_var.set(f"FAILED: {result.message}")
                self.log(f"Manual Click: Failed - {result.message}", "error")
        
        except ValueError:
            self.manual_click_result_var.set("ERROR: Invalid coordinates (must be numbers)")
        except Exception as e:
            self.manual_click_result_var.set(f"ERROR: {str(e)}")
            self.log(f"Manual Click Error: {e}", "error")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def launch_gui():
    """Launch the GUI application."""
    app = AgentGUI()
    app.run()


if __name__ == "__main__":
    launch_gui()
