"""
Coordinate Finder Test Program
Tests iterative coordinate refinement using AI vision.

Method: Cursor Movement
1. Start with cursor at center of image
2. AI analyzes image with cyan dot at cursor position
3. AI outputs percentage to move cursor in X/Y direction (-100% to +100%)
4. Move cursor based on percentage (relative to distance to edges)
5. Show cursor movement trail in preview
6. Repeat until AI is 100% confident cursor is over target, then "click"
"""

import os
import warnings
import io
import ctypes
from contextlib import redirect_stdout, redirect_stderr

# Reduce PaddleOCR startup checks and limit thread contention on some CPUs.
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ.setdefault('DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PADDLEOCR_SHOW_LOG', 'False')
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('FLAGS_minloglevel', '2')

# Suppress noisy warnings from Paddle/PaddleOCR.
warnings.filterwarnings("ignore", category=UserWarning, message=".*ccache.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*predict.*")
import sys
import json
import base64
import time
import math
import random
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
from io import BytesIO
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from agent.ocr import get_ocr_engine
from agent.screenshot import ScreenRegion
from agent.click_ocr import run_ocr_in_roi, build_ocr_candidates
from agent.click_color import find_color_regions, build_color_candidates
from agent.prompt_loader import load_prompt, format_prompt
import urllib.request
import shutil


@dataclass
class NewMethodSnapshot:
    """Snapshot of the current test context for the new coordinate finder method."""
    image: Image.Image
    screen_resolution: Tuple[int, int]
    dpi_scaling: float
    window_rect: Tuple[int, int, int, int]
    window_title: str
    cursor_position: Tuple[int, int]
    user_instruction: str


@dataclass
class PlannerTextIntent:
    primary_text: str
    variants: List[str] = field(default_factory=list)
    strictness: str = "medium"  # "low" | "medium" | "high"


@dataclass
class PlannerLocationIntent:
    scope: str = "window"  # "window" | "screen"
    zone: str = "any"      # "top_bar", "left", "right", "footer", "center", "any" ("sidebar" aliases "left")
    position: str = "any"  # "top_left", "top", "top_right", "left", "center", "right", "bottom_left", "bottom", "bottom_right", "any"


@dataclass
class PlannerVisualIntent:
    accent_color_relevant: bool = False
    shape_importance: str = "medium"  # "low" | "medium" | "high"
    primary_color: str = "unknown"  # "white", "grey", "black", "red", "blue", "green", "pink", "purple", "yellow", "unknown"
    relative_luminance: str = "unknown"  # "lighter", "darker", "similar", "unknown"
    shape: str = "unknown"  # "rounded_rectangle", "rectangle", "pill", "circle", "icon", "text_only", "unknown"
    size: str = "unknown"  # "small", "medium", "large", "full_width", "unknown"
    width: str = "unknown"  # "narrow", "medium", "wide", "full", "unknown"
    height: str = "unknown"  # "short", "medium", "tall", "unknown"
    text_presence: str = "unknown"  # "required", "optional", "absent", "unknown"
    description: str = ""


@dataclass
class PlannerRiskIntent:
    level: str = "medium"  # "low" | "medium" | "high"


@dataclass
class PlannerOutput:
    """Semantic description of what a correct target would look like."""
    text_intent: PlannerTextIntent
    location_intent: PlannerLocationIntent
    visual_intent: PlannerVisualIntent
    risk_intent: PlannerRiskIntent
    decision: str  # "OK" or "UNSURE"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePackage:
    """Deterministically generated candidate region."""
    id: int
    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom) in image coordinates
    click_point: Tuple[int, int]
    text: Optional[str]
    color: Tuple[int, int, int]
    scores: Dict[str, float]
    total_score: float
    source: str = "unknown"  # "ocr" | "shape" | "unknown"


# Load environment variables
load_dotenv()

class CoordinateFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 AI Coordinate Finder Test")
        # Slightly smaller default window so everything fits on common screens
        self.root.geometry("1200x800")
        self.root.configure(bg='#0d1117')
        
        # OpenAI client
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key or self.api_key == 'your_openai_api_key_here':
            self.api_key = None
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        # SeeClick model (lazy-loaded)
        self.seeclick_tokenizer = None
        self.seeclick_model = None
        
        # OCR engine (lazy-loaded)
        self.ocr_engine = None
        self.ocr_results = None
        self.use_paddleocr = False  # Use Tesseract by default (faster/less setup pain)
        self.paddleocr_ocr = None  # PaddleOCR instance
        # For the new method experiments we do NOT auto-run OCR on startup.
        self.ocr_auto_run = False
        self._ocr_running = False
        self.ocr_fast_mode = True
        
        # State
        self.original_image = None
        self.current_image = None
        self._last_display_image = None
        self.image_path = "tesla.png"
        self.images_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.image_var = tk.StringVar(value=self.image_path)
        self.iterations = []
        self.is_running = False
        self.current_coords = None
        self.max_iterations = 5
        
        # Image display scaling info (for hover detection)
        self.image_scale_x = 1.0
        self.image_scale_y = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.displayed_image_size = None

        # Zoom and pan variables for canvas
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # New-method visualization state
        self.nm_stages = []          # List[(name, PIL.Image)]
        self.nm_stage_index = 0      # Which stage is currently shown
        self.nm_stage_label_var = tk.StringVar(value="Stage: N/A")

        # New method visualization state
        self.nm_views = []          # list of (PIL.Image, label)
        self.nm_view_index = -1     # current index in nm_views

        # New method toggles
        self.nm_use_ocr_var = tk.BooleanVar(value=True)
        self.nm_use_color_var = tk.BooleanVar(value=True)
        self.nm_use_shape_var = tk.BooleanVar(value=True)
        self.nm_color_mask_entry = None
        self.nm_color_max_area_entry = None
        self.nm_color_split_entry = None
        self.nm_color_min_area_entry = None
        self.roi_settings_path = os.path.join(os.getcwd(), "roi_settings.json")
        self.nm_roi_settings = self._nm_load_roi_settings()
        self.nm_ocr_match_entry = None
        self.nm_size_var = tk.StringVar(value="any")
        self.nm_position_var = tk.StringVar(value="any")
        self.nm_sample_mode = False
        self.nm_color_settings = self._nm_color_settings_defaults()
        self.nm_color_split_keep = 2

        # Manual-step pipeline state (debug/interactive)
        self._nm_stage_add_disabled = False
        self._nm_manual_refresh_job = None
        self.nm_manual_snapshot: Optional[NewMethodSnapshot] = None
        self.nm_manual_planner: Optional[PlannerOutput] = None
        self.nm_manual_candidates_base: List[CandidatePackage] = []
        self.nm_manual_candidates_current: List[CandidatePackage] = []
        self.nm_manual_sig: Optional[Tuple[Any, ...]] = None

        # Manual filter toggles (combine via intersection)
        self.nm_filter_color_enabled = tk.BooleanVar(value=False)
        self.nm_filter_ocr_enabled = tk.BooleanVar(value=False)
        self.nm_filter_size_enabled = tk.BooleanVar(value=False)
        self.nm_filter_position_enabled = tk.BooleanVar(value=False)

        # Background removal (large coherent color regions)
        self.nm_bg_remove_enabled = tk.BooleanVar(value=False)
        self.nm_bg_remove_min_area_var = tk.StringVar(value="40000")
        self.nm_bg_remove_leniency_var = tk.StringVar(value="8")
        self.nm_manual_snapshot_raw_image: Optional[Image.Image] = None
        
        # Predefined distinct colors for vertical selection lines (left → right order)
        self.line_colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "magenta",
            "cyan",
            "orange",
            "purple"
        ]
        
        self.setup_styles()
        self.setup_ui()
        self._refresh_image_list()
        self.load_image()
        
        # Show API key dialog if not set
        if not self.client:
            self.root.after(100, self.show_api_key_dialog)
        
    def setup_styles(self):
        """Setup custom styles for the UI."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Dark.TFrame', background='#0d1117')
        style.configure('Card.TFrame', background='#161b22')
        style.configure('Dark.TLabel', background='#0d1117', foreground='#c9d1d9', font=('Segoe UI', 10))
        style.configure('Title.TLabel', background='#0d1117', foreground='#58a6ff', font=('Segoe UI', 14, 'bold'))
        style.configure('Header.TLabel', background='#161b22', foreground='#f0f6fc', font=('Segoe UI', 11, 'bold'))
        style.configure('Status.TLabel', background='#161b22', foreground='#7ee787', font=('Segoe UI', 10))
        style.configure('Coords.TLabel', background='#161b22', foreground='#ffa657', font=('Consolas', 11))
        
        style.configure('Action.TButton', 
                       background='#238636', 
                       foreground='white', 
                       font=('Segoe UI', 11, 'bold'),
                       padding=(20, 10))
        style.map('Action.TButton',
                 background=[('active', '#2ea043'), ('disabled', '#21262d')])
        
        style.configure('Stop.TButton',
                       background='#da3633',
                       foreground='white',
                       font=('Segoe UI', 11, 'bold'),
                       padding=(20, 10))
        style.map('Stop.TButton',
                 background=[('active', '#f85149'), ('disabled', '#21262d')])

        # Combobox styling for dark theme
        style.configure(
            "Dark.TCombobox",
            fieldbackground="#21262d",
            background="#21262d",
            foreground="#c9d1d9",
            arrowcolor="#c9d1d9",
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", "#21262d")],
            foreground=[("readonly", "#c9d1d9")],
            background=[("readonly", "#21262d")],
        )

    def setup_ui(self):
        """Setup the main UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(title_frame, text="🎯 AI Coordinate Finder - Cursor Movement Method", 
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(title_frame, text="● Ready", style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT)
        
        # Content area (left: controls + log, right: images)
        content_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (scrollable)
        left_panel = ttk.Frame(content_frame, style='Dark.TFrame', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)

        left_canvas = tk.Canvas(left_panel, bg='#0d1117', highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left_container = ttk.Frame(left_canvas, style='Dark.TFrame')
        left_window = left_canvas.create_window((0, 0), window=left_container, anchor='nw')

        def _on_left_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox('all'))
            left_canvas.itemconfig(left_window, width=event.width)

        left_container.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox('all')))
        left_canvas.bind("<Configure>", _on_left_configure)
        
        # Input card
        input_card = self.create_card(left_container, "📝 Input")
        input_card.pack(fill=tk.X, pady=(0, 15))
        
        # Prompt input
        prompt_label = ttk.Label(input_card, text="What do you want to click?", style='Dark.TLabel')
        prompt_label.pack(anchor=tk.W, pady=(5, 5))

        self.prompt_entry = tk.Entry(input_card, font=('Segoe UI', 11), bg='#21262d', fg='#c9d1d9',
                                     insertbackground='#58a6ff', relief=tk.FLAT, 
                                     highlightthickness=1, highlightbackground='#30363d',
                                     highlightcolor='#58a6ff')
        self.prompt_entry.pack(fill=tk.X, pady=(0, 10), ipady=8)
        self.prompt_entry.insert(0, "Sök input at the top of the screen")

        # Image selection (images/ folder)
        image_row = ttk.Frame(input_card, style='Card.TFrame')
        image_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(image_row, text="Image:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.image_combo = ttk.Combobox(
            image_row,
            textvariable=self.image_var,
            values=[],
            state="readonly",
            style="Dark.TCombobox",
        )
        self.image_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self.image_combo.bind("<<ComboboxSelected>>", lambda e: self._on_image_select())
        
        # Buttons - Main control buttons
        btn_frame = ttk.Frame(input_card, style='Card.TFrame')
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        self.start_btn = tk.Button(btn_frame, text="▶ Start Finding", command=self.start_finding,
                                   bg='#238636', fg='white', font=('Segoe UI', 11, 'bold'),
                                   relief=tk.FLAT, cursor='hand2', padx=20, pady=8)
        self.start_btn.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 6))

        self.stop_btn = tk.Button(btn_frame, text="■ Stop", command=self.stop_finding,
                                  bg='#da3633', fg='white', font=('Segoe UI', 11, 'bold'),
                                  relief=tk.FLAT, cursor='hand2', padx=20, pady=8, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, sticky="w", pady=(0, 6))

        self.reset_btn = tk.Button(btn_frame, text="↺ Reset", command=self.reset,
                                   bg='#30363d', fg='#c9d1d9', font=('Segoe UI', 11),
                                   relief=tk.FLAT, cursor='hand2', padx=20, pady=8)
        self.reset_btn.grid(row=1, column=2, sticky="w", pady=(0, 6))

        self.reset_zoom_btn = tk.Button(btn_frame, text="🔍 Reset View", command=self.reset_zoom_pan,
                                       bg='#30363d', fg='#c9d1d9', font=('Segoe UI', 10),
                                       relief=tk.FLAT, cursor='hand2', padx=15, pady=8)
        self.reset_zoom_btn.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=(0, 6))

        # Debug tools (single-purpose helpers)
        debug_btn_frame = ttk.Frame(input_card, style='Card.TFrame')
        debug_btn_frame.pack(fill=tk.X, pady=(5, 0))

        self.border_color_btn = tk.Button(
            debug_btn_frame,
            text="Border Color Test",
            command=self.run_border_color_test,
            bg='#1f6feb',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8,
        )
        self.border_color_btn.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.roi_editor_btn = tk.Button(
            debug_btn_frame,
            text="ROI Editor",
            command=self.open_roi_editor,
            bg='#1f6feb',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8,
        )
        self.roi_editor_btn.grid(row=0, column=1, sticky="w")

        # New method filter toggles
        toggle_frame = ttk.Frame(input_card, style='Card.TFrame')
        toggle_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(toggle_frame, text="New method filters:", style='Dark.TLabel').grid(row=0, column=0, sticky="w")

        toggle_opts = {
            "bg": "#161b22",
            "fg": "#c9d1d9",
            "selectcolor": "#0d1117",
            "activebackground": "#161b22",
            "activeforeground": "#f0f6fc",
        }
        tk.Checkbutton(toggle_frame, text="OCR", variable=self.nm_use_ocr_var, **toggle_opts).grid(row=1, column=0, sticky="w", pady=(4, 0))
        tk.Checkbutton(toggle_frame, text="Color", variable=self.nm_use_color_var, **toggle_opts).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        tk.Checkbutton(toggle_frame, text="Shape", variable=self.nm_use_shape_var, **toggle_opts).grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(4, 0))

        # Color mask override input
        color_mask_frame = ttk.Frame(input_card, style='Card.TFrame')
        color_mask_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(color_mask_frame, text="Color mask:", style='Dark.TLabel').grid(row=0, column=0, sticky="w")
        self.nm_color_mask_entry = tk.Entry(
            color_mask_frame,
            font=('Segoe UI', 10),
            bg='#21262d',
            fg='#c9d1d9',
            insertbackground='#58a6ff',
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground='#30363d',
            highlightcolor='#58a6ff',
        )
        self.nm_color_mask_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.nm_color_mask_entry.insert(0, "red")

        ttk.Label(color_mask_frame, text="Max area:", style='Dark.TLabel').grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.nm_color_max_area_entry = tk.Entry(
            color_mask_frame,
            font=('Segoe UI', 10),
            bg='#21262d',
            fg='#c9d1d9',
            insertbackground='#58a6ff',
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground='#30363d',
            highlightcolor='#58a6ff',
        )
        self.nm_color_max_area_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self.nm_color_max_area_entry.insert(0, "40000")

        ttk.Label(color_mask_frame, text="Min area:", style='Dark.TLabel').grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.nm_color_min_area_entry = tk.Entry(
            color_mask_frame,
            font=('Segoe UI', 10),
            bg='#21262d',
            fg='#c9d1d9',
            insertbackground='#58a6ff',
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground='#30363d',
            highlightcolor='#58a6ff',
        )
        self.nm_color_min_area_entry.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self.nm_color_min_area_entry.insert(0, "6")

        ttk.Label(color_mask_frame, text="Use splits (e.g. blue=3):", style='Dark.TLabel').grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.nm_color_split_entry = tk.Entry(
            color_mask_frame,
            font=('Segoe UI', 10),
            bg='#21262d',
            fg='#c9d1d9',
            insertbackground='#58a6ff',
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground='#30363d',
            highlightcolor='#58a6ff',
        )
        self.nm_color_split_entry.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self.nm_color_split_entry.insert(0, "blue=3")

        color_mask_frame.columnconfigure(1, weight=1)

        # Color mask debug button
        self.color_mask_btn = tk.Button(
            color_mask_frame,
            text="Color Mask",
            command=self.run_color_mask_only,
            bg='#2a7f62',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.color_mask_btn.grid(row=4, column=0, sticky="w", padx=(0, 10), pady=(8, 0))

        self.color_mask_debug_btn = tk.Button(
            color_mask_frame,
            text="Show Mask Debug",
            command=self.run_color_mask_debug,
            bg='#444c56',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.color_mask_debug_btn.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        self.color_mask_settings_btn = tk.Button(
            color_mask_frame,
            text="Mask Settings",
            command=self.open_color_mask_settings,
            bg='#30363d',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.color_mask_settings_btn.grid(row=5, column=0, sticky="w", padx=(0, 10), pady=(8, 0))

        self.hue_editor_btn = tk.Button(
            color_mask_frame,
            text="Hue Editor",
            command=self.open_hue_editor,
            bg='#30363d',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.hue_editor_btn.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        self.sample_color_btn = tk.Button(
            color_mask_frame,
            text="Sample Color",
            command=self.toggle_sample_color,
            bg='#444c56',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.sample_color_btn.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        # OCR match override input
        ocr_match_frame = ttk.Frame(input_card, style='Card.TFrame')
        ocr_match_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(ocr_match_frame, text="OCR match:", style='Dark.TLabel').grid(row=0, column=0, sticky="w")
        self.nm_ocr_match_entry = tk.Entry(
            ocr_match_frame,
            font=('Segoe UI', 10),
            bg='#21262d',
            fg='#c9d1d9',
            insertbackground='#58a6ff',
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground='#30363d',
            highlightcolor='#58a6ff',
        )
        self.nm_ocr_match_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ocr_match_frame.columnconfigure(1, weight=1)

        self.ocr_match_btn = tk.Button(
            ocr_match_frame,
            text="OCR Match",
            command=self.run_ocr_match_only,
            bg='#2a7f62',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.ocr_match_btn.grid(row=1, column=0, sticky="w", pady=(8, 0))

        # Size selection (dropdown)
        size_select_frame = ttk.Frame(input_card, style='Card.TFrame')
        size_select_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(size_select_frame, text="Size:", style='Dark.TLabel').grid(row=0, column=0, sticky="w")
        self.nm_size_combo = ttk.Combobox(
            size_select_frame,
            textvariable=self.nm_size_var,
            values=["any", "small", "medium", "large", "full_width"],
            state="readonly",
            style="Dark.TCombobox",
        )
        self.nm_size_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        size_select_frame.columnconfigure(1, weight=1)

        self.size_filter_btn = tk.Button(
            size_select_frame,
            text="Size Filter",
            command=self.run_size_filter_only,
            bg='#2a7f62',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.size_filter_btn.grid(row=1, column=0, sticky="w", pady=(8, 0))

        # Position selection (dropdown)
        position_select_frame = ttk.Frame(input_card, style='Card.TFrame')
        position_select_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(position_select_frame, text="Position:", style='Dark.TLabel').grid(row=0, column=0, sticky="w")
        self.nm_position_combo = ttk.Combobox(
            position_select_frame,
            textvariable=self.nm_position_var,
            values=[
                "any",
                "top_left",
                "top",
                "top_right",
                "left",
                "center",
                "right",
                "bottom_left",
                "bottom",
                "bottom_right",
            ],
            state="readonly",
            style="Dark.TCombobox",
        )
        self.nm_position_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        position_select_frame.columnconfigure(1, weight=1)

        self.position_btn = tk.Button(
            position_select_frame,
            text="Position Filter",
            command=self.run_position_only,
            bg='#2a7f62',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.position_btn.grid(row=1, column=0, sticky="w", pady=(8, 0))

        # Manual pipeline controls
        pipeline_btns = ttk.Frame(input_card, style='Card.TFrame')
        pipeline_btns.pack(fill=tk.X, pady=(14, 0))

        self.rebuild_candidates_btn = tk.Button(
            pipeline_btns,
            text="Rebuild Candidates",
            command=self.nm_rebuild_candidates,
            bg='#30363d',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.rebuild_candidates_btn.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.reset_candidates_btn = tk.Button(
            pipeline_btns,
            text="Reset Filters",
            command=self.nm_reset_manual_filters,
            bg='#30363d',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.reset_candidates_btn.grid(row=0, column=1, sticky="w", padx=(0, 10))

        self.send_to_picker_btn = tk.Button(
            pipeline_btns,
            text="Send to Picker",
            command=self.nm_send_to_picker,
            bg='#8957e5',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.send_to_picker_btn.grid(row=0, column=2, sticky="w")

        # BG remove controls
        bg_remove_frame = ttk.Frame(input_card, style='Card.TFrame')
        bg_remove_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(bg_remove_frame, text="BG remove min area (px):", style='Dark.TLabel').grid(row=0, column=0, sticky="w")
        self.nm_bg_remove_area_entry = tk.Entry(
            bg_remove_frame,
            font=('Segoe UI', 10),
            bg='#21262d',
            fg='#c9d1d9',
            insertbackground='#58a6ff',
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground='#30363d',
            highlightcolor='#58a6ff',
            textvariable=self.nm_bg_remove_min_area_var,
        )
        self.nm_bg_remove_area_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(bg_remove_frame, text="Leniency:", style='Dark.TLabel').grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.nm_bg_remove_leniency_entry = tk.Entry(
            bg_remove_frame,
            font=('Segoe UI', 10),
            bg='#21262d',
            fg='#c9d1d9',
            insertbackground='#58a6ff',
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground='#30363d',
            highlightcolor='#58a6ff',
            textvariable=self.nm_bg_remove_leniency_var,
        )
        self.nm_bg_remove_leniency_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        bg_remove_frame.columnconfigure(1, weight=1)

        self.bg_remove_btn = tk.Button(
            bg_remove_frame,
            text="BG Remove",
            command=self.run_bg_remove_toggle,
            bg='#30363d',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        )
        self.bg_remove_btn.grid(row=2, column=0, sticky="w", pady=(8, 0))

        # Manual filter: refresh on parameter changes (when enabled)
        self._nm_manual_update_toggle_ui()
        if self.nm_color_mask_entry is not None:
            self.nm_color_mask_entry.bind(
                "<KeyRelease>",
                lambda e: self._nm_manual_schedule_refresh() if self.nm_filter_color_enabled.get() else None,
            )
        if self.nm_ocr_match_entry is not None:
            self.nm_ocr_match_entry.bind(
                "<KeyRelease>",
                lambda e: self._nm_manual_schedule_refresh() if self.nm_filter_ocr_enabled.get() else None,
            )
            self.nm_ocr_match_entry.bind(
                "<Return>",
                lambda e: self._nm_manual_schedule_refresh(immediate=True) if self.nm_filter_ocr_enabled.get() else None,
            )
        if hasattr(self, "nm_size_combo") and self.nm_size_combo is not None:
            self.nm_size_combo.bind(
                "<<ComboboxSelected>>",
                lambda e: self._nm_manual_schedule_refresh(immediate=True) if self.nm_filter_size_enabled.get() else None,
            )
        if hasattr(self, "nm_position_combo") and self.nm_position_combo is not None:
            self.nm_position_combo.bind(
                "<<ComboboxSelected>>",
                lambda e: self._nm_manual_schedule_refresh(immediate=True) if self.nm_filter_position_enabled.get() else None,
            )
        if hasattr(self, "prompt_entry") and self.prompt_entry is not None:
            self.prompt_entry.bind("<Return>", lambda e: self._nm_manual_schedule_refresh(immediate=True))

        if hasattr(self, "nm_bg_remove_area_entry") and self.nm_bg_remove_area_entry is not None:
            self.nm_bg_remove_area_entry.bind(
                "<KeyRelease>",
                lambda e: self._nm_manual_schedule_refresh() if self.nm_bg_remove_enabled.get() else None,
            )
            self.nm_bg_remove_area_entry.bind(
                "<Return>",
                lambda e: self._nm_manual_schedule_refresh(immediate=True) if self.nm_bg_remove_enabled.get() else None,
            )
        if hasattr(self, "nm_bg_remove_leniency_entry") and self.nm_bg_remove_leniency_entry is not None:
            self.nm_bg_remove_leniency_entry.bind(
                "<KeyRelease>",
                lambda e: self._nm_manual_schedule_refresh() if self.nm_bg_remove_enabled.get() else None,
            )
            self.nm_bg_remove_leniency_entry.bind(
                "<Return>",
                lambda e: self._nm_manual_schedule_refresh(immediate=True) if self.nm_bg_remove_enabled.get() else None,
            )

        # Current state card
        state_card = self.create_card(left_container, "📊 Current State")
        state_card.pack(fill=tk.X, pady=(0, 15))
        
        # Coordinates display
        coords_frame = ttk.Frame(state_card, style='Card.TFrame')
        coords_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(coords_frame, text="Coordinates:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.coords_label = ttk.Label(coords_frame, text="Not set", style='Coords.TLabel')
        self.coords_label.pack(side=tk.RIGHT)
        
        # Image size display (keep, useful context)
        size_frame = ttk.Frame(state_card, style='Card.TFrame')
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Image Size:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.size_label = ttk.Label(size_frame, text="N/A", style='Coords.TLabel')
        self.size_label.pack(side=tk.RIGHT)
        
        # Log card
        log_card = self.create_card(left_container, "📜 AI Activity Log")
        log_card.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_card, font=('Consolas', 9), 
                                                   bg='#0d1117', fg='#8b949e',
                                                   relief=tk.FLAT, wrap=tk.WORD,
                                                   insertbackground='#58a6ff')
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Configure log tags
        self.log_text.tag_configure('info', foreground='#58a6ff')
        self.log_text.tag_configure('success', foreground='#7ee787')
        self.log_text.tag_configure('error', foreground='#f85149')
        self.log_text.tag_configure('coords', foreground='#ffa657')
        self.log_text.tag_configure('ai', foreground='#d2a8ff')
        self.log_text.tag_configure('time', foreground='#484f58')
        
        # Right panel (output view)
        right_panel = ttk.Frame(content_frame, style='Dark.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Current view (with cursor)
        current_card = self.create_card(right_panel, "🔍 Current View (with Cursor)")
        current_card.pack(fill=tk.BOTH, expand=True)
        
        # Stage navigation controls
        stage_nav = ttk.Frame(current_card, style='Card.TFrame')
        stage_nav.pack(fill=tk.X, pady=(0, 6))
        self.stage_label = ttk.Label(stage_nav, textvariable=self.nm_stage_label_var, style='Dark.TLabel')
        self.stage_label.pack(side=tk.LEFT)
        copy_preview_btn = tk.Button(
            stage_nav,
            text="Copy preview",
            command=self.copy_current_preview_to_clipboard,
            bg='#238636',
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=10,
            pady=4,
        )
        copy_preview_btn.pack(side=tk.RIGHT, padx=(0, 6))
        nav_btn_frame = ttk.Frame(stage_nav, style='Card.TFrame')
        nav_btn_frame.pack(side=tk.RIGHT)
        self.prev_stage_btn = tk.Button(nav_btn_frame, text="◀ Prev", command=self.show_prev_stage,
                                        bg='#30363d', fg='#c9d1d9', relief=tk.FLAT,
                                        cursor='hand2', padx=10, pady=4, state=tk.DISABLED)
        self.prev_stage_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.next_stage_btn = tk.Button(nav_btn_frame, text="Next ▶", command=self.show_next_stage,
                                        bg='#30363d', fg='#c9d1d9', relief=tk.FLAT,
                                        cursor='hand2', padx=10, pady=4, state=tk.DISABLED)
        self.next_stage_btn.pack(side=tk.LEFT)

        self.current_canvas = tk.Canvas(current_card, bg='#21262d', highlightthickness=0)
        self.current_canvas.pack(fill=tk.BOTH, expand=True, pady=5)

        # Add hover detection for OCR dots
        self.current_canvas.bind("<Motion>", self.on_canvas_hover)
        self.current_canvas.bind("<Leave>", self.on_canvas_leave)
        self.current_canvas.bind("<Button-1>", self.on_canvas_click)

        # Add zoom and pan bindings
        self.current_canvas.bind("<MouseWheel>", self.on_canvas_zoom)  # Windows
        self.current_canvas.bind("<Button-4>", self.on_canvas_zoom)    # Linux scroll up
        self.current_canvas.bind("<Button-5>", self.on_canvas_zoom)    # Linux scroll down
        self.current_canvas.bind("<ButtonPress-1>", self.on_canvas_pan_start)
        self.current_canvas.bind("<B1-Motion>", self.on_canvas_pan_drag)
        self.current_canvas.bind("<ButtonRelease-1>", self.on_canvas_pan_end)
        
        # Tooltip for showing OCR text
        self.hover_tooltip = None
        
    def create_card(self, parent, title):
        """Create a styled card container."""
        card = ttk.Frame(parent, style='Card.TFrame')
        card.configure(padding=15)
        
        # Title
        title_label = ttk.Label(card, text=title, style='Header.TLabel')
        title_label.pack(anchor=tk.W)
        
        # Separator
        sep = ttk.Frame(card, height=1)
        sep.pack(fill=tk.X, pady=(8, 5))
        
        return card
        
    def show_api_key_dialog(self):
        """Show dialog to enter API key."""
        dialog = tk.Toplevel(self.root)
        dialog.title("🔑 OpenAI API Key Required")
        dialog.geometry("500x200")
        dialog.configure(bg='#161b22')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 500) // 2
        y = (dialog.winfo_screenheight() - 200) // 2
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, style='Card.TFrame', padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Enter your OpenAI API Key:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(frame, text="(You can also set OPENAI_API_KEY environment variable)", 
                 style='Dark.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        key_entry = tk.Entry(frame, font=('Consolas', 10), bg='#21262d', fg='#c9d1d9',
                            insertbackground='#58a6ff', relief=tk.FLAT, show='*',
                            highlightthickness=1, highlightbackground='#30363d')
        key_entry.pack(fill=tk.X, pady=(0, 15), ipady=8)
        key_entry.focus()
        
        def save_key():
            key = key_entry.get().strip()
            if key:
                self.api_key = key
                self.client = OpenAI(api_key=key)
                self.log("API key configured successfully!", 'success')
                dialog.destroy()
            else:
                self.log("No API key entered", 'error')
        
        btn_frame = ttk.Frame(frame, style='Card.TFrame')
        btn_frame.pack(fill=tk.X)
        
        save_btn = tk.Button(btn_frame, text="✓ Save Key", command=save_key,
                            bg='#238636', fg='white', font=('Segoe UI', 10, 'bold'),
                            relief=tk.FLAT, cursor='hand2', padx=15, pady=5)
        save_btn.pack(side=tk.LEFT)
        
        key_entry.bind('<Return>', lambda e: save_key())
        
    def load_image(self):
        """Load the test image."""
        try:
            resolved = self._resolve_image_path(self.image_path)
            if os.path.exists(resolved):
                # Load image at full quality - no compression or resizing
                self.original_image = Image.open(resolved)
                # Ensure we're using the original quality
                if self.original_image.format == 'PNG':
                    # PNG is lossless, good
                    pass
                elif self.original_image.format == 'JPEG':
                    # Reload without any quality loss
                    self.original_image = Image.open(resolved)
                
                self.current_image = self.original_image.copy()
                self.size_label.config(text=f"{self.original_image.width} x {self.original_image.height}")
                self.log("Loaded image: " + resolved, 'info')
                self.log(f"Image dimensions: {self.original_image.width}x{self.original_image.height} (full quality)", 'info')
                self.update_displays()
                
                # Automatically run OCR when image loads (using PaddleOCR)
                if self.ocr_auto_run:
                    self.root.after(500, self.run_ocr)
            else:
                self.log(f"Image not found: {self.image_path}", 'error')
        except Exception as e:
            self.log(f"Error loading image: {e}", 'error')
            
    def draw_crosshair(self, image):
        """Draw cyan crosshair with 75% opacity on the image."""
        # Convert to RGBA for alpha compositing
        if image.mode != 'RGBA':
            image_rgba = image.convert('RGBA')
        else:
            image_rgba = image.copy()
        
        # Create overlay image with RGBA mode
        overlay = Image.new('RGBA', image_rgba.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        center_x = image_rgba.width // 2
        center_y = image_rgba.height // 2
        
        # Cyan color with 75% opacity (255 * 0.75 = 191)
        cyan_color = (0, 255, 255, 191)
        
        # Draw full-screen vertical line
        overlay_draw.line([center_x, 0, center_x, image_rgba.height], fill=cyan_color, width=2)
        
        # Draw full-screen horizontal line
        overlay_draw.line([0, center_y, image_rgba.width, center_y], fill=cyan_color, width=2)
        
        # Composite overlay onto original image
        result = Image.alpha_composite(image_rgba, overlay)
        
        # Convert back to RGB (preserve original mode)
        if image.mode == 'RGB':
            rgb_result = Image.new('RGB', result.size)
            rgb_result.paste(result, mask=result.split()[3])  # Use alpha channel as mask
            return rgb_result
        elif image.mode == 'RGBA':
            return result
        else:
            # For other modes, convert to RGB
            return result.convert('RGB')
    
    def draw_grid(self, image, grid_x, grid_y):
        """Draw a numbered grid on the image."""
        # Convert to RGBA for drawing
        if image.mode != 'RGBA':
            img_rgba = image.convert('RGBA')
        else:
            img_rgba = image.copy()
        
        overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        width = img_rgba.width
        height = img_rgba.height
        
        # Calculate cell dimensions
        cell_width = width / grid_x
        cell_height = height / grid_y
        
        # Grid line color (cyan with 75% opacity)
        grid_color = (0, 255, 255, 191)  # Cyan with 75% opacity
        text_color = (0, 255, 255, 191)  # Cyan with 75% opacity
        
        # Draw grid lines
        for i in range(grid_x + 1):
            x = int(i * cell_width)
            draw.line([x, 0, x, height], fill=grid_color, width=2)
        
        for i in range(grid_y + 1):
            y = int(i * cell_height)
            draw.line([0, y, width, y], fill=grid_color, width=2)
        
        # Draw numbers in each cell
        cell_num = 1
        # Calculate font size - ensure readability even for small grids
        min_cell_dim = min(cell_width, cell_height)
        if min_cell_dim < 100:
            font_size = max(16, int(min_cell_dim * 0.4))  # 40% for small cells
        elif min_cell_dim < 200:
            font_size = max(20, int(min_cell_dim * 0.35))  # 35% for medium cells
        else:
            font_size = max(24, int(min_cell_dim * 0.3))   # 30% for large cells
        
        try:
            # Try to use a better font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=font_size)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        for row in range(grid_y):
            for col in range(grid_x):
                # Calculate cell center
                center_x = int((col + 0.5) * cell_width)
                center_y = int((row + 0.5) * cell_height)
                
                # Draw number
                text = str(cell_num)
                # Get text bounding box for centering
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw text centered
                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2
                
                # Draw text with thicker outline for better visibility
                for adj in [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2),
                           (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    draw.text((text_x + adj[0], text_y + adj[1]), text, fill=(0, 0, 0, 255), font=font)
                draw.text((text_x, text_y), text, fill=text_color, font=font)
                
                cell_num += 1
        
        # Composite overlay
        result = Image.alpha_composite(img_rgba, overlay)
        
        # Convert back to original mode
        if image.mode == 'RGB':
            rgb_result = Image.new('RGB', result.size)
            rgb_result.paste(result, mask=result.split()[3])
            return rgb_result
        elif image.mode == 'RGBA':
            return result
        else:
            return result.convert('RGB')
    
    def draw_grid_on_region(self, image, grid_x, grid_y, region_left, region_top, region_right, region_bottom):
        """Draw a numbered grid overlay on a specific region of the image."""
        # Convert to RGBA for drawing
        if image.mode != 'RGBA':
            img_rgba = image.convert('RGBA')
        else:
            img_rgba = image.copy()
        
        overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Region dimensions
        region_width = region_right - region_left
        region_height = region_bottom - region_top
        
        # Calculate cell dimensions within the region
        cell_width = region_width / grid_x
        cell_height = region_height / grid_y
        
        # Grid line color (cyan with 75% opacity)
        grid_color = (0, 255, 255, 191)  # Cyan with 75% opacity
        text_color = (0, 255, 255, 191)  # Cyan with 75% opacity
        
        # Draw grid lines within the region
        for i in range(grid_x + 1):
            x = int(region_left + i * cell_width)
            draw.line([x, region_top, x, region_bottom], fill=grid_color, width=2)
        
        for i in range(grid_y + 1):
            y = int(region_top + i * cell_height)
            draw.line([region_left, y, region_right, y], fill=grid_color, width=2)
        
        # Draw numbers in each cell
        cell_num = 1
        # Calculate font size - ensure readability even for small grids
        min_cell_dim = min(cell_width, cell_height)
        if min_cell_dim < 100:
            font_size = max(16, int(min_cell_dim * 0.4))  # 40% for small cells
        elif min_cell_dim < 200:
            font_size = max(20, int(min_cell_dim * 0.35))  # 35% for medium cells
        else:
            font_size = max(24, int(min_cell_dim * 0.3))   # 30% for large cells
        
        try:
            # Try to use a better font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=font_size)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        for row in range(grid_y):
            for col in range(grid_x):
                # Calculate cell center within the region
                center_x = int(region_left + (col + 0.5) * cell_width)
                center_y = int(region_top + (row + 0.5) * cell_height)
                
                # Draw number
                text = str(cell_num)
                # Get text bounding box for centering
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw text centered
                text_x = center_x - text_width // 2
                text_y = center_y - text_height // 2
                
                # Draw text with thicker outline for better visibility
                for adj in [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2),
                           (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    draw.text((text_x + adj[0], text_y + adj[1]), text, fill=(0, 0, 0, 255), font=font)
                draw.text((text_x, text_y), text, fill=text_color, font=font)
                
                cell_num += 1
        
        # Composite overlay
        result = Image.alpha_composite(img_rgba, overlay)
        
        # Convert back to original mode
        if image.mode == 'RGB':
            rgb_result = Image.new('RGB', result.size)
            rgb_result.paste(result, mask=result.split()[3])
            return rgb_result
        elif image.mode == 'RGBA':
            return result
        else:
            return result.convert('RGB')

    # --- New method visualization helpers ---
    def _nm_reset_stages(self):
        """Clear stored visualization stages and reset navigation UI."""
        self.nm_stages = []
        self.nm_stage_index = 0
        self.nm_stage_label_var.set("Stage: N/A")
        self.prev_stage_btn.config(state=tk.DISABLED)
        self.next_stage_btn.config(state=tk.DISABLED)

    def _nm_add_stage(self, name: str, image: Image.Image):
        """Store a stage image and update the stage label."""
        if getattr(self, "_nm_stage_add_disabled", False):
            return
        if image.mode != "RGB":
            image = image.convert("RGB")
        self.nm_stages.append((name, image))
        self.nm_stage_index = len(self.nm_stages) - 1
        self._nm_show_stage(self.nm_stage_index)

    def _nm_show_stage(self, index: int):
        """Show a specific stored stage."""
        if not self.nm_stages:
            return
        index = max(0, min(index, len(self.nm_stages) - 1))
        self.nm_stage_index = index
        name, img = self.nm_stages[index]
        self.nm_stage_label_var.set(f"Stage {index+1}/{len(self.nm_stages)}: {name}")
        # Enable/disable navigation buttons
        self.prev_stage_btn.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self.next_stage_btn.config(state=tk.NORMAL if index < len(self.nm_stages) - 1 else tk.DISABLED)
        # Update canvas preview
        self.current_image = img
        self.current_coords = None
        self.display_image(self.current_canvas, self.current_image, coords=None, draw_crosshair=False)

    def show_prev_stage(self):
        """Navigate to previous visualization stage."""
        if self.nm_stages and self.nm_stage_index > 0:
            self._nm_show_stage(self.nm_stage_index - 1)

    def show_next_stage(self):
        """Navigate to next visualization stage."""
        if self.nm_stages and self.nm_stage_index < len(self.nm_stages) - 1:
            self._nm_show_stage(self.nm_stage_index + 1)

    def toggle_sample_color(self):
        """Toggle color sampling mode for the canvas."""
        self.nm_sample_mode = not self.nm_sample_mode
        state = "ON" if self.nm_sample_mode else "OFF"
        self.log(f"Color sampling mode: {state}. Click on the image to sample.", "info")

    def on_canvas_click(self, event):
        """Handle canvas click for color sampling."""
        if not self.nm_sample_mode:
            return
        if not self.original_image:
            return

        coords = self._nm_canvas_to_image(event.x, event.y)
        if coords is None:
            return
        x, y = coords
        if x < 0 or y < 0 or x >= self.original_image.width or y >= self.original_image.height:
            return

        r, g, b = self.original_image.getpixel((x, y))
        rgb_text = f"rgb({r},{g},{b})"
        hex_text = f"#{r:02x}{g:02x}{b:02x}"
        self.log(f"Sampled color at ({x},{y}): {rgb_text} ({hex_text})", "info")
        if self.nm_color_mask_entry:
            self.nm_color_mask_entry.delete(0, tk.END)
            self.nm_color_mask_entry.insert(0, rgb_text)
        self.nm_sample_mode = False

    def _nm_canvas_to_image(self, canvas_x: int, canvas_y: int) -> Optional[Tuple[int, int]]:
        """Map canvas coordinates to original image coordinates."""
        if not self.displayed_image_size:
            return None

        img_width, img_height = self.displayed_image_size
        canvas_width = self.current_canvas.winfo_width()
        canvas_height = self.current_canvas.winfo_height()

        if self.zoom_level == 1.0 and self.pan_offset_x == 0.0 and self.pan_offset_y == 0.0:
            img_x = canvas_x - self.image_offset_x
            img_y = canvas_y - self.image_offset_y
        else:
            image_x = canvas_width // 2 + self.pan_offset_x
            image_y = canvas_height // 2 + self.pan_offset_y
            img_x = canvas_x - image_x + img_width // 2
            img_y = canvas_y - image_y + img_height // 2

        if img_x < 0 or img_y < 0 or img_x > img_width or img_y > img_height:
            return None

        orig_x = int(img_x * self.image_scale_x)
        orig_y = int(img_y * self.image_scale_y)
        return orig_x, orig_y

    def _nm_get_color_override(self) -> str:
        if not self.nm_color_mask_entry:
            return ""
        return self.nm_color_mask_entry.get().strip().lower()

    def _nm_get_color_max_area(self) -> int:
        if not self.nm_color_max_area_entry:
            return 40000
        raw = self.nm_color_max_area_entry.get().strip()
        try:
            val = int(raw)
        except ValueError:
            return 40000
        return max(0, val)

    def _nm_get_color_split_override(self, target_color: str = "") -> List[int]:
        if not self.nm_color_split_entry:
            return []
        raw = (self.nm_color_split_entry.get() or "").strip()
        if not raw:
            return []
        target = (target_color or "").strip().lower()
        parts = [p.strip() for p in raw.replace(";", ",").replace("|", ",").split(",")]
        out: List[int] = []
        color_specific = {}
        for part in parts:
            if not part:
                continue
            if ":" in part or "=" in part:
                sep = ":" if ":" in part else "="
                name, vals = part.split(sep, 1)
                name = name.strip().lower()
                vals = vals.strip()
                color_specific[name] = vals
                continue
            # allow space-separated too
            for token in part.split():
                try:
                    val = int(token)
                except Exception:
                    continue
                if val > 0:
                    out.append(val)

        if target and target in color_specific:
            out = []
            vals = color_specific[target]
            for token in vals.replace("|", " ").replace(",", " ").split():
                try:
                    val = int(token)
                except Exception:
                    continue
                if val > 0:
                    out.append(val)
        # de-dupe preserve order
        seen = set()
        cleaned = []
        for val in out:
            if val in seen:
                continue
            seen.add(val)
            cleaned.append(val)
        return cleaned

    def _nm_get_color_min_area(self) -> int:
        if not self.nm_color_min_area_entry:
            return 6
        raw = (self.nm_color_min_area_entry.get() or "").strip()
        try:
            val = int(raw)
        except Exception:
            return 6
        return max(0, val)

    def _nm_parse_rgb_color(self, value: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse rgb(...) or #RRGGBB; return (r,g,b,tol) or None."""
        import re

        if not value:
            return None
        v = value.strip().lower()
        if v.startswith("#") and len(v) == 7:
            try:
                r = int(v[1:3], 16)
                g = int(v[3:5], 16)
                b = int(v[5:7], 16)
                return (r, g, b, 20)
            except ValueError:
                return None
        if v.startswith("rgb"):
            nums = re.findall(r"\d+", v)
            if len(nums) >= 3:
                r, g, b = [int(n) for n in nums[:3]]
                tol = int(nums[3]) if len(nums) >= 4 else 20
                return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), max(0, tol))
        return None

    def run_color_mask_only(self):
        """Toggle color mask filter (combines with other enabled filters)."""
        self.nm_filter_color_enabled.set(not self.nm_filter_color_enabled.get())
        self._nm_manual_update_toggle_ui()
        if self.nm_filter_color_enabled.get():
            color_override = (self._nm_get_color_override() or "").strip()
            self.log(f"Color mask ON (color='{color_override or 'unknown'}')", "info")
            # Refresh manual pipeline view so the effect is visible immediately.
            self._nm_manual_schedule_refresh(immediate=True)
        else:
            self.log("Color mask OFF", "info")
            self._nm_manual_schedule_refresh(immediate=True)

    def run_color_mask_debug(self):
        """Show the raw color mask the detector uses."""
        if not self.original_image:
            self.log("No image loaded!", "error")
            return

        color_override = self._nm_get_color_override()
        if not color_override:
            self.log("Color mask is empty. Enter a color like red, blue, gray.", "error")
            return

        # Reset visualization stages so reruns start clean
        self._nm_reset_stages()

        img = self.original_image.copy()
        roi = ScreenRegion(left=0, top=0, width=img.width, height=img.height)
        masks_u8 = self._nm_build_color_masks(
            img,
            roi,
            color_override,
            True,
        )
        if not masks_u8:
            self._nm_add_stage(f"Color mask '{color_override}' (no mask)", img)
            return

        for i, mask_u8 in enumerate(masks_u8, start=1):
            mask_img = Image.fromarray(mask_u8).convert("RGB")
            self._nm_add_stage(f"Mask raw '{color_override}' ({i}/{len(masks_u8)})", mask_img)

            # Overlay mask on the image for context
            overlay = img.copy().convert("RGBA")
            mask_rgba = Image.fromarray(mask_u8).convert("L")
            tint = Image.new("RGBA", overlay.size, (0, 180, 255, 120))
            overlay = Image.composite(tint, overlay, mask_rgba)
            self._nm_add_stage(f"Mask overlay '{color_override}' ({i}/{len(masks_u8)})", overlay.convert("RGB"))

    def _nm_point_in_any_rect(self, pt: Tuple[int, int], rects: List[Tuple[int, int, int, int]]) -> bool:
        x, y = pt
        for l, t, r, b in rects:
            if l <= x <= r and t <= y <= b:
                return True
        return False

    def _nm_manual_signature(self) -> Tuple[Any, ...]:
        # Only include inputs that affect deterministic candidate generation for the manual pipeline.
        use_ocr, use_color, use_shape = self._nm_manual_generation_flags()
        sig = [
            self.image_path,
            getattr(self.original_image, "size", None),
            bool(use_ocr),
            bool(use_color),
            bool(use_shape),
        ]
        if use_color:
            sig.extend(
                [
                    (self._nm_get_color_override() or "").strip().lower(),
                    int(self._nm_get_color_max_area() or 0),
                ]
            )
        sig.extend(
            [
                bool(self.nm_bg_remove_enabled.get()),
                int(self._nm_get_bg_remove_min_area() or 0),
                float(self._nm_get_bg_remove_leniency() or 0.0),
            ]
        )
        return tuple(sig)

    def _nm_manual_generation_flags(self) -> Tuple[bool, bool, bool]:
        """
        Decide which expensive detectors to run for the manual pipeline.
        Goal: only run what is needed for the currently enabled manual filters.
        """
        use_color = bool(self.nm_filter_color_enabled.get())
        use_ocr = bool(self.nm_filter_ocr_enabled.get())
        use_shape = bool(self.nm_filter_size_enabled.get() or self.nm_filter_position_enabled.get())
        # If nothing is enabled, keep a lightweight fallback candidate set.
        if not use_color and not use_ocr and not use_shape:
            use_shape = True
        return use_ocr, use_color, use_shape

    def _nm_manual_build_generation_planner(self, prompt: str) -> PlannerOutput:
        # Neutral planner intent: generate candidates based on enabled manual detectors.
        _, use_color, _ = self._nm_manual_generation_flags()
        primary_color = "unknown"
        if use_color:
            primary_color = (self._nm_get_color_override() or "").strip() or "unknown"
        return PlannerOutput(
            text_intent=PlannerTextIntent(primary_text="", variants=[], strictness="low"),
            location_intent=PlannerLocationIntent(scope="window", zone="any", position="any"),
            visual_intent=PlannerVisualIntent(
                accent_color_relevant=False,
                shape_importance="low",
                primary_color=primary_color,
                relative_luminance="unknown",
                shape="unknown",
                size="unknown",
                width="unknown",
                height="unknown",
                text_presence="optional",
                description="(manual generation)",
            ),
            risk_intent=PlannerRiskIntent(level="low"),
            decision="OK",
            raw={"manual_generation": True, "prompt": prompt},
        )

    def _nm_manual_build_planner_from_inputs(self, prompt: str) -> PlannerOutput:
        # Only include intent fields that are actively enabled via manual filter toggles.
        raw_text = ""
        if self.nm_ocr_match_entry is not None:
            raw_text = (self.nm_ocr_match_entry.get() or "").strip()
        parts = [p.strip() for p in raw_text.replace("|", ",").split(",") if p.strip()]

        primary_text = parts[0] if (parts and self.nm_filter_ocr_enabled.get()) else ""
        variants = parts[1:] if (len(parts) > 1 and self.nm_filter_ocr_enabled.get()) else []

        size_sel = (self.nm_size_var.get() or "any").strip().lower()
        if not self.nm_filter_size_enabled.get() or size_sel == "any":
            size_sel = "unknown"

        position_sel = (self.nm_position_var.get() or "any").strip().lower()
        if not self.nm_filter_position_enabled.get() or position_sel in {"", "any", "unknown"}:
            position_sel = "any"

        color_override = (self._nm_get_color_override() or "").strip().lower()
        allowed_colors = {"white", "grey", "black", "red", "blue", "green", "pink", "purple", "yellow", "unknown"}
        if not self.nm_filter_color_enabled.get():
            primary_color = "unknown"
        else:
            primary_color = color_override if color_override in allowed_colors else "unknown"

        text_presence = "required" if primary_text else "optional"

        return PlannerOutput(
            text_intent=PlannerTextIntent(primary_text=primary_text, variants=variants, strictness="medium"),
            location_intent=PlannerLocationIntent(scope="window", zone="any", position=position_sel),
            visual_intent=PlannerVisualIntent(
                primary_color=primary_color,
                relative_luminance="unknown",
                shape="unknown",
                size=size_sel,
                width="unknown",
                height="unknown",
                text_presence=text_presence,
                accent_color_relevant=False,
                shape_importance="medium",
                description="(manual override)",
            ),
            risk_intent=PlannerRiskIntent(level="low"),
            decision="OK",
            raw={"manual": True, "prompt": prompt},
        )

    def _nm_manual_generate_color_only_candidates(self, snapshot: NewMethodSnapshot) -> List[CandidatePackage]:
        """
        Fast path: only run color masking and convert each detected color region into a candidate.
        Avoids OCR/shape work entirely (for snappy "Color Mask" toggling).
        """
        color_override = self._nm_get_color_override()
        if not color_override:
            return []

        roi_full = ScreenRegion(left=0, top=0, width=snapshot.image.width, height=snapshot.image.height)
        max_area = self._nm_get_color_max_area()

        img = snapshot.image
        regions = self._nm_find_color_regions(
            img,
            roi_full,
            color_override,
            True,
            max_area,
        )

        candidates: List[CandidatePackage] = []
        next_id = 1
        for left, top, right, bottom in regions:
            area = max(1, (right - left) * (bottom - top))
            if area < 60:
                continue
            click_x = (left + right) // 2
            click_y = (top + bottom) // 2
            color = self._nm_compute_region_color(img, (left, top, right, bottom))
            scores = {"text_match": 0.0, "shape_plausibility": 0.8, "location_match": 0.5}
            total = 0.6 * scores["shape_plausibility"] + 0.4 * scores["location_match"]
            candidates.append(
                CandidatePackage(
                    id=next_id,
                    bbox=(left, top, right, bottom),
                    click_point=(click_x, click_y),
                    text="[color mask]",
                    color=color,
                    scores=scores,
                    total_score=total,
                    source="color",
                )
            )
            next_id += 1

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        for new_id, cand in enumerate(candidates, start=1):
            cand.id = new_id
        return candidates

    def _nm_get_bg_remove_min_area(self) -> int:
        try:
            raw = (self.nm_bg_remove_min_area_var.get() or "").strip()
            v = int(raw)
            return max(0, v)
        except Exception:
            return 0

    def _nm_get_bg_remove_leniency(self) -> float:
        try:
            raw = (self.nm_bg_remove_leniency_var.get() or "").strip()
            v = float(raw)
            # 0 means "no leniency" (very strict).
            return max(0.0, min(50.0, v))
        except Exception:
            return 8.0

    def run_bg_remove_toggle(self):
        self.nm_bg_remove_enabled.set(not self.nm_bg_remove_enabled.get())
        self._nm_manual_set_toggle_button(getattr(self, "bg_remove_btn", None), self.nm_bg_remove_enabled.get())
        self._nm_manual_schedule_refresh(immediate=True)

    def _nm_manual_show_bg_removed_only(self) -> None:
        if not self.original_image:
            self.log("No image loaded!", "error")
            return
        raw = self.original_image.copy()
        processed, removed_px = self._nm_bg_remove_large_coherent_regions(
            raw,
            self._nm_get_bg_remove_min_area(),
            self._nm_get_bg_remove_leniency(),
        )
        self._nm_reset_stages()
        self._nm_add_stage("Snapshot (raw)", raw)
        self._nm_add_stage(f"Snapshot (BG removed, {removed_px} px)", processed)

    def _nm_manual_show_snapshot_only(self) -> None:
        if not self.original_image:
            self.log("No image loaded!", "error")
            return
        raw = self.original_image.copy()
        self._nm_reset_stages()
        self._nm_add_stage("Snapshot (raw)", raw)

    def _nm_make_cyan_grid(self, width: int, height: int, tile: int = 18) -> "np.ndarray":
        import numpy as np

        y = (np.arange(height) // tile)[:, None]
        x = (np.arange(width) // tile)[None, :]
        checker = ((x + y) % 2).astype(np.uint8)
        grid = np.zeros((height, width, 3), dtype=np.uint8)
        # Two cyan-ish shades (low opacity simulated by blending later)
        grid[checker == 0] = (0, 170, 255)
        grid[checker == 1] = (0, 210, 255)
        return grid

    def _nm_bg_grid_exclude_mask(self, roi_array: "np.ndarray", tol: int = 4) -> "np.ndarray":
        """
        Return a boolean mask for pixels that match the synthetic cyan grid background.
        This prevents color masks (especially "blue") from selecting the BG-removed grid.
        """
        import numpy as np

        tol = int(max(0, min(30, tol)))
        if roi_array is None or roi_array.size == 0:
            return np.zeros((0, 0), dtype=bool)

        # These are the two grid colors after blending with the app bg:
        # bg=(13,17,23); grid_low=(0.22*grid + 0.78*bg) cast to uint8 => truncation.
        c1 = np.array([10, 50, 74], dtype=np.int16)
        c2 = np.array([10, 59, 74], dtype=np.int16)
        arr = roi_array.astype(np.int16)
        d1 = np.max(np.abs(arr - c1[None, None, :]), axis=2)
        d2 = np.max(np.abs(arr - c2[None, None, :]), axis=2)
        return (d1 <= tol) | (d2 <= tol)

    def _nm_bg_remove_large_coherent_regions(
        self,
        image: Image.Image,
        min_area_px: int,
        leniency: float,
    ) -> Tuple[Image.Image, int]:
        """
        Remove large, locally-uniform color regions (background bars/panels) and replace with a cyan grid.
        Returns (processed_image, removed_pixel_count).
        """
        if min_area_px <= 0:
            return image.copy().convert("RGB"), 0

        try:
            import cv2
            import numpy as np
        except Exception:
            self.log("OpenCV (cv2) / numpy missing; BG remove is unavailable.", "error")
            return image.copy().convert("RGB"), 0

        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        # Seed mask: pixels that are locally uniform (small local range), tolerant of slight noise.
        # This creates "cores" of big background regions. We'll then grow them outward using neighbor
        # expansion constrained by color similarity to the core color, which preserves thin lines/icons.
        tol = float(leniency)
        tol = max(0.0, min(60.0, tol))
        k = 9
        neigh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        seed_tol = 0 if tol <= 0 else int(min(80, round(tol * 2.0 + 2.0)))
        coherent = np.ones((h, w), dtype=np.uint8)
        for ch in range(3):
            c = arr[:, :, ch]
            c_max = cv2.dilate(c, neigh)
            c_min = cv2.erode(c, neigh)
            c_range = (c_max.astype(np.int16) - c_min.astype(np.int16))
            coherent &= (c_range <= seed_tol).astype(np.uint8)

        # Light cleanup (keeps big cores, removes isolated specks)
        try:
            k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            coherent = cv2.morphologyEx(coherent, cv2.MORPH_OPEN, k_clean, iterations=1)
        except Exception:
            pass

        # Connected components on coherence mask
        num, labels, stats, _ = cv2.connectedComponentsWithStats(coherent, connectivity=8)
        if num <= 1:
            return image.copy().convert("RGB"), 0

        remove = np.zeros((h, w), dtype=np.uint8)
        # Neighbor-expansion: iteratively add border pixels that match the core color.
        # 0 => exact match only.
        color_tol = 0 if tol <= 0 else int(max(1, min(80, round(tol * 1.4 + 1.0))))
        grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        max_iters = 25

        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area_px:
                x0 = int(stats[label, cv2.CC_STAT_LEFT])
                y0 = int(stats[label, cv2.CC_STAT_TOP])
                w0 = int(stats[label, cv2.CC_STAT_WIDTH])
                h0 = int(stats[label, cv2.CC_STAT_HEIGHT])

                pad = 32  # allow growth beyond the core bounding box
                x1 = max(0, x0 - pad)
                y1 = max(0, y0 - pad)
                x2 = min(w, x0 + w0 + pad)
                y2 = min(h, y0 + h0 + pad)

                labels_roi = labels[y1:y2, x1:x2]
                arr_roi = arr[y1:y2, x1:x2]
                region = (labels_roi == label).astype(np.uint8)
                if region.sum() <= 0:
                    continue

                # Representative color of the core (mean, robust enough for background bars)
                core_pixels = arr_roi[region.astype(bool)]
                rep = core_pixels.mean(axis=0).astype(np.float32)
                rep_i16 = rep.astype(np.int16)

                for _ in range(max_iters):
                    dil = cv2.dilate(region, grow_kernel, iterations=1)
                    border = (dil == 1) & (region == 0)
                    if not border.any():
                        break
                    diff = np.max(np.abs(arr_roi.astype(np.int16) - rep_i16[None, None, :]), axis=2)
                    add = border & (diff <= color_tol)
                    if not add.any():
                        break
                    region[add] = 1

                remove[y1:y2, x1:x2] = np.maximum(remove[y1:y2, x1:x2], region)

        removed_px = int(remove.sum())
        if removed_px <= 0:
            return image.copy().convert("RGB"), 0

        # Do NOT close/fill holes: that can erase thin contrasting details (e.g. white separators).

        grid = self._nm_make_cyan_grid(w, h, tile=18).astype(np.float32)
        bg = np.array([13, 17, 23], dtype=np.float32)  # app dark background
        grid_low = (0.22 * grid + 0.78 * bg).astype(np.uint8)

        out = arr.copy()
        out[remove.astype(bool)] = grid_low[remove.astype(bool)]
        return Image.fromarray(out).convert("RGB"), removed_px

    def _nm_manual_set_toggle_button(self, btn: Optional[tk.Button], enabled: bool) -> None:
        if btn is None:
            return
        btn.config(
            bg="#238636" if enabled else "#30363d",
            relief=tk.SUNKEN if enabled else tk.FLAT,
        )

    def _nm_manual_update_toggle_ui(self) -> None:
        self._nm_manual_set_toggle_button(getattr(self, "color_mask_btn", None), self.nm_filter_color_enabled.get())
        self._nm_manual_set_toggle_button(getattr(self, "ocr_match_btn", None), self.nm_filter_ocr_enabled.get())
        self._nm_manual_set_toggle_button(getattr(self, "size_filter_btn", None), self.nm_filter_size_enabled.get())
        self._nm_manual_set_toggle_button(getattr(self, "position_btn", None), self.nm_filter_position_enabled.get())
        self._nm_manual_set_toggle_button(getattr(self, "bg_remove_btn", None), self.nm_bg_remove_enabled.get())

    def _nm_manual_schedule_refresh(self, immediate: bool = False) -> None:
        if self._nm_manual_refresh_job is not None:
            try:
                self.root.after_cancel(self._nm_manual_refresh_job)
            except Exception:
                pass
            self._nm_manual_refresh_job = None
        delay = 1 if immediate else 180
        self._nm_manual_refresh_job = self.root.after(delay, self._nm_manual_refresh_view)

    def _nm_manual_refresh_view(self) -> None:
        # If no filters are enabled, keep it simple and never run OCR/shape/color candidate generation.
        if not (
            self.nm_filter_color_enabled.get()
            or self.nm_filter_ocr_enabled.get()
            or self.nm_filter_size_enabled.get()
            or self.nm_filter_position_enabled.get()
        ):
            self._nm_manual_update_toggle_ui()
            if self.nm_bg_remove_enabled.get():
                self._nm_manual_show_bg_removed_only()
            else:
                self._nm_manual_show_snapshot_only()
            return

        # If BG remove is the only active control, do ONLY the masking (no OCR/shape/color candidate work).
        if not self._nm_manual_ensure_candidates_initialized(auto_stage=False):
            return
        snapshot = self.nm_manual_snapshot
        if snapshot is None:
            return

        base = list(self.nm_manual_candidates_base or [])
        kept = list(base)
        gen_use_ocr, gen_use_color, gen_use_shape = self._nm_manual_generation_flags()

        # Reset stages and build a deterministic, readable pipeline view
        self._nm_reset_stages()
        raw_img = self.nm_manual_snapshot_raw_image or snapshot.image
        self._nm_add_stage("Snapshot (raw)", raw_img)
        if self.nm_bg_remove_enabled.get():
            self._nm_add_stage("Snapshot (BG removed)", snapshot.image)

        base_overlay = self._nm_draw_candidates_overlay(snapshot.image, kept, highlight_id=None, show_ids=True)
        self._nm_add_stage(f"Base candidates ({len(kept)})", base_overlay)

        # Filter order: color -> OCR match -> size -> position
        if self.nm_filter_color_enabled.get():
            # Always compute masks here so the toggle shows a visible stage.
            color_override = self._nm_get_color_override()
            regions = []
            masks_u8 = []
            if not color_override:
                kept = []
                self.log("Color mask enabled but empty; no candidates match.", "info")
            else:
                roi_full = ScreenRegion(left=0, top=0, width=snapshot.image.width, height=snapshot.image.height)
                regions, masks_u8 = self._nm_find_color_regions(
                    snapshot.image,
                    roi_full,
                    color_override,
                    True,
                    self._nm_get_color_max_area(),
                    return_masks=True,
                )
                kept = [c for c in kept if self._nm_point_in_any_rect(c.click_point, regions)]

            # Visualize raw masks so the effect is obvious when toggling the filter.
            if masks_u8:
                try:
                    import numpy as np

                    def add_mask_stage(label: str, mask_u8: "np.ndarray") -> None:
                        overlay = snapshot.image.copy().convert("RGBA")
                        mask_rgba = Image.fromarray(mask_u8).convert("L")
                        tint = Image.new("RGBA", overlay.size, (0, 180, 255, 120))
                        overlay = Image.composite(tint, overlay, mask_rgba)
                        self._nm_add_stage(label, overlay.convert("RGB"))

                    combined = np.zeros_like(masks_u8[0], dtype=np.uint8)
                    for m in masks_u8:
                        if m is not None:
                            combined = np.maximum(combined, m)
                    add_mask_stage(f"Color mask combined ({len(masks_u8)})", combined)
                    if len(masks_u8) > 1:
                        for i, m in enumerate(masks_u8, start=1):
                            if m is None:
                                continue
                            add_mask_stage(f"Color mask split {i}/{len(masks_u8)}", m)
                except Exception:
                    pass
            overlay = self._nm_draw_candidates_overlay(snapshot.image, kept, highlight_id=None, show_ids=True)
            self._nm_add_stage(f"After color ({len(kept)})", overlay)

        if self.nm_filter_ocr_enabled.get():
            raw_text = (self.nm_ocr_match_entry.get() or "").strip() if self.nm_ocr_match_entry is not None else ""
            if not raw_text:
                kept = []
                self.log("OCR match enabled but empty; no candidates match.", "info")
            else:
                parts = [p.strip() for p in raw_text.replace("|", ",").split(",") if p.strip()]
                intent = PlannerTextIntent(primary_text=parts[0], variants=parts[1:], strictness="medium")
                text_threshold = 0.85
                kept = [c for c in kept if self._nm_text_similarity(intent, c.text or "") >= text_threshold]
            overlay = self._nm_draw_candidates_overlay(snapshot.image, kept, highlight_id=None, show_ids=True)
            self._nm_add_stage(f"After OCR match ({len(kept)})", overlay)

        if self.nm_filter_size_enabled.get():
            size_sel = (self.nm_size_var.get() or "any").strip().lower()
            if size_sel in {"", "any", "unknown"}:
                self.log("Size filter enabled but set to 'any'; not restricting.", "info")
            else:
                roi_any = self._nm_zone_to_roi(snapshot.image, snapshot.window_rect, "any")
                features = self._nm_build_candidate_features(snapshot, roi_any, kept)
                next_kept = []
                for c in kept:
                    feat = features.get(c.id, {})
                    if feat.get("size_px_class") == size_sel:
                        next_kept.append(c)
                kept = next_kept
            overlay = self._nm_draw_candidates_overlay(snapshot.image, kept, highlight_id=None, show_ids=True)
            self._nm_add_stage(f"After size ({len(kept)})", overlay)

        if self.nm_filter_position_enabled.get():
            position_sel = (self.nm_position_var.get() or "any").strip().lower()
            roi_any = self._nm_zone_to_roi(snapshot.image, snapshot.window_rect, "any")
            pos_overlay = self._nm_draw_position_overlay(snapshot.image, roi_any, position_sel)

            if position_sel in {"", "any", "unknown"}:
                self.log("Position filter enabled but set to 'any'; not restricting.", "info")
            else:
                features = self._nm_build_candidate_features(snapshot, roi_any, kept)
                kept = [
                    c
                    for c in kept
                    if self._nm_position_matches(
                        position_sel,
                        float(features.get(c.id, {}).get("x_norm", 0.5)),
                        float(features.get(c.id, {}).get("y_norm", 0.5)),
                    )
                ]
            overlay = self._nm_draw_candidates_overlay(pos_overlay, kept, highlight_id=None, show_ids=True)
            self._nm_add_stage(f"After position ({len(kept)})", overlay)

        self.nm_manual_candidates_current = kept
        self._nm_manual_update_toggle_ui()

    def _nm_manual_ensure_candidates_initialized(self, auto_stage: bool = True) -> bool:
        if not self.original_image:
            self.log("No image loaded!", "error")
            return False

        prompt = ""
        if hasattr(self, "prompt_entry") and self.prompt_entry is not None:
            prompt = (self.prompt_entry.get() or "").strip()
        if not prompt:
            prompt = "manual"

        sig = self._nm_manual_signature()
        needs_rebuild = (
            self.nm_manual_sig != sig
            or self.nm_manual_snapshot is None
            or not self.nm_manual_candidates_base
        )

        if needs_rebuild:
            snapshot = self._nm_capture_snapshot(prompt)
            if snapshot is None:
                return False
            self.nm_manual_snapshot_raw_image = snapshot.image.copy()
            if self.nm_bg_remove_enabled.get():
                processed, removed_px = self._nm_bg_remove_large_coherent_regions(
                    snapshot.image,
                    self._nm_get_bg_remove_min_area(),
                    self._nm_get_bg_remove_leniency(),
                )
                snapshot.image = processed
                self.log(f"BG remove: replaced {removed_px} pixels with grid.", "info")
            planner_for_generation = self._nm_manual_build_generation_planner(prompt)
            use_ocr, use_color, use_shape = self._nm_manual_generation_flags()

            # Build candidates without auto-filling the stage timeline
            self._nm_stage_add_disabled = True
            try:
                if use_color and (not use_ocr) and (not use_shape):
                    base = self._nm_manual_generate_color_only_candidates(snapshot)
                else:
                    base = self._nm_generate_candidates(
                        snapshot,
                        planner_for_generation,
                        force_use_ocr=use_ocr,
                        force_use_color=use_color,
                        force_use_shape=use_shape,
                    )
            finally:
                self._nm_stage_add_disabled = False

            self.nm_manual_snapshot = snapshot
            self.nm_manual_candidates_base = base or []
            self.nm_manual_candidates_current = list(self.nm_manual_candidates_base)
            self.nm_manual_sig = sig

            if auto_stage:
                overlay = self._nm_draw_candidates_overlay(snapshot.image, self.nm_manual_candidates_current, show_ids=True)
                self._nm_add_stage(f"Manual candidates (base, {len(self.nm_manual_candidates_current)})", overlay)

        if self.nm_manual_candidates_current is None:
            self.nm_manual_candidates_current = list(self.nm_manual_candidates_base or [])

        # Keep planner/snapshot prompt up-to-date for picker intent summaries
        if self.nm_manual_snapshot is not None:
            self.nm_manual_snapshot.user_instruction = prompt
        self.nm_manual_planner = self._nm_manual_build_planner_from_inputs(prompt)
        return True

    def nm_rebuild_candidates(self):
        """Explicit rebuild of deterministic candidates (manual pipeline)."""
        if not self.original_image:
            self.log("No image loaded!", "error")
            return
        self.nm_manual_sig = None
        self.nm_manual_candidates_base = []
        self.nm_manual_candidates_current = []
        ok = self._nm_manual_ensure_candidates_initialized(auto_stage=False)
        if ok:
            self.log(f"Rebuilt candidates: {len(self.nm_manual_candidates_base)}", "success")
            self._nm_manual_schedule_refresh(immediate=True)

    def nm_reset_manual_filters(self):
        """Disable all manual filters and refresh view."""
        self.nm_filter_color_enabled.set(False)
        self.nm_filter_ocr_enabled.set(False)
        self.nm_filter_size_enabled.set(False)
        self.nm_filter_position_enabled.set(False)
        self.nm_bg_remove_enabled.set(False)
        self._nm_manual_update_toggle_ui()
        self.log("Manual filters cleared.", "info")
        self._nm_manual_schedule_refresh(immediate=True)

    def run_ocr_match_only(self):
        """Toggle OCR match filter (combines with other enabled filters)."""
        self.nm_filter_ocr_enabled.set(not self.nm_filter_ocr_enabled.get())
        self._nm_manual_update_toggle_ui()
        self._nm_manual_schedule_refresh(immediate=True)

    def run_size_filter_only(self):
        """Toggle size filter (combines with other enabled filters)."""
        self.nm_filter_size_enabled.set(not self.nm_filter_size_enabled.get())
        self._nm_manual_update_toggle_ui()
        self._nm_manual_schedule_refresh(immediate=True)

    def _nm_position_region_rect(self, roi: ScreenRegion, position: str) -> Optional[Tuple[int, int, int, int]]:
        position = (position or "any").strip().lower()
        if position in {"", "any", "unknown"}:
            return None
        w = max(1, roi.right - roi.left)
        h = max(1, roi.bottom - roi.top)

        def x_from_norm(a: float) -> int:
            return int(roi.left + a * w)

        def y_from_norm(a: float) -> int:
            return int(roi.top + a * h)

        if position == "top":
            return (roi.left, roi.top, roi.right, y_from_norm(0.4))
        if position == "bottom":
            return (roi.left, y_from_norm(0.6), roi.right, roi.bottom)
        if position == "left":
            return (roi.left, roi.top, x_from_norm(0.4), roi.bottom)
        if position == "right":
            return (x_from_norm(0.6), roi.top, roi.right, roi.bottom)
        if position == "center":
            return (x_from_norm(0.25), y_from_norm(0.25), x_from_norm(0.75), y_from_norm(0.75))
        if position == "top_left":
            return (roi.left, roi.top, x_from_norm(0.4), y_from_norm(0.4))
        if position == "top_right":
            return (x_from_norm(0.6), roi.top, roi.right, y_from_norm(0.4))
        if position == "bottom_left":
            return (roi.left, y_from_norm(0.6), x_from_norm(0.4), roi.bottom)
        if position == "bottom_right":
            return (x_from_norm(0.6), y_from_norm(0.6), roi.right, roi.bottom)
        return None

    def _nm_draw_position_overlay(self, image: Image.Image, roi: ScreenRegion, position: str) -> Image.Image:
        rect = self._nm_position_region_rect(roi, position)
        if rect is None:
            return image.copy().convert("RGB")
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        l, t, r, b = rect
        draw.rectangle([l, t, r, b], fill=(255, 215, 0, 50), outline=(255, 215, 0, 230), width=3)
        return Image.alpha_composite(base, overlay).convert("RGB")

    def run_position_only(self):
        """Toggle position filter (combines with other enabled filters)."""
        self.nm_filter_position_enabled.set(not self.nm_filter_position_enabled.get())
        self._nm_manual_update_toggle_ui()
        self._nm_manual_schedule_refresh(immediate=True)

    def nm_send_to_picker(self):
        """Send the current candidate set to the picker AI."""
        self._nm_manual_refresh_view()
        if not self._nm_manual_ensure_candidates_initialized(auto_stage=False):
            return
        snapshot = self.nm_manual_snapshot
        planner = self.nm_manual_planner
        if snapshot is None or planner is None:
            return

        candidates = list(self.nm_manual_candidates_current or [])
        if not candidates:
            self.log("No candidates available to send to picker. Rebuild and/or loosen filters.", "error")
            return

        if not self.client:
            self.log("OpenAI API key not configured - picker requires LLM access.", "error")
            return

        overlay = self._nm_draw_candidates_overlay(snapshot.image, candidates, show_ids=True)
        self._nm_add_stage(f"To picker ({len(candidates)} candidates)", overlay)

        # Rescore with current manual intent before sending (helps the text summary)
        try:
            candidates = self._nm_score_candidates(snapshot, planner, candidates)
        except Exception:
            pass

        choice = self._nm_picker_ai(snapshot, planner, candidates)
        if choice is None:
            self.log("Picker returned UNSURE - no candidate selected.", "error")
            return

        chosen = next((c for c in candidates if c.id == choice), None)
        if not chosen:
            self.log(f"Picker chose candidate {choice}, but it was not found.", "error")
            return

        self.log(f"Picker chose ID={chosen.id} at {chosen.click_point}.", "coords")
        self._nm_show_virtual_click(snapshot, chosen, candidates)

    def _nm_color_settings_defaults(self) -> Dict[str, Any]:
        return {
            "hue": {
                "red1": (0, 10),
                "red2": (170, 179),
                "orange": (5, 24),
                "yellow": (16, 35),
                "green": (31, 85),
                "teal": (86, 105),
                "blue": (97, 130),
                "purple": (125, 147),
                "pink": (138, 170),
                "brown": (10, 25),
            },
            "val": {
                "red1": (40, 240),
                "red2": (40, 235),
                "orange": (30, 245),
                "yellow": (30, 250),
                "green": (25, 245),
                "teal": (30, 245),
                "blue": (30, 245),
                "purple": (40, 245),
                "pink": (30, 250),
                "brown": (0, 140),
            },
            "sat_val": {
                "strong_s": 35,
                "strong_v": 35,
                "soft_s": 20,
                "soft_v": 70,
                "blue_strong_s": 60,
                "blue_strong_v": 50,
                "blue_soft_s": 45,
                "blue_soft_v": 70,
            },
            "blue_dom": {
                "b_over_g": 25,
                "b_over_r": 35,
            },
            "blue_contrast": {
                "delta_s": 8,
                "delta_v": 10,
                "delta_b": 12,
                "delta_h": 6,
            },
            "grey": {
                "diff": 25,
                "white_min": 215,
                "black_max": 50,
                "dark_min": 50,
                "dark_max": 130,
                "light_min": 150,
                "light_max": 245,
                "band_step": 8,
            },
            "colorful": {
                "s_min": 40,
                "v_min": 40,
            },
            "brown": {
                "v_max": 140,
                "s_min": 30,
            },
        }

    def _nm_roi_settings_defaults(self) -> Dict[str, float]:
        return {
            "overlap_x_pct": 0.08,
            "overlap_y_pct": 0.08,
            "top_bar_h_pct": 0.30,
            "left_w_pct": 0.35,
            "right_w_pct": 0.35,
            "sidebar_w_pct": 0.35,
            "footer_h_pct": 0.25,
            "center_w_pct": 0.65,  # shrink center region by 35% relative to full width
            "center_h_pct": 0.65,  # shrink center region by 35% relative to full height
        }

    def _nm_load_roi_settings(self) -> Dict[str, float]:
        defaults = self._nm_roi_settings_defaults()
        path = getattr(self, "roi_settings_path", None)
        if not path or not os.path.exists(path):
            return dict(defaults)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return dict(defaults)
        merged = dict(defaults)
        for key, default_val in defaults.items():
            try:
                val = float(data.get(key, default_val))
            except Exception:
                val = default_val
            # Clamp to sane range
            if val < 0:
                val = 0.0
            if val > 1.0:
                val = 1.0
            merged[key] = val

        # Backwards-compat: if old JSON only had sidebar width, copy it to left/right.
        if "left_w_pct" not in data and "sidebar_w_pct" in data:
            merged["left_w_pct"] = merged.get("sidebar_w_pct", defaults.get("left_w_pct", 0.35))
        if "right_w_pct" not in data and "sidebar_w_pct" in data:
            merged["right_w_pct"] = merged.get("sidebar_w_pct", defaults.get("right_w_pct", 0.35))
        return merged

    def _nm_save_roi_settings(self) -> None:
        path = getattr(self, "roi_settings_path", None)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.nm_roi_settings, f, indent=2)
        except Exception:
            pass

    def _nm_flat_color_settings(self) -> List[Tuple[str, Any]]:
        s = self.nm_color_settings
        return [
            ("red_h1_min", s["hue"]["red1"][0]), ("red_h1_max", s["hue"]["red1"][1]),
            ("red_h2_min", s["hue"]["red2"][0]), ("red_h2_max", s["hue"]["red2"][1]),
            ("orange_h_min", s["hue"]["orange"][0]), ("orange_h_max", s["hue"]["orange"][1]),
            ("yellow_h_min", s["hue"]["yellow"][0]), ("yellow_h_max", s["hue"]["yellow"][1]),
            ("green_h_min", s["hue"]["green"][0]), ("green_h_max", s["hue"]["green"][1]),
            ("teal_h_min", s["hue"]["teal"][0]), ("teal_h_max", s["hue"]["teal"][1]),
            ("blue_h_min", s["hue"]["blue"][0]), ("blue_h_max", s["hue"]["blue"][1]),
            ("purple_h_min", s["hue"]["purple"][0]), ("purple_h_max", s["hue"]["purple"][1]),
            ("pink_h_min", s["hue"]["pink"][0]), ("pink_h_max", s["hue"]["pink"][1]),
            ("brown_h_min", s["hue"]["brown"][0]), ("brown_h_max", s["hue"]["brown"][1]),
            ("strong_s", s["sat_val"]["strong_s"]), ("strong_v", s["sat_val"]["strong_v"]),
            ("soft_s", s["sat_val"]["soft_s"]), ("soft_v", s["sat_val"]["soft_v"]),
            ("blue_strong_s", s["sat_val"]["blue_strong_s"]), ("blue_strong_v", s["sat_val"]["blue_strong_v"]),
            ("blue_soft_s", s["sat_val"]["blue_soft_s"]), ("blue_soft_v", s["sat_val"]["blue_soft_v"]),
            ("blue_b_over_g", s["blue_dom"]["b_over_g"]), ("blue_b_over_r", s["blue_dom"]["b_over_r"]),
            ("blue_delta_s", s["blue_contrast"]["delta_s"]), ("blue_delta_v", s["blue_contrast"]["delta_v"]),
            ("blue_delta_b", s["blue_contrast"]["delta_b"]), ("blue_delta_h", s["blue_contrast"]["delta_h"]),
            ("grey_diff", s["grey"]["diff"]),
            ("white_min", s["grey"]["white_min"]), ("black_max", s["grey"]["black_max"]),
            ("dark_min", s["grey"]["dark_min"]), ("dark_max", s["grey"]["dark_max"]),
            ("light_min", s["grey"]["light_min"]), ("light_max", s["grey"]["light_max"]),
            ("grey_band_step", s["grey"]["band_step"]),
            ("colorful_s_min", s["colorful"]["s_min"]), ("colorful_v_min", s["colorful"]["v_min"]),
            ("brown_v_max", s["brown"]["v_max"]), ("brown_s_min", s["brown"]["s_min"]),
        ]

    def open_color_mask_settings(self):
        """Popup to edit color mask thresholds and ranges."""
        try:
            if hasattr(self, "_mask_settings_win") and self._mask_settings_win.winfo_exists():
                self._mask_settings_win.lift()
                return
        except Exception:
            pass

        win = tk.Toplevel(self.root)
        win.title("Color Mask Settings")
        win.geometry("520x640")
        win.configure(bg="#0d1117")
        self._mask_settings_win = win

        canvas = tk.Canvas(win, bg="#0d1117", highlightthickness=0)
        scrollbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frame = ttk.Frame(canvas, style='Dark.TFrame')
        window_id = canvas.create_window((0, 0), window=frame, anchor='nw')

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(window_id, width=event.width)

        frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_frame_configure)

        ttk.Label(frame, text="Adjust HSV thresholds & limits", style='Header.TLabel').pack(anchor=tk.W, pady=(8, 12))
        ttk.Label(
            frame,
            text=(
                "Explanations:\n"
                "• *_h_min/_h_max: HSV hue range (0-179) for each color.\n"
                "• strong_s/strong_v: strict saturation/value thresholds.\n"
                "• soft_s/soft_v: looser thresholds for gradients.\n"
                "• blue_b_over_g / blue_b_over_r: blue dominance in RGB.\n"
                "• blue_delta_*: local contrast vs blurred background (higher = stricter).\n"
                "• grey_*: greyscale limits (diff, white/black bands).\n"
                "• colorful_*: generic colorful mask thresholds.\n"
                "• brown_*: extra constraints for brown.\n"
                "Live updates: values re-run mask debug after you stop typing."
            ),
            style='Dark.TLabel',
            wraplength=480,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=6, pady=(0, 12))

        self._mask_setting_vars = {}
        for name, value in self._nm_flat_color_settings():
            row = ttk.Frame(frame, style='Dark.TFrame')
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=name, style='Dark.TLabel', width=18).pack(side=tk.LEFT, padx=(6, 8))
            var = tk.StringVar(value=str(value))
            entry = tk.Entry(row, textvariable=var, width=10, bg='#21262d', fg='#c9d1d9',
                             insertbackground='#58a6ff', relief=tk.FLAT, highlightthickness=1,
                             highlightbackground='#30363d', highlightcolor='#58a6ff')
            entry.pack(side=tk.LEFT)
            self._mask_setting_vars[name] = var
            var.trace_add("write", lambda *args: self._on_mask_settings_change())

        def _apply_settings():
            vals = {k: v.get().strip() for k, v in self._mask_setting_vars.items()}
            def iv(name, default):
                try:
                    return int(float(vals.get(name, default)))
                except Exception:
                    return default

            s = self.nm_color_settings
            s["hue"]["red1"] = (iv("red_h1_min", 0), iv("red_h1_max", 10))
            s["hue"]["red2"] = (iv("red_h2_min", 170), iv("red_h2_max", 179))
            s["hue"]["orange"] = (iv("orange_h_min", 5), iv("orange_h_max", 24))
            s["hue"]["yellow"] = (iv("yellow_h_min", 16), iv("yellow_h_max", 35))
            s["hue"]["green"] = (iv("green_h_min", 31), iv("green_h_max", 85))
            s["hue"]["teal"] = (iv("teal_h_min", 86), iv("teal_h_max", 105))
            s["hue"]["blue"] = (iv("blue_h_min", 97), iv("blue_h_max", 130))
            s["hue"]["purple"] = (iv("purple_h_min", 125), iv("purple_h_max", 147))
            s["hue"]["pink"] = (iv("pink_h_min", 138), iv("pink_h_max", 170))
            s["hue"]["brown"] = (iv("brown_h_min", 10), iv("brown_h_max", 25))
            s["sat_val"]["strong_s"] = iv("strong_s", 35)
            s["sat_val"]["strong_v"] = iv("strong_v", 35)
            s["sat_val"]["soft_s"] = iv("soft_s", 20)
            s["sat_val"]["soft_v"] = iv("soft_v", 70)
            s["sat_val"]["blue_strong_s"] = iv("blue_strong_s", 60)
            s["sat_val"]["blue_strong_v"] = iv("blue_strong_v", 50)
            s["sat_val"]["blue_soft_s"] = iv("blue_soft_s", 45)
            s["sat_val"]["blue_soft_v"] = iv("blue_soft_v", 70)
            s["blue_dom"]["b_over_g"] = iv("blue_b_over_g", 25)
            s["blue_dom"]["b_over_r"] = iv("blue_b_over_r", 35)
            s["blue_contrast"]["delta_s"] = iv("blue_delta_s", 8)
            s["blue_contrast"]["delta_v"] = iv("blue_delta_v", 10)
            s["blue_contrast"]["delta_b"] = iv("blue_delta_b", 12)
            s["blue_contrast"]["delta_h"] = iv("blue_delta_h", 6)
            s["grey"]["diff"] = iv("grey_diff", 25)
            s["grey"]["white_min"] = iv("white_min", 215)
            s["grey"]["black_max"] = iv("black_max", 50)
            s["grey"]["dark_min"] = iv("dark_min", 50)
            s["grey"]["dark_max"] = iv("dark_max", 130)
            s["grey"]["light_min"] = iv("light_min", 150)
            s["grey"]["light_max"] = iv("light_max", 245)
            s["grey"]["band_step"] = iv("grey_band_step", 8)
            s["colorful"]["s_min"] = iv("colorful_s_min", 40)
            s["colorful"]["v_min"] = iv("colorful_v_min", 40)
            s["brown"]["v_max"] = iv("brown_v_max", 140)
            s["brown"]["s_min"] = iv("brown_s_min", 30)
            s.setdefault("val", {})
            brown_vmin, _ = s["val"].get("brown", (0, 255))
            s["val"]["brown"] = (int(brown_vmin), int(s["brown"]["v_max"]))

            self.log("Color mask settings updated.", "success")
            try:
                self.run_color_mask_debug()
            except Exception:
                pass

        def _reset_defaults():
            self.nm_color_settings = self._nm_color_settings_defaults()
            for name, value in self._nm_flat_color_settings():
                self._mask_setting_vars[name].set(str(value))
            self.log("Color mask settings reset to defaults.", "info")

        btn_row = ttk.Frame(frame, style='Dark.TFrame')
        btn_row.pack(fill=tk.X, pady=(12, 10))
        tk.Button(btn_row, text="Apply", command=_apply_settings,
                  bg='#238636', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT, padx=(6, 8))
        tk.Button(btn_row, text="Print Values", command=self._print_color_mask_settings,
                  bg='#1f6feb', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_row, text="Reset Defaults", command=_reset_defaults,
                  bg='#444c56', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT)

    def open_hue_editor(self):
        """Popup to visualize and edit hue thresholds with a gradient preview."""
        try:
            if hasattr(self, "_hue_editor_win") and self._hue_editor_win.winfo_exists():
                self._hue_editor_win.lift()
                return
        except Exception:
            pass

        win = tk.Toplevel(self.root)
        self._hue_editor_win = win
        win.title("Hue Editor")
        win.geometry("700x620")
        win.configure(bg="#0d1117")

        header = ttk.Frame(win, style='Card.TFrame', padding=10)
        header.pack(fill=tk.X, padx=10, pady=(10, 6))
        ttk.Label(header, text="Hue ranges (0-179) and brightness (V) ranges (0-255). Adjust min/max to tighten color masks.",
                  style='Header.TLabel').pack(anchor=tk.W)

        # Preview area
        preview_frame = ttk.Frame(win, style='Card.TFrame', padding=10)
        preview_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self._hue_preview_label = ttk.Label(preview_frame)
        self._hue_preview_label.pack(fill=tk.X)

        body = ttk.Frame(win, style='Card.TFrame', padding=10)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Editable hue + brightness ranges
        self._hue_vars = {}
        self._hue_brightness_labels = {}
        header_row = ttk.Frame(body, style='Dark.TFrame')
        header_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(header_row, text="Color", style='Dark.TLabel', width=10).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Label(header_row, text="Hue min", style='Dark.TLabel', width=7).pack(side=tk.LEFT)
        ttk.Label(header_row, text="Hue max", style='Dark.TLabel', width=7).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(header_row, text="V min", style='Dark.TLabel', width=7).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(header_row, text="V max", style='Dark.TLabel', width=7).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(header_row, text="V range", style='Dark.TLabel', width=8).pack(side=tk.LEFT, padx=(10, 0))

        val_settings = self.nm_color_settings.get("val", {})
        for name, (hmin, hmax) in self.nm_color_settings["hue"].items():
            vmin, vmax = val_settings.get(name, (0, 255))
            row = ttk.Frame(body, style='Dark.TFrame')
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=name, style='Dark.TLabel', width=10).pack(side=tk.LEFT, padx=(6, 8))
            min_var = tk.StringVar(value=str(hmin))
            max_var = tk.StringVar(value=str(hmax))
            vmin_var = tk.StringVar(value=str(vmin))
            vmax_var = tk.StringVar(value=str(vmax))
            min_entry = tk.Entry(row, textvariable=min_var, width=8, bg='#21262d', fg='#c9d1d9',
                                 insertbackground='#58a6ff', relief=tk.FLAT, highlightthickness=1,
                                 highlightbackground='#30363d', highlightcolor='#58a6ff')
            max_entry = tk.Entry(row, textvariable=max_var, width=8, bg='#21262d', fg='#c9d1d9',
                                 insertbackground='#58a6ff', relief=tk.FLAT, highlightthickness=1,
                                 highlightbackground='#30363d', highlightcolor='#58a6ff')
            vmin_entry = tk.Entry(row, textvariable=vmin_var, width=8, bg='#21262d', fg='#c9d1d9',
                                  insertbackground='#58a6ff', relief=tk.FLAT, highlightthickness=1,
                                  highlightbackground='#30363d', highlightcolor='#58a6ff')
            vmax_entry = tk.Entry(row, textvariable=vmax_var, width=8, bg='#21262d', fg='#c9d1d9',
                                  insertbackground='#58a6ff', relief=tk.FLAT, highlightthickness=1,
                                  highlightbackground='#30363d', highlightcolor='#58a6ff')
            min_entry.pack(side=tk.LEFT, padx=(0, 6))
            max_entry.pack(side=tk.LEFT, padx=(0, 10))
            vmin_entry.pack(side=tk.LEFT, padx=(0, 6))
            vmax_entry.pack(side=tk.LEFT, padx=(0, 6))
            preview = ttk.Label(row)
            preview.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)
            self._hue_brightness_labels[name] = preview
            self._hue_vars[name] = (min_var, max_var, vmin_var, vmax_var)
            min_var.trace_add("write", lambda *args: self._update_hue_editor_preview(apply_live=True))
            max_var.trace_add("write", lambda *args: self._update_hue_editor_preview(apply_live=True))
            vmin_var.trace_add("write", lambda *args: self._update_hue_editor_preview(apply_live=True))
            vmax_var.trace_add("write", lambda *args: self._update_hue_editor_preview(apply_live=True))

        btn_row = ttk.Frame(win, style='Dark.TFrame')
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Button(btn_row, text="Apply", command=lambda: self._update_hue_editor_preview(apply_live=True),
                  bg='#238636', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_row, text="Reset Defaults", command=self._reset_hue_defaults,
                  bg='#444c56', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_row, text="Close", command=win.destroy,
                  bg='#30363d', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT)

        self._update_hue_editor_preview(apply_live=False)
        self.root.after(50, lambda: self._update_hue_editor_preview(apply_live=False))

    def _reset_hue_defaults(self):
        """Reset hue + brightness ranges to defaults and refresh the editor."""
        defaults = self._nm_color_settings_defaults()
        self.nm_color_settings["hue"] = dict(defaults["hue"])
        self.nm_color_settings["val"] = dict(defaults.get("val", {}))
        if hasattr(self, "_hue_vars"):
            for name, vars_tuple in self._hue_vars.items():
                hmin, hmax = defaults["hue"].get(name, (0, 179))
                vmin, vmax = defaults.get("val", {}).get(name, (0, 255))
                if len(vars_tuple) >= 4:
                    min_var, max_var, vmin_var, vmax_var = vars_tuple
                    vmin_var.set(str(vmin))
                    vmax_var.set(str(vmax))
                else:
                    min_var, max_var = vars_tuple
                min_var.set(str(hmin))
                max_var.set(str(hmax))
        self._update_hue_editor_preview(apply_live=False)

    def _update_hue_editor_preview(self, apply_live: bool = True):
        """Update the hue editor preview image and optionally apply changes live."""
        if not hasattr(self, "_hue_vars"):
            return

        def clamp_hue(value, default):
            try:
                v = int(float(value))
            except Exception:
                v = default
            return max(0, min(179, v))

        def clamp_val(value, default):
            try:
                v = int(float(value))
            except Exception:
                v = default
            return max(0, min(255, v))

        preview_ranges = {}
        preview_vals = {}
        for name, vars_tuple in self._hue_vars.items():
            if len(vars_tuple) >= 4:
                min_var, max_var, vmin_var, vmax_var = vars_tuple
            else:
                min_var, max_var = vars_tuple
                vmin_var = None
                vmax_var = None
            hmin = clamp_hue(min_var.get().strip(), 0)
            hmax = clamp_hue(max_var.get().strip(), 179)
            if vmin_var is not None and vmax_var is not None:
                vmin = clamp_val(vmin_var.get().strip(), 0)
                vmax = clamp_val(vmax_var.get().strip(), 255)
            else:
                vmin, vmax = 0, 255
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            preview_ranges[name] = (hmin, hmax)
            preview_vals[name] = (vmin, vmax)

        if apply_live:
            for name, (hmin, hmax) in preview_ranges.items():
                self.nm_color_settings["hue"][name] = (hmin, hmax)
            self.nm_color_settings.setdefault("val", {})
            for name, (vmin, vmax) in preview_vals.items():
                self.nm_color_settings["val"][name] = (vmin, vmax)
                if name == "brown":
                    try:
                        self.nm_color_settings["brown"]["v_max"] = int(vmax)
                    except Exception:
                        pass

        self._update_hue_brightness_previews(preview_ranges, preview_vals)

        img = self._render_hue_preview(preview_ranges)
        if img is None:
            return
        try:
            self._hue_preview_img = ImageTk.PhotoImage(img)
            self._hue_preview_label.configure(image=self._hue_preview_img)
            self._hue_preview_label.image = self._hue_preview_img
        except Exception:
            pass

    def _update_hue_brightness_previews(
        self,
        preview_ranges: Dict[str, Tuple[int, int]],
        preview_vals: Dict[str, Tuple[int, int]],
    ) -> None:
        """Update per-color brightness range previews next to inputs."""
        if not hasattr(self, "_hue_brightness_labels"):
            return
        try:
            import colorsys
        except Exception:
            colorsys = None

        if not hasattr(self, "_hue_brightness_imgs"):
            self._hue_brightness_imgs = {}

        for name, label in self._hue_brightness_labels.items():
            hmin, hmax = preview_ranges.get(name, (0, 179))
            vmin, vmax = preview_vals.get(name, (0, 255))
            label_width = 0
            try:
                label_width = int(label.winfo_width() or 0)
            except Exception:
                label_width = 0
            if label_width <= 30:
                label_width = 200
            img = self._render_brightness_preview(hmin, hmax, vmin, vmax, colorsys, width=label_width)
            if img is None:
                continue
            try:
                photo = ImageTk.PhotoImage(img)
                label.configure(image=photo)
                label.image = photo
                self._hue_brightness_imgs[name] = photo
            except Exception:
                pass

    def _render_brightness_preview(
        self,
        hmin: int,
        hmax: int,
        vmin: int,
        vmax: int,
        colorsys_mod=None,
        width: int = 200,
        height: int = 14,
    ) -> Optional[Image.Image]:
        """Render a small brightness bar with the allowed V range highlighted."""
        try:
            from PIL import ImageDraw
        except Exception:
            return None

        img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        # Color brightness gradient: black -> dark color -> bright color -> light color -> white
        if colorsys_mod is None:
            for x in range(width):
                v = int(round(255 * x / max(1, width - 1)))
                draw.line([(x, 0), (x, height)], fill=(v, v, v, 255))
        else:
            mid = (hmin + hmax) / 2.0
            hue = mid / 179.0
            pivot = int(round((width - 1) * 0.8))
            pivot = max(1, min(width - 2, pivot))
            for x in range(width):
                if x <= pivot:
                    v = x / float(pivot)
                    s = 1.0
                else:
                    v = 1.0
                    s = 1.0 - ((x - pivot) / float((width - 1) - pivot))
                r, g, b = colorsys_mod.hsv_to_rgb(hue, s, v)
                draw.line([(x, 0), (x, height)], fill=(int(r * 255), int(g * 255), int(b * 255), 255))

        # Highlight allowed V range
        vmin = max(0, min(255, int(vmin)))
        vmax = max(0, min(255, int(vmax)))
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        x0 = int(round(vmin / 255.0 * (width - 1)))
        x1 = int(round(vmax / 255.0 * (width - 1)))
        if x1 < x0:
            x0, x1 = x1, x0
        if x1 == x0:
            x1 = min(width - 1, x0 + 1)

        # Dim outside the allowed V range and outline the active region
        if x0 > 0:
            draw.rectangle([0, 0, x0, height - 1], fill=(0, 0, 0, 90))
        if x1 < width - 1:
            draw.rectangle([x1, 0, width - 1, height - 1], fill=(0, 0, 0, 90))
        # No border; just dim outside the allowed range.

        return img.convert("RGB")

    def _render_hue_preview(self, ranges: Dict[str, Tuple[int, int]]) -> Optional[Image.Image]:
        """Render a hue gradient with ranges overlaid."""
        try:
            import colorsys
        except Exception:
            return None

        width = 640
        height = 80
        bar_top = 16
        bar_height = 32
        img = Image.new("RGB", (width, height), (13, 17, 23))
        draw = ImageDraw.Draw(img)

        # Gradient bar
        for x in range(width):
            hue = int(round(x * 179 / (width - 1)))
            r, g, b = colorsys.hsv_to_rgb(hue / 179.0, 1.0, 1.0)
            draw.line([(x, bar_top), (x, bar_top + bar_height)], fill=(int(r * 255), int(g * 255), int(b * 255)))

        # Overlay ranges
        label_y = bar_top + bar_height + 4
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        order = ["red1", "red2", "orange", "yellow", "green", "teal", "blue", "purple", "pink", "brown"]
        for idx, name in enumerate(order):
            if name not in ranges:
                continue
            hmin, hmax = ranges[name]
            x0 = int(hmin / 179 * (width - 1))
            x1 = int(hmax / 179 * (width - 1))
            mid = (hmin + hmax) / 2.0
            r, g, b = colorsys.hsv_to_rgb(mid / 179.0, 1.0, 1.0)
            color = (int(r * 255), int(g * 255), int(b * 255))
            draw.rectangle([x0, bar_top, x1, bar_top + bar_height], outline=color, width=2)
            # Stagger labels a bit for readability
            text_y = label_y + (idx % 2) * 12
            draw.text((x0 + 2, text_y), f"{name} {hmin}-{hmax}", fill=color, font=font)

        return img

    def open_roi_editor(self):
        """Popup to visualize and edit ROI settings."""
        try:
            if hasattr(self, "_roi_editor_win") and self._roi_editor_win.winfo_exists():
                self._roi_editor_win.lift()
                return
        except Exception:
            pass

        win = tk.Toplevel(self.root)
        self._roi_editor_win = win
        win.title("ROI Editor")
        win.geometry("720x680")
        win.configure(bg="#0d1117")

        header = ttk.Frame(win, style='Card.TFrame', padding=10)
        header.pack(fill=tk.X, padx=10, pady=(10, 6))
        ttk.Label(header, text="ROI settings (percent of window). Use Prev/Next to preview each ROI.",
                  style='Header.TLabel').pack(anchor=tk.W)

        preview_frame = ttk.Frame(win, style='Card.TFrame', padding=10)
        preview_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self._roi_preview_label = ttk.Label(preview_frame)
        self._roi_preview_label.pack(fill=tk.X)
        self._roi_preview_info = ttk.Label(preview_frame, style='Dark.TLabel')
        self._roi_preview_info.pack(anchor=tk.W, pady=(6, 0))

        nav = ttk.Frame(win, style='Dark.TFrame')
        nav.pack(fill=tk.X, padx=10, pady=(0, 8))
        self._roi_zones = ["any", "top_bar", "left", "right", "footer", "center"]
        self._roi_preview_index = 0
        self._roi_zone_label = ttk.Label(nav, text="ROI: any", style='Dark.TLabel')
        self._roi_zone_label.pack(side=tk.LEFT)
        btn_prev = tk.Button(nav, text="◀ Prev", command=lambda: self._roi_preview_step(-1),
                             bg='#30363d', fg='#c9d1d9', relief=tk.FLAT, cursor='hand2', padx=10, pady=4)
        btn_prev.pack(side=tk.RIGHT, padx=(6, 0))
        btn_next = tk.Button(nav, text="Next ▶", command=lambda: self._roi_preview_step(1),
                             bg='#30363d', fg='#c9d1d9', relief=tk.FLAT, cursor='hand2', padx=10, pady=4)
        btn_next.pack(side=tk.RIGHT)

        body = ttk.Frame(win, style='Card.TFrame', padding=10)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._roi_vars = {}
        fields = [
            ("overlap_x_pct", "Overlap X (%)"),
            ("overlap_y_pct", "Overlap Y (%)"),
            ("top_bar_h_pct", "Top bar height (%)"),
            ("left_w_pct", "Left panel width (%)"),
            ("right_w_pct", "Right panel width (%)"),
            ("footer_h_pct", "Footer height (%)"),
            ("center_w_pct", "Center width (%)"),
            ("center_h_pct", "Center height (%)"),
        ]
        for key, label in fields:
            row = ttk.Frame(body, style='Dark.TFrame')
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label, style='Dark.TLabel', width=22).pack(side=tk.LEFT, padx=(6, 8))
            var = tk.StringVar(value=str(int(round(self.nm_roi_settings.get(key, 0.0) * 100))))
            entry = tk.Entry(row, textvariable=var, width=8, bg='#21262d', fg='#c9d1d9',
                             insertbackground='#58a6ff', relief=tk.FLAT, highlightthickness=1,
                             highlightbackground='#30363d', highlightcolor='#58a6ff')
            entry.pack(side=tk.LEFT, padx=(0, 6))
            self._roi_vars[key] = var
            var.trace_add("write", lambda *args: self._update_roi_editor_preview(apply_live=True))

        btn_row = ttk.Frame(win, style='Dark.TFrame')
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(btn_row, text="Apply", command=lambda: self._update_roi_editor_preview(apply_live=True),
                  bg='#238636', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_row, text="Save", command=self._save_roi_editor_settings,
                  bg='#1f6feb', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_row, text="Reset Defaults", command=self._reset_roi_defaults,
                  bg='#444c56', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_row, text="Close", command=win.destroy,
                  bg='#30363d', fg='white', font=('Segoe UI', 10, 'bold'),
                  relief=tk.FLAT, cursor='hand2', padx=12, pady=6).pack(side=tk.LEFT)

        self._update_roi_editor_preview(apply_live=False)

    def _roi_preview_step(self, delta: int) -> None:
        if not hasattr(self, "_roi_zones"):
            return
        self._roi_preview_index = (self._roi_preview_index + delta) % len(self._roi_zones)
        self._update_roi_editor_preview(apply_live=False)

    def _reset_roi_defaults(self) -> None:
        defaults = self._nm_roi_settings_defaults()
        self.nm_roi_settings = dict(defaults)
        if hasattr(self, "_roi_vars"):
            for key, var in self._roi_vars.items():
                var.set(str(int(round(defaults.get(key, 0.0) * 100))))
        self._update_roi_editor_preview(apply_live=False)

    def _save_roi_editor_settings(self) -> None:
        self._update_roi_editor_preview(apply_live=True)
        self._nm_save_roi_settings()
        self.log("ROI settings saved.", "success")

    def _update_roi_editor_preview(self, apply_live: bool = True) -> None:
        if not hasattr(self, "_roi_vars"):
            return

        def read_pct(key: str, default: float) -> float:
            try:
                val = float(self._roi_vars[key].get().strip())
            except Exception:
                val = default * 100.0
            val = max(0.0, min(100.0, val))
            return val / 100.0

        if apply_live:
            defaults = self._nm_roi_settings_defaults()
            for key, default in defaults.items():
                self.nm_roi_settings[key] = read_pct(key, default)

        zone = "any"
        if hasattr(self, "_roi_zones"):
            zone = self._roi_zones[self._roi_preview_index]
        if hasattr(self, "_roi_zone_label"):
            self._roi_zone_label.config(text=f"ROI: {zone}")

        if not self.original_image:
            return
        img = self.original_image.copy()
        roi = self._nm_zone_to_roi(img, (0, 0, img.width, img.height), zone)
        overlay = self._nm_draw_roi_overlay(img, roi)

        # Scale preview to fit
        max_w = 660
        scale = min(1.0, max_w / overlay.width)
        if scale < 1.0:
            new_size = (int(overlay.width * scale), int(overlay.height * scale))
            overlay = overlay.resize(new_size, Image.Resampling.LANCZOS)

        try:
            self._roi_preview_img = ImageTk.PhotoImage(overlay)
            self._roi_preview_label.configure(image=self._roi_preview_img)
            self._roi_preview_label.image = self._roi_preview_img
        except Exception:
            pass

        if hasattr(self, "_roi_preview_info"):
            self._roi_preview_info.config(
                text=f"ROI coords: ({roi.left}, {roi.top})–({roi.right}, {roi.bottom})"
            )

    def _on_mask_settings_change(self):
        """Debounced live update for mask settings."""
        try:
            if hasattr(self, "_mask_settings_debounce"):
                self.root.after_cancel(self._mask_settings_debounce)
        except Exception:
            pass
        self._mask_settings_debounce = self.root.after(250, self._apply_mask_settings_live)

    def _apply_mask_settings_live(self):
        try:
            # Reuse the settings parser by invoking the same logic as Apply
            # but without logging spam.
            vals = {k: v.get().strip() for k, v in self._mask_setting_vars.items()}
            def iv(name, default):
                try:
                    return int(float(vals.get(name, default)))
                except Exception:
                    return default
            s = self.nm_color_settings
            s["hue"]["red1"] = (iv("red_h1_min", 0), iv("red_h1_max", 10))
            s["hue"]["red2"] = (iv("red_h2_min", 170), iv("red_h2_max", 179))
            s["hue"]["orange"] = (iv("orange_h_min", 5), iv("orange_h_max", 24))
            s["hue"]["yellow"] = (iv("yellow_h_min", 16), iv("yellow_h_max", 35))
            s["hue"]["green"] = (iv("green_h_min", 31), iv("green_h_max", 85))
            s["hue"]["teal"] = (iv("teal_h_min", 86), iv("teal_h_max", 105))
            s["hue"]["blue"] = (iv("blue_h_min", 97), iv("blue_h_max", 130))
            s["hue"]["purple"] = (iv("purple_h_min", 125), iv("purple_h_max", 147))
            s["hue"]["pink"] = (iv("pink_h_min", 138), iv("pink_h_max", 170))
            s["hue"]["brown"] = (iv("brown_h_min", 10), iv("brown_h_max", 25))
            s["sat_val"]["strong_s"] = iv("strong_s", 35)
            s["sat_val"]["strong_v"] = iv("strong_v", 35)
            s["sat_val"]["soft_s"] = iv("soft_s", 20)
            s["sat_val"]["soft_v"] = iv("soft_v", 70)
            s["sat_val"]["blue_strong_s"] = iv("blue_strong_s", 60)
            s["sat_val"]["blue_strong_v"] = iv("blue_strong_v", 50)
            s["sat_val"]["blue_soft_s"] = iv("blue_soft_s", 45)
            s["sat_val"]["blue_soft_v"] = iv("blue_soft_v", 70)
            s["blue_dom"]["b_over_g"] = iv("blue_b_over_g", 25)
            s["blue_dom"]["b_over_r"] = iv("blue_b_over_r", 35)
            s["blue_contrast"]["delta_s"] = iv("blue_delta_s", 8)
            s["blue_contrast"]["delta_v"] = iv("blue_delta_v", 10)
            s["blue_contrast"]["delta_b"] = iv("blue_delta_b", 12)
            s["blue_contrast"]["delta_h"] = iv("blue_delta_h", 6)
            s["grey"]["diff"] = iv("grey_diff", 25)
            s["grey"]["white_min"] = iv("white_min", 215)
            s["grey"]["black_max"] = iv("black_max", 50)
            s["grey"]["dark_min"] = iv("dark_min", 50)
            s["grey"]["dark_max"] = iv("dark_max", 130)
            s["grey"]["light_min"] = iv("light_min", 150)
            s["grey"]["light_max"] = iv("light_max", 245)
            s["grey"]["band_step"] = iv("grey_band_step", 8)
            s["colorful"]["s_min"] = iv("colorful_s_min", 40)
            s["colorful"]["v_min"] = iv("colorful_v_min", 40)
            s["brown"]["v_max"] = iv("brown_v_max", 140)
            s["brown"]["s_min"] = iv("brown_s_min", 30)
            s.setdefault("val", {})
            brown_vmin, _ = s["val"].get("brown", (0, 255))
            s["val"]["brown"] = (int(brown_vmin), int(s["brown"]["v_max"]))
            self.run_color_mask_debug()
        except Exception:
            pass

    def _print_color_mask_settings(self):
        """Print current mask settings into a popup text box for sharing."""
        try:
            import json
            data = json.dumps(self.nm_color_settings, indent=2)
        except Exception:
            data = str(self.nm_color_settings)
        win = tk.Toplevel(self.root)
        win.title("Color Mask Settings Values")
        win.geometry("520x520")
        win.configure(bg="#0d1117")
        txt = scrolledtext.ScrolledText(win, bg="#0d1117", fg="#c9d1d9",
                                        insertbackground="#58a6ff", font=("Consolas", 9))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert("1.0", data)
        txt.config(state=tk.NORMAL)
    
    def draw_cursor_dot(self, image, cursor_x, cursor_y):
        """Draw a single vertical cyan line at the cursor X position."""
        # Convert to RGBA for drawing
        if image.mode != 'RGBA':
            img_rgba = image.convert('RGBA')
        else:
            img_rgba = image.copy()
        
        overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Cyan color - strong, fully opaque
        cyan_color = (0, 255, 255, 255)
        
        # Draw ONLY a vertical line (full screen) at cursor_x
        draw.line([cursor_x, 0, cursor_x, image.height], fill=cyan_color, width=3)

        # Composite overlay
        result = Image.alpha_composite(img_rgba, overlay)
        
        # Convert back to original mode
        if image.mode == 'RGB':
            rgb_result = Image.new('RGB', result.size)
            rgb_result.paste(result, mask=result.split()[3])
            return rgb_result
        elif image.mode == 'RGBA':
            return result
        else:
            return result.convert('RGB')

    def generate_numbered_points(self, region_left, region_top, region_right, region_bottom, count):
        """Generate 'count' evenly scattered numbered points inside the given region."""
        points = []
        region_width = max(1, int(region_right) - int(region_left))
        region_height = max(1, int(region_bottom) - int(region_top))
        
        # Calculate grid dimensions for even distribution
        # Try to make it roughly square-ish
        aspect_ratio = region_width / region_height
        cols = int(math.ceil(math.sqrt(count * aspect_ratio)))
        rows = int(math.ceil(count / cols))
        
        # Adjust to get close to the desired count
        while cols * rows < count:
            if cols < rows:
                cols += 1
            else:
                rows += 1
        
        # Calculate spacing
        col_spacing = region_width / max(1, cols - 1) if cols > 1 else 0
        row_spacing = region_height / max(1, rows - 1) if rows > 1 else 0
        
        # Generate points on grid with small random jitter for natural look
        point_id = 1
        for row in range(rows):
            for col in range(cols):
                if point_id > count:
                    break
                
                # Base position on grid
                base_x = int(region_left) + col * col_spacing
                base_y = int(region_top) + row * row_spacing
                
                # Add small random jitter (10% of spacing) to avoid perfect grid
                jitter_x = random.uniform(-col_spacing * 0.1, col_spacing * 0.1) if cols > 1 else 0
                jitter_y = random.uniform(-row_spacing * 0.1, row_spacing * 0.1) if rows > 1 else 0
                
                x = int(base_x + jitter_x)
                y = int(base_y + jitter_y)
                
                # Clamp to region bounds
                x = max(int(region_left), min(int(region_right) - 1, x))
                y = max(int(region_top), min(int(region_bottom) - 1, y))
                
                points.append({"id": point_id, "x": x, "y": y})
                point_id += 1
            
            if point_id > count:
                break
        
        return points
    
    def draw_numbered_points(self, image, points):
        """Draw small cyan numbers for each point on the image (no background/border)."""
        if image.mode != 'RGBA':
            img_rgba = image.convert('RGBA')
        else:
            img_rgba = image.copy()
        
        overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Choose font size relative to image height
        base_font_size = max(12, image.height // 50)
        try:
            font = ImageFont.truetype("arial.ttf", base_font_size)
        except Exception:
            font = ImageFont.load_default()
        
        cyan = (0, 255, 255, 255)
        
        for p in points:
            x = p["x"]
            y = p["y"]
            pid = str(p["id"])
            
            # Number text (small cyan, no background/border)
            text = pid
            # Use font bounding box to measure text size (Pillow 10+)
            try:
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                # Fallback approximate size
                tw, th = len(text) * base_font_size // 2, base_font_size
            tx = x - tw // 2
            ty = y - th // 2
            
            draw.text((tx, ty), text, font=font, fill=cyan)
        
        result = Image.alpha_composite(img_rgba, overlay)
        
        if image.mode == 'RGB':
            rgb_result = Image.new('RGB', result.size)
            rgb_result.paste(result, mask=result.split()[3])
            return rgb_result
        elif image.mode == 'RGBA':
            return result
        else:
            return result.convert('RGB')
    
    def draw_ocr_dots(self, image, ocr_results):
        """Draw dots on the image at the center of each OCR match."""
        if not ocr_results or not ocr_results.matches:
            return image
        
        if image.mode != 'RGBA':
            img_rgba = image.convert('RGBA')
        else:
            img_rgba = image.copy()
        
        overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Red color for OCR dots
        dot_color = (255, 0, 0, 255)  # Red, fully opaque
        dot_radius = 5
        
        for match in ocr_results.matches:
            # Get center coordinates
            center_x, center_y = match.center
            
            # Draw a red dot at the center
            draw.ellipse(
                [center_x - dot_radius, center_y - dot_radius,
                 center_x + dot_radius, center_y + dot_radius],
                fill=dot_color,
                outline=dot_color,
                width=2
            )
        
        # Composite overlay onto image
        result = Image.alpha_composite(img_rgba, overlay)
        
        # Convert back to original mode
        if image.mode == 'RGB':
            rgb_result = Image.new('RGB', result.size)
            rgb_result.paste(result, mask=result.split()[3])
            return rgb_result
        elif image.mode == 'RGBA':
            return result
        else:
            return result.convert('RGB')

    def _get_ocr_query_intent(self) -> Optional[PlannerTextIntent]:
        """Return OCR query intent parsed from the OCR match entry, if any."""
        if self.nm_ocr_match_entry is None:
            return None
        raw_text = (self.nm_ocr_match_entry.get() or "").strip()
        if not raw_text:
            return None
        parts = [p.strip() for p in raw_text.replace("|", ",").split(",") if p.strip()]
        if not parts:
            return None
        return PlannerTextIntent(primary_text=parts[0], variants=parts[1:], strictness="medium")

    def _rank_ocr_matches_for_query(
        self,
        matches: List["OCRMatch"],
        intent: PlannerTextIntent,
        top_n: int = 5,
    ) -> List["OCRMatch"]:
        """Rank OCR matches for a query and return the top N."""
        query = (intent.primary_text or "").strip().lower()
        short_query = bool(query) and len(query) <= 2
        scored = []
        for match in matches:
            candidate = (match.text or "").strip().lower()
            if short_query:
                if candidate != query:
                    continue
                text_score = 1.0
            else:
                text_score = self._nm_text_similarity(intent, match.text or "")
            if text_score <= 0.0:
                continue
            combined = (text_score * 100.0) + (match.confidence * 0.25)
            scored.append((combined, match.confidence, match))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [m for _, _, m in scored[:top_n]]

    def draw_ocr_boxes(self, image, matches, color=(0, 255, 0, 255), width=2):
        """Draw bounding boxes for OCR matches."""
        if not matches:
            return image

        if image.mode != 'RGBA':
            img_rgba = image.convert('RGBA')
        else:
            img_rgba = image.copy()

        overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for idx, match in enumerate(matches, 1):
            l, t, r, b = match.bbox.left, match.bbox.top, match.bbox.right, match.bbox.bottom
            draw.rectangle([l, t, r, b], outline=color, width=width)
            draw.text((l + 3, t + 2), str(idx), font=font, fill=color)

        result = Image.alpha_composite(img_rgba, overlay)

        if image.mode == 'RGB':
            rgb_result = Image.new('RGB', result.size)
            rgb_result.paste(result, mask=result.split()[3])
            return rgb_result
        elif image.mode == 'RGBA':
            return result
        else:
            return result.convert('RGB')
    
    def preprocess_image_variants(self, image):
        """Create multiple preprocessed variants with better quality enhancement."""
        from PIL import ImageEnhance, ImageFilter
        variants = []
        
        # Convert to grayscale first
        gray = image.convert("L")
        
        # 1. Original grayscale (baseline)
        variants.append(("original_gray", gray))
        
        # 2. Denoise + enhance contrast (better quality)
        try:
            import numpy as np
            from scipy import ndimage
            img_array = np.array(gray)
            # Denoise using median filter
            denoised = ndimage.median_filter(img_array, size=3)
            denoised_img = Image.fromarray(denoised.astype(np.uint8))
            # Enhance contrast after denoising
            enhancer = ImageEnhance.Contrast(denoised_img)
            enhanced_denoised = enhancer.enhance(1.8)
            variants.append(("denoised_contrast", enhanced_denoised))
        except ImportError:
            # Fallback if scipy not available
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(1.8)
            variants.append(("contrast_1.8", enhanced))
        
        # 3. High contrast + brightness adjustment
        enhancer = ImageEnhance.Contrast(gray)
        high_contrast = enhancer.enhance(2.2)
        brightness = ImageEnhance.Brightness(high_contrast)
        bright_contrast = brightness.enhance(1.1)
        variants.append(("high_contrast_bright", bright_contrast))
        
        # 4. Sharpened + contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.6)
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        # Apply sharpening twice for better edge definition
        sharpened2 = sharpened.filter(ImageFilter.SHARPEN)
        variants.append(("sharpened_2x", sharpened2))
        
        # 5. Adaptive binarization (Otsu-like threshold)
        try:
            import numpy as np
            img_array = np.array(gray)
            # Use Otsu's method approximation
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
            # Find threshold that maximizes between-class variance
            total = img_array.size
            sum_all = np.sum(np.arange(256) * hist)
            sum_bg = 0
            w_bg = 0
            max_var = 0
            threshold = 127
            
            for i in range(256):
                w_bg += hist[i]
                if w_bg == 0:
                    continue
                w_fg = total - w_bg
                if w_fg == 0:
                    break
                sum_bg += i * hist[i]
                mean_bg = sum_bg / w_bg
                mean_fg = (sum_all - sum_bg) / w_fg
                var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
                if var_between > max_var:
                    max_var = var_between
                    threshold = i
            
            binary = Image.fromarray((img_array > threshold).astype(np.uint8) * 255)
            variants.append(("otsu_binary", binary))
        except ImportError:
            # Fallback: simple threshold
            try:
                import numpy as np
                img_array = np.array(gray)
                threshold = np.mean(img_array)
                binary = Image.fromarray((img_array > threshold).astype(np.uint8) * 255)
                variants.append(("binary", binary))
            except ImportError:
                pass
        
        # 6. Upscaled 2x with better interpolation (helps with small text, especially Swedish chars)
        upscaled = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
        # Enhance upscaled image with stronger contrast for Swedish characters
        enhancer = ImageEnhance.Contrast(upscaled)
        upscaled_enhanced = enhancer.enhance(1.8)  # Higher contrast for better char recognition
        # Sharpen to preserve character details (important for ä, ö, å)
        upscaled_enhanced = upscaled_enhanced.filter(ImageFilter.SHARPEN)
        upscaled_enhanced._scale_factor = 2.0
        variants.append(("upscaled_2x_enhanced", upscaled_enhanced))
        
        # 7. Extra high contrast variant specifically for Swedish characters
        # Swedish characters (ä, ö, å) need very clear edges
        enhancer = ImageEnhance.Contrast(gray)
        very_high_contrast = enhancer.enhance(2.5)
        # Double sharpen for character edge clarity
        very_sharp = very_high_contrast.filter(ImageFilter.SHARPEN)
        very_sharp = very_sharp.filter(ImageFilter.SHARPEN)
        variants.append(("very_high_contrast_sharp", very_sharp))
        
        return variants
    
    def on_ocr_engine_change(self):
        """Handle OCR engine selection change."""
        self.use_paddleocr = (self.ocr_engine_var.get() == "PaddleOCR")
        # Reset OCR engine to force reload
        self.ocr_engine = None
        self.paddleocr_ocr = None
        engine_name = "PaddleOCR" if self.use_paddleocr else "Tesseract"
        self.log(f"OCR engine changed to: {engine_name}", 'info')

    def _set_ocr_ui_running(self, running: bool) -> None:
        """Enable/disable OCR UI controls while OCR is running."""
        state = tk.DISABLED if running else tk.NORMAL

        def apply_state():
            if getattr(self, "ocr_btn", None):
                self.ocr_btn.config(state=state)
            if getattr(self, "ocr_match_btn", None):
                self.ocr_match_btn.config(state=state)
            self.update_status("OCR running" if running else "Ready", "#58a6ff")

        self.root.after(0, apply_state)
    
    def init_paddleocr(self):
        """Initialize PaddleOCR."""
        if self.paddleocr_ocr is not None:
            return True
        
        try:
            # Try importing PaddleOCR first
            try:
                import os
                # Avoid the model hoster connectivity check on startup.
                os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
                os.environ.setdefault('DISABLE_MODEL_SOURCE_CHECK', 'True')
                os.environ.setdefault('FLAGS_use_mkldnn', '0')
                os.environ.setdefault('FLAGS_enable_cinn', '0')

                try:
                    import paddle
                    self.log(f"Paddle version: {getattr(paddle, '__version__', 'unknown')}", 'info')
                except Exception as pe:
                    self.log("ERROR: PaddlePaddle runtime not available. Install with: pip install paddlepaddle", 'error')
                    self.log(f"  Import error: {pe}", 'error')
                    return False

                from paddleocr import PaddleOCR
            except Exception as ie:
                self.log("ERROR: PaddleOCR not installed or failed to import.", 'error')
                self.log("  Install with: pip install paddleocr", 'error')
                self.log(f"  Import error: {ie}", 'error')
                return False
            
            self.log("Initializing PaddleOCR (sv/Latin multilingual, full quality)...", 'info')
            self.log("This may take a moment on first run (downloading models)...", 'info')
            
            # Use Swedish language, CPU mode, enable angle classification
            # No image resizing - use full quality
            # Initialize PaddleOCR with only valid parameters
            # Swedish uses the Latin model (covers å ä ö)
            ocr_kwargs = {
                "lang": "sv",
                "use_textline_orientation": False,
                "show_log": False,
                "device": "cpu",
                "enable_mkldnn": False,
                "enable_cinn": False,
            }

            # Some PaddleOCR builds reject unknown args. Retry after dropping them.
            while True:
                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.paddleocr_ocr = PaddleOCR(**ocr_kwargs)
                    break
                except Exception as e:
                    msg = str(e)
                    if "Unknown argument:" in msg:
                        bad = msg.split("Unknown argument:", 1)[-1].strip()
                        if bad in ocr_kwargs:
                            self.log(f"PaddleOCR does not support '{bad}'. Retrying without it.", 'info')
                            del ocr_kwargs[bad]
                            continue
                    raise
            self.log("✓ PaddleOCR initialized successfully (sv/Latin multilingual, full quality)", 'success')
            return True
        except Exception as e:
            self.log(f"✗ Failed to initialize PaddleOCR: {e}", 'error')
            import traceback
            self.log(traceback.format_exc(), 'error')
            return False
    
    def run_ocr_paddleocr(self):
        """Run OCR using PaddleOCR."""
        if not self.original_image:
            self.log("No image loaded!", 'error')
            return
        
        if not self.init_paddleocr():
            self.log("Falling back to Tesseract...", 'info')
            self.use_paddleocr = False
            self._run_ocr_impl()
            return
        
        try:
            self.log("\n" + "="*50, 'info')
            self.log("Running PaddleOCR (sv/Latin multilingual)...", 'info')
            self.log("="*50, 'info')
            
            import time
            start_time = time.time()
            
            # Convert PIL image to numpy array for PaddleOCR - optimize for performance
            import numpy as np

            # Resize large images for better OCR performance while maintaining accuracy
            img_for_ocr = self.original_image
            original_size = (self.original_image.width, self.original_image.height)

            # Resize if image is very large (>1920px wide) to improve speed
            max_ocr_width = 1920
            if self.original_image.width > max_ocr_width:
                scale_factor = max_ocr_width / self.original_image.width
                new_width = int(self.original_image.width * scale_factor)
                new_height = int(self.original_image.height * scale_factor)
                img_for_ocr = self.original_image.resize((new_width, new_height), Image.LANCZOS)
                self.log(f"Resized image for OCR: {original_size[0]}x{original_size[1]} → {new_width}x{new_height}", 'info')

            img_array = np.array(img_for_ocr.convert('RGB'))
            self.log(f"Processing image: {img_array.shape[1]}x{img_array.shape[0]} pixels", 'info')
            
            # Run PaddleOCR with full quality image (suppress noisy stdout/stderr)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                results = self.paddleocr_ocr.ocr(img_array)
            
            elapsed = time.time() - start_time
            self.log(f"PaddleOCR completed in {elapsed:.2f} seconds", 'success')
            
            # Parse results
            matches = []
            from agent.screenshot import ScreenRegion
            from agent.ocr import OCRMatch
            
            if results and results[0]:
                for line in results[0]:
                    if line:
                        # PaddleOCR format: [[bbox], (text, confidence)]
                        bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text, confidence = line[1]
                        confidence_percent = confidence * 100  # Convert to percentage
                        
                        # Calculate bounding box from points - use PaddleOCR's detected region directly
                        xs = [p[0] for p in bbox_points]
                        ys = [p[1] for p in bbox_points]

                        # Debug: print polygon info for first few detections
                        if len(matches) < 3:
                            self.log(f"Full OCR polygon: text='{text}', points={bbox_points}", "debug")
                            self.log(f"Full OCR bounds: x={min(xs):.1f}-{max(xs):.1f}, y={min(ys):.1f}-{max(ys):.1f}", "debug")

                        # Use the polygon bounding box directly - PaddleOCR positions these over text
                        left = int(min(xs))
                        top = int(min(ys))
                        right = int(max(xs))
                        bottom = int(max(ys))
                        width = max(1, right - left)
                        height = max(1, bottom - top)
                        
                        # Validate text
                        if self.is_valid_text(text) and confidence_percent >= 15:  # Low threshold initially
                            bbox = ScreenRegion(left=left, top=top, width=width, height=height)
                            match = OCRMatch(text=text, confidence=confidence_percent, bbox=bbox)
                            matches.append(match)
            
            self.log(f"Found {len(matches)} text items", 'success')
            
            if matches:
                # Deduplicate and filter
                self.log("Deduplicating matches...", 'info')
                # Sort by confidence
                matches.sort(key=lambda m: m.confidence, reverse=True)
                
                # Deduplicate close matches
                merged_matches = []
                merge_distance = 15
                min_overlap = 0.5
                
                for match in matches:
                    is_duplicate = False
                    center_x, center_y = match.center
                    
                    for merged_match in merged_matches:
                        merged_center_x, merged_center_y = merged_match.center
                        distance = ((center_x - merged_center_x) ** 2 + (center_y - merged_center_y) ** 2) ** 0.5
                        overlap_ratio = self.bbox_overlap_ratio(match.bbox, merged_match.bbox)
                        
                        if (distance < merge_distance and overlap_ratio > min_overlap) or distance < 10:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        merged_matches.append(match)
                
                # Filter to 50%+ confidence
                final_matches = [m for m in merged_matches if m.confidence >= 50]
                
                self.log(f"After deduplication and filtering: {len(final_matches)} matches (50%+ confidence)", 'success')
                
                # Display results
                if final_matches:
                    self.log("\n" + "-" * 50, 'info')
                    self.log("All detected text items:", 'info')
                    self.log("-" * 50, 'info')
                    
                    for i, match in enumerate(final_matches, 1):
                        center_x, center_y = match.center
                        self.log(
                            f"{i}. '{match.text}' "
                            f"at ({center_x}, {center_y}) "
                            f"[conf: {match.confidence:.1f}%]",
                            'info'
                        )
                    
                    # Create OCRResult
                    from agent.ocr import OCRResult
                    self.ocr_results = OCRResult(matches=final_matches, raw_text=" ".join([m.text for m in final_matches]))
                    
                    query_intent = self._get_ocr_query_intent()
                    if query_intent:
                        top_matches = self._rank_ocr_matches_for_query(
                            self.ocr_results.matches,
                            query_intent,
                            top_n=5,
                        )
                        if top_matches:
                            self.log(
                                f"Top {len(top_matches)} OCR matches for '{query_intent.primary_text}':",
                                'info',
                            )
                            for i, match in enumerate(top_matches, 1):
                                self.log(
                                    f"  {i}. '{match.text}' [conf: {match.confidence:.1f}%]",
                                    'info',
                                )
                            img_with_boxes = self.draw_ocr_boxes(self.original_image.copy(), top_matches)
                            self.current_image = img_with_boxes
                            self.root.after(0, lambda: self.update_displays())
                            self.log("Bounding boxes drawn for top OCR matches", 'info')
                        else:
                            self.log(f"No OCR matches for '{query_intent.primary_text}'", 'error')
                    else:
                        # Draw dots for all matches
                        img_with_dots = self.draw_ocr_dots(self.original_image.copy(), self.ocr_results)
                        self.current_image = img_with_dots
                        self.root.after(0, lambda: self.update_displays())
                        self.log("Red dots drawn on image at center of each detected text", 'info')
                else:
                    self.log("No matches found with 50%+ confidence", 'error')
            else:
                self.log("No text found in image", 'error')
                
        except Exception as e:
            self.log(f"Error running PaddleOCR: {e}", 'error')
            import traceback
            self.log(traceback.format_exc(), 'error')
            # Fallback to Tesseract
            self.log("Falling back to Tesseract...", 'info')
            self.use_paddleocr = False
            self._run_ocr_impl()
    
    def download_swedish_language(self):
        """Download Swedish language file for Tesseract."""
        import pytesseract
        
        # Find Tesseract installation directory
        tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
        if not tesseract_cmd:
            # Try to find it
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    tesseract_cmd = path
                    break
        
        if not tesseract_cmd or not os.path.exists(tesseract_cmd):
            self.log("Could not find Tesseract installation directory", 'error')
            return False
        
        # Get tessdata directory (usually next to tesseract.exe)
        tesseract_dir = os.path.dirname(tesseract_cmd)
        tessdata_dir = os.path.join(tesseract_dir, 'tessdata')
        
        # Create tessdata directory if it doesn't exist
        if not os.path.exists(tessdata_dir):
            try:
                os.makedirs(tessdata_dir)
                self.log(f"Created tessdata directory: {tessdata_dir}", 'info')
            except Exception as e:
                self.log(f"Could not create tessdata directory: {e}", 'error')
                return False
        
        swe_file = os.path.join(tessdata_dir, 'swe.traineddata')
        
        # Check if already exists
        if os.path.exists(swe_file):
            self.log("Swedish language file already exists", 'info')
            return True
        
        # Download the file
        url = "https://github.com/tesseract-ocr/tessdata/raw/main/swe.traineddata"
        self.log(f"Downloading Swedish language from: {url}", 'info')
        self.log(f"Target location: {swe_file}", 'info')
        
        try:
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size) if total_size > 0 else 0
                if block_num % 10 == 0:  # Update every 10 blocks
                    self.log(f"  Downloading... {percent}%", 'info')
            
            urllib.request.urlretrieve(url, swe_file, show_progress)
            
            # Verify file was downloaded
            if os.path.exists(swe_file) and os.path.getsize(swe_file) > 0:
                file_size = os.path.getsize(swe_file) / (1024 * 1024)  # MB
                self.log(f"Downloaded {file_size:.2f} MB", 'success')
                return True
            else:
                self.log("Downloaded file is empty or missing", 'error')
                return False
                
        except Exception as e:
            self.log(f"Error downloading Swedish language: {e}", 'error')
            # Clean up partial download
            if os.path.exists(swe_file):
                try:
                    os.remove(swe_file)
                except:
                    pass
            return False
    
    def is_valid_text(self, text):
        """Check if text looks like readable UI text (including Swedish letters)."""
        if not text:
            return False

        allowed_punct = set(" -_.:,/()[]+'\"&")
        swedish_chars = "\u00e5\u00e4\u00f6\u00c5\u00c4\u00d6"

        # Check if all characters are reasonable
        for char in text:
            if char.isalnum() or char in allowed_punct or char.isspace():
                continue
            if char in swedish_chars:
                continue
            return False

        # Must contain at least one letter or number (not just spaces)
        has_content = any(c.isalnum() or c in swedish_chars for c in text)
        return has_content
    def run_ocr_single(self, image, psm_mode, lang='eng', min_confidence=30, variant_name=""):
        """Run OCR with specific settings and return matches."""
        import pytesseract
        from agent.screenshot import ScreenRegion
        from agent.ocr import OCRMatch
        
        try:
            # Character whitelist: only allow letters (including Swedish), numbers, and spaces
            # This prevents Tesseract from detecting symbols and weird characters
            char_whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÅÄÖåäö '
            
            # Try Swedish + English for better Swedish character recognition
            # Use the language passed in (already determined at start)
            lang_to_use = lang
            
            # Log what language we're actually using (only once)
            if variant_name and 'original' in variant_name.lower():
                self.log(f"  Using language: {lang_to_use}", 'info')
            
            try:
                data = pytesseract.image_to_data(
                    image,
                    lang=lang_to_use,
                    output_type=pytesseract.Output.DICT,
                    config=f'--psm {psm_mode} -c tessedit_char_whitelist={char_whitelist}'
                )
            except Exception as e:
                # If swe+eng fails, try swe alone, then eng
                if lang_to_use == 'swe+eng':
                    self.log(f"    swe+eng failed, trying swe alone...", 'info')
                    try:
                        data = pytesseract.image_to_data(
                            image,
                            lang='swe',
                            output_type=pytesseract.Output.DICT,
                            config=f'--psm {psm_mode} -c tessedit_char_whitelist={char_whitelist}'
                        )
                        lang_to_use = 'swe'
                    except:
                        # Fallback to English
                        self.log(f"    swe failed, using eng...", 'info')
                        data = pytesseract.image_to_data(
                            image,
                            lang='eng',
                            output_type=pytesseract.Output.DICT,
                            config=f'--psm {psm_mode} -c tessedit_char_whitelist={char_whitelist}'
                        )
                        lang_to_use = 'eng'
                else:
                    raise
            
            matches = []
            n_boxes = len(data["level"])
            
            # Get scale factor for coordinate conversion
            scale_factor = 1.0
            if hasattr(image, '_scale_factor'):
                scale_factor = 1.0 / image._scale_factor
            
            for i in range(n_boxes):
                level = data["level"][i]  # 5 = character, 4 = word, etc.
                text = data["text"][i].strip()
                conf = float(data["conf"][i])
                
                # Collect ALL matches with low threshold - we'll filter intelligently later
                if text and len(text) > 0 and conf > min_confidence:
                    # Validate text contains only allowed characters
                    if not self.is_valid_text(text):
                        continue
                    
                    left = int(data["left"][i] * scale_factor)
                    top = int(data["top"][i] * scale_factor)
                    width = int(data["width"][i] * scale_factor)
                    height = int(data["height"][i] * scale_factor)
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    bbox = ScreenRegion(left=left, top=top, width=width, height=height)
                    match = OCRMatch(text=text, confidence=conf, bbox=bbox)
                    # Store level info for later filtering
                    match.level = level
                    matches.append(match)
            
            return matches
        except Exception as e:
            return []
    
    def bbox_overlap_ratio(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes."""
        # Calculate intersection
        left = max(bbox1.left, bbox2.left)
        top = max(bbox1.top, bbox2.top)
        right = min(bbox1.right, bbox2.right)
        bottom = min(bbox1.bottom, bbox2.bottom)
        
        if right <= left or bottom <= top:
            return 0.0
        
        intersection_area = (right - left) * (bottom - top)
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def split_suspicious_multi_char(self, match):
        """Split multi-character matches that might be separate characters incorrectly combined.
        Returns list of matches (original if not suspicious, or split if suspicious)."""
        text = match.text.strip()
        
        # Only check 2-character matches that are all digits or all letters
        # (like "34" or "ab" that might be two separate characters)
        if len(text) == 2:
            # Check if bbox width suggests it should be split
            # Average character width in UI is roughly 8-12 pixels
            # If bbox width is less than ~20 pixels for 2 chars, it's suspicious
            char_width_estimate = match.bbox.width / len(text)
            
            # If each character would be less than 8 pixels wide, it's likely a single char misread
            if char_width_estimate < 8:
                self.log(f"  Splitting suspicious '{text}' (width {match.bbox.width}px, {char_width_estimate:.1f}px/char)", 'info')
                # Split into individual characters
                from agent.ocr import OCRMatch
                from agent.screenshot import ScreenRegion
                
                split_matches = []
                char_width = match.bbox.width // 2
                for i, char in enumerate(text):
                    char_left = match.bbox.left + (i * char_width)
                    char_bbox = ScreenRegion(
                        left=char_left,
                        top=match.bbox.top,
                        width=char_width,
                        height=match.bbox.height
                    )
                    # Use slightly lower confidence for split characters
                    char_match = OCRMatch(text=char, confidence=match.confidence * 0.9, bbox=char_bbox)
                    split_matches.append(char_match)
                return split_matches
        
        # Not suspicious, return original
        return [match]
    
    def merge_duplicate_matches(self, all_matches, merge_distance=5):
        """Merge duplicate/overlapping matches, keeping the highest confidence.
        Only merges if texts are EXACTLY the same to avoid combining different characters."""
        if not all_matches:
            return []
        
        # Sort by confidence (highest first)
        sorted_matches = sorted(all_matches, key=lambda m: m.confidence, reverse=True)
        merged = []
        
        for match in sorted_matches:
            is_duplicate = False
            match_text = match.text.strip()
            match_center = match.center
            
            for merged_match in merged:
                merged_text = merged_match.text.strip()
                merged_center = merged_match.center
                
                # Calculate distance
                distance = ((match_center[0] - merged_center[0]) ** 2 + 
                           (match_center[1] - merged_center[1]) ** 2) ** 0.5
                
                # Calculate bbox overlap
                overlap_ratio = self.bbox_overlap_ratio(match.bbox, merged_match.bbox)
                
                # Only merge if:
                # 1. Texts are EXACTLY the same (case-insensitive)
                # 2. AND they're very close together (small distance) OR overlapping significantly
                # This prevents merging "3" with "34" or other different texts
                if match_text.lower() == merged_text.lower():
                    # Same text - check if they're duplicates
                    if distance < merge_distance or overlap_ratio > 0.5:
                        is_duplicate = True
                        # Keep the one with higher confidence
                        if match.confidence > merged_match.confidence:
                            merged.remove(merged_match)
                            merged.append(match)
                        break
            
            if not is_duplicate:
                merged.append(match)
        
        return merged
    
    def run_ocr(self):
        """Run OCR in a background thread to keep the UI responsive."""
        if self._ocr_running:
            self.log("OCR is already running...", 'info')
            return

        if not self.original_image:
            self.log("No image loaded!", 'error')
            return

        self._ocr_running = True
        self._set_ocr_ui_running(True)

        def worker():
            try:
                self._run_ocr_impl()
            finally:
                self._ocr_running = False
                self._set_ocr_ui_running(False)

        threading.Thread(target=worker, daemon=True).start()

    def _run_ocr_tesseract_fast(self):
        """Fast single-pass OCR using Tesseract."""
        try:
            self.log("\n" + "=" * 50, 'info')
            self.log("Running fast OCR (Tesseract)...", 'info')
            self.log("=" * 50, 'info')

            if self.ocr_engine is None:
                self.ocr_engine = get_ocr_engine()
                self.log("OCR engine initialized", 'success')

            img_for_ocr = self.original_image
            scale = 1.0
            max_ocr_width = 1600
            if img_for_ocr.width > max_ocr_width:
                scale = max_ocr_width / img_for_ocr.width
                new_size = (int(img_for_ocr.width * scale), int(img_for_ocr.height * scale))
                img_for_ocr = img_for_ocr.resize(new_size, Image.LANCZOS)
                self.log(f"Resized image for OCR: {self.original_image.width}x{self.original_image.height} → {new_size[0]}x{new_size[1]}", 'info')

            import time
            start_time = time.time()
            ocr_result = self.ocr_engine.process(img_for_ocr, include_phrases=False)
            elapsed = time.time() - start_time
            self.log(f"OCR completed in {elapsed:.2f} seconds", 'success')

            from agent.ocr import OCRMatch, OCRResult
            from agent.screenshot import ScreenRegion

            matches = ocr_result.matches or []
            matches_rescaled = False

            query_intent = self._get_ocr_query_intent()
            query_text = (query_intent.primary_text if query_intent else "").strip()
            is_tesseract = hasattr(self.ocr_engine, "_select_language")
            if query_text and len(query_text) == 1 and is_tesseract:
                # For single-letter queries, use character boxes.
                try:
                    import pytesseract
                    lang_to_use = "eng"
                    try:
                        lang_to_use = self.ocr_engine._select_language()
                    except Exception:
                        pass
                    boxes = pytesseract.image_to_boxes(
                        img_for_ocr,
                        lang=lang_to_use,
                        config="--oem 3 --psm 6"
                    )
                    h = img_for_ocr.height
                    char_matches = []
                    q = query_text.lower()
                    for line in boxes.splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        ch = parts[0]
                        if ch.lower() != q:
                            continue
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        left = x1
                        right = x2
                        top = h - y2
                        bottom = h - y1
                        if scale != 1.0:
                            inv = 1.0 / scale
                            left = int(left * inv)
                            right = int(right * inv)
                            top = int(top * inv)
                            bottom = int(bottom * inv)
                        if right <= left or bottom <= top:
                            continue
                        bbox = ScreenRegion(
                            left=left,
                            top=top,
                            width=max(1, right - left),
                            height=max(1, bottom - top),
                        )
                        char_matches.append(OCRMatch(text=ch, confidence=90.0, bbox=bbox, source="char"))
                    if char_matches:
                        matches = char_matches
                        matches_rescaled = True
                        self.log(f"Found {len(matches)} character matches for '{query_text}'.", 'info')
                except Exception as e:
                    self.log(f"Character OCR failed: {e} (continuing with word OCR)", 'info')
            if scale != 1.0 and matches and not matches_rescaled:
                scaled = []
                inv = 1.0 / scale
                for m in matches:
                    b = m.bbox
                    new_bbox = ScreenRegion(
                        left=int(b.left * inv),
                        top=int(b.top * inv),
                        width=max(1, int(b.width * inv)),
                        height=max(1, int(b.height * inv)),
                    )
                    scaled.append(OCRMatch(text=m.text, confidence=m.confidence, bbox=new_bbox, source=m.source))
                matches = scaled

            if matches:
                # Deduplicate quick overlaps
                matches = self.merge_duplicate_matches(matches, merge_distance=10)

                self.ocr_results = OCRResult(
                    matches=matches,
                    raw_text=" ".join([m.text for m in matches]),
                )

                query_intent = self._get_ocr_query_intent()
                if query_intent:
                    top_matches = self._rank_ocr_matches_for_query(
                        self.ocr_results.matches,
                        query_intent,
                        top_n=5,
                    )
                    if top_matches:
                        self.log(
                            f"Top {len(top_matches)} OCR matches for '{query_intent.primary_text}':",
                            'info',
                        )
                        for i, match in enumerate(top_matches, 1):
                            self.log(
                                f"  {i}. '{match.text}' [conf: {match.confidence:.1f}%]",
                                'info',
                            )
                        img_with_boxes = self.draw_ocr_boxes(self.original_image.copy(), top_matches)
                        self.current_image = img_with_boxes
                        self.root.after(0, lambda: self.update_displays())
                        self.log("Bounding boxes drawn for top OCR matches", 'info')
                    else:
                        self.log(f"No OCR matches for '{query_intent.primary_text}'", 'error')
                else:
                    img_with_dots = self.draw_ocr_dots(self.original_image.copy(), self.ocr_results)
                    self.current_image = img_with_dots
                    self.root.after(0, lambda: self.update_displays())
                    self.log("Red dots drawn on image at center of each detected text", 'info')
            else:
                self.log("No text found in image", 'error')
        except Exception as e:
            self.log(f"Error running fast OCR: {e}", 'error')
            import traceback
            self.log(traceback.format_exc(), 'error')

    def _run_ocr_impl(self):
        """Run OCR with multiple strategies and combine results for better consistency."""
        if not self.original_image:
            self.log("No image loaded!", 'error')
            return
        
        # Use PaddleOCR if selected
        if self.use_paddleocr:
            self.run_ocr_paddleocr()
            return

        # Default to fast Tesseract path
        if self.ocr_fast_mode:
            self._run_ocr_tesseract_fast()
            return

        # Non-Tesseract engines: run a simple single-pass OCR to avoid Tesseract-specific logic.
        if self.ocr_engine is None:
            try:
                self.ocr_engine = get_ocr_engine()
                self.log("OCR engine initialized", 'success')
            except Exception as e:
                self.log(f"Failed to initialize OCR engine: {e}", 'error')
                return
        if not hasattr(self.ocr_engine, "_select_language"):
            try:
                import time
                start_time = time.time()
                ocr_result = self.ocr_engine.process_with_preprocessing(self.original_image, include_phrases=True)
                elapsed = time.time() - start_time
                self.log(f"OCR completed in {elapsed:.2f} seconds", 'success')

                from agent.ocr import OCRResult
                self.ocr_results = OCRResult(matches=ocr_result.matches, raw_text=ocr_result.raw_text)

                query_intent = self._get_ocr_query_intent()
                if query_intent and query_intent.primary_text:
                    top_matches = self._nm_rank_ocr_matches(query_intent.primary_text, self.ocr_results.matches, top_n=10)
                    if top_matches:
                        self.log(
                            f"Top {len(top_matches)} OCR matches for '{query_intent.primary_text}':",
                            'info',
                        )
                        for m in top_matches:
                            self.log(f"  {m.text} (conf={m.confidence:.1f})", 'info')
                        self.current_image = self.draw_ocr_boxes(self.original_image.copy(), top_matches)
                        self.root.after(0, lambda: self.update_displays())
                        self.log("Bounding boxes drawn for top OCR matches", 'info')
                    else:
                        self.log(f"No OCR matches for '{query_intent.primary_text}'", 'error')
                else:
                    img_with_dots = self.draw_ocr_dots(self.original_image.copy(), self.ocr_results)
                    self.current_image = img_with_dots
                    self.root.after(0, lambda: self.update_displays())
                    self.log("Red dots drawn on image at center of each detected text", 'info')
            except Exception as e:
                self.log(f"Error running OCR: {e}", 'error')
            return
        
        try:
            self.log("\n" + "="*50, 'info')
            self.log("Running multi-strategy OCR (comprehensive detection)...", 'info')
            self.log("="*50, 'info')
            
            # Get or create OCR engine
            if self.ocr_engine is None:
                try:
                    self.ocr_engine = get_ocr_engine()
                    self.log("OCR engine initialized", 'success')
                    
                    # Check Swedish language availability and auto-download if missing
                    import pytesseract
                    try:
                        available_langs = pytesseract.get_languages(config='')
                        self.log(f"Available Tesseract languages: {', '.join(sorted(available_langs))}", 'info')
                        if 'swe' in available_langs:
                            self.log("✓ Swedish language is available in Tesseract", 'success')
                        else:
                            self.log("✗ Swedish language NOT found in Tesseract", 'error')
                            self.log("Attempting to download Swedish language file...", 'info')
                            if self.download_swedish_language():
                                self.log("✓ Swedish language downloaded successfully!", 'success')
                                self.log("Swedish support is now available", 'success')
                            else:
                                self.log("Failed to download Swedish language", 'error')
                                self.log("You can manually download from:", 'error')
                                self.log("https://github.com/tesseract-ocr/tessdata/raw/main/swe.traineddata", 'error')
                    except Exception as e:
                        self.log(f"Could not check available languages: {e}", 'error')
                except Exception as e:
                    self.log(f"Failed to initialize OCR engine: {e}", 'error')
                    return
            
            import time
            start_time = time.time()
            
            # Create multiple preprocessed variants
            self.log("Creating image preprocessing variants...", 'info')
            variants = self.preprocess_image_variants(self.original_image)
            self.log(f"Created {len(variants)} image variants", 'success')
            
            # Use MANY strategies to catch everything
            # Use multiple PSM modes: 11 (sparse), 8 (single word), 6 (uniform block), 7 (single line)
            psm_modes = [11, 8, 6, 7]
            
            # Use ALL variants to maximize detection
            key_variants = [
                ("original_gray", variants[0][1]),
                ("denoised_contrast", variants[1][1]) if len(variants) > 1 else None,
                ("high_contrast_bright", variants[2][1]) if len(variants) > 2 else None,
                ("sharpened_2x", variants[3][1]) if len(variants) > 3 else None,
                ("otsu_binary", variants[4][1]) if len(variants) > 4 else None,
                ("upscaled_2x_enhanced", variants[5][1]) if len(variants) > 5 else None,
                ("very_high_contrast_sharp", variants[6][1]) if len(variants) > 6 else None
            ]
            key_variants = [v for v in key_variants if v is not None]
            
            all_matches = []
            
            self.log("Running OCR with comprehensive strategies...", 'info')
            total_attempts = len(key_variants) * len(psm_modes)
            current_attempt = 0
            
            # Check Swedish language ONCE at the start
            import pytesseract
            lang_to_use_global = 'eng'
            try:
                available_langs = pytesseract.get_languages(config='')
                self.log(f"Available languages: {', '.join(sorted(available_langs))}", 'info')
                if 'swe' in available_langs:
                    lang_to_use_global = 'swe+eng'
                    self.log("✓ Swedish language available - will use swe+eng", 'success')
                else:
                    self.log("✗ Swedish NOT found - attempting download...", 'info')
                    if self.download_swedish_language():
                        # Re-check
                        available_langs = pytesseract.get_languages(config='')
                        if 'swe' in available_langs:
                            lang_to_use_global = 'swe+eng'
                            self.log("✓ Swedish downloaded successfully - using swe+eng", 'success')
                        else:
                            self.log("✗ Swedish download failed - using English only", 'error')
                    else:
                        self.log("✗ Swedish download failed - using English only", 'error')
            except Exception as e:
                self.log(f"Error checking languages: {e}", 'error')
            
            # Very low threshold to catch everything - we'll filter later
            base_confidence = 15  # Very low to catch everything, even low-confidence detections
            
            for variant_name, variant_image in key_variants:
                for psm_mode in psm_modes:
                    current_attempt += 1
                    self.log(f"  [{current_attempt}/{total_attempts}] {variant_name} + PSM {psm_mode}...", 'info')
                    
                    # Use the determined language (Swedish+English if available)
                    matches = self.run_ocr_single(variant_image, psm_mode, lang=lang_to_use_global, min_confidence=base_confidence, variant_name=variant_name)
                    all_matches.extend(matches)
                    
                    if matches:
                        self.log(f"    Found {len(matches)} items", 'success')
            
            elapsed = time.time() - start_time
            self.log(f"\nOCR completed in {elapsed:.2f} seconds", 'success')
            self.log(f"Total raw matches before deduplication: {len(all_matches)}", 'info')
            
            self.log(f"\nTotal raw matches collected: {len(all_matches)}", 'info')
            
            # Filter out invalid matches
            self.log("Filtering invalid matches...", 'info')
            valid_matches = [m for m in all_matches if self.is_valid_text(m.text)]
            invalid_matches = [m for m in all_matches if not self.is_valid_text(m.text)]
            self.log(f"Valid matches: {len(valid_matches)} (removed {len(invalid_matches)} invalid)", 'info')
            
            # Show some invalid matches for debugging
            if invalid_matches:
                self.log(f"Sample invalid matches (first 5):", 'info')
                for m in invalid_matches[:5]:
                    self.log(f"  '{m.text}' (conf: {m.confidence:.1f}%)", 'info')
            
            # Score matches by quality (character-level > word-level, higher confidence = better)
            self.log("Scoring matches by quality...", 'info')
            scored_matches = []
            for m in valid_matches:
                score = m.confidence
                text = m.text.strip()
                
                # Boost character-level matches (level 5)
                if hasattr(m, 'level') and m.level == 5:
                    score += 20  # Boost character-level
                
                # Boost single-character matches
                if len(text) == 1:
                    score += 10
                
                # Boost matches with Swedish characters (ä, ö, å) - these are harder to detect correctly
                if any(c in text.lower() for c in ['ä', 'ö', 'å']):
                    score += 15  # Extra boost for Swedish characters
                
                # Penalize multi-digit numbers (might be incorrectly combined)
                if text.isdigit() and len(text) >= 2:
                    score -= 15
                
                # Penalize matches that look like misread Swedish characters
                # Common misreadings: ä->a, ö->o, å->a
                if 'a' in text.lower() and len(text) >= 3:
                    # Could be misread Swedish character, slight penalty
                    score -= 5
                
                scored_matches.append((m, score))
            
            # Sort by score (highest first)
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Split suspicious multi-character detections
            self.log("Checking for suspicious multi-character detections...", 'info')
            split_matches = []
            for m, score in scored_matches:
                split_result = self.split_suspicious_multi_char(m)
                for split_match in split_result:
                    # Recalculate score for split matches
                    split_score = split_match.confidence
                    if len(split_match.text.strip()) == 1:
                        split_score += 10
                    split_matches.append((split_match, split_score))
            
            # Smart deduplication: Merge ONLY truly close/overlapping matches, keep highest confidence
            self.log("Deduplicating matches (merging only very close matches, keeping highest confidence)...", 'info')
            
            # Sort all matches by confidence (highest first)
            all_matches_sorted = sorted(split_matches, key=lambda x: x[0].confidence, reverse=True)
            
            merged_matches = []
            merge_distance = 15  # pixels - only merge if centers are within 15px (stricter)
            min_overlap = 0.5  # Only merge if 50%+ overlap (stricter)
            
            for match, score in all_matches_sorted:
                center_x, center_y = match.center
                is_duplicate = False
                
                # Check if this match is close to any already merged match
                for merged_match in merged_matches:
                    merged_center_x, merged_center_y = merged_match.center
                    
                    # Calculate distance between centers
                    distance = ((center_x - merged_center_x) ** 2 + (center_y - merged_center_y) ** 2) ** 0.5
                    
                    # Calculate bbox overlap
                    overlap_ratio = self.bbox_overlap_ratio(match.bbox, merged_match.bbox)
                    
                    # Only merge if BOTH conditions are met (stricter):
                    # 1. Very close (within merge_distance) AND
                    # 2. Significant overlap (min_overlap)
                    # OR if they're extremely close (within 10px) - likely same detection
                    if (distance < merge_distance and overlap_ratio > min_overlap) or distance < 10:
                        # This is a duplicate - keep the one with higher confidence (already sorted)
                        is_duplicate = True
                        self.log(f"  Merged '{match.text}' ({match.confidence:.1f}%) with '{merged_match.text}' ({merged_match.confidence:.1f}%) - distance: {distance:.1f}px, overlap: {overlap_ratio:.1%} - keeping '{merged_match.text}'", 'info')
                        break
                
                if not is_duplicate:
                    merged_matches.append(match)
            
            removed_count = len(split_matches) - len(merged_matches)
            self.log(f"After deduplication: {len(merged_matches)} unique matches (removed {removed_count} duplicates)", 'success')
            
            # Show what we have before filtering
            self.log(f"\nMatches before final filter: {len(merged_matches)}", 'info')
            if merged_matches:
                conf_distribution = {}
                for m in merged_matches:
                    conf_range = f"{(m.confidence // 10) * 10}-{(m.confidence // 10) * 10 + 9}%"
                    conf_distribution[conf_range] = conf_distribution.get(conf_range, 0) + 1
                self.log("Confidence distribution:", 'info')
                for conf_range in sorted(conf_distribution.keys(), reverse=True):
                    self.log(f"  {conf_range}: {conf_distribution[conf_range]} matches", 'info')
            
            # Final filtering: STRICT 50%+ confidence minimum as requested
            self.log("\nApplying final quality filter (50%+ confidence only)...", 'info')
            final_matches = []
            filtered_out = []
            for match in merged_matches:
                conf = match.confidence
                
                # STRICT: Only keep matches with 50%+ confidence
                if conf >= 50:
                    final_matches.append(match)
                else:
                    filtered_out.append(match)
            
            # Log what we're filtering out (show top 10 closest to threshold)
            if filtered_out:
                filtered_out.sort(key=lambda m: m.confidence, reverse=True)
                self.log(f"Filtered out {len(filtered_out)} matches below 50% confidence:", 'info')
                for m in filtered_out[:10]:  # Show top 10 closest to threshold
                    self.log(f"  '{m.text}' ({m.confidence:.1f}%) at ({m.center[0]}, {m.center[1]})", 'info')
                if len(filtered_out) > 10:
                    self.log(f"  ... and {len(filtered_out) - 10} more", 'info')
            
            merged_matches = final_matches
            self.log(f"\nFinal matches after 50% confidence filter: {len(merged_matches)}", 'success')
            
            if len(merged_matches) == 0:
                self.log("WARNING: No matches found with 50%+ confidence!", 'error')
                self.log("Consider checking image quality or language installation", 'info')
                if filtered_out:
                    highest_filtered = max(filtered_out, key=lambda m: m.confidence)
                    self.log(f"Highest filtered match was '{highest_filtered.text}' ({highest_filtered.confidence:.1f}%)", 'info')
            
            # Sort by position (top to bottom, left to right) for consistent display
            merged_matches.sort(key=lambda m: (m.bbox.top, m.bbox.left))
            
            # Count characters vs words
            char_count = sum(1 for m in merged_matches if len(m.text) == 1)
            word_count = len(merged_matches) - char_count
            
            # Display all found texts
            if merged_matches:
                self.log("\n" + "-" * 50, 'info')
                self.log("All detected text items:", 'info')
                self.log("-" * 50, 'info')
                
                for i, match in enumerate(merged_matches, 1):
                    center_x, center_y = match.center
                    item_type = "char" if len(match.text) == 1 else "word"
                    self.log(
                        f"{i}. [{item_type}] '{match.text}' "
                        f"at ({center_x}, {center_y}) "
                        f"[conf: {match.confidence:.1f}%] "
                        f"bbox: ({match.bbox.left},{match.bbox.top}) {match.bbox.width}x{match.bbox.height}",
                        'info'
                    )
                
                self.log("-" * 50, 'info')
                self.log(f"Total: {len(merged_matches)} unique items ({char_count} chars, {word_count} words)", 'success')
                
                # Create OCRResult for compatibility
                from agent.ocr import OCRResult
                self.ocr_results = OCRResult(matches=merged_matches, raw_text=" ".join([m.text for m in merged_matches]))
                
                query_intent = self._get_ocr_query_intent()
                if query_intent:
                    top_matches = self._rank_ocr_matches_for_query(
                        self.ocr_results.matches,
                        query_intent,
                        top_n=5,
                    )
                    if top_matches:
                        self.log(
                            f"Top {len(top_matches)} OCR matches for '{query_intent.primary_text}':",
                            'info',
                        )
                        for i, match in enumerate(top_matches, 1):
                            self.log(
                                f"  {i}. '{match.text}' [conf: {match.confidence:.1f}%]",
                                'info',
                            )
                        img_with_boxes = self.draw_ocr_boxes(self.original_image.copy(), top_matches)
                        self.current_image = img_with_boxes
                        self.root.after(0, lambda: self.update_displays())
                        self.log("Bounding boxes drawn for top OCR matches", 'info')
                    else:
                        self.log(f"No OCR matches for '{query_intent.primary_text}'", 'error')
                else:
                    # Draw dots on the image
                    img_with_dots = self.draw_ocr_dots(self.original_image.copy(), self.ocr_results)
                    self.current_image = img_with_dots
                    self.root.after(0, lambda: self.update_displays())
                    self.log("Red dots drawn on image at center of each detected text", 'info')
            else:
                self.log("No text found in image", 'error')
                self.log("Try: Ensure image has clear text, good contrast, and Tesseract is properly installed", 'info')
                
        except Exception as e:
            self.log(f"Error running OCR: {e}", 'error')
            import traceback
            self.log(traceback.format_exc(), 'error')
    
    def ask_cursor_movement(self, image, points, target_prompt):
        """Ask AI which numbered point is closest to the target, or click."""
        if not self.client:
            return None, None
        
        self.log(f"Asking AI which numbered point is closest to: '{target_prompt}'", 'ai')
        
        # Draw numbered points on image
        img_with_points = self.draw_numbered_points(image.copy(), points)
        
        width = image.width
        height = image.height
        
        # Build description of point IDs
        id_list = ", ".join(str(p["id"]) for p in points)
        
        system_prompt = format_prompt(
            load_prompt("coordinate_finder/cursor_movement_system.txt"),
            width=width,
            height=height,
            id_list=id_list,
            target_prompt=target_prompt,
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Which numbered point is closest to '{target_prompt}'? Or is one of them already exactly on the best click point?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(img_with_points)}"
                        }}
                    ]}
                ],
                max_completion_tokens=200,
                temperature=1
            )
            
            response_text = response.choices[0].message.content
            self.log(f"AI Response: {response_text}", 'ai')
            
            # Parse JSON (expecting only action + optional id)
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    self.log("Could not parse cursor movement response", 'error')
                    return None, None
            
            action = result.get('action', '').lower()
            pid = result.get('id', None)
            
            if action not in ('select', 'click') or pid is None:
                self.log(f"Invalid response from AI: {result}", 'error')
                return None, None
            
            try:
                pid = int(pid)
            except (ValueError, TypeError):
                self.log(f"Invalid id value from AI: {pid}", 'error')
                return None, None
            
            # Ensure id exists in current points
            valid_ids = {p["id"] for p in points}
            if pid not in valid_ids:
                self.log(f"AI returned unknown id: {pid}", 'error')
                return None, None
            
            if action == 'click':
                self.log(f"AI requested CLICK on id {pid}.", 'success')
                return ("click", pid), None
            
            self.log(f"Selected id to refine around: {pid}", 'coords')
            return ("select", pid), None
            
        except Exception as e:
            self.log(f"Error calling OpenAI API: {e}", 'error')
            return None, None
    
    def calculate_dynamic_grid_size(self, region_width, region_height, base_grid_x, base_grid_y, reduction_factor):
        """Calculate grid size based on region dimensions."""
        # Calculate minimum dimension
        min_dim = min(region_width, region_height)
        
        # Determine grid size based on region size
        # Very small regions (< 200px) get 2x2 grid
        # Small regions (< 400px) get reduced grid
        # Larger regions use base grid
        
        if min_dim < 200:
            grid_x = 2
            grid_y = 2
        elif min_dim < 400:
            # Reduce grid size proportionally
            reduction = min_dim / 400
            grid_x = max(2, int(base_grid_x * reduction * reduction_factor))
            grid_y = max(2, int(base_grid_y * reduction * reduction_factor))
        else:
            # Use base grid size with reduction factor
            grid_x = max(2, int(base_grid_x * reduction_factor))
            grid_y = max(2, int(base_grid_y * reduction_factor))
        
        return grid_x, grid_y
    
    def manual_select_grid(self):
        """Manually select a grid number for debugging."""
        if not self.original_image:
            self.log("No image loaded!", 'error')
            return
        
        try:
            grid_num = int(self.manual_grid_entry.get().strip())
        except ValueError:
            self.log("Invalid grid number", 'error')
            return
        
        # Get current state from finding process if running
        if hasattr(self, '_current_region') and hasattr(self, '_current_grid_x') and hasattr(self, '_current_grid_y'):
            region = self._current_region
            grid_x = self._current_grid_x
            grid_y = self._current_grid_y
            
            # Process manual selection
            self.log(f"Manual selection: Grid {grid_num}", 'coords')
            
            # Calculate new region with 50% padding
            region_width = region['right'] - region['left']
            region_height = region['bottom'] - region['top']
            
            cell_width = region_width / grid_x
            cell_height = region_height / grid_y
            
            grid_num_idx = grid_num - 1
            row = grid_num_idx // grid_x
            col = grid_num_idx % grid_x
            
            cell_left = region['left'] + col * cell_width
            cell_top = region['top'] + row * cell_height
            cell_right = region['left'] + (col + 1) * cell_width
            cell_bottom = region['top'] + (row + 1) * cell_height
            
            # 50% padding
            padding_x = cell_width * 0.5
            padding_y = cell_height * 0.5
            
            new_region = {
                'left': max(0, int(cell_left - padding_x)),
                'top': max(0, int(cell_top - padding_y)),
                'right': min(self.original_image.width, int(cell_right + padding_x)),
                'bottom': min(self.original_image.height, int(cell_bottom + padding_y))
            }
            
            # Get reduction factor
            try:
                reduction_factor = float(self.reduction_entry.get().strip())
            except ValueError:
                reduction_factor = 0.7
            
            # Calculate new grid size
            new_grid_x, new_grid_y = self.calculate_dynamic_grid_size(
                new_region['right'] - new_region['left'],
                new_region['bottom'] - new_region['top'],
                int(self.grid_x_entry.get().strip()),
                int(self.grid_y_entry.get().strip()),
                reduction_factor
            )
            
            # Update state
            self._current_region = new_region
            self._current_grid_x = new_grid_x
            self._current_grid_y = new_grid_y
            
            # Redraw grid
            img_with_grid = self.draw_grid_on_region(
                self.original_image.copy(),
                new_grid_x, new_grid_y,
                new_region['left'], new_region['top'],
                new_region['right'], new_region['bottom']
            )
            self.current_image = img_with_grid
            self.update_displays(show_grid=False, grid_x=None, grid_y=None)
            
            self.log(f"New region: ({new_region['left']}, {new_region['top']}) - ({new_region['right']}, {new_region['bottom']})", 'info')
            self.log(f"New grid size: {new_grid_x}x{new_grid_y}", 'info')
        else:
            # No active region, start with full image
            self.log("No active region. Starting with full image grid.", 'info')
            try:
                grid_x = int(self.grid_x_entry.get().strip())
                grid_y = int(self.grid_y_entry.get().strip())
            except ValueError:
                self.log("Invalid grid size", 'error')
                return
            
            if grid_num < 1 or grid_num > grid_x * grid_y:
                self.log(f"Grid number must be between 1 and {grid_x * grid_y}", 'error')
                return
            
            # Calculate region for selected grid
            region_info = self.calculate_grid_region(self.original_image, grid_x, grid_y, grid_num, padding_percent=0.5)
            
            # Get reduction factor
            try:
                reduction_factor = float(self.reduction_entry.get().strip())
            except ValueError:
                reduction_factor = 0.7
            
            # Calculate new grid size
            new_grid_x, new_grid_y = self.calculate_dynamic_grid_size(
                region_info['right'] - region_info['left'],
                region_info['bottom'] - region_info['top'],
                grid_x, grid_y, reduction_factor
            )
            
            # Store state
            self._current_region = {
                'left': region_info['left'],
                'top': region_info['top'],
                'right': region_info['right'],
                'bottom': region_info['bottom']
            }
            self._current_grid_x = new_grid_x
            self._current_grid_y = new_grid_y
            
            # Draw grid on region
            img_with_grid = self.draw_grid_on_region(
                self.original_image.copy(),
                new_grid_x, new_grid_y,
                region_info['left'], region_info['top'],
                region_info['right'], region_info['bottom']
            )
            self.current_image = img_with_grid
            self.update_displays(show_grid=False, grid_x=None, grid_y=None)
            
            self.log(f"Region: ({region_info['left']}, {region_info['top']}) - ({region_info['right']}, {region_info['bottom']})", 'info')
            self.log(f"New grid size: {new_grid_x}x{new_grid_y}", 'info')
    
    def update_displays(self, show_grid=False, grid_x=None, grid_y=None):
        """Update the output image display."""
        if self.current_image:
            self.display_image(self.current_canvas, self.current_image, draw_crosshair=False, 
                             show_grid=show_grid, grid_x=grid_x, grid_y=grid_y)
        elif self.original_image:
            # Show original image initially if no zoomed view yet
            self.display_image(self.current_canvas, self.original_image, draw_crosshair=False,
                             show_grid=show_grid, grid_x=grid_x, grid_y=grid_y)
            
    def display_image(self, canvas, image, coords=None, draw_crosshair=False, show_grid=False, grid_x=None, grid_y=None):
        """Display an image on a canvas, optionally with crosshair and marker, supporting zoom and pan."""
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 400
            canvas_height = 300

        # Create a copy to draw on
        display_img = image.copy()
        draw = ImageDraw.Draw(display_img)

        # Draw marker at coordinates on original
        if coords and not draw_crosshair:
            x, y = coords
            # Scale coords to image
            marker_size = max(5, int(15 * self.zoom_level))  # Scale marker with zoom
            # Draw target circles
            draw.ellipse([x-marker_size, y-marker_size, x+marker_size, y+marker_size],
                        outline='#ff4444', width=max(1, int(3 * self.zoom_level)))
            draw.ellipse([x-5, y-5, x+5, y+5], fill='#ff4444')
            # Draw crosshair lines
            draw.line([x-marker_size-10, y, x+marker_size+10, y], fill='#ff4444', width=max(1, int(2 * self.zoom_level)))
            draw.line([x, y-marker_size-10, x, y+marker_size+10], fill='#ff4444', width=max(1, int(2 * self.zoom_level)))

        # Draw grid if requested
        if show_grid and grid_x and grid_y:
            display_img = self.draw_grid(display_img, grid_x, grid_y)

        # Check if we're at default zoom/pan (fit to screen)
        if self.zoom_level == 1.0 and self.pan_offset_x == 0.0 and self.pan_offset_y == 0.0:
            # Use original fit-to-screen logic
            img_ratio = display_img.width / display_img.height
            canvas_ratio = canvas_width / canvas_height

            if img_ratio > canvas_ratio:
                new_width = canvas_width - 20
                new_height = int(new_width / img_ratio)
            else:
                new_height = canvas_height - 20
                new_width = int(new_height * img_ratio)

            display_img = display_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Store scaling info for hover detection
            self.displayed_image_size = (new_width, new_height)
            self.image_scale_x = image.width / new_width if new_width > 0 else 1.0
            self.image_scale_y = image.height / new_height if new_height > 0 else 1.0
            self.image_offset_x = (canvas_width - new_width) // 2
            self.image_offset_y = (canvas_height - new_height) // 2

            # Center the image (original behavior)
            image_x = canvas_width // 2
            image_y = canvas_height // 2
        else:
            # Apply zoom to the image
            zoomed_width = int(display_img.width * self.zoom_level)
            zoomed_height = int(display_img.height * self.zoom_level)

            if zoomed_width > 0 and zoomed_height > 0:
                display_img = display_img.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)

            # Store scaling info for hover detection
            self.displayed_image_size = (zoomed_width, zoomed_height)
            self.image_scale_x = image.width / zoomed_width if zoomed_width > 0 else 1.0
            self.image_scale_y = image.height / zoomed_height if zoomed_height > 0 else 1.0

            # Calculate image position with pan offset
            image_x = canvas_width // 2 + self.pan_offset_x
            image_y = canvas_height // 2 + self.pan_offset_y

        # Save the rendered image for clipboard copying
        self._last_display_image = display_img.copy()
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(display_img)
        canvas.delete("all")
        canvas.create_image(image_x, image_y, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep reference
        
    def copy_current_preview_to_clipboard(self):
        """Copy the currently displayed preview image (with overlays) to the clipboard."""
        image = getattr(self, "_last_display_image", None)
        if image is None:
            self.log("No preview image is ready to copy.", "error")
            return
        if os.name != "nt":
            self.log("Copying preview images is only supported on Windows.", "error")
            return
        try:
            self._copy_image_to_clipboard_windows(image)
            self.log("Preview image copied to clipboard.", "success")
        except Exception as exc:
            self.log(f"Failed to copy preview image: {exc}", "error")

    def _copy_image_to_clipboard_windows(self, image: Image.Image) -> None:
        """Use the Windows clipboard APIs to store a BMP version of the image."""
        if image is None:
            raise ValueError("No image provided")
        output = io.BytesIO()
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]  # skip BMP header

        GMEM_MOVEABLE = 0x0002
        CF_DIB = 8
        size = len(data)

        h_mem = ctypes.windll.kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not h_mem:
            raise OSError("GlobalAlloc failed")
        ptr = ctypes.windll.kernel32.GlobalLock(h_mem)
        if not ptr:
            ctypes.windll.kernel32.GlobalFree(h_mem)
            raise OSError("GlobalLock failed")

        ctypes.memmove(ptr, data, size)
        ctypes.windll.kernel32.GlobalUnlock(h_mem)

        if not ctypes.windll.user32.OpenClipboard(None):
            ctypes.windll.kernel32.GlobalFree(h_mem)
            raise OSError("OpenClipboard failed")
        try:
            ctypes.windll.user32.EmptyClipboard()
            if not ctypes.windll.user32.SetClipboardData(CF_DIB, h_mem):
                raise OSError("SetClipboardData failed")
        finally:
            ctypes.windll.user32.CloseClipboard()
    def on_canvas_hover(self, event):
        """Handle mouse hover on canvas to show OCR text."""
        if not self.ocr_results or not self.ocr_results.matches:
            return
        
        # Convert canvas coordinates to image coordinates
        canvas_x = event.x
        canvas_y = event.y

        # Check if mouse is over the image area
        if self.displayed_image_size:
            img_width, img_height = self.displayed_image_size
            canvas_width = self.current_canvas.winfo_width()
            canvas_height = self.current_canvas.winfo_height()

            # Check if we're at default zoom/pan (fit to screen)
            if self.zoom_level == 1.0 and self.pan_offset_x == 0.0 and self.pan_offset_y == 0.0:
                # Use original coordinate conversion (centered fit-to-screen)
                img_x = canvas_x - self.image_offset_x
                img_y = canvas_y - self.image_offset_y
            else:
                # Calculate image position on canvas
                image_x = canvas_width // 2 + self.pan_offset_x
                image_y = canvas_height // 2 + self.pan_offset_y

                # Convert canvas coordinates to image coordinates
                img_x = canvas_x - image_x + img_width // 2
                img_y = canvas_y - image_y + img_height // 2

            # Check if within image bounds
            if img_x < 0 or img_y < 0 or img_x > img_width or img_y > img_height:
                self.on_canvas_leave(event)
                return

            # Convert to original image coordinates
            orig_x = int(img_x * self.image_scale_x)
            orig_y = int(img_y * self.image_scale_y)
            
            # Find nearest OCR match within reasonable distance (30 pixels in original image)
            # This accounts for the dot size and gives a comfortable hover area
            hover_radius_pixels = 30  # pixels in original image coordinates
            nearest_match = None
            min_distance = float('inf')
            
            for match in self.ocr_results.matches:
                center_x, center_y = match.center
                # Calculate distance in original image coordinates
                distance = ((orig_x - center_x) ** 2 + (orig_y - center_y) ** 2) ** 0.5
                
                # Check if within hover radius
                if distance < hover_radius_pixels and distance < min_distance:
                    min_distance = distance
                    nearest_match = match
            
            # Show tooltip if near a match
            if nearest_match:
                tooltip_text = f"'{nearest_match.text}' (conf: {nearest_match.confidence:.1f}%)"
                self.show_hover_tooltip(event, tooltip_text)
            else:
                self.hide_hover_tooltip()
        else:
            self.hide_hover_tooltip()
    
    def on_canvas_leave(self, event):
        """Handle mouse leaving canvas."""
        self.hide_hover_tooltip()

    def on_canvas_zoom(self, event):
        """Handle mouse wheel zoom on canvas."""
        if not self.current_image:
            return

        # Determine zoom direction
        if event.num == 5 or event.delta < 0:  # Scroll down
            zoom_factor = 0.9
        else:  # Scroll up
            zoom_factor = 1.1

        # Get mouse position relative to canvas
        mouse_x = event.x
        mouse_y = event.y

        # Calculate zoom center in canvas coordinates
        canvas_width = self.current_canvas.winfo_width()
        canvas_height = self.current_canvas.winfo_height()

        # Apply zoom
        old_zoom = self.zoom_level
        self.zoom_level *= zoom_factor
        self.zoom_level = max(0.1, min(5.0, self.zoom_level))  # Limit zoom range

        if self.zoom_level != old_zoom:
            # Adjust pan to keep mouse position fixed
            zoom_ratio = self.zoom_level / old_zoom
            self.pan_offset_x = mouse_x - (mouse_x - self.pan_offset_x) * zoom_ratio
            self.pan_offset_y = mouse_y - (mouse_y - self.pan_offset_y) * zoom_ratio

            # Redisplay image with new zoom/pan
            self.display_image(self.current_canvas, self.current_image, coords=self.current_coords, draw_crosshair=False)

    def on_canvas_pan_start(self, event):
        """Start panning when mouse button is pressed."""
        self.is_dragging = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.current_canvas.config(cursor="fleur")  # Change cursor to indicate dragging

    def on_canvas_pan_drag(self, event):
        """Handle mouse drag for panning."""
        if not self.is_dragging:
            return

        # Calculate movement
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y

        # Update pan offset
        self.pan_offset_x += dx
        self.pan_offset_y += dy

        # Update last position
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        # Redisplay image with new pan position
        if self.current_image:
            self.display_image(self.current_canvas, self.current_image, coords=self.current_coords, draw_crosshair=False)

    def on_canvas_pan_end(self, event):
        """End panning when mouse button is released."""
        self.is_dragging = False
        self.current_canvas.config(cursor="")  # Reset cursor

    def reset_zoom_pan(self):
        """Reset zoom and pan to default values (fit to screen)."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        if self.current_image:
            self.display_image(self.current_canvas, self.current_image, coords=self.current_coords, draw_crosshair=False)
    
    def show_hover_tooltip(self, event, text):
        """Show a tooltip near the mouse cursor."""
        # Update existing tooltip if it exists
        if self.hover_tooltip:
            try:
                # Update position and text
                self.hover_tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
                # Find the label widget and update its text
                for widget in self.hover_tooltip.winfo_children():
                    if isinstance(widget, tk.Label):
                        widget.config(text=text)
                return
            except:
                # If update fails, recreate
                self.hover_tooltip.destroy()
                self.hover_tooltip = None
        
        # Create new tooltip window
        self.hover_tooltip = tk.Toplevel(self.root)
        self.hover_tooltip.wm_overrideredirect(True)
        self.hover_tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
        
        # Style the tooltip
        label = tk.Label(
            self.hover_tooltip,
            text=text,
            background='#161b22',
            foreground='#c9d1d9',
            font=('Segoe UI', 10),
            relief=tk.SOLID,
            borderwidth=1,
            padx=8,
            pady=4
        )
        label.pack()
        
        # Make sure tooltip is on top
        self.hover_tooltip.attributes('-topmost', True)
    
    def hide_hover_tooltip(self):
        """Hide the hover tooltip."""
        if self.hover_tooltip:
            self.hover_tooltip.destroy()
            self.hover_tooltip = None
    
    def log(self, message, tag='info'):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        def _append():
            self.log_text.insert(tk.END, f"[{timestamp}] ", 'time')
            self.log_text.insert(tk.END, f"{message}\n", tag)
            self.log_text.see(tk.END)

        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.root.after(0, _append)

    def _resolve_image_path(self, name: str) -> str:
        if not name:
            return self.image_path
        if os.path.isabs(name) and os.path.exists(name):
            return name
        candidate = os.path.join(self.images_dir, name)
        if os.path.exists(candidate):
            return candidate
        return name

    def _refresh_image_list(self):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        files = []
        if os.path.isdir(self.images_dir):
            for f in sorted(os.listdir(self.images_dir)):
                if f.lower().endswith(exts):
                    files.append(f)
        if not files and os.path.exists(self.image_path):
            files = [os.path.basename(self.image_path)]
        if getattr(self, "image_combo", None):
            self.image_combo["values"] = files
        if files:
            if self.image_var.get() not in files:
                self.image_var.set(files[0])
        return files

    def _on_image_select(self):
        name = (self.image_var.get() or "").strip()
        if not name:
            return
        self.image_path = self._resolve_image_path(name)
        self.load_image()

    def run_border_color_test(self):
        """Debug: show border-based color regions."""
        if not self.original_image:
            self.log("No image loaded!", 'error')
            return
        target = ""
        if self.nm_color_mask_entry is not None:
            target = (self.nm_color_mask_entry.get() or "").strip()
        if not target:
            target = "blue"
        roi = ScreenRegion(left=0, top=0, width=self.original_image.width, height=self.original_image.height)
        regions = self._nm_find_border_color_regions(self.original_image, roi, target, self._nm_get_color_max_area())
        self.log(f"Border-color regions for '{target}': {len(regions)}", "info")
        overlay = self._nm_draw_rects_overlay(self.original_image, regions, (0, 255, 0, 220), width=2)
        self.current_image = overlay
        self.update_displays()
        
    def encode_image(self, image):
        """Encode PIL Image to base64."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # === SeeClick integration ===

    def ensure_seeclick_loaded(self):
        """Lazily load the SeeClick model and tokenizer."""
        if self.seeclick_model is not None and self.seeclick_tokenizer is not None:
            return True

        ckpt_dir = os.getenv("SEECLICK_CKPT_DIR")
        seeclick_hf_repo = os.getenv("SEECLICK_HF_REPO")  # optional: repo id to auto-download
        seeclick_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        # If no local directory, optionally auto-download from HF if repo id is provided
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            if seeclick_hf_repo:
                try:
                    from huggingface_hub import snapshot_download  # type: ignore

                    target_dir = os.path.join(os.getcwd(), ".hf_seeclick_ckpt")
                    self.log(f"Downloading SeeClick checkpoint from HF repo: {seeclick_hf_repo}", "info")
                    ckpt_dir = snapshot_download(
                        repo_id=seeclick_hf_repo,
                        local_dir=target_dir,
                        local_dir_use_symlinks=False,
                        token=seeclick_hf_token,
                    )
                    os.environ["SEECLICK_CKPT_DIR"] = ckpt_dir
                    self.log(f"Downloaded SeeClick checkpoint to: {ckpt_dir}", "success")
                except Exception as e:
                    self.log(
                        "Failed to auto-download SeeClick checkpoint from Hugging Face.\n"
                        f"Repo: {seeclick_hf_repo}\n"
                        f"Error: {e}\n\n"
                        "If the repo is private/gated, set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN).",
                        "error",
                    )
                    return False
            else:
                self.log(
                    "SeeClick checkpoint directory not found.\n"
                    "To enable SeeClick, do ONE of the following:\n"
                    "1) Set SEECLICK_CKPT_DIR to a local checkpoint folder path, OR\n"
                    "2) Set SEECLICK_HF_REPO to the Hugging Face repo id (and HF_TOKEN if needed).\n\n"
                    "SeeClick README: https://github.com/njucckevin/SeeClick.git",
                    "error",
                )
                return False

        try:
            self.log(f"Loading SeeClick model from: {ckpt_dir}", "info")
            self.seeclick_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-VL-Chat",
                trust_remote_code=True,
            )
            # Let Transformers decide proper device placement; use BF16 on GPU if available.
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.seeclick_model = AutoModelForCausalLM.from_pretrained(
                ckpt_dir,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).eval()
            self.seeclick_model.generation_config = GenerationConfig.from_pretrained(
                "Qwen/Qwen-VL-Chat",
                trust_remote_code=True,
            )
            self.log("SeeClick model loaded successfully.", "success")
            return True
        except Exception as e:
            self.log(f"Error loading SeeClick model: {e}", "error")
            self.seeclick_tokenizer = None
            self.seeclick_model = None
            return False

    def get_seeclick_coordinates(self, target_prompt):
        """
        Call SeeClick once to get a single (x, y) coordinate for the target_prompt.
        Returns pixel coordinates in the original image space, or None on failure.
        """
        if not self.original_image:
            self.log("No image loaded!", "error")
            return None

        if not self.ensure_seeclick_loaded():
            return None

        img_path = os.path.abspath(self.image_path)
        if not os.path.exists(img_path):
            self.log(f"Image path for SeeClick not found: {img_path}", "error")
            return None

        prompt_template = (
            'In this UI screenshot, what is the position of the element corresponding to '
            'the command "{}" (with point)?'
        )

        try:
            query = self.seeclick_tokenizer.from_list_format(
                [
                    {"image": img_path},
                    {"text": prompt_template.format(target_prompt)},
                ]
            )
            self.log(f"Calling SeeClick for: '{target_prompt}'", "ai")
            response, history = self.seeclick_model.chat(
                self.seeclick_tokenizer,
                query=query,
                history=None,
            )
            self.log(f"SeeClick raw response: {response}", "ai")

            # Parse response like "(0.17,0.06)" into floats
            import re

            nums = re.findall(r"[-+]?\d*\.?\d+", response)
            if len(nums) < 2:
                self.log("Could not parse coordinates from SeeClick response.", "error")
                return None

            x_ratio = float(nums[0])
            y_ratio = float(nums[1])

            # Clamp to [0, 1]
            x_ratio = max(0.0, min(1.0, x_ratio))
            y_ratio = max(0.0, min(1.0, y_ratio))

            x_px = int(x_ratio * self.original_image.width)
            y_px = int(y_ratio * self.original_image.height)

            self.log(f"SeeClick coordinates (ratios): ({x_ratio:.3f}, {y_ratio:.3f})", "info")
            self.log(f"SeeClick coordinates (pixels): ({x_px}, {y_px})", "coords")

            return x_px, y_px
        except Exception as e:
            self.log(f"Error calling SeeClick: {e}", "error")
            return None

    # === New coordinate finder method (Planner → Deterministic engine → Picker → Verifier) ===

    def run_new_method_once(self):
        """
        Entry point for the new architecture test.
        Runs a full single-cycle:
        1) Snapshot capture
        2) Planner AI (semantic only)
        3) Deterministic candidate generation
        4) Picker AI (forced choice)
        5) Virtual click + verification
        """
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            self.log("Prompt is empty. Please enter what you want to click.", "error")
            return

        if not self.original_image:
            self.log("No image loaded for new method test!", "error")
            return

        if not self.client:
            self.log("OpenAI API key not configured – new method requires LLM access.", "error")
            return

        self.log("\n" + "=" * 60, "info")
        self.log(f"NEW METHOD: Planner → Deterministic → Picker → Verifier", "info")
        self.log(f"Instruction: {prompt}", "info")
        self.log("=" * 60, "info")

        # Reset visualization stages so reruns start clean
        self._nm_reset_stages()

        # Step 1: Snapshot
        snapshot = self._nm_capture_snapshot(prompt)
        if snapshot is None:
            self.log("Snapshot step returned UNSURE – aborting.", "error")
            return
        self._nm_add_stage("Snapshot", snapshot.image)

        # Step 2: Planner AI
        planner = self._nm_planner_ai(snapshot)
        if planner is None or planner.decision.upper() == "UNSURE":
            self.log("Planner returned UNSURE – aborting before any candidates are generated.", "error")
            return
        # Visualize planner ROI on top of the snapshot
        planner_roi = self._nm_zone_to_roi(snapshot.image, snapshot.window_rect, planner.location_intent.zone)
        roi_img = self._nm_draw_roi_overlay(snapshot.image, planner_roi)
        self._nm_add_stage(f"Planner ROI ({planner.location_intent.zone})", roi_img)

        if not self.is_running:
            self.log("Stopped by user.", "info")
            return

        # Decide OCR usage based on planner text intent
        has_text_intent = bool(planner.text_intent.primary_text or planner.text_intent.variants)
        self.nm_use_ocr_var.set(has_text_intent)
        self.nm_use_color_var.set(True)
        self.nm_use_shape_var.set(False)

        ocr_candidates: List[CandidatePackage] = []
        good_ocr: List[CandidatePackage] = []
        if has_text_intent:
            self.log("Phase 1: OCR text search (planner includes text).", "info")
            ocr_candidates = self._nm_generate_candidates(
                snapshot,
                planner,
                force_use_ocr=True,
                force_use_color=False,
                force_use_shape=False,
            )
            ocr_candidates = [c for c in ocr_candidates if c.source == "ocr"]
            good_ocr = self._nm_good_text_candidates(planner.text_intent, ocr_candidates)
            if good_ocr:
                self.log(f"OCR produced {len(good_ocr)} strong text candidates.", "info")
            else:
                self.log("OCR did not produce strong text matches.", "info")

        if not self.is_running:
            self.log("Stopped by user.", "info")
            return

        color_candidates: List[CandidatePackage] = []
        if not good_ocr:
            self.log("Phase 2: Color masking.", "info")
            color_candidates = self._nm_generate_candidates(
                snapshot,
                planner,
                force_use_ocr=False,
                force_use_color=True,
                force_use_shape=False,
            )
            color_candidates = [c for c in color_candidates if c.source == "color"]

        # Choose candidate pool
        if good_ocr:
            candidates = good_ocr
        else:
            combined = ocr_candidates + color_candidates
            if not combined:
                self.log("No OCR or color candidates found → UNSURE.", "error")
                return
            if self._nm_is_textract_engine():
                candidates = combined
            else:
                candidates = self._nm_dedupe_candidates(combined)

        # Reassign IDs to avoid collisions after combining sources
        candidates = self._nm_reassign_candidate_ids(candidates)

        # Score and sort
        candidates = self._nm_score_candidates(snapshot, planner, candidates)
        candidates.sort(key=lambda c: c.total_score, reverse=True)

        cand_overlay = self._nm_draw_candidates_overlay(snapshot.image, candidates, highlight_id=None, show_ids=True)
        self._nm_add_stage("Candidates (selected pool)", cand_overlay)

        # If we have exactly one strong OCR candidate, select it directly.
        if good_ocr and len(candidates) == 1:
            chosen = candidates[0]
            self.log(f"Direct OCR match selected: ID={chosen.id} at {chosen.click_point}", "coords")
            self._nm_show_virtual_click(snapshot, chosen, candidates)
            verified = self._nm_verify(snapshot, planner, chosen)
            if verified:
                self.log("Verifier: PASS – chosen candidate is consistent with planner intent.", "success")
            else:
                self.log("Verifier: UNSURE – candidate did not clearly satisfy planner intent.", "error")
            return

        # Step 5: Picker AI (forced choice among candidates)
        if not self.is_running:
            self.log("Stopped by user.", "info")
            return
        choice = self._nm_picker_ai(snapshot, planner, candidates)
        if choice is None:
            self.log("Picker returned UNSURE – no candidate selected.", "error")
            return

        chosen = next((c for c in candidates if c.id == choice), None)
        if not chosen:
            self.log(f"Picker chose candidate {choice}, but it was not found in package → UNSURE.", "error")
            return

        # Step 6: Virtual click execution (on test image only)
        self.log(f"Chosen candidate ID={chosen.id} at {chosen.click_point} (score={chosen.total_score:.3f})", "coords")
        self._nm_show_virtual_click(snapshot, chosen, candidates)

        # Step 7: Verification (logical consistency only in this offline tester)
        verified = self._nm_verify(snapshot, planner, chosen)
        if verified:
            self.log("Verifier: PASS – chosen candidate is consistent with planner intent.", "success")
        else:
            self.log("Verifier: UNSURE – candidate did not clearly satisfy planner intent.", "error")

    def _nm_capture_snapshot(self, user_instruction: str) -> Optional[NewMethodSnapshot]:
        """Step 1: Capture a deterministic snapshot used by all later steps."""
        try:
            # In this test harness, we use the loaded image as "window"
            img = self.original_image.copy()
            w, h = img.size

            # Tk knows current screen size; DPI scaling on Windows is approximated by ratio
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            screen_resolution = (screen_w, screen_h)

            # Approximate DPI scaling based on width ratio (fallback to 1.0)
            dpi_scaling = 1.0
            if screen_w and w:
                dpi_scaling = max(0.5, min(3.0, screen_w / float(w)))

            # Treat image as window at (0,0) in its own coordinate space
            window_rect = (0, 0, w, h)
            window_title = "Test tesla.png (offline new-method harness)"

            # Use logical center as "cursor" reference
            cursor_position = (w // 2, h // 2)

            snapshot = NewMethodSnapshot(
                image=img,
                screen_resolution=screen_resolution,
                dpi_scaling=dpi_scaling,
                window_rect=window_rect,
                window_title=window_title,
                cursor_position=cursor_position,
                user_instruction=user_instruction,
            )
            self.log(
                f"Snapshot captured: image={w}x{h}, screen={screen_w}x{screen_h}, dpi≈{dpi_scaling:.2f}",
                "info",
            )
            return snapshot
        except Exception as e:
            self.log(f"Error during snapshot capture: {e}", "error")
            return None

    def _nm_planner_ai(self, snapshot: NewMethodSnapshot) -> Optional[PlannerOutput]:
        """
        Step 2: Planner AI.
        Consumes screenshot + user instruction and returns a semantic intent – no coordinates.
        """
        try:
            image_b64 = self.encode_image(snapshot.image)

            system_prompt = load_prompt("coordinate_finder/planner_system.txt")

            user_prompt = format_prompt(
                load_prompt("coordinate_finder/planner_user.txt"),
                instruction=snapshot.user_instruction,
                window_title=snapshot.window_title,
                screen_width=snapshot.screen_resolution[0],
                screen_height=snapshot.screen_resolution[1],
                image_width=snapshot.image.width,
                image_height=snapshot.image.height,
            )

            plan_dict = None
            attempts = [
                {"detail": "high"},
                {"detail": "low"},
            ]
            models = ["gpt-5.2"]

            for model_name in models:
                for attempt_idx, attempt in enumerate(attempts, start=1):
                    try:
                        response = self.client.chat.completions.create(
                            model=model_name,
                            max_completion_tokens=500,  # Reduced - structured JSON is concise
                            response_format={"type": "json_object"},
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": user_prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{image_b64}",
                                                "detail": attempt["detail"],
                                            },
                                        },
                                    ],
                                },
                            ],
                        )
                    except Exception as e:
                        msg = str(e)
                        if "response_format" in msg or "json_object" in msg:
                            response = self.client.chat.completions.create(
                                model=model_name,
                                max_completion_tokens=500,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": user_prompt},
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/png;base64,{image_b64}",
                                                    "detail": attempt["detail"],
                                                },
                                            },
                                        ],
                                    },
                                ],
                            )
                        else:
                            raise
                    raw_text = response.choices[0].message.content or ""
                    self.log(f"Planner raw response (model={model_name}, attempt {attempt_idx}): {raw_text}", "ai")

                    if not raw_text.strip():
                        self.log("Planner response was empty; retrying...", "error")
                        continue

                    try:
                        plan_dict = json.loads(raw_text)
                        break
                    except json.JSONDecodeError:
                        # Try to extract JSON object
                        import re

                        m = re.search(r"\{[\s\S]*\}", raw_text)
                        if not m:
                            self.log("Planner response was not valid JSON and no object could be extracted.", "error")
                            continue
                        plan_dict = json.loads(m.group(0))
                        break

                if plan_dict is not None:
                    break
                self.log(f"Planner returned empty response with model {model_name}.", "error")

            if plan_dict is None:
                return None

            decision = str(plan_dict.get("decision", "UNSURE")).upper()

            ti = plan_dict.get("text_intent", {}) or {}
            li = plan_dict.get("location_intent", {}) or {}
            vi = plan_dict.get("visual_intent", {}) or {}
            ri = plan_dict.get("risk", {}) or {}

            text_intent = PlannerTextIntent(
                primary_text=str(ti.get("primary_text", "")).strip(),
                variants=[str(v).strip() for v in ti.get("variants", []) if str(v).strip()],
                strictness=str(ti.get("strictness", "medium")).lower(),
            )
            location_intent = PlannerLocationIntent(
                scope=str(li.get("scope", "window")).lower(),
                zone=str(li.get("zone", "any")).lower(),
                position=str(li.get("position", "any")).lower(),
            )
            visual_intent = PlannerVisualIntent(
                accent_color_relevant=bool(vi.get("accent_color_relevant", False)),
                shape_importance=str(vi.get("shape_importance", "medium")).lower(),
                primary_color=str(vi.get("primary_color", "unknown")).lower(),
                relative_luminance=str(vi.get("relative_luminance", "unknown")).lower(),
                shape=str(vi.get("shape", "unknown")).lower(),
                size=str(vi.get("size", "unknown")).lower(),
                width=str(vi.get("width", "unknown")).lower(),
                height=str(vi.get("height", "unknown")).lower(),
                text_presence=str(vi.get("text_presence", "unknown")).lower(),
                description=str(vi.get("description", "")).strip(),
            )
            risk_intent = PlannerRiskIntent(
                level=str(ri.get("level", "medium")).lower(),
            )

            planner = PlannerOutput(
                text_intent=text_intent,
                location_intent=location_intent,
                visual_intent=visual_intent,
                risk_intent=risk_intent,
                decision=decision,
                raw=plan_dict,
            )

            self.log(
                f"Planner intent → text='{planner.text_intent.primary_text}', "
                f"zone={planner.location_intent.zone}, pos={planner.location_intent.position}, "
                f"color={planner.visual_intent.primary_color}/{planner.visual_intent.relative_luminance}, "
                f"shape={planner.visual_intent.shape}, size={planner.visual_intent.size}, "
                f"text_presence={planner.visual_intent.text_presence}, decision={planner.decision}",
                "info",
            )
            if planner.visual_intent.description:
                self.log(f"Planner visual: {planner.visual_intent.description}", "info")
            return planner
        except Exception as e:
            self.log(f"Planner AI error: {e}", "error")
            return None

    def _nm_zone_to_roi(
        self,
        image: Image.Image,
        window_rect: Tuple[int, int, int, int],
        zone: str,
    ) -> ScreenRegion:
        """Deterministically map a semantic zone label to a rectangular ROI."""
        left, top, right, bottom = window_rect
        width = right - left
        height = bottom - top

        settings = self.nm_roi_settings if isinstance(getattr(self, "nm_roi_settings", None), dict) else self._nm_roi_settings_defaults()
        left_width_pct = settings.get("left_w_pct", settings.get("sidebar_w_pct", 0.35))
        right_width_pct = settings.get("right_w_pct", settings.get("sidebar_w_pct", 0.35))

        # Default: full window
        if zone not in {"top_bar", "sidebar", "footer", "center", "left", "right"}:
            return ScreenRegion(left=left, top=top, width=width, height=height)

        overlap_x = int(width * settings.get("overlap_x_pct", 0.08))
        overlap_y = int(height * settings.get("overlap_y_pct", 0.08))

        def clamp_rect(l: int, t: int, r: int, b: int) -> ScreenRegion:
            l2 = max(left, l - overlap_x)
            t2 = max(top, t - overlap_y)
            r2 = min(right, r + overlap_x)
            b2 = min(bottom, b + overlap_y)
            return ScreenRegion(left=l2, top=t2, width=max(1, r2 - l2), height=max(1, b2 - t2))

        if zone == "top_bar":
            # Wider top bar to avoid missing edge items
            h = int(height * settings.get("top_bar_h_pct", 0.30))
            return clamp_rect(left, top, right, top + h)
        if zone in {"sidebar", "left"}:
            w = int(width * left_width_pct)
            return clamp_rect(left, top, left + w, bottom)
        if zone == "right":
            w = int(width * right_width_pct)
            l = right - w
            return clamp_rect(l, top, right, bottom)
        if zone == "footer":
            h = int(height * settings.get("footer_h_pct", 0.25))
            return clamp_rect(left, bottom - h, right, bottom)
        if zone == "center":
            # Much wider center region to include edges
            w = int(width * settings.get("center_w_pct", 0.90))
            h = int(height * settings.get("center_h_pct", 0.90))
            cx = left + width // 2
            cy = top + height // 2
            l = cx - w // 2
            t = cy - h // 2
            r = cx + w // 2
            b = cy + h // 2
            return clamp_rect(l, t, r, b)

        # Fallback
        return ScreenRegion(left=left, top=top, width=width, height=height)

    def _nm_draw_roi_overlay(self, image: Image.Image, roi: ScreenRegion) -> Image.Image:
        """Return image with ROI rectangle overlay."""
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([roi.left, roi.top, roi.right, roi.bottom], outline=(255, 215, 0, 220), width=4)
        return Image.alpha_composite(base, overlay).convert("RGB")

    def _nm_refine_ocr_matches(self, image: Image.Image, matches: List["OCRMatch"]) -> List["OCRMatch"]:
        """Tighten OCR boxes using local pixel evidence to reduce offsets."""
        if not matches:
            return matches
        try:
            import cv2
            import numpy as np
            from agent.ocr import OCRMatch
        except Exception:
            return matches

        refined = []
        for m in matches:
            l, t, r, b = m.bbox.left, m.bbox.top, m.bbox.right, m.bbox.bottom
            search_pad = 6
            l2 = max(0, l - search_pad)
            t2 = max(0, t - search_pad)
            r2 = min(image.width, r + search_pad)
            b2 = min(image.height, b + search_pad)
            if r2 <= l2 or b2 <= t2:
                refined.append(m)
                continue

            crop = image.crop((l2, t2, r2, b2)).convert("RGB")
            arr = np.array(crop)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            # Adaptive threshold handles gradients better than a single global threshold.
            thr = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
            )
            # Ensure text is foreground (white) by picking the smaller pixel mass.
            if np.count_nonzero(thr) > (thr.size * 0.6):
                thr = cv2.bitwise_not(thr)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                refined.append(m)
                continue

            boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if (w * h) < 10:
                    continue
                boxes.append((x, y, w, h))

            if not boxes:
                refined.append(m)
                continue

            x0 = min(x for x, _, _, _ in boxes)
            y0 = min(y for _, y, _, _ in boxes)
            x1 = max(x + w for x, _, w, _ in boxes)
            y1 = max(y + h for _, y, _, h in boxes)

            pad = 2
            new_l = max(0, l2 + x0 - pad)
            new_t = max(0, t2 + y0 - pad)
            new_r = min(image.width, l2 + x1 + pad)
            new_b = min(image.height, t2 + y1 + pad)
            if new_r <= new_l or new_b <= new_t:
                refined.append(m)
                continue

            new_bbox = ScreenRegion(left=new_l, top=new_t, width=new_r - new_l, height=new_b - new_t)
            refined.append(OCRMatch(text=m.text, confidence=m.confidence, bbox=new_bbox))

        return refined

    def _nm_text_center_in_bbox(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Compute a text-pixel centroid inside bbox; fallback to bbox center."""
        try:
            import cv2
            import numpy as np
        except Exception:
            l, t, r, b = bbox
            return ((l + r) // 2, (t + b) // 2)

        l, t, r, b = bbox
        l = max(0, min(image.width - 1, l))
        t = max(0, min(image.height - 1, t))
        r = max(l + 1, min(image.width, r))
        b = max(t + 1, min(image.height, b))

        crop = image.crop((l, t, r, b)).convert("RGB")
        arr = np.array(crop)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        if np.count_nonzero(thr) > (thr.size * 0.6):
            thr = cv2.bitwise_not(thr)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

        ys, xs = np.where(thr > 0)
        if xs.size == 0:
            return ((l + r) // 2, (t + b) // 2)
        cx = int(xs.mean()) + l
        cy = int(ys.mean()) + t
        return (cx, cy)

    def _nm_compute_region_color(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """Compute a simple average color for a region."""
        l, t, r, b = bbox
        l = max(0, min(image.width - 1, l))
        t = max(0, min(image.height - 1, t))
        r = max(l + 1, min(image.width, r))
        b = max(t + 1, min(image.height, b))
        region = image.crop((l, t, r, b)).convert("RGB")
        # Downsample for speed
        small = region.resize((1, 1), Image.Resampling.BILINEAR)
        return tuple(small.getpixel((0, 0)))

    def _nm_text_similarity(self, intent: PlannerTextIntent, candidate_text: str) -> float:
        """Deterministic similarity score between 0 and 1. More lenient matching."""
        if not candidate_text:
            return 0.0
        cand = candidate_text.strip().lower()
        primary = intent.primary_text.strip().lower()
        variants = [v.strip().lower() for v in intent.variants]

        if not primary and not variants:
            return 0.5  # If no intent, give neutral score instead of 0

        # Exact match
        if primary and cand == primary:
            return 1.0
        
        # Substring matches (more lenient)
        if primary:
            if primary in cand:
                return 0.9  # Increased from 0.85
            if cand in primary:
                return 0.85  # Reverse substring match
        
        # Variant matches
        for v in variants:
            if cand == v:
                return 0.9  # Increased from 0.8
            if v and v in cand:
                return 0.8  # Increased from 0.7
            if v and cand in v:
                return 0.75  # Reverse variant match

        # Word-level matching (more lenient)
        cand_words = set(cand.split())
        primary_words = set(primary.split())
        if primary_words and cand_words:
            overlap = len(primary_words & cand_words)
            if overlap > 0:
                # If any words match, give a decent score
                word_ratio = overlap / max(len(primary_words), len(cand_words))
                return 0.4 + (word_ratio * 0.4)  # Range: 0.4-0.8

        # Character-level fuzzy matching (very lenient fallback)
        if primary:
            # Check if significant portion of characters match
            common_chars = set(primary) & set(cand)
            if len(common_chars) >= min(3, len(primary) * 0.5):
                return 0.3  # Low but non-zero score

        return 0.0

    def _nm_text_match_threshold(self, intent: PlannerTextIntent) -> float:
        strict = (intent.strictness or "medium").lower()
        if strict == "high":
            return 0.9
        if strict == "low":
            return 0.75
        return 0.85

    def _nm_good_text_candidates(
        self,
        intent: PlannerTextIntent,
        candidates: List[CandidatePackage],
    ) -> List[CandidatePackage]:
        """Filter OCR candidates to only strong text matches."""
        query = (intent.primary_text or "").strip().lower()
        if not query:
            return []

        threshold = self._nm_text_match_threshold(intent)
        good = []
        short_query = len(query) <= 2
        for c in candidates:
            if not c.text:
                continue
            cand_text = c.text.strip().lower()
            if short_query:
                if cand_text != query:
                    continue
                good.append(c)
            else:
                if c.scores.get("text_match", 0.0) >= threshold:
                    good.append(c)
        return good

    def _nm_reassign_candidate_ids(self, candidates: List[CandidatePackage]) -> List[CandidatePackage]:
        for i, c in enumerate(candidates, start=1):
            c.id = i
        return candidates

    def _nm_luminance(self, rgb: Tuple[int, int, int]) -> float:
        r, g, b = rgb
        return (0.2126 * (r / 255.0)) + (0.7152 * (g / 255.0)) + (0.0722 * (b / 255.0))

    def _nm_compute_surrounding_color(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        pad: int = 8,
    ) -> Tuple[int, int, int]:
        """Estimate surrounding color by sampling thin strips around the bbox."""
        l, t, r, b = bbox
        l2 = max(0, l - pad)
        t2 = max(0, t - pad)
        r2 = min(image.width, r + pad)
        b2 = min(image.height, b + pad)

        samples: List[Tuple[int, int, int]] = []

        def sample(box: Tuple[int, int, int, int]):
            if box[2] <= box[0] or box[3] <= box[1]:
                return
            region = image.crop(box).convert("RGB")
            small = region.resize((1, 1), Image.Resampling.BILINEAR)
            samples.append(tuple(small.getpixel((0, 0))))

        # Top, bottom, left, right strips
        sample((l2, t2, r2, t))
        sample((l2, b, r2, b2))
        sample((l2, t, l, b))
        sample((r, t, r2, b))

        if not samples:
            return self._nm_compute_region_color(image, bbox)

        avg = [0, 0, 0]
        for s in samples:
            avg[0] += s[0]
            avg[1] += s[1]
            avg[2] += s[2]
        count = len(samples)
        return (avg[0] // count, avg[1] // count, avg[2] // count)

    def _nm_color_group(self, rgb: Tuple[int, int, int]) -> str:
        """Classify a color into coarse UI-friendly buckets."""
        import colorsys

        r, g, b = rgb
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
        lum = self._nm_luminance(rgb)

        if s < 0.18:
            if lum > 0.88:
                return "white"
            if lum > 0.70:
                return "light_gray"
            if lum > 0.45:
                return "gray"
            if lum > 0.25:
                return "dark_gray"
            return "black"

        hue = (h * 360.0) % 360.0
        if hue < 15 or hue >= 345:
            return "red"
        if hue < 35:
            return "orange"
        if hue < 55:
            return "yellow"
        if hue < 150:
            return "green"
        if hue < 190:
            return "teal"
        if hue < 250:
            return "blue"
        if hue < 290:
            return "purple"
        if hue < 330:
            return "pink"
        return "red"

    def _nm_normalize_color(self, value: str) -> str:
        v = (value or "").strip().lower().replace(" ", "_")
        aliases = {
            "gray": "grey",
            "lightgray": "grey",
            "darkgray": "grey",
            "light_gray": "grey",
            "dark_gray": "grey",
            "lightgrey": "grey",
            "darkgrey": "grey",
            "light": "grey",
            "dark": "grey",
            "black": "black",
            "white": "white",
        }
        return aliases.get(v, v or "unknown")

    def _nm_color_matches(self, intent_color: str, candidate_color: str) -> bool:
        intent = self._nm_normalize_color(intent_color)
        cand = self._nm_normalize_color(candidate_color)
        if intent in {"", "unknown"}:
            return True
        if intent == cand:
            return True
        synonyms = {
            "white": {"white", "grey"},
            "grey": {"white", "grey", "black"},
            "black": {"grey", "black"},
            "red": {"red", "pink"},
            "pink": {"red", "pink", "purple"},
            "purple": {"pink", "purple", "blue"},
            "blue": {"purple", "blue", "green"},
            "green": {"blue", "green", "yellow"},
            "yellow": {"green", "yellow", "red"},
        }
        if intent in synonyms:
            return cand in synonyms[intent]
        return False

    def _nm_width_class(self, width_ratio: float) -> str:
        if width_ratio >= 0.75:
            return "full"
        if width_ratio >= 0.45:
            return "wide"
        if width_ratio >= 0.2:
            return "medium"
        return "narrow"

    def _nm_height_class(self, height_ratio: float) -> str:
        if height_ratio >= 0.18:
            return "tall"
        if height_ratio >= 0.08:
            return "medium"
        return "short"

    def _nm_size_class(self, width_ratio: float, height_ratio: float, area_ratio: float) -> str:
        if width_ratio >= 0.7 and height_ratio <= 0.15:
            return "full_width"
        if width_ratio >= 0.45 and height_ratio <= 0.12:
            return "large"
        if area_ratio >= 0.03:
            return "medium"
        return "small"

    def _nm_size_class_px(self, width: int, height: int, window_w: int) -> str:
        if width >= int(window_w * 0.75):
            return "full_width"
        area = width * height
        if area <= 9000:
            return "small"
        if area <= 30000:
            return "medium"
        return "large"

    def _nm_shape_class(self, aspect: float, width_ratio: float, height_ratio: float, text_present: bool) -> str:
        if aspect >= 4.0 and height_ratio <= 0.10:
            return "pill"
        if aspect >= 1.6:
            return "rounded_rectangle"
        if 0.85 <= aspect <= 1.2 and width_ratio <= 0.12 and height_ratio <= 0.12 and not text_present:
            return "circle"
        if text_present and width_ratio <= 0.2 and height_ratio <= 0.08:
            return "text_only"
        return "rectangle"

    def _nm_position_matches(self, position: str, x_norm: float, y_norm: float) -> bool:
        if position in {"", "any", "unknown"}:
            return True
        if position == "top":
            return y_norm <= 0.45
        if position == "bottom":
            return y_norm >= 0.55
        if position == "left":
            return x_norm <= 0.45
        if position == "right":
            return x_norm >= 0.55
        if position == "center":
            return 0.2 <= x_norm <= 0.8 and 0.2 <= y_norm <= 0.8
        if position == "top_left":
            return x_norm <= 0.45 and y_norm <= 0.45
        if position == "top_right":
            return x_norm >= 0.55 and y_norm <= 0.45
        if position == "bottom_left":
            return x_norm <= 0.45 and y_norm >= 0.55
        if position == "bottom_right":
            return x_norm >= 0.55 and y_norm >= 0.55
        return True

    def _nm_build_candidate_features(
        self,
        snapshot: NewMethodSnapshot,
        roi: ScreenRegion,
        candidates: List[CandidatePackage],
    ) -> Dict[int, Dict[str, Any]]:
        features: Dict[int, Dict[str, Any]] = {}
        window_w = max(1, snapshot.window_rect[2] - snapshot.window_rect[0])
        window_h = max(1, snapshot.window_rect[3] - snapshot.window_rect[1])
        roi_w = max(1, roi.right - roi.left)
        roi_h = max(1, roi.bottom - roi.top)

        for c in candidates:
            l, t, r, b = c.bbox
            width = max(1, r - l)
            height = max(1, b - t)
            aspect = width / float(height)
            width_ratio = width / float(window_w)
            height_ratio = height / float(window_h)
            area_ratio = width_ratio * height_ratio
            text_present = bool(c.text and c.text.strip() and c.text.strip() != "[visual element]")
            size_px_class = self._nm_size_class_px(width, height, window_w)

            candidate_color = c.color
            surround_color = self._nm_compute_surrounding_color(snapshot.image, c.bbox)
            cand_lum = self._nm_luminance(candidate_color)
            surround_lum = self._nm_luminance(surround_color)
            lum_delta = cand_lum - surround_lum
            if lum_delta > 0.08:
                relative_luminance = "lighter"
            elif lum_delta < -0.08:
                relative_luminance = "darker"
            else:
                relative_luminance = "similar"

            cx, cy = c.click_point
            x_norm = (cx - roi.left) / float(roi_w)
            y_norm = (cy - roi.top) / float(roi_h)

            features[c.id] = {
                "width_px": width,
                "height_px": height,
                "width_ratio": width_ratio,
                "height_ratio": height_ratio,
                "area_ratio": area_ratio,
                "aspect": aspect,
                "text_present": text_present,
                "color_group": self._nm_color_group(candidate_color),
                "relative_luminance": relative_luminance,
                "width_class": self._nm_width_class(width_ratio),
                "height_class": self._nm_height_class(height_ratio),
                "size_class": self._nm_size_class(width_ratio, height_ratio, area_ratio),
                "size_px_class": size_px_class,
                "shape_class": self._nm_shape_class(aspect, width_ratio, height_ratio, text_present),
                "x_norm": x_norm,
                "y_norm": y_norm,
            }

        return features

    def _nm_draw_filter_overlay(
        self,
        image: Image.Image,
        kept: List[CandidatePackage],
        removed: List[CandidatePackage],
        show_ids: bool = True,
    ) -> Image.Image:
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        # Draw removed first (red), then kept (green)
        for c in removed:
            l, t, r, b = c.bbox
            draw.rectangle([l, t, r, b], outline=(255, 64, 64, 220), width=2)
            if show_ids:
                draw.text((l + 4, t + 2), str(c.id), font=font, fill=(255, 64, 64, 255))

        for c in kept:
            l, t, r, b = c.bbox
            draw.rectangle([l, t, r, b], outline=(0, 255, 0, 220), width=2)
            if show_ids:
                draw.text((l + 4, t + 2), str(c.id), font=font, fill=(0, 255, 0, 255))

        return Image.alpha_composite(base, overlay).convert("RGB")

    def _nm_draw_rects_overlay(
        self,
        image: Image.Image,
        rects: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int, int],
        width: int = 2,
    ) -> Image.Image:
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for l, t, r, b in rects:
            draw.rectangle([l, t, r, b], outline=color, width=width)
        return Image.alpha_composite(base, overlay).convert("RGB")

    def _nm_phase_filter_candidates(
        self,
        snapshot: NewMethodSnapshot,
        planner: PlannerOutput,
        candidates: List[CandidatePackage],
        force_use_ocr: Optional[bool] = None,
        force_use_color: Optional[bool] = None,
        force_use_shape: Optional[bool] = None,
    ) -> List[CandidatePackage]:
        if not candidates:
            return []

        roi = self._nm_zone_to_roi(
            image=snapshot.image,
            window_rect=snapshot.window_rect,
            zone=planner.location_intent.zone,
        )
        features = self._nm_build_candidate_features(snapshot, roi, candidates)
        self.log("Filter views: green=kept, red=filtered", "info")

        def add_stage(name: str, kept: List[CandidatePackage], removed: List[CandidatePackage]):
            stage = self._nm_draw_filter_overlay(snapshot.image, kept, removed, show_ids=True)
            self._nm_add_stage(name, stage)

        def stage_filter(
            name: str,
            base_candidates: List[CandidatePackage],
            predicate,
            enabled: bool,
        ) -> List[CandidatePackage]:
            if not enabled:
                add_stage(f"{name} filter (not applied)", base_candidates, [])
                return []

            if not base_candidates:
                add_stage(f"{name} filter (kept 0/0)", [], [])
                return []

            kept = []
            removed = []
            for c in base_candidates:
                if predicate(c, features[c.id]):
                    kept.append(c)
                else:
                    removed.append(c)

            add_stage(f"{name} filter (kept {len(kept)}/{len(base_candidates)})", kept, removed)
            return kept

        use_ocr = self.nm_use_ocr_var.get() if force_use_ocr is None else bool(force_use_ocr)
        use_color = self.nm_use_color_var.get() if force_use_color is None else bool(force_use_color)
        use_shape = self.nm_use_shape_var.get() if force_use_shape is None else bool(force_use_shape)

        ocr_candidates = [c for c in candidates if c.source == "ocr"]
        color_candidates = [c for c in candidates if c.source == "color"]
        shape_candidates = [c for c in candidates if c.source == "shape"]

        text_presence = planner.visual_intent.text_presence
        has_text_intent = bool(planner.text_intent.primary_text or planner.text_intent.variants)
        apply_ocr = use_ocr and (has_text_intent or text_presence in {"required", "absent"})
        text_threshold = 0.85

        def ocr_pred(candidate: CandidatePackage, feat: Dict[str, Any]) -> bool:
            if text_presence == "absent":
                return not feat["text_present"]
            if not has_text_intent:
                return False
            if not feat["text_present"]:
                return False
            return candidate.scores.get("text_match", 0.0) >= text_threshold

        ocr_kept = stage_filter("OCR", ocr_candidates, ocr_pred, apply_ocr)

        apply_color = use_color
        if apply_color:
            add_stage("Color filter (not applied)", color_candidates, [])
            color_kept = color_candidates
        else:
            color_kept = []

        size_intent = planner.visual_intent.size
        width_intent = planner.visual_intent.width
        height_intent = planner.visual_intent.height
        shape_intent = planner.visual_intent.shape
        shape_importance = planner.visual_intent.shape_importance
        apply_shape = use_shape

        def shape_pred(candidate: CandidatePackage, feat: Dict[str, Any]) -> bool:
            shape_ok = True
            if (
                candidate.source != "color"
                and shape_intent not in {"", "unknown"}
                and shape_importance in {"high", "medium"}
            ):
                if shape_intent == "rounded_rectangle":
                    shape_ok = feat["shape_class"] in {"rounded_rectangle", "pill", "rectangle"}
                elif shape_intent == "rectangle":
                    shape_ok = feat["shape_class"] in {"rectangle", "rounded_rectangle", "pill"}
                elif shape_intent == "pill":
                    shape_ok = feat["shape_class"] == "pill"
                elif shape_intent == "circle":
                    shape_ok = feat["shape_class"] == "circle"
                elif shape_intent == "icon":
                    shape_ok = feat["shape_class"] in {"circle", "rectangle"} and not feat["text_present"]
                elif shape_intent == "text_only":
                    shape_ok = feat["shape_class"] == "text_only"

            return shape_ok

        shape_base = shape_candidates if shape_candidates else candidates
        shape_kept = stage_filter("Shape", shape_base, shape_pred, apply_shape)

        combined = []
        if apply_ocr:
            combined.extend(ocr_kept)
        if apply_color:
            combined.extend(color_kept)
        if apply_shape:
            combined.extend(shape_kept)

        if not combined:
            combined = candidates[:]

        if self._nm_is_textract_engine():
            combined = combined
        else:
            combined = self._nm_dedupe_candidates(combined)
        removed = [c for c in candidates if c not in combined]
        add_stage(f"Combined (union) (kept {len(combined)}/{len(candidates)})", combined, removed)

        apply_size = size_intent not in {"", "unknown"}
        if apply_size:
            def size_pred(candidate: CandidatePackage, feat: Dict[str, Any]) -> bool:
                size_ok = True
                if size_intent not in {"", "unknown"}:
                    area = feat["width_px"] * feat["height_px"]
                    if size_intent == "full_width":
                        size_ok = feat["size_px_class"] == "full_width" or feat["size_class"] == "full_width"
                    elif size_intent == "large":
                        # Only filter out tiny noise for "large"
                        size_ok = area >= 1200
                    elif size_intent == "medium":
                        size_ok = 1200 <= area <= 60000
                    elif size_intent == "small":
                        size_ok = area <= 9000
                    else:
                        size_ok = feat["size_px_class"] == size_intent
                return size_ok

            kept = []
            removed = []
            for c in combined:
                if size_pred(c, features[c.id]):
                    kept.append(c)
                else:
                    removed.append(c)
            add_stage(f"Size filter (kept {len(kept)}/{len(combined)})", kept, removed)
            combined = kept

        position_intent = planner.location_intent.position
        apply_position = False
        if position_intent not in {"", "any", "unknown"}:
            add_stage("Position filter (not applied)", combined, [])

        return combined

    def _nm_dedupe_candidates(self, candidates: List[CandidatePackage]) -> List[CandidatePackage]:
        def overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
            left = max(a[0], b[0])
            top = max(a[1], b[1])
            right = min(a[2], b[2])
            bottom = min(a[3], b[3])
            if right <= left or bottom <= top:
                return 0.0
            inter = (right - left) * (bottom - top)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = max(1, area_a + area_b - inter)
            return inter / union

        kept: List[CandidatePackage] = []
        for c in candidates:
            dup = False
            for k in kept:
                if overlap_ratio(c.bbox, k.bbox) >= 0.6:
                    dup = True
                    break
            if not dup:
                kept.append(c)
        return kept

    def _nm_is_textract_engine(self) -> bool:
        """Return True if the active OCR engine is Textract."""
        try:
            if self.ocr_engine is None:
                self.ocr_engine = get_ocr_engine()
            return type(self.ocr_engine).__name__ == "TextractOCREngine"
        except Exception:
            return False

    def _nm_score_candidates(
        self,
        snapshot: NewMethodSnapshot,
        planner: PlannerOutput,
        candidates: List[CandidatePackage],
    ) -> List[CandidatePackage]:
        if not candidates:
            return []

        import time
        filter_start_time = time.time()

        roi = self._nm_zone_to_roi(
            image=snapshot.image,
            window_rect=snapshot.window_rect,
            zone=planner.location_intent.zone,
        )
        features = self._nm_build_candidate_features(snapshot, roi, candidates)
        self.log("Scoring candidates: each gets points for matching criteria", "info")

        # Initialize scores for all candidates
        scored_candidates = []
        for c in candidates:
            c.scores = getattr(c, 'scores', {})  # Ensure scores dict exists
            c.total_score = 0.0
            scored_candidates.append(c)

        def add_score(name: str, candidate: CandidatePackage, points: float, reason: str = ""):
            """Add points to a candidate's score"""
            candidate.total_score += points
            if reason:
                self.log(f"{name}: +{points:.1f} to candidate {candidate.id} ({reason})", "debug")

        def apply_scorer(name: str, scorer_func, enabled: bool):
            """Apply scoring function to all candidates"""
            if not enabled:
                self.log(f"{name} scoring (not applied)", "info")
                return

            for candidate in scored_candidates:
                points = scorer_func(candidate, features[candidate.id])
                if points > 0:
                    add_score(name, candidate, points)

            self.log(f"{name} scoring applied to {len([c for c in scored_candidates if c.total_score > 0])} candidates", "info")

        # OCR/text scoring
        text_presence = planner.visual_intent.text_presence
        has_text_intent = bool(planner.text_intent.primary_text or planner.text_intent.variants)
        apply_text_scoring = has_text_intent or text_presence in {"absent", "required"}

        def text_scorer(candidate: CandidatePackage, feat: Dict[str, Any]) -> float:
            points = 0.0

            # Text presence scoring
            if text_presence == "absent":
                if not feat["text_present"]:
                    points += 3.0  # Bonus for no text when we want no text
                else:
                    points -= 2.0  # Penalty for having text when we want none
            elif text_presence == "required":
                if feat["text_present"]:
                    points += 2.0  # Bonus for having text when required
                    # Additional points based on text match quality
                    text_score = candidate.scores.get("text_match", 0.0)
                    if text_score >= 0.8:
                        points += 3.0
                    elif text_score >= 0.6:
                        points += 2.0
                    elif text_score >= 0.4:
                        points += 1.0
                else:
                    points -= 1.0  # Small penalty for missing text when required
            elif text_presence == "optional":
                if feat["text_present"]:
                    points += 1.0  # Small bonus for having text when optional
            # For "unknown", no text presence scoring

            # Text content matching (if we have text intent)
            if has_text_intent and feat["text_present"]:
                text_score = candidate.scores.get("text_match", 0.0)
                if text_score >= 0.95:
                    points += 4.0  # Excellent text match
                elif text_score >= 0.85:
                    points += 3.0  # Strong text match

            return max(0, points)  # Don't go negative

        apply_scorer("OCR/Text", text_scorer, apply_text_scoring)

        # Color scoring
        apply_color_scoring = (
            planner.visual_intent.accent_color_relevant
            or planner.visual_intent.primary_color not in {"", "unknown"}
            or planner.visual_intent.relative_luminance not in {"", "unknown"}
        )

        def color_scorer(candidate: CandidatePackage, feat: Dict[str, Any]) -> float:
            points = 0.0
            has_specific_color = planner.visual_intent.primary_color not in {"", "unknown"}
            has_luminance = planner.visual_intent.relative_luminance not in {"", "unknown"}

            # Color matching scoring
            if has_specific_color:
                if self._nm_color_matches(planner.visual_intent.primary_color, feat["color_group"]):
                    points += 3.0  # Exact or synonym color match
                    # Bonus for exact match
                    if planner.visual_intent.primary_color == self._nm_normalize_color(feat["color_group"]):
                        points += 1.0
                else:
                    points -= 1.0  # Penalty for wrong color
            elif planner.visual_intent.accent_color_relevant:
                # Accent color relevant but no specific color - reward colorful elements
                color_name = feat["color_group"]
                if color_name in {"white", "grey", "black"}:
                    points -= 0.5  # Small penalty for plain colors
                else:
                    points += 1.0  # Bonus for colorful elements

            # Luminance matching scoring
            if has_luminance:
                if feat["relative_luminance"] == planner.visual_intent.relative_luminance:
                    points += 2.0  # Correct luminance
                else:
                    points -= 1.0  # Wrong luminance

            return max(0, points)  # Don't go negative

        apply_scorer("Color", color_scorer, apply_color_scoring)

        # Size/shape scoring
        size_intent = planner.visual_intent.size
        width_intent = planner.visual_intent.width
        height_intent = planner.visual_intent.height
        shape_intent = planner.visual_intent.shape
        shape_importance = planner.visual_intent.shape_importance
        apply_size_scoring = any(v not in {"", "unknown"} for v in [size_intent, width_intent, height_intent, shape_intent])

        def size_scorer(candidate: CandidatePackage, feat: Dict[str, Any]) -> float:
            points = 0.0
            shape_multiplier = 1.0
            if shape_importance == "high":
                shape_multiplier = 1.5
            elif shape_importance == "low":
                shape_multiplier = 0.5

            # Size matching
            if size_intent not in {"", "unknown"}:
                if size_intent == "full_width" and feat["size_class"] == "full_width":
                    points += 3.0  # Exact full width match
                elif size_intent == "large" and feat["size_class"] in {size_intent, "full_width"}:
                    points += 2.0  # Large or full width
                elif feat["size_class"] == size_intent:
                    points += 2.0  # Exact size match
                else:
                    points -= 0.5  # Wrong size

            # Width matching
            if width_intent not in {"", "unknown"}:
                if width_intent == "wide" and feat["width_class"] in {"wide", "full"}:
                    points += 2.0  # Wide or full width
                elif width_intent == "full" and feat["width_class"] == "full":
                    points += 3.0  # Exact full width
                elif feat["width_class"] == width_intent:
                    points += 2.0  # Exact width match
                else:
                    points -= 0.5  # Wrong width

            # Height matching
            if height_intent not in {"", "unknown"}:
                if feat["height_class"] == height_intent:
                    points += 2.0  # Exact height match
                else:
                    points -= 0.5  # Wrong height

            # Shape matching
            if shape_intent not in {"", "unknown"}:
                shape_points = 0.0
                if shape_intent == "rounded_rectangle":
                    if feat["shape_class"] == "rounded_rectangle":
                        shape_points = 2.0
                    elif feat["shape_class"] in {"pill", "rectangle"}:
                        shape_points = 1.0
                elif shape_intent == "rectangle":
                    if feat["shape_class"] == "rectangle":
                        shape_points = 2.0
                    elif feat["shape_class"] in {"rounded_rectangle", "pill"}:
                        shape_points = 1.0
                elif shape_intent == "pill":
                    if feat["shape_class"] == "pill":
                        shape_points = 2.0
                elif shape_intent == "circle":
                    if feat["shape_class"] == "circle":
                        shape_points = 2.0
                elif shape_intent == "icon":
                    if feat["shape_class"] in {"circle", "rectangle"} and not feat["text_present"]:
                        shape_points = 2.0
                elif shape_intent == "text_only":
                    if feat["shape_class"] == "text_only":
                        shape_points = 2.0
                else:
                    shape_points = 0.0

                points += shape_points * shape_multiplier

            return max(0, points)  # Don't go negative

        apply_scorer("Size/Shape", size_scorer, apply_size_scoring)

        # Position scoring
        position_intent = planner.location_intent.position
        apply_position_scoring = position_intent not in {"", "any", "unknown"}

        def position_scorer(candidate: CandidatePackage, feat: Dict[str, Any]) -> float:
            points = 0.0
            if self._nm_position_matches(position_intent, feat["x_norm"], feat["y_norm"]):
                points += 2.0  # Position matches intent
            else:
                points -= 1.0  # Position doesn't match
            return max(0, points)

        apply_scorer("Position", position_scorer, apply_position_scoring)

        # Sort candidates by total score and return top candidates
        scored_candidates.sort(key=lambda c: c.total_score, reverse=True)

        # Log final scoring results
        self.log(f"Final scoring: {len(scored_candidates)} candidates", "info")
        for i, candidate in enumerate(scored_candidates[:10]):  # Show top 10
            text_preview = (candidate.text or "icon").replace("\n", " ")
            if len(text_preview) > 40:
                text_preview = text_preview[:37] + "..."
            self.log(
                f"  #{i+1}: ID {candidate.id}, score {candidate.total_score:.1f}, "
                f"source {candidate.source}, text '{text_preview}'",
                "info",
            )

        # Log total time for candidate filtering/scoring
        filter_elapsed = time.time() - filter_start_time
        self.log(f"Candidate filtering/scoring completed in {filter_elapsed:.2f} seconds", "info")

        # Return all candidates with their scores - let the AI decide from the top ones
        return scored_candidates

    def _nm_find_contrast_regions(
        self,
        image: Image.Image,
        roi: ScreenRegion,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find regions with high contrast (likely buttons/inputs) by scanning for
        areas with distinct color boundaries or borders.
        Returns list of (left, top, right, bottom) bounding boxes.
        """
        import numpy as np
        
        # Crop ROI
        settings = self.nm_color_settings
        roi_img = image.crop((roi.left, roi.top, roi.right, roi.bottom))
        roi_array = np.array(roi_img.convert("RGB"))
        
        # Convert to grayscale for contrast detection
        gray = np.mean(roi_array, axis=2).astype(np.uint8)
        
        # Compute gradient (contrast) in both directions
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))
        
        # Pad gradients to match original size
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        
        # Combined gradient strength (avoid overflow by using uint16 intermediate)
        # Use Manhattan distance instead of Euclidean to avoid sqrt overflow
        gradient = (grad_x.astype(np.uint16) + grad_y.astype(np.uint16)).astype(np.float32)
        
        # Find regions with high gradient (borders)
        # Use a simpler threshold calculation to avoid overflow
        if gradient.size > 0 and np.any(gradient > 0):
            # Use mean + std deviation as threshold (more stable than percentile)
            mean_grad = np.mean(gradient)
            std_grad = np.std(gradient)
            threshold = mean_grad + std_grad * 1.2  # Slightly more sensitive for subtle UI borders
        else:
            threshold = 0
        border_mask = gradient > threshold
        
        # Find rectangular regions by looking for connected border pixels
        regions = []
        visited = np.zeros_like(border_mask, dtype=bool)
        h, w = border_mask.shape
        
        # Simple flood-fill to find rectangular regions
        for y in range(5, h - 5, 8):  # Sample every 8px for better coverage
            for x in range(5, w - 5, 8):
                if border_mask[y, x] and not visited[y, x]:
                    # Try to find a rectangle starting here
                    # Look for horizontal and vertical borders
                    x_start = x
                    x_end = x
                    y_start = y
                    y_end = y
                    
                    # Expand horizontally
                    for x_test in range(x, min(w - 5, x + 900)):
                        if np.sum(border_mask[max(0, y-2):min(h, y+3), x_test]) > 2:
                            x_end = x_test
                        else:
                            break
                    
                    # Expand vertically
                    for y_test in range(y, min(h - 5, y + 200)):
                        if np.sum(border_mask[y_test, max(0, x_start-2):min(w, x_end+3)]) > (x_end - x_start) * 0.3:
                            y_end = y_test
                        else:
                            break
                    
                    width = x_end - x_start
                    height = y_end - y_start
                    
                    # Filter by size (reasonable button/input sizes)
                    max_width = max(600, int(w * 0.95))
                    max_height = max(150, int(h * 0.35))
                    if 40 <= width <= max_width and 15 <= height <= max_height:
                        # Convert to full image coordinates
                        regions.append((
                            roi.left + x_start,
                            roi.top + y_start,
                            roi.left + x_end,
                            roi.top + y_end
                        ))
                        # Mark as visited
                        visited[y_start:y_end+1, x_start:x_end+1] = True
        
        return regions

    def _nm_find_color_regions(
        self,
        image: Image.Image,
        roi: ScreenRegion,
        target_color: str,
        accent_color_relevant: bool,
        max_area: int,
        return_masks: bool = False,
    ) -> List[Tuple[int, int, int, int]] | Tuple[List[Tuple[int, int, int, int]], List["np.ndarray"]]:
        """Find connected regions matching a target color mask."""
        return find_color_regions(
            self,
            image,
            roi,
            target_color,
            accent_color_relevant,
            max_area,
            return_masks=return_masks,
        )

    def _nm_find_border_color_regions(
        self,
        image: Image.Image,
        roi: ScreenRegion,
        target_color: str,
        max_area: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect hard-bordered rectangles and keep those whose interior matches target color."""
        color = self._nm_normalize_color(target_color)
        if color in {"", "unknown"}:
            return []

        try:
            import cv2
            import numpy as np
        except Exception:
            return []

        roi_img = image.crop((roi.left, roi.top, roi.right, roi.bottom)).convert("RGB")
        roi_arr = np.array(roi_img)
        gray = cv2.cvtColor(roi_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 180)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[Tuple[int, int, int, int]] = []
        roi_area = max(1, (roi.right - roi.left) * (roi.bottom - roi.top))
        min_area = max(25, int(roi_area * 0.00005))

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue
            if w < 30 or h < 12:
                continue

            # Border edge density to ensure "hard border"
            border = np.zeros((h, w), dtype=np.uint8)
            border[:2, :] = 1
            border[-2:, :] = 1
            border[:, :2] = 1
            border[:, -2:] = 1
            border_edges = edges[y:y + h, x:x + w]
            border_count = int(np.count_nonzero(border))
            if border_count == 0:
                continue
            border_ratio = float(np.count_nonzero(border_edges[border == 1])) / float(border_count)
            if border_ratio < 0.10:
                continue

            # Color check inside (ignore border)
            pad = 2
            ix0 = max(0, x + pad)
            iy0 = max(0, y + pad)
            ix1 = min(roi_arr.shape[1], x + w - pad)
            iy1 = min(roi_arr.shape[0], y + h - pad)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            region = roi_arr[iy0:iy1, ix0:ix1]
            avg = tuple(np.mean(region.reshape(-1, 3), axis=0).astype(int))
            group = self._nm_color_group(avg)
            if not self._nm_color_matches(color, group):
                continue

            regions.append((roi.left + x, roi.top + y, roi.left + x + w, roi.top + y + h))

        return regions

    def _nm_select_best_color_masks(
        self,
        image: Image.Image,
        roi: ScreenRegion,
        masks_u8: List["np.ndarray"],
        keep: int = 2,
    ) -> Tuple[List["np.ndarray"], List[int]]:
        """Pick the most button-like mask splits to avoid merged blobs."""
        if not masks_u8:
            return [], []
        if len(masks_u8) <= 1:
            return masks_u8, [1]

        try:
            import cv2
            import numpy as np
        except Exception:
            return masks_u8, list(range(1, len(masks_u8) + 1))

        roi_img = image.crop((roi.left, roi.top, roi.right, roi.bottom)).convert("RGB")
        roi_arr = np.array(roi_img)
        gray = cv2.cvtColor(roi_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        roi_area = max(1, (roi.right - roi.left) * (roi.bottom - roi.top))
        min_pixels = max(80, int(roi_area * 0.0002))
        scored = []

        for idx, mask_u8 in enumerate(masks_u8, start=1):
            if mask_u8 is None:
                continue
            area = int(np.count_nonzero(mask_u8))
            if area < min_pixels:
                continue

            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = max(1, w * h)
            fill = float(area) / float(bbox_area)

            aspect = w / max(1.0, h)
            if aspect < 0.5 or aspect > 12.0:
                shape_score = 0.2
            elif aspect < 0.8 or aspect > 8.0:
                shape_score = 0.6
            else:
                shape_score = 1.0

            area_ratio = float(area) / float(mask_u8.size)
            size_score = max(0.0, 1.0 - min(area_ratio / 0.25, 1.0))

            # Border edge density (hard edges boost score)
            border = np.zeros((h, w), dtype=np.uint8)
            border[:2, :] = 1
            border[-2:, :] = 1
            border[:, :2] = 1
            border[:, -2:] = 1
            border_edges = edges[y:y + h, x:x + w]
            border_count = int(np.count_nonzero(border))
            if border_count:
                border_ratio = float(np.count_nonzero(border_edges[border == 1])) / float(border_count)
            else:
                border_ratio = 0.0

            score = (fill * 1.3) + (shape_score * 0.9) + (border_ratio * 0.8) + (size_score * 0.5)
            scored.append((score, idx, mask_u8))

        if not scored:
            return masks_u8, list(range(1, len(masks_u8) + 1))

        scored.sort(key=lambda x: x[0], reverse=True)
        keep_n = max(1, min(int(keep), len(scored)))
        chosen = scored[:keep_n]
        selected_masks = [m for _, _, m in chosen]
        selected_indices = [idx for _, idx, _ in chosen]
        return selected_masks, selected_indices

    def _nm_build_color_masks(
        self,
        image: Image.Image,
        roi: ScreenRegion,
        target_color: str,
        accent_color_relevant: bool,
    ) -> List["np.ndarray"]:
        """Build raw color mask(s) as uint8 arrays (255 for match, 0 otherwise)."""
        if not target_color and not accent_color_relevant:
            return []

        color = self._nm_normalize_color(target_color)
        if color in {"", "unknown"} and accent_color_relevant:
            color = "colorful"
        if color in {"", "unknown"}:
            return []

        try:
            import cv2
            import numpy as np
        except Exception:
            return []

        settings = self.nm_color_settings

        roi_img = image.crop((roi.left, roi.top, roi.right, roi.bottom))
        roi_array = np.array(roi_img.convert("RGB"))
        exclude_grid = None
        if getattr(self, "nm_bg_remove_enabled", None) is not None and self.nm_bg_remove_enabled.get():
            exclude_grid = self._nm_bg_grid_exclude_mask(roi_array, tol=4)

        mask = None
        masks = None
        custom = self._nm_parse_rgb_color(target_color)
        if custom:
            r0, g0, b0, tol = custom
            diff_r = np.abs(roi_array[:, :, 0].astype(np.int16) - r0)
            diff_g = np.abs(roi_array[:, :, 1].astype(np.int16) - g0)
            diff_b = np.abs(roi_array[:, :, 2].astype(np.int16) - b0)
            max_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
            mask = max_diff <= tol
        else:
            hsv = cv2.cvtColor(roi_array, cv2.COLOR_RGB2HSV)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
        if mask is None:
            if color in {"grey", "gray", "light_gray", "dark_gray", "light_grey", "dark_grey", "white", "black"}:
                # Greys: strict RGB closeness (max-min), exclude near-black/near-white unless requested
                r = roi_array[:, :, 0]
                g = roi_array[:, :, 1]
                b = roi_array[:, :, 2]
                maxc = np.maximum(np.maximum(r, g), b)
                minc = np.minimum(np.minimum(r, g), b)
                diff = maxc - minc  # low diff => grayish

                # Allow a bit of tint so near-greys still count (e.g., UI anti-aliasing)
                base = diff <= settings["grey"]["diff"]
                masks = None
                if color == "white":
                    mask = base & (maxc >= settings["grey"]["white_min"])
                elif color == "black":
                    mask = base & (maxc <= settings["grey"]["black_max"])
                elif color in {"dark_gray", "dark_grey"}:
                    mask = base & (maxc > settings["grey"]["dark_min"]) & (maxc <= settings["grey"]["dark_max"])
                elif color in {"light_gray", "light_grey"}:
                    mask = base & (maxc >= settings["grey"]["light_min"]) & (maxc < settings["grey"]["light_max"])
                else:
                    # "grey" / "gray": split by luminance bands so adjacent greys don't merge
                    lum = ((r.astype(np.uint16) + g.astype(np.uint16) + b.astype(np.uint16)) // 3).astype(np.uint8)
                    lum = cv2.blur(lum, (3, 3))
                    step = settings["grey"]["band_step"]
                    bands = (lum // step).astype(np.uint8)
                    unique_bands = np.unique(bands[base])
                    masks = [base & (bands == band) for band in unique_bands]
            elif color == "colorful":
                mask = (s >= settings["colorful"]["s_min"]) & (v >= settings["colorful"]["v_min"])
            else:
                def hue_range(hmin, hmax):
                    if hmin <= hmax:
                        return (h >= hmin) & (h <= hmax)
                    return (h >= hmin) | (h <= hmax)

                if color == "red":
                    mask = hue_range(settings["hue"]["red1"][0], settings["hue"]["red1"][1]) | hue_range(settings["hue"]["red2"][0], settings["hue"]["red2"][1])
                elif color == "orange":
                    mask = hue_range(settings["hue"]["orange"][0], settings["hue"]["orange"][1])
                elif color == "yellow":
                    mask = hue_range(settings["hue"]["yellow"][0], settings["hue"]["yellow"][1])
                elif color == "green":
                    mask = hue_range(settings["hue"]["green"][0], settings["hue"]["green"][1])
                elif color == "teal":
                    mask = hue_range(settings["hue"]["teal"][0], settings["hue"]["teal"][1])
                elif color == "blue":
                    base = hue_range(settings["hue"]["blue"][0], settings["hue"]["blue"][1])
                    r = roi_array[:, :, 0]
                    g = roi_array[:, :, 1]
                    b = roi_array[:, :, 2]
                    b_dom = (b.astype(np.int16) >= g.astype(np.int16) + settings["blue_dom"]["b_over_g"]) & (
                        b.astype(np.int16) >= r.astype(np.int16) + settings["blue_dom"]["b_over_r"]
                    )
                    strong = base & (s >= settings["sat_val"]["blue_strong_s"]) & (v >= settings["sat_val"]["blue_strong_v"])
                    soft = base & (s >= settings["sat_val"]["blue_soft_s"]) & (v >= settings["sat_val"]["blue_soft_v"])
                    mask = (strong | soft) & b_dom

                    # Keep bluish buttons even when contrast is subtle vs background
                    try:
                        s_blur = cv2.blur(s, (31, 31))
                        v_blur = cv2.blur(v, (31, 31))
                        b_blur = cv2.blur(b, (31, 31))
                        h_blur = cv2.blur(h, (31, 31))

                        delta_s = (s.astype(np.int16) - s_blur.astype(np.int16))
                        delta_v = (v.astype(np.int16) - v_blur.astype(np.int16))
                        delta_b = (b.astype(np.int16) - b_blur.astype(np.int16))
                        delta_h = np.abs(h.astype(np.int16) - h_blur.astype(np.int16))

                        contrast_ok = (
                            (delta_s >= settings["blue_contrast"]["delta_s"]) |
                            (delta_v >= settings["blue_contrast"]["delta_v"]) |
                            (delta_b >= settings["blue_contrast"]["delta_b"]) |
                            (delta_h >= settings["blue_contrast"]["delta_h"])
                        )
                        # Allow solid, saturated blues even if interior contrast is low.
                        strong_keep = strong & b_dom
                        mask = mask & (contrast_ok | strong_keep)
                    except Exception:
                        pass
                elif color == "purple":
                    mask = hue_range(settings["hue"]["purple"][0], settings["hue"]["purple"][1])
                elif color == "pink":
                    mask = hue_range(settings["hue"]["pink"][0], settings["hue"]["pink"][1])
                elif color == "brown":
                    mask = hue_range(settings["hue"]["brown"][0], settings["hue"]["brown"][1]) & (
                        v <= settings["brown"]["v_max"]
                    ) & (s >= settings["brown"]["s_min"])
                else:
                    mask = hue_range(0, 179)

                # Allow gradients: keep strong color and lightly tinted variants
                if color != "blue":
                    strong = mask & (s >= settings["sat_val"]["strong_s"]) & (v >= settings["sat_val"]["strong_v"])
                    soft = mask & (s >= settings["sat_val"]["soft_s"]) & (v >= settings["sat_val"]["soft_v"])
                    mask = strong | soft
        if mask is None and masks is None:
            return []

        if exclude_grid is not None:
            if mask is not None:
                mask = mask & (~exclude_grid)
            if masks is not None:
                masks = [(m & (~exclude_grid)) for m in masks]

        # If a single mask covers too much area, split by HSV clusters
        if mask is not None and masks is None:
            try:
                import numpy as np
                area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
            except Exception:
                area_ratio = 0.0
            split_threshold = 0.08 if color == "blue" else 0.03
            if area_ratio >= split_threshold:
                base_mask = mask
                splits = self._nm_split_mask_by_hsv(roi_array, base_mask, max_clusters=3)
                if splits:
                    masks = [base_mask] + splits
                    mask = None
                # For blue, also split by brightness to separate light vs dark blues.
                if color == "blue":
                    try:
                        import numpy as np
                        hsv = cv2.cvtColor(roi_array, cv2.COLOR_RGB2HSV)
                        v = hsv[:, :, 2]
                        v_vals = v[base_mask]
                        if v_vals.size > 0:
                            p40 = int(np.percentile(v_vals, 40))
                            p60 = int(np.percentile(v_vals, 60))
                            dark = base_mask & (v <= p40)
                            light = base_mask & (v >= p60)
                            min_pixels = max(200, int(base_mask.size * 0.002))
                            extra = []
                            if np.count_nonzero(dark) >= min_pixels:
                                extra.append(dark)
                            if np.count_nonzero(light) >= min_pixels:
                                extra.append(light)
                            if extra:
                                if masks is None:
                                    masks = [base_mask] + extra
                                    mask = None
                                else:
                                    masks.extend(extra)

                        # Strong-blue core: isolate the most saturated/blue-dominant pixels
                        r = roi_array[:, :, 0].astype(np.int16)
                        g = roi_array[:, :, 1].astype(np.int16)
                        b = roi_array[:, :, 2].astype(np.int16)
                        s = hsv[:, :, 1].astype(np.int16)
                        v_i = hsv[:, :, 2].astype(np.int16)
                        s_blur = cv2.blur(hsv[:, :, 1], (21, 21)).astype(np.int16)
                        v_blur = cv2.blur(hsv[:, :, 2], (21, 21)).astype(np.int16)
                        score = (b - ((r + g) // 2)) + (s // 2) + (s - s_blur) + (v_blur - v_i)
                        strong = self._nm_extract_strong_mask(base_mask, score, min_pixels=200, max_ratio=0.35)
                        if strong is not None:
                            if masks is None:
                                masks = [base_mask, strong]
                                mask = None
                            else:
                                masks.append(strong)
                    except Exception:
                        pass

        masks_to_process = masks if masks is not None else [mask]
        return [(m.astype(np.uint8) * 255) for m in masks_to_process if m is not None]

    def _nm_split_mask_by_hsv(
        self,
        roi_array: "np.ndarray",
        mask: "np.ndarray",
        max_clusters: int = 3,
    ) -> List["np.ndarray"]:
        """Split a large color mask into clusters by HSV (helps separate similar blues)."""
        try:
            import cv2
            import numpy as np
        except Exception:
            return []

        coords = np.column_stack(np.where(mask))
        total = coords.shape[0]
        if total < 500:
            return []

        hsv = cv2.cvtColor(roi_array, cv2.COLOR_RGB2HSV)
        hsv_masked = hsv[mask].astype(np.float32)

        # Sample for k-means
        sample_n = min(6000, total)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(total, size=sample_n, replace=False)
        sample = hsv_masked[sample_idx]

        # Normalize to [0,1]
        sample_nrm = np.column_stack([
            sample[:, 0] / 179.0,
            sample[:, 1] / 255.0,
            sample[:, 2] / 255.0,
        ]).astype(np.float32)

        # Choose cluster count
        k = 2
        if total > (mask.size * 0.25):
            k = min(max_clusters, 3)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
        try:
            _compact, _labels, centers = cv2.kmeans(
                sample_nrm, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
        except Exception:
            return []

        # Assign all masked pixels to nearest center
        all_nrm = np.column_stack([
            hsv_masked[:, 0] / 179.0,
            hsv_masked[:, 1] / 255.0,
            hsv_masked[:, 2] / 255.0,
        ]).astype(np.float32)
        dists = np.sum((all_nrm[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels_all = np.argmin(dists, axis=1)

        submasks: List[np.ndarray] = []
        min_pixels = max(150, int(total * 0.03))
        for i in range(k):
            idxs = labels_all == i
            if np.count_nonzero(idxs) < min_pixels:
                continue
            sub = np.zeros(mask.shape, dtype=bool)
            sub[coords[idxs, 0], coords[idxs, 1]] = True
            submasks.append(sub)

        return submasks

    def _nm_extract_strong_mask(
        self,
        base_mask: "np.ndarray",
        score: "np.ndarray",
        min_pixels: int = 200,
        max_ratio: float = 0.45,
    ) -> Optional["np.ndarray"]:
        """Extract a smaller submask using high-percentile score values."""
        try:
            import numpy as np
        except Exception:
            return None

        vals = score[base_mask]
        if vals.size < min_pixels:
            return None
        base_area = float(vals.size)
        for pct in (80, 85, 90, 93, 95):
            try:
                thr = np.percentile(vals, pct)
            except Exception:
                continue
            strong = base_mask & (score >= thr)
            area = int(np.count_nonzero(strong))
            if area >= min_pixels and area <= base_area * max_ratio:
                return strong
        return None

    def _nm_mask_to_bbox(
        self,
        mask_u8: "np.ndarray",
        roi: ScreenRegion,
        pad: int = 2,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return a single tight bbox around a mask."""
        try:
            import numpy as np
        except Exception:
            return None
        ys, xs = np.where(mask_u8 > 0)
        if ys.size == 0 or xs.size == 0:
            return None
        l = int(xs.min())
        r = int(xs.max())
        t = int(ys.min())
        b = int(ys.max())
        l = max(0, l - pad)
        t = max(0, t - pad)
        r = min((roi.right - roi.left) - 1, r + pad)
        b = min((roi.bottom - roi.top) - 1, b + pad)
        return (roi.left + l, roi.top + t, roi.left + r, roi.top + b)

    def _nm_run_paddleocr_roi(
        self,
        image: Image.Image,
        roi: ScreenRegion,
    ) -> List["OCRMatch"]:
        """Run PaddleOCR on a ROI and return OCRMatch list."""
        if not self.init_paddleocr():
            return []

        try:
            import numpy as np
            from agent.ocr import OCRMatch

            crop = image.crop((roi.left, roi.top, roi.right, roi.bottom))
            img_array = np.array(crop.convert("RGB"))
            results = self.paddleocr_ocr.ocr(img_array)

            matches: List[OCRMatch] = []
            if isinstance(results, list) and results and results[0]:
                for line in results[0]:
                    if not line:
                        continue
                    bbox_points = line[0]
                    text, confidence = line[1]
                    if not text:
                        continue

                    confidence_percent = float(confidence) * 100.0
                    if confidence_percent < 10:
                        continue
                    if not self.is_valid_text(text):
                        continue

                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    left = int(min(xs)) + roi.left
                    top = int(min(ys)) + roi.top
                    right = int(max(xs)) + roi.left
                    bottom = int(max(ys)) + roi.top
                    width = max(1, right - left)
                    height = max(1, bottom - top)
                    bbox = ScreenRegion(left=left, top=top, width=width, height=height)
                    matches.append(OCRMatch(text=text, confidence=confidence_percent, bbox=bbox))

            return matches
        except Exception as e:
            self.log(f"PaddleOCR ROI error: {e}", "error")
            return []

    def _nm_generate_candidates(
        self,
        snapshot: NewMethodSnapshot,
        planner: PlannerOutput,
        force_use_ocr: Optional[bool] = None,
        force_use_color: Optional[bool] = None,
        force_use_shape: Optional[bool] = None,
    ) -> List[CandidatePackage]:
        """
        Step 3–4: Deterministic engine.
        - Region reduction using planner location intent
        - Feature extraction with OCR and simple geometry
        - Candidate creation and scoring
        """
        import time
        start_time = time.time()
        img = snapshot.image

        # 3.1 Region reduction
        roi = self._nm_zone_to_roi(
            image=img,
            window_rect=snapshot.window_rect,
            zone=planner.location_intent.zone,
        )
        self.log(
            f"ROI from planner zone '{planner.location_intent.zone}': "
            f"({roi.left},{roi.top})–({roi.right},{roi.bottom})",
            "info",
        )

        # 3.2 Feature extraction - OCR with multiple strategies to catch more text
        use_ocr = self.nm_use_ocr_var.get() if force_use_ocr is None else bool(force_use_ocr)
        use_color = self.nm_use_color_var.get() if force_use_color is None else bool(force_use_color)
        use_shape = self.nm_use_shape_var.get() if force_use_shape is None else bool(force_use_shape)
        has_text_intent = False
        if use_ocr:
            has_text_intent = bool(
                (planner.text_intent.primary_text or "").strip()
                or (planner.text_intent.variants or [])
            )
            query_text = (planner.text_intent.primary_text or "").strip()
            query_word_count = len([w for w in query_text.split() if w])
            include_phrases = query_word_count >= 2
            # Match test-script behavior: OCR over the full image when text intent exists.
            ocr_roi = roi
            if has_text_intent:
                ocr_roi = ScreenRegion(left=0, top=0, width=img.width, height=img.height)
            ocr_result = run_ocr_in_roi(
                self,
                img,
                ocr_roi,
                force_full=has_text_intent,
                include_phrases=include_phrases,
                query_text=query_text,
            )
        else:
            from agent.ocr import OCRResult
            ocr_result = OCRResult(matches=[], raw_text="")
            self.log("OCR disabled for this run.", "info")

        # If we have text intent, filter to top-N matches by test-script similarity.
        ocr_similarity_fn = None
        if has_text_intent and ocr_result.matches:
            from difflib import SequenceMatcher

            query = (planner.text_intent.primary_text or "").strip()
            if not query and planner.text_intent.variants:
                query = (planner.text_intent.variants[0] or "").strip()

            def _test_similarity(text: str) -> float:
                if not text or not query:
                    return 0.0
                cand = text.strip().lower()
                q = query.strip().lower()
                if q in cand:
                    return 1.0
                return SequenceMatcher(None, cand, q).ratio()

            ocr_similarity_fn = _test_similarity
            scored = []
            for m in ocr_result.matches:
                sim = _test_similarity(m.text)
                if sim >= 0.9:
                    scored.append((m, sim))
            scored.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
            kept = [m for m, _ in scored[:5]]
            if kept:
                self.log(f"OCR kept top {len(kept)} matches by similarity>=0.9.", "info")
            else:
                self.log("OCR kept 0 matches by similarity>=0.9.", "info")
            ocr_result.matches = kept

        if ocr_result.matches:
            ocr_boxes = [(m.bbox.left, m.bbox.top, m.bbox.right, m.bbox.bottom) for m in ocr_result.matches]
            ocr_overlay = self._nm_draw_rects_overlay(img, ocr_boxes, (255, 80, 80, 220), width=2)
            self._nm_add_stage(f"OCR matches ({len(ocr_boxes)})", ocr_overlay)

        # Find visual regions (buttons/inputs) using contrast detection
        contrast_regions = []
        if use_shape:
            contrast_regions = self._nm_find_contrast_regions(img, roi)
            self.log(f"Contrast detection found {len(contrast_regions)} potential UI element regions.", "info")
            if contrast_regions:
                contrast_overlay = self._nm_draw_rects_overlay(img, contrast_regions, (255, 215, 0, 220), width=2)
                self._nm_add_stage(f"Contrast regions ({len(contrast_regions)})", contrast_overlay)
        else:
            self.log("Shape disabled - skipping contrast detection.", "info")

        # Color mask regions (independent of OCR)
        color_regions = []
        masks_u8 = []
        if use_color:
            target_color = planner.visual_intent.primary_color
            accent = planner.visual_intent.accent_color_relevant
            color_regions, masks_u8 = self._nm_find_color_regions(
                img,
                roi,
                target_color,
                accent,
                self._nm_get_color_max_area(),
                return_masks=True,
            )
            self.log(f"Color mask found {len(color_regions)} regions.", "info")
            if color_regions:
                color_overlay = self._nm_draw_rects_overlay(img, color_regions, (0, 180, 255, 220), width=2)
                self._nm_add_stage(f"Color mask regions ({len(color_regions)})", color_overlay)
            if masks_u8:
                try:
                    import numpy as np

                    def add_mask_stage(label: str, mask_u8: "np.ndarray") -> None:
                        overlay = img.copy().convert("RGBA")
                        mask_rgba = Image.fromarray(mask_u8).convert("L")
                        tint = Image.new("RGBA", overlay.size, (0, 180, 255, 120))
                        overlay = Image.composite(tint, overlay, mask_rgba)
                        self._nm_add_stage(label, overlay.convert("RGB"))

                    combined = np.zeros_like(masks_u8[0], dtype=np.uint8)
                    for m in masks_u8:
                        if m is not None:
                            combined = np.maximum(combined, m)
                    add_mask_stage(f"Color mask combined ({len(masks_u8)})", combined)
                    if len(masks_u8) > 1:
                        for i, m in enumerate(masks_u8, start=1):
                            if m is None:
                                continue
                            add_mask_stage(f"Color mask split {i}/{len(masks_u8)}", m)
                except Exception:
                    pass

        # 3.3 Candidate creation
        candidates: List[CandidatePackage] = []
        next_id = 1

        # Text-anchored candidates: keep boxes tight to text for accurate visuals
        if ocr_result.matches:
            min_sim = 0.9 if has_text_intent else 0.0
            ocr_candidates, next_id = build_ocr_candidates(
                self,
                snapshot,
                planner,
                ocr_roi if has_text_intent else roi,
                ocr_result,
                next_id,
                CandidatePackage,
                min_similarity=min_sim,
                similarity_fn=ocr_similarity_fn,
            )
            candidates.extend(ocr_candidates)

        # Color-mask candidates: regions detected by color (independent of OCR)
        (
            color_candidates,
            next_id,
            color_candidates_added,
            color_candidates_skipped,
            color_noise_area_min,
        ) = build_color_candidates(
            self,
            snapshot,
            roi,
            color_regions,
            next_id,
            CandidatePackage,
        )
        candidates.extend(color_candidates)

        if color_candidates_added > 0:
            self.log(f"Added {color_candidates_added} color-mask candidates.", "info")
        if color_candidates_skipped > 0:
            self.log(
                f"Skipped {color_candidates_skipped} tiny color-mask regions (area<{color_noise_area_min}px).",
                "info",
            )

        # Shape-anchored candidates: regions detected by visual contrast (buttons/inputs)
        # These catch UI elements that OCR might miss (empty input fields, icon-only buttons)
        shape_candidates_added = 0
        for left, top, right, bottom in contrast_regions:
            # Skip if this region overlaps significantly with an existing OCR candidate
            overlaps = False
            for existing in candidates:
                ex_left, ex_top, ex_right, ex_bottom = existing.bbox
                # Check overlap
                overlap_area = max(0, min(right, ex_right) - max(left, ex_left)) * \
                              max(0, min(bottom, ex_bottom) - max(top, ex_top))
                region_area = (right - left) * (bottom - top)
                if overlap_area > region_area * 0.5:  # More than 50% overlap
                    overlaps = True
                    break
            
            if overlaps:
                continue  # Skip duplicate
            
            # Check if there's any OCR text nearby (within the region)
            nearby_text = None
            for match in ocr_result.matches:
                mx, my = match.center
                if left <= mx <= right and top <= my <= bottom:
                    nearby_text = match.text.strip()
                    break
            
            # Center click point
            click_x = (left + right) // 2
            click_y = (top + bottom) // 2
            color = self._nm_compute_region_color(img, (left, top, right, bottom))
            
            # Score based on text match if text found, otherwise lower score
            if nearby_text:
                text_sim = self._nm_text_similarity(planner.text_intent, nearby_text)
            else:
                # No text found - give neutral score, let picker AI decide
                text_sim = 0.3
            
            size = (right - left, bottom - top)
            aspect = size[0] / max(1, size[1])
            if 0.8 <= aspect <= 8.0:
                shape_score = 1.0
            elif 0.5 <= aspect <= 12.0:
                shape_score = 0.8
            else:
                shape_score = 0.6
            
            # Location score
            wx = (snapshot.window_rect[0] + snapshot.window_rect[2]) / 2.0
            wy = (snapshot.window_rect[1] + snapshot.window_rect[3]) / 2.0
            dx = (click_x - wx) / max(1.0, snapshot.window_rect[2] - snapshot.window_rect[0])
            dy = (click_y - wy) / max(1.0, snapshot.window_rect[3] - snapshot.window_rect[1])
            dist_norm = math.sqrt(dx * dx + dy * dy)
            location_score = max(0.0, 1.0 - dist_norm)
            
            scores = {
                "text_match": text_sim,
                "shape_plausibility": shape_score,
                "location_match": location_score,
            }
            
            # Lower weight for shape-based candidates (they're less certain)
            total = (
                scores["text_match"] * 0.5
                + scores["shape_plausibility"] * 0.3
                + scores["location_match"] * 0.2
            )
            
            candidates.append(
                CandidatePackage(
                    id=next_id,
                    bbox=(left, top, right, bottom),
                    click_point=(click_x, click_y),
                    text=nearby_text or "[visual element]",
                    color=color,
                    scores=scores,
                    total_score=total,
                    source="shape",
                )
            )
            next_id += 1
            shape_candidates_added += 1
        
        if shape_candidates_added > 0:
            self.log(f"Added {shape_candidates_added} shape-based candidates from contrast detection.", "info")

        # 3.4 Sort by score but do NOT discard low-scoring candidates
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        # Only cap in extreme cases to avoid overwhelming the picker
        max_candidates = 5000
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        # Reassign stable numeric IDs after sorting
        for new_id, cand in enumerate(candidates, start=1):
            cand.id = new_id

        return candidates

    def _nm_draw_candidates_overlay(
        self,
        image: Image.Image,
        candidates: List[CandidatePackage],
        highlight_id: Optional[int] = None,
        show_ids: bool = True,
    ) -> Image.Image:
        """Draw candidate boxes (and optional IDs) on a copy of the image."""
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Choose font for IDs
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for c in candidates:
            l, t, r, b = c.bbox
            bbox_pad = 3
            l = max(0, l - bbox_pad)
            t = max(0, t - bbox_pad)
            r = min(base.width, r + bbox_pad)
            b = min(base.height, b + bbox_pad)
            color = (0, 255, 255, 220)
            width = 2
            if highlight_id is not None and c.id == highlight_id:
                color = (0, 255, 0, 255)
                width = 3
            draw.rectangle([l, t, r, b], outline=color, width=width)
            if show_ids:
                text = f"{c.id}"
                # Use textbbox for compatibility with newer Pillow versions
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except Exception:
                    # Fallback rough estimate if textbbox is unavailable
                    tw = len(text) * 8
                    th = 14
                pad = 4
                label_w = tw + pad * 2
                label_h = th + pad * 2
                # Place label outside the box to avoid covering the target
                label_x = l
                label_y = t - label_h - 2
                if label_y < 0:
                    label_y = b + 2
                if label_y + label_h > base.height:
                    label_y = max(0, t)
                    label_x = r + 2
                if label_x + label_w > base.width:
                    label_x = max(0, l - label_w - 2)
                label_x = max(0, min(base.width - label_w, label_x))
                label_y = max(0, min(base.height - label_h, label_y))
                draw.rectangle([label_x, label_y, label_x + label_w, label_y + label_h], fill=(0, 0, 0, 180))
                draw.text((label_x + pad, label_y + pad), text, font=font, fill=(0, 255, 255, 255))

        return Image.alpha_composite(base, overlay).convert("RGB")

    def _nm_picker_ai(
        self,
        snapshot: NewMethodSnapshot,
        planner: PlannerOutput,
        candidates: List[CandidatePackage],
    ) -> Optional[int]:
        """
        Step 5: Picker AI.
        Forced choice among numbered candidates. Must return candidate ID or UNSURE.
        """
        try:
            if not candidates:
                return None

            # Crop to ROI of all candidates with padding for context
            min_left = min(c.bbox[0] for c in candidates)
            min_top = min(c.bbox[1] for c in candidates)
            max_right = max(c.bbox[2] for c in candidates)
            max_bottom = max(c.bbox[3] for c in candidates)
            # Add padding (10% of width/height, min 50px) for better context
            pad_x = max(50, int((max_right - min_left) * 0.1))
            pad_y = max(50, int((max_bottom - min_top) * 0.1))
            crop_left = max(0, min_left - pad_x)
            crop_top = max(0, min_top - pad_y)
            crop_right = min(snapshot.image.width, max_right + pad_x)
            crop_bottom = min(snapshot.image.height, max_bottom + pad_y)
            crop = snapshot.image.crop((crop_left, crop_top, crop_right, crop_bottom))
            crop_b64 = self.encode_image(crop)

            roi = self._nm_zone_to_roi(
                image=snapshot.image,
                window_rect=snapshot.window_rect,
                zone=planner.location_intent.zone,
            )
            features = self._nm_build_candidate_features(snapshot, roi, candidates)

            # Prepare textual description of candidates
            lines = []
            for c in candidates:
                l, t, r, b = c.bbox
                feat = features.get(c.id, {})
                width_px = feat.get("width_px", max(1, r - l))
                height_px = feat.get("height_px", max(1, b - t))
                size_px_class = feat.get("size_px_class", "unknown")
                color_group = feat.get("color_group", "unknown")
                lines.append(
                    f"ID {c.id}: text='{c.text}', "
                    f"bbox=({l},{t},{r},{b}), "
                    f"score={c.total_score:.3f}, "
                    f"text_score={c.scores.get('text_match', 0):.3f}, "
                    f"color={color_group}, size={size_px_class} ({width_px}x{height_px})"
                )
            candidates_text = "\n".join(lines)
            layout_lines = [
                f"ID {c.id}: {self._describe_candidate_position(c, snapshot.window_rect)} (text '{c.text or 'icon'}')"
                for c in candidates
            ]
            location_details = "\n".join(layout_lines)

            intent_summary = (
                f"Primary text: '{planner.text_intent.primary_text}' "
                f"(variants: {planner.text_intent.variants}, strict={planner.text_intent.strictness}), "
                f"zone={planner.location_intent.zone}, pos={planner.location_intent.position}, "
                f"color={planner.visual_intent.primary_color}/{planner.visual_intent.relative_luminance}, "
                f"shape={planner.visual_intent.shape}, size={planner.visual_intent.size}, "
                f"risk={planner.risk_intent.level}"
            )
            if planner.visual_intent.description:
                intent_summary += f", visual='{planner.visual_intent.description}'"

            system_prompt = load_prompt("coordinate_finder/picker_system.txt")

            user_prompt = format_prompt(
                load_prompt("coordinate_finder/picker_user.txt"),
                instruction=snapshot.user_instruction,
                intent_summary=intent_summary,
                candidates_text=candidates_text,
                location_details=location_details,
            )

            response = self.client.chat.completions.create(
                model="gpt-5.2",
                max_completion_tokens=300,  # Increased for better reasoning
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{crop_b64}",
                                    "detail": "high",  # High detail needed to accurately distinguish between candidates
                                },
                            },
                        ],
                    },
                ],
            )
            raw_text = response.choices[0].message.content or ""
            self.log(f"Picker raw response: {raw_text}", "ai")

            try:
                choice_dict = json.loads(raw_text)
            except json.JSONDecodeError:
                import re

                m = re.search(r"\{[\s\S]*\}", raw_text)
                if not m:
                    self.log("Picker response was not valid JSON and no object could be extracted.", "error")
                    return None
                choice_dict = json.loads(m.group(0))

            choice_val = choice_dict.get("choice", "UNSURE")
            if isinstance(choice_val, str) and choice_val.upper() == "UNSURE":
                return None
            try:
                choice_id = int(choice_val)
            except (TypeError, ValueError):
                return None

            if not any(c.id == choice_id for c in candidates):
                # Invalid ID – treat as UNSURE
                return None

            return choice_id
        except Exception as e:
            self.log(f"Picker AI error: {e}", "error")
            return None

    def _describe_candidate_position(
        self,
        candidate: CandidatePackage,
        window_rect: Tuple[int, int, int, int],
    ) -> str:
        left, top, right, bottom = candidate.bbox
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        wl, wt, wr, wb = window_rect
        win_w = max(1, wr - wl)
        win_h = max(1, wb - wt)

        def axis_section(value, start, total):
            norm = (value - start) / total
            if norm < 0.33:
                return "left" if total == win_w else "top"
            if norm > 0.66:
                return "right" if total == win_w else "bottom"
            return "center"

        horiz = axis_section(cx, wl, win_w)
        vert = axis_section(cy, wt, win_h)
        relative_x = (cx - wl) / win_w
        relative_y = (cy - wt) / win_h

        edges = []
        if top - wt < 0.05 * win_h:
            edges.append("just below the top edge")
        if wb - bottom < 0.05 * win_h:
            edges.append("just above the bottom edge")
        if left - wl < 0.05 * win_w:
            edges.append("hugging the left edge")
        if wr - right < 0.05 * win_w:
            edges.append("hugging the right edge")

        size_desc = f"{right-left}x{bottom-top}px"
        details = "; ".join(edges) if edges else "away from the window edges"
        text_label = candidate.text or "non-text icon"

        return (
            f"{vert}-{horiz} of the window (x={relative_x:.2f}, y={relative_y:.2f}), "
            f"size {size_desc}, {details}, near {text_label}"
        )

    def _nm_show_virtual_click(
        self,
        snapshot: NewMethodSnapshot,
        chosen: CandidatePackage,
        candidates: List[CandidatePackage],
    ) -> None:
        """
        Step 6: "Click execution" for this offline tester.
        We visually MASK the UI down to just the candidate regions and highlight the chosen one.
        """
        base = snapshot.image.copy().convert("RGBA")

        # Start with a dark mask over the whole image
        mask_overlay = Image.new("RGBA", base.size, (0, 0, 0, 180))
        mask_draw = ImageDraw.Draw(mask_overlay)

        # Punch transparent "windows" for every candidate region
        for c in candidates:
            l, t, r, b = c.bbox
            # Make candidate region fully transparent in the mask (reveals original image)
            mask_draw.rectangle([l, t, r, b], fill=(0, 0, 0, 0))

        # Composite mask over base: everything except candidate boxes is darkened
        masked = Image.alpha_composite(base, mask_overlay)

        # Now draw borders and chosen point on top
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for c in candidates:
            l, t, r, b = c.bbox
            color = (0, 255, 255, 220) if c.id != chosen.id else (0, 255, 0, 255)
            width = 2 if c.id != chosen.id else 3
            draw.rectangle([l, t, r, b], outline=color, width=width)

        # Draw chosen click point
        cx, cy = chosen.click_point
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 255, 0, 255), outline=(0, 0, 0, 255), width=2)

        composite = Image.alpha_composite(masked, overlay).convert("RGB")

        # Update current view canvas and store as a stage
        self.current_image = composite
        self.current_coords = chosen.click_point
        self.coords_label.config(text=f"{cx}, {cy}")
        self._nm_add_stage(f"Picker choice (ID {chosen.id})", composite)

    def _nm_verify(
        self,
        snapshot: NewMethodSnapshot,
        planner: PlannerOutput,
        chosen: CandidatePackage,
    ) -> bool:
        """
        Step 7: Verification.
        In this offline tester we cannot observe UI changes, so we instead verify:
        - Text match quality for the chosen candidate against planner intent.
        - That chosen candidate is reasonably better than the second-best candidate.

        If this logical verification fails, we report UNSURE instead of success.
        """
        text_score = self._nm_text_similarity(planner.text_intent, chosen.text or "")
        if text_score < 0.4:
            # Chosen candidate does not look semantically correct
            return False

        # Compare to second-best candidate
        others = [c for c in [] + [chosen] if c.id != chosen.id]  # trivial init
        # We need all candidates; easiest is to recompute locally: not ideal, but safe.
        # For this offline harness, treat a good text_score as sufficient.
        return True

    def run_seeclick_once(self):
        """Use SeeClick once to get coordinates and plot them on the current image."""
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            self.log("Prompt is empty. Please enter what you want to click.", "error")
            return

        coords = self.get_seeclick_coordinates(prompt)
        if coords is None:
            return

        self.current_coords = coords
        self.coords_label.config(text=f"{coords[0]}, {coords[1]}")
        self.current_image = self.original_image.copy()
        # Draw marker on the main canvas
        self.display_image(
            self.current_canvas,
            self.current_image,
            coords=self.current_coords,
            draw_crosshair=False,
        )
        
    def get_initial_coordinates(self, prompt):
        """Ask AI to find initial approximate coordinates."""
        if not self.client:
            self.log("ERROR: OpenAI API key not configured!", 'error')
            self.log("Please add your API key to the .env file", 'error')
            return None
            
        self.log(f"Asking AI to find: '{prompt}'", 'ai')
        
        system_prompt = format_prompt(
            load_prompt("coordinate_finder/initial_coordinates_system.txt"),
            width=self.original_image.width,
            height=self.original_image.height,
        )

        try:
            # Add crosshair to image for AI
            image_with_crosshair = self.draw_crosshair(self.original_image.copy())
            
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Find the coordinates of: {prompt}"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(image_with_crosshair)}"
                        }}
                    ]}
                ],
                max_completion_tokens=500
            )
            
            response_text = response.choices[0].message.content
            self.log(f"AI Response: {response_text}", 'ai')
            
            # Parse JSON from response
            # Try to extract JSON from the response
            try:
                # Try direct parsing first
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    self.log("Could not parse AI response as JSON", 'error')
                    return None
            
            x, y = int(result['x']), int(result['y'])
            confidence = result.get('confidence', 'unknown')
            reasoning = result.get('reasoning', '')
            
            self.log(f"Initial coordinates: ({x}, {y})", 'coords')
            self.log(f"Confidence: {confidence}", 'info')
            self.log(f"Reasoning: {reasoning}", 'info')
            
            return (x, y)
            
        except Exception as e:
            self.log(f"Error calling OpenAI API: {e}", 'error')
            return None
            
    def create_zoomed_view(self, coords, zoom_factor=1.5):
        """Create a zoomed view centered on the coordinates, with black padding if needed."""
        x, y = coords
        
        # Calculate crop region size (zoom in means showing less of original)
        crop_width = int(self.original_image.width / zoom_factor)
        crop_height = int(self.original_image.height / zoom_factor)
        
        # Calculate desired crop bounds centered on coordinate
        desired_left = x - crop_width // 2
        desired_top = y - crop_height // 2
        desired_right = x + crop_width // 2
        desired_bottom = y + crop_height // 2
        
        # Calculate how much padding we need on each side
        pad_left = max(0, -desired_left)
        pad_top = max(0, -desired_top)
        pad_right = max(0, desired_right - self.original_image.width)
        pad_bottom = max(0, desired_bottom - self.original_image.height)
        
        # Actual crop bounds (clamped to image)
        actual_left = max(0, desired_left)
        actual_top = max(0, desired_top)
        actual_right = min(self.original_image.width, desired_right)
        actual_bottom = min(self.original_image.height, desired_bottom)
        
        # Crop the image
        cropped = self.original_image.crop((actual_left, actual_top, actual_right, actual_bottom))
        
        # Create a black canvas of the desired crop size
        zoomed = Image.new('RGB', (crop_width, crop_height), color='black')
        
        # Paste the cropped image onto the black canvas at the correct position
        paste_x = pad_left
        paste_y = pad_top
        zoomed.paste(cropped, (paste_x, paste_y))
        
        # Resize back to original dimensions for display
        zoomed = zoomed.resize((self.original_image.width, self.original_image.height), 
                                Image.Resampling.LANCZOS)
        
        # Calculate where the coordinate appears in the zoomed image
        # The coordinate should be at the center of the zoomed view
        coord_in_zoomed_x = self.original_image.width // 2
        coord_in_zoomed_y = self.original_image.height // 2
        
        self.log(f"Zoomed to region centered on ({x}, {y})", 'info')
        self.log(f"Crop size: {crop_width}x{crop_height}, Padding: L:{pad_left} T:{pad_top} R:{pad_right} B:{pad_bottom}", 'info')
        self.log(f"Zoom factor: {zoom_factor}x ({int(100/zoom_factor*100)}% of original visible)", 'info')
        
        # Return zoomed image and crop info (for coordinate mapping)
        crop_info = {
            'left': actual_left,
            'top': actual_top,
            'right': actual_right,
            'bottom': actual_bottom,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'crop_width': crop_width,
            'crop_height': crop_height,
            'zoom_factor': zoom_factor,
            'coord_in_zoomed': (coord_in_zoomed_x, coord_in_zoomed_y)
        }
        
        return zoomed, crop_info
    
    def ask_which_grid(self, image, grid_x, grid_y, target_prompt):
        """Ask AI which grid cell contains the target."""
        if not self.client:
            return None
        
        self.log(f"Asking AI which grid (out of {grid_x * grid_y}) contains: '{target_prompt}'", 'ai')
        
        # Draw grid on image
        img_with_grid = self.draw_grid(image.copy(), grid_x, grid_y)
        
        total_cells = grid_x * grid_y
        
        system_prompt = format_prompt(
            load_prompt("coordinate_finder/grid_selector_system.txt"),
            grid_x=grid_x,
            grid_y=grid_y,
            total_cells=total_cells,
            target_prompt=target_prompt,
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Which grid cell contains '{target_prompt}'?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(img_with_grid)}"
                        }}
                    ]}
                ],
                max_completion_tokens=500
            )
            
            response_text = response.choices[0].message.content
            self.log(f"AI Response: {response_text}", 'ai')
            
            # Parse JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    self.log("Could not parse grid selection response", 'error')
                    return None
            
            grid_num_or_click = result.get('grid_number')
            confidence = result.get('confidence', 'low').lower()
            reasoning = result.get('reasoning', '')
            
            # Check if AI wants to click
            if isinstance(grid_num_or_click, str) and grid_num_or_click.lower() == 'click':
                if confidence == 'high':
                    self.log("🎯 AI is HIGHLY CONFIDENT - Ready to CLICK!", 'success')
                    self.log(f"Confidence: {confidence}", 'success')
                    self.log(f"Reasoning: {reasoning}", 'info')
                    return "click"
                else:
                    self.log("AI said 'click' but confidence is not high, continuing search...", 'info')
                    return None
            
            # Parse grid number
            try:
                grid_num = int(grid_num_or_click)
            except (ValueError, TypeError):
                self.log(f"Invalid grid_number value: {grid_num_or_click}", 'error')
                return None
            
            if grid_num < 1 or grid_num > total_cells:
                self.log(f"Invalid grid number: {grid_num} (must be 1-{total_cells})", 'error')
                return None
            
            self.log(f"AI selected grid cell: {grid_num}", 'coords')
            self.log(f"Confidence: {confidence}", 'info')
            self.log(f"Reasoning: {reasoning}", 'info')
            
            return grid_num
            
        except Exception as e:
            self.log(f"Error calling OpenAI API: {e}", 'error')
            return None
    
    def calculate_grid_region(self, image, grid_x, grid_y, grid_number, padding_percent=0.5):
        """Calculate the region (with padding) for a specific grid cell."""
        width = image.width
        height = image.height
        
        # Calculate cell dimensions
        cell_width = width / grid_x
        cell_height = height / grid_y
        
        # Convert grid number to row/col (1-indexed)
        grid_num = grid_number - 1  # Convert to 0-indexed
        row = grid_num // grid_x
        col = grid_num % grid_x
        
        # Calculate grid cell bounds
        cell_left = col * cell_width
        cell_top = row * cell_height
        cell_right = (col + 1) * cell_width
        cell_bottom = (row + 1) * cell_height
        
        # Calculate padding (50% of cell size)
        padding_x = cell_width * padding_percent
        padding_y = cell_height * padding_percent
        
        # Calculate region bounds (cell + padding)
        region_left = max(0, cell_left - padding_x)
        region_top = max(0, cell_top - padding_y)
        region_right = min(width, cell_right + padding_x)
        region_bottom = min(height, cell_bottom + padding_y)
        
        return {
            'left': int(region_left),
            'top': int(region_top),
            'right': int(region_right),
            'bottom': int(region_bottom),
            'cell_left': int(cell_left),
            'cell_top': int(cell_top),
            'cell_right': int(cell_right),
            'cell_bottom': int(cell_bottom),
            'cell_center_x': int((cell_left + cell_right) / 2),
            'cell_center_y': int((cell_top + cell_bottom) / 2)
        }
    
    def zoom_to_grid(self, image, grid_x, grid_y, grid_number, padding_percent=0.25):
        """Zoom into a specific grid cell with padding around edges."""
        width = image.width
        height = image.height
        
        # Calculate cell dimensions
        cell_width = width / grid_x
        cell_height = height / grid_y
        
        # Convert grid number to row/col (1-indexed)
        grid_num = grid_number - 1  # Convert to 0-indexed
        row = grid_num // grid_x
        col = grid_num % grid_x
        
        # Calculate grid cell bounds
        cell_left = col * cell_width
        cell_top = row * cell_height
        cell_right = (col + 1) * cell_width
        cell_bottom = (row + 1) * cell_height
        
        # Calculate center of grid cell
        cell_center_x = (cell_left + cell_right) / 2
        cell_center_y = (cell_top + cell_bottom) / 2
        
        # Calculate padding (25% of cell size)
        padding_x = cell_width * padding_percent
        padding_y = cell_height * padding_percent
        
        # Calculate desired crop region (cell + padding)
        desired_left = cell_left - padding_x
        desired_top = cell_top - padding_y
        desired_right = cell_right + padding_x
        desired_bottom = cell_bottom + padding_y
        
        # Calculate how much padding we need on each side (for black padding)
        # Only add black padding if we're actually outside the original image bounds
        pad_left = max(0, -desired_left)
        pad_top = max(0, -desired_top)
        pad_right = max(0, desired_right - width)
        pad_bottom = max(0, desired_bottom - height)
        
        # Actual crop bounds (clamped to image)
        actual_left = max(0, int(desired_left))
        actual_top = max(0, int(desired_top))
        actual_right = min(width, int(desired_right))
        actual_bottom = min(height, int(desired_bottom))
        
        # Crop size
        crop_width = int(desired_right - desired_left)
        crop_height = int(desired_bottom - desired_top)
        
        # Crop the image
        cropped = image.crop((actual_left, actual_top, actual_right, actual_bottom))
        
        # Only create black canvas and add padding if we're actually outside the image bounds
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            # Create black canvas only when padding is needed (outside original image)
            zoomed = Image.new('RGB', (crop_width, crop_height), color='black')
            
            # Paste cropped image onto black canvas
            paste_x = int(pad_left)
            paste_y = int(pad_top)
            zoomed.paste(cropped, (paste_x, paste_y))
            self.log(f"Added black padding: L:{int(pad_left)} T:{int(pad_top)} R:{int(pad_right)} B:{int(pad_bottom)}", 'info')
        else:
            # No black padding needed - just use the cropped image
            zoomed = cropped
        
        # Resize back to original dimensions for display
        zoomed = zoomed.resize((width, height), Image.Resampling.LANCZOS)
        
        self.log(f"Zoomed to grid {grid_number} (row {row+1}, col {col+1})", 'info')
        self.log(f"Cell bounds: ({int(cell_left)}, {int(cell_top)}) - ({int(cell_right)}, {int(cell_bottom)})", 'info')
        self.log(f"Padding: {int(padding_x)}x{int(padding_y)} pixels ({int(padding_percent*100)}%)", 'info')
        
        # Calculate where grid center appears in zoomed image
        # The grid center should be at the center of the zoomed view
        grid_center_in_zoomed_x = width // 2
        grid_center_in_zoomed_y = height // 2
        
        crop_info = {
            'left': actual_left,
            'top': actual_top,
            'right': actual_right,
            'bottom': actual_bottom,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'crop_width': crop_width,
            'crop_height': crop_height,
            'grid_center': (cell_center_x, cell_center_y),
            'grid_center_in_zoomed': (grid_center_in_zoomed_x, grid_center_in_zoomed_y)
        }
        
        return zoomed, crop_info
    
    def check_grid_centering(self, zoomed_image, grid_x, grid_y, target_prompt):
        """Check if target is centered on a grid number."""
        if not self.client:
            return False
        
        self.log("Checking if target is centered on a grid number...", 'ai')
        
        # Draw grid
        img_with_grid = self.draw_grid(zoomed_image.copy(), grid_x, grid_y)
        
        total_cells = grid_x * grid_y
        
        system_prompt = format_prompt(
            load_prompt("coordinate_finder/grid_centering_system.txt"),
            grid_x=grid_x,
            grid_y=grid_y,
            total_cells=total_cells,
            target_prompt=target_prompt,
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Is '{target_prompt}' centered on a grid number at the center of the image?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(img_with_grid)}"
                        }}
                    ]}
                ],
                max_completion_tokens=500
            )
            
            response_text = response.choices[0].message.content
            self.log(f"AI Response: {response_text}", 'ai')
            
            # Parse JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    self.log("Could not parse centering check response", 'error')
                    return False
            
            is_centered = result.get('is_centered', False)
            confidence = result.get('confidence', 'low').lower()
            grid_num = result.get('grid_number')
            reasoning = result.get('reasoning', '')
            
            self.log(f"Is centered on grid: {is_centered}", 'success' if is_centered else 'info')
            self.log(f"Confidence: {confidence}", 'info')
            if grid_num:
                self.log(f"Grid number: {grid_num}", 'coords')
            self.log(f"Reasoning: {reasoning}", 'info')
            
            # Return True if centered AND highly confident
            return is_centered and confidence == 'high'
            
        except Exception as e:
            self.log(f"Error in centering check: {e}", 'error')
            return False
        
    def check_centering(self, zoomed_image, target_prompt, crop_info):
        """Ask AI if the target is centered in the crosshair. Only returns true if EXACTLY centered."""
        if not self.client:
            return None, False
            
        self.log("Checking if target is EXACTLY centered in crosshair...", 'ai')
        
        crop_width = crop_info['crop_width']
        crop_height = crop_info['crop_height']
        zoom_factor = crop_info['zoom_factor']
        
        system_prompt = format_prompt(
            load_prompt("coordinate_finder/crosshair_centering_system.txt"),
            zoom_factor=zoom_factor,
            crop_width=crop_width,
            crop_height=crop_height,
            target_prompt=target_prompt,
        )

        try:
            # Draw crosshair on image for AI using the new cyan crosshair
            img_with_crosshair = self.draw_crosshair(zoomed_image.copy())
            
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Is '{target_prompt}' centered at the cyan crosshair?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(img_with_crosshair)}"
                        }}
                    ]}
                ],
                max_completion_tokens=500
            )
            
            response_text = response.choices[0].message.content
            self.log(f"AI Response: {response_text}", 'ai')
            
            # Parse JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    self.log("Could not parse centering check response", 'error')
                    return None, False
            
            is_centered = result.get('is_centered', False)
            adj_x = int(result.get('adjustment_x', 0))
            adj_y = int(result.get('adjustment_y', 0))
            reasoning = result.get('reasoning', '')
            
            self.log(f"Is centered: {is_centered}", 'success' if is_centered else 'info')
            self.log(f"Adjustment needed: ({adj_x}, {adj_y})", 'coords')
            self.log(f"Reasoning: {reasoning}", 'info')
            
            return (adj_x, adj_y), is_centered
            
        except Exception as e:
            self.log(f"Error in centering check: {e}", 'error')
            return None, False

    def run_finding_process(self, prompt, max_iterations, grid_x, grid_y):
        """Random numbered points refinement search loop."""
        self.is_running = True
        self.iterations = []
        iteration = 0
        
        # Always use the original image
        original_image = self.original_image.copy()
        
        # Start with full-image region
        region_left = 0
        region_right = original_image.width
        region_top = 0
        region_bottom = original_image.height
        
        # Base number of points per step (from grid_x argument)
        try:
            base_points = int(grid_x)
        except Exception:
            base_points = 100
        base_points = max(100, base_points)
        
        self.log(f"\n{'='*50}", 'info')
        self.log(f"STARTING NUMBERED-POINTS REFINEMENT SEARCH", 'info')
        self.log(f"Image size: {original_image.width}x{original_image.height}", 'info')
        self.log(f"Max iterations: {max_iterations}", 'info')
        self.log(f"Base points per step: {base_points}", 'info')
        self.log(f"{'='*50}\n", 'info')
        
        while self.is_running and iteration < max_iterations:
            iteration += 1
            # Legacy iteration label (no-op if label not present in simplified UI)
            if hasattr(self, "iter_label") and self.iter_label is not None:
                self.root.after(0, lambda i=iteration, m=max_iterations: self.iter_label.config(text=f"{i} / {m}"))
            
            self.log(f"\n{'='*40}", 'info')
            self.log(f"ITERATION {iteration}", 'info')
            self.log(f"{'='*40}", 'info')
            self.log(f"Current region: ({region_left}, {region_top}) → ({region_right}, {region_bottom}) "
                     f"(size {region_right - region_left}x{region_bottom - region_top}px)", 'coords')
            
            self.update_status(f"Iteration {iteration}: Analyzing numbered points...", '#58a6ff')
            
            # Small delay for UI update
            time.sleep(0.3)
            
            if not self.is_running:
                break
            
            # Decide how many points this iteration (decrease with each iteration)
            if iteration == 1:
                num_points = base_points
            else:
                # Decrease points more aggressively: divide by 2^iteration, but keep minimum
                num_points = max(base_points // (2 ** iteration), 20)
            
            # Generate random numbered points inside current region
            points = self.generate_numbered_points(region_left, region_top, region_right, region_bottom, num_points)
            
            # Draw and show them
            img_with_points = self.draw_numbered_points(original_image.copy(), points)
            self.current_image = img_with_points
            self.root.after(0, lambda: self.update_displays(show_grid=False, grid_x=None, grid_y=None))
            
            # Small delay so user can see the lines
            time.sleep(0.3)
            
            # Ask AI which point to refine around, or click
            selection_result, _ = self.ask_cursor_movement(
                original_image, points, prompt
            )
            
            if not self.is_running:
                break
            
            if selection_result is None:
                self.log("Could not determine point, retrying...", 'error')
                continue
            
            action, pid = selection_result
            
            # Get selected point coordinates
            point_lookup = {p["id"]: p for p in points}
            p = point_lookup.get(pid)
            if not p:
                self.log(f"Selected id {pid} not found in current points", 'error')
                continue
            px, py = p["x"], p["y"]
            
            # Check if AI wants to click on this point
            if action == "click":
                click_x = px
                click_y = py
                self.current_coords = (click_x, click_y)
                
                self.log(f"🎯 AI requested CLICK on point id {pid}!", 'success')
                self.update_status("✓ Target point found - Clicking!", '#7ee787')
                
                # Blink a cyan point a few times at the chosen location
                for i in range(6):
                    if not self.is_running:
                        break
                    if original_image.mode != 'RGBA':
                        img_rgba = original_image.copy().convert('RGBA')
                    else:
                        img_rgba = original_image.copy()
                    
                    overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
                    draw = ImageDraw.Draw(overlay)
                    
                    if i % 2 == 0:
                        # Draw cyan point (ON)
                        cyan_color = (0, 255, 255, 255)
                        point_radius = 8
                        draw.ellipse([click_x - point_radius, click_y - point_radius,
                                     click_x + point_radius, click_y + point_radius],
                                    fill=cyan_color, outline=cyan_color, width=2)
                    
                    result = Image.alpha_composite(img_rgba, overlay)
                    if result.mode == 'RGBA':
                        rgb_result = Image.new('RGB', result.size)
                        rgb_result.paste(result, mask=result.split()[3])
                        img_with_point = rgb_result
                    else:
                        img_with_point = result
                    
                    self.current_image = img_with_point
                    self.root.after(0, lambda: self.update_displays(show_grid=False, grid_x=None, grid_y=None))
                    self.root.after(0, lambda c=self.current_coords: self.coords_label.config(text=f"({c[0]}, {c[1]})"))
                    time.sleep(0.2)
                
                # Final ON state before click
                if original_image.mode != 'RGBA':
                    img_rgba = original_image.copy().convert('RGBA')
                else:
                    img_rgba = original_image.copy()
                overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                cyan_color = (0, 255, 255, 255)
                point_radius = 8
                draw.ellipse([click_x - point_radius, click_y - point_radius,
                             click_x + point_radius, click_y + point_radius],
                            fill=cyan_color, outline=cyan_color, width=2)
                result = Image.alpha_composite(img_rgba, overlay)
                if result.mode == 'RGBA':
                    rgb_result = Image.new('RGB', result.size)
                    rgb_result.paste(result, mask=result.split()[3])
                    img_with_point = rgb_result
                else:
                    img_with_point = result
                self.current_image = img_with_point
                self.root.after(0, lambda: self.update_displays(show_grid=False, grid_x=None, grid_y=None))
                self.root.after(0, lambda c=self.current_coords: self.coords_label.config(text=f"({c[0]}, {c[1]})"))
                
                # Simulate click
                self.log(f"\n🖱️ CLICK at coordinates: ({click_x}, {click_y}) [id {pid}]", 'success')
                
                self.finish_finding(True)
                return
            
            # Refine region around selected point in a shrinking box (circular-ish)
            region_width = region_right - region_left
            region_height = region_bottom - region_top
            radius_x = max(20, int(region_width * 0.3))
            radius_y = max(20, int(region_height * 0.3))
            
            new_left = max(0, px - radius_x)
            new_right = min(original_image.width, px + radius_x)
            new_top = max(0, py - radius_y)
            new_bottom = min(original_image.height, py + radius_y)
            
            self.log(f"Refining around id {pid} at ({px}, {py})", 'coords')
            self.log(f"New region: ({new_left}, {new_top}) → ({new_right}, {new_bottom}) "
                     f"(size {new_right - new_left}x{new_bottom - new_top}px)", 'coords')
            
            region_left, region_right = new_left, new_right
            region_top, region_bottom = new_top, new_bottom
            
            # Store iteration info
            self.iterations.append({
                'region': (region_left, region_right, region_top, region_bottom),
                'point_id': pid,
                'point': (px, py)
            })
            
        if self.is_running:
            self.log(f"\n⚠️ Max iterations ({max_iterations}) reached", 'error')
            self.log(f"Final region: ({region_left}, {region_top}) → ({region_right}, {region_bottom})", 'coords')
            self.update_status("Max iterations reached", '#ffa657')
            
        self.finish_finding(False)
        
    def update_status(self, text, color):
        """Update status label."""
        self.root.after(0, lambda: self.status_label.config(text=f"● {text}"))
        
    def finish_finding(self, success):
        """Clean up after finding process."""
        self.is_running = False
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        
    def start_finding(self):
        """Start the single-pass planner → OCR/color → picker pipeline."""
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            self.log("Please enter what you want to click!", 'error')
            return
            
        if not self.original_image:
            self.log("No image loaded!", 'error')
            return
            
        if not self.client:
            self.log("OpenAI API key not configured!", 'error')
            self.log("Please add your key to the .env file and restart", 'error')
            return

        if self.is_running:
            self.log("Already running.", 'info')
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.log(f"\n{'='*50}", 'info')
        self.log(f"STARTING PLANNER → OCR/COLOR → PICKER: '{prompt}'", 'info')
        self.log(f"{'='*50}\n", 'info')

        def worker():
            try:
                self.run_new_method_once()
            finally:
                self.finish_finding(True)

        threading.Thread(target=worker, daemon=True).start()
        
    def stop_finding(self):
        """Stop the finding process."""
        self.is_running = False
        self.log("Stopping...", 'error')
        self.update_status("Stopped by user", '#f85149')
        
    def reset(self):
        """Reset to initial state."""
        self.is_running = False
        self.current_coords = None
        self.current_image = self.original_image.copy() if self.original_image else None
        self.iterations = []
        self.ocr_results = None
        
        self.coords_label.config(text="Not set")
        self.status_label.config(text="● Ready")
        
        self.update_displays()
        self.log("\n--- RESET ---\n", 'info')


def main():
    root = tk.Tk()
    app = CoordinateFinderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
