"""
Headless click pipeline that reuses the coordinate finder test logic
to choose a click point from a screenshot and a target description.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Callable

from openai import OpenAI
from PIL import Image

from .config import config
from .screenshot import get_screen_capture

# Reuse the deterministic + picker pipeline from the test harness.
from coordinate_finder_test import (
    CoordinateFinderApp,
    NewMethodSnapshot,
    CandidatePackage,
    PlannerOutput,
)


class _BoolVar:
    def __init__(self, value: bool = True) -> None:
        self._value = bool(value)

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        self._value = bool(value)


class HeadlessClickPipeline(CoordinateFinderApp):
    """
    Minimal, headless wrapper around CoordinateFinderApp's pipeline methods.
    Avoids any UI/Tk calls while keeping the full candidate + picker logic.
    """

    def __init__(
        self,
        client: OpenAI,
        screen_size: Optional[Tuple[int, int]] = None,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        # Do NOT call super().__init__ (it builds a full Tk UI).
        self.client = client
        self._log_callback = log_callback
        self._screen_size = screen_size

        # Core state used by pipeline
        self.original_image: Optional[Image.Image] = None
        self.current_image: Optional[Image.Image] = None
        self.ocr_results = None
        self.ocr_fast_mode = True
        self.use_paddleocr = False
        self.paddleocr_ocr = None
        self.ocr_engine = None

        # ROI + mask settings
        self.roi_settings_path = os.path.join(os.getcwd(), "roi_settings.json")
        self.nm_roi_settings = self._nm_load_roi_settings()
        self.nm_color_settings = self._nm_color_settings_defaults()
        self.nm_color_split_keep = 2

        # UI-backed entries (unused in headless; defaults are used)
        self.nm_color_mask_entry = None
        self.nm_color_max_area_entry = None
        self.nm_color_min_area_entry = None
        self.nm_color_split_entry = None
        self.nm_ocr_match_entry = None

        # Toggle vars (use full pipeline by default)
        self.nm_use_ocr_var = _BoolVar(True)
        self.nm_use_color_var = _BoolVar(True)
        self.nm_use_shape_var = _BoolVar(True)

        # Disable stage rendering in headless mode
        self._nm_stage_add_disabled = True
        self.nm_stages = []
        self.nm_stage_index = 0

    def set_log_callback(self, callback: Optional[Callable[[str, str], None]]) -> None:
        self._log_callback = callback

    def log(self, message, tag="info"):
        if self._log_callback:
            self._log_callback(str(message), str(tag))

    # Override UI/preview methods as no-ops for headless use.
    def _nm_add_stage(self, name: str, image: Image.Image):
        return

    def _nm_reset_stages(self):
        return

    def _nm_show_stage(self, index: int):
        return

    def update_displays(self, *args, **kwargs):
        return

    def display_image(self, *args, **kwargs):
        return

    def _nm_capture_snapshot(self, user_instruction: str) -> Optional[NewMethodSnapshot]:
        """Headless snapshot using only the provided image."""
        try:
            if self.original_image is None:
                return None
            img = self.original_image.copy()
            w, h = img.size
            screen_resolution = self._screen_size or (w, h)
            dpi_scaling = 1.0
            window_rect = (0, 0, w, h)
            window_title = "Agent screen"
            cursor_position = (w // 2, h // 2)
            return NewMethodSnapshot(
                image=img,
                screen_resolution=screen_resolution,
                dpi_scaling=dpi_scaling,
                window_rect=window_rect,
                window_title=window_title,
                cursor_position=cursor_position,
                user_instruction=user_instruction,
            )
        except Exception as exc:
            self.log(f"Snapshot error: {exc}", "error")
            return None

    def find_click(
        self,
        screenshot: Image.Image,
        instruction: str,
    ) -> Optional[Tuple[int, int]]:
        """Run the full planner -> deterministic -> picker pipeline and return click coordinates."""
        if not instruction:
            return None

        self.original_image = screenshot
        snapshot = self._nm_capture_snapshot(instruction)
        if snapshot is None:
            return None

        planner = self._nm_planner_ai(snapshot)
        if planner is None or planner.decision.upper() == "UNSURE":
            return None

        candidates = self._nm_generate_candidates(snapshot, planner)
        if not candidates:
            return None

        choice_id = self._nm_picker_ai(snapshot, planner, candidates)
        if choice_id is None:
            return None

        chosen = next((c for c in candidates if c.id == choice_id), None)
        if not chosen:
            return None

        return chosen.click_point


_pipeline_instance: Optional[HeadlessClickPipeline] = None


def get_click_pipeline() -> HeadlessClickPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured for the click pipeline.")
        client = OpenAI(api_key=config.openai_api_key)
        screen_size = get_screen_capture().get_screen_size()
        _pipeline_instance = HeadlessClickPipeline(client=client, screen_size=screen_size)
    return _pipeline_instance
