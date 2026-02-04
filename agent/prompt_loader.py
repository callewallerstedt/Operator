from __future__ import annotations

from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(relative_path: str) -> str:
    path = PROMPTS_DIR / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {path}") from None


def format_prompt(template: str, **values: Any) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing prompt placeholder: {missing}") from None
