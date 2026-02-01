"""
Configuration management for the AI Computer Agent.
Put your OpenAI API key in a .env file or set it as an environment variable.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Load .env file if it exists
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_env()


@dataclass
class AgentConfig:
    """Configuration for the AI Agent."""
    
    # OpenAI API Configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = "gpt-4o"  # Vision-capable model
    openai_vision_model: str = "gpt-5.2"  # Separate model for vision clicking (use gpt-5.2 for better precision)
    openai_max_tokens: int = 1024
    
    # Screenshot settings
    screenshot_quality: int = 85  # JPEG quality for compression
    screenshot_scale: float = 1.0  # DPI scaling factor
    
    # OCR settings
    ocr_language: str = "eng"
    ocr_confidence_threshold: float = 50.0  # Lowered for better detection
    
    # Action execution settings
    click_delay: float = 0.3  # Delay after clicks (increased for UI to respond)
    type_delay: float = 0.02  # Delay between keystrokes
    action_timeout: float = 10.0  # Max wait for action verification
    post_action_delay: float = 0.8  # Delay after any action before next step
    
    # Agent loop settings
    max_steps: int = 50  # Maximum steps before stopping
    verification_retries: int = 3  # Retries for action verification
    loop_delay: float = 0.5  # Delay between loop iterations
    
    # Vision click sub-agent settings
    vision_marker_spacing: int = 50  # Pixels between markers in initial pass
    vision_refine_spacing: int = 25  # Pixels between markers in zoomed pass (tighter grid)
    vision_crop_size: int = 300  # Size of crop for refine pass
    vision_max_zoom_steps: int = 2  # Maximum number of zoom steps (1 = one zoom, 2 = two zooms for extra precision)
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found!\n"
                "Please set it in one of these ways:\n"
                "1. Create a .env file in the project root with: OPENAI_API_KEY=your-key-here\n"
                "2. Set the OPENAI_API_KEY environment variable"
            )
        return True


# Global config instance
config = AgentConfig()
