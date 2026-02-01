"""
Smart Vision Click - Uses GPT-5 to directly identify click coordinates.
Uses numbered overlay approach with step-by-step debug mode.
"""

import json
import re
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

from openai import OpenAI

from .config import config
from .screenshot import get_screen_capture


@dataclass
class ClickTarget:
    """A target location to click."""
    x: int
    y: int
    confidence: float
    description: str


class VisionClickEngine:
    """
    Uses GPT-5 to identify click targets using numbered overlays.
    Supports step-by-step debug mode for visualization.
    """
    
    def __init__(self):
        config.validate()
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_vision_model  # Use GPT-5.2 for vision
        self.screen = get_screen_capture()
        
        # Log which model we're using
        print(f"[VisionClickEngine] Using model: {self.model}")
    
    def create_numbered_overlay(
        self,
        image: Image.Image,
        spacing: int = 40
    ) -> Tuple[Image.Image, dict]:
        """
        Create numbered overlay on image with improved visibility.
        
        Returns:
            (overlay_image, number_map) where number_map is {number: (x, y)}
        """
        overlay = image.copy()
        
        # Create a layer for numbers
        number_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(number_layer)
        
        # Use smaller font for less intrusive numbers
        font_size = 11
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("calibri.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        
        # Generate numbers across the screen
        number_map = {}  # number -> (x, y)
        number = 1
        
        # Calculate grid of numbers with offset for better coverage
        cols = image.width // spacing
        rows = image.height // spacing
        
        # Start with a small offset to avoid edges
        offset_x = spacing // 2
        offset_y = spacing // 2
        
        for row in range(rows + 1):
            for col in range(cols + 1):
                x = offset_x + col * spacing
                y = offset_y + row * spacing
                
                # Skip if too close to edges
                if x < 15 or x > image.width - 15 or y < 15 or y > image.height - 15:
                    continue
                
                # Draw number in bright cyan, no outline (cleaner look)
                num_text = str(number)
                text_bbox = draw.textbbox((0, 0), num_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                
                # Center the number
                text_x = x - text_w // 2
                text_y = y - text_h // 2
                
                # Draw bright cyan number (no outline - cleaner)
                draw.text((text_x, text_y), num_text, fill=(0, 255, 255, 255), font=font)
                
                number_map[number] = (x, y)
                number += 1
        
        # Composite the number layer onto the image
        overlay = Image.alpha_composite(
            overlay.convert("RGBA"),
            number_layer
        ).convert("RGB")
        
        return overlay, number_map
    
    def analyze_overlay(
        self,
        overlay_image: Image.Image,
        number_map: dict,
        description: str,
        context: str,
        prefer_primary: bool,
        is_zoomed: bool = False
    ) -> Union[ClickTarget, Tuple[str, Tuple[int, int]], None]:
        """
        Send overlay to AI and get response (click or zoom).
        
        Returns:
            - ClickTarget if AI wants to click
            - ("zoom", (x, y)) if AI wants to zoom
            - None if not found
        """
        overlay_b64 = self.screen.to_base64(overlay_image, format="PNG")
        
        primary_hint = ""
        if prefer_primary:
            primary_hint = """
IMPORTANT: If there are multiple similar buttons, choose the PRIMARY action:
- Usually the more prominent button (bigger, colored, highlighted)
- Usually on the right side or bottom
- "Accept All", "OK", "Confirm" are typically primary
"""
        
        zoom_note = "This is a zoomed-in view. " if is_zoomed else ""
        
        prompt = f"""Look at this screenshot with bright cyan numbers overlaid. Each number marks a clickable point.

{zoom_note}TARGET TO CLICK: "{description}"
{f"Context: {context}" if context else ""}
{primary_hint}

There are {len(number_map)} numbered points on the screen.

**YOUR TASK: Find the number that is CLOSEST TO THE CENTER of the target element.**

**ANALYSIS STEPS:**
1. First, LOCATE the target element in the screenshot (button, link, thumbnail, etc.)
2. Then, find the number that is CLOSEST TO THE CENTER of that element
3. The click will happen EXACTLY at that number's position

**CLICK vs ZOOM DECISION:**
{"**Since this is already a ZOOMED view, you should CLICK now** - pick the number closest to the target center." if is_zoomed else "**PREFER ZOOM** for precision unless you are 100% certain a number is EXACTLY on the target center."}

- **CLICK** (confidence = 1.0 / 100% certain): A number is EXACTLY on the center of the target element
- **ZOOM** (confidence < 1.0): Zoom in for better precision - this gives you MORE numbers to choose from!
- {"After zoom, there will be many more numbers in a tighter grid for precise clicking." if not is_zoomed else ""}

**WHEN TO CLICK (only if 100% sure):**
- A number is directly ON the button text
- A number is clearly in the CENTER of a large clickable area (thumbnail, big button)

**WHEN TO ZOOM (preferred for precision):**
- You can see the element but numbers are spread out around it
- The element is small (icons, small buttons, links)
- You want to be more precise about the exact click location
- Any uncertainty at all - just zoom!

**IMPORTANT:**
- {"This is first pass - prefer ZOOM to get more precise numbers" if not is_zoomed else "This is zoomed view - you should CLICK now with the number closest to center"}
- Buttons can be clicked on their text OR anywhere in the button area
- Thumbnails/images are fully clickable - any point on them works
- When in doubt, ZOOM for better precision

Respond with ONLY this JSON format:
{{"action": "click", "number": <num>, "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}
{{"action": "zoom", "number": <num>, "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}
{{"action": "not_found", "number": 0, "confidence": 0, "reasoning": "Element not visible"}}"""
        
        try:
            # Log the API call
            print(f"[VisionClickEngine] Calling {self.model} with {len(number_map)} points...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=1500,  # Increased to allow for reasoning tokens + response
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{overlay_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Log response info
            if hasattr(response, 'usage') and response.usage:
                print(f"[VisionClickEngine] Tokens: {response.usage.total_tokens} (completion: {response.usage.completion_tokens})")
            
            # Get response content - handle empty or None
            if not response.choices:
                self._last_ai_response = f"ERROR: No choices in response. Response object: {response}"
                self._last_ai_error = "API returned no choices"
                return None
            
            if not response.choices[0].message:
                self._last_ai_response = f"ERROR: No message in choice. Choice: {response.choices[0]}"
                self._last_ai_error = "API returned no message"
                return None
            
            response_text = response.choices[0].message.content
            if response_text is None:
                self._last_ai_response = f"ERROR: Response content is None. Full response: {response}"
                self._last_ai_error = "API returned None for message content"
                return None
            
            response_text = response_text.strip()
            
            # Store response for debugging (will be accessed by GUI)
            self._last_ai_response = response_text if response_text else "(empty response)"
            
            if not response_text:
                self._last_ai_error = f"Response is empty. Full response object: {response}. Model: {response.model if hasattr(response, 'model') else 'unknown'}"
                return None
            
            # Try to extract JSON - multiple patterns
            json_match = None
            
            # Try full JSON object first
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            
            # If that fails, try simpler pattern
            if not json_match:
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            
            # If still no match, try to find JSON in code blocks
            if not json_match:
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if code_block_match:
                    json_match = code_block_match
            
            if json_match:
                json_str = json_match.group(1) if json_match.lastindex else json_match.group()
                try:
                    data = json.loads(json_str)
                    action = data.get("action", "")
                    num = data.get("number", 0)
                    conf = data.get("confidence", 0)
                    reasoning = data.get("reasoning", "")
                    
                    # Store parsed data for debugging
                    self._last_ai_data = {
                        "action": action,
                        "number": num,
                        "confidence": conf,
                        "reasoning": reasoning,
                        "raw_response": response_text
                    }
                    
                    if action == "not_found" or num == 0:
                        return None
                    
                    if num not in number_map:
                        # Number not in map - log this
                        max_num = max(number_map.keys()) if number_map else 0
                        self._last_ai_error = f"Number {num} not found in number_map (available: 1-{max_num})"
                        return None
                    
                    x, y = number_map[num]
                    
                    if action == "click":
                        target = ClickTarget(
                            x=x,
                            y=y,
                            confidence=conf,
                            description=f"{description} (number {num}: {reasoning})"
                        )
                        return target
                    
                    elif action == "zoom":
                        return ("zoom", (x, y))
                except json.JSONDecodeError as e:
                    self._last_ai_error = f"JSON decode error: {str(e)}. JSON string: {json_str[:200]}"
            else:
                # No JSON found
                self._last_ai_error = f"No JSON found in response. Full response: {response_text[:500]}"
        except Exception as e:
            import traceback
            error_msg = f"API call error: {str(e)}\n{traceback.format_exc()}"
            self._last_ai_response = error_msg
            self._last_ai_error = error_msg
        
        return None
    
    def draw_crosshair(
        self,
        image: Image.Image,
        x: int,
        y: int,
        color: str = "green"
    ) -> Image.Image:
        """Draw a crosshair on the image at the specified point."""
        result = image.copy()
        draw = ImageDraw.Draw(result)
        
        # Draw crosshair
        size = 20
        width = 4
        draw.line([(x-size, y), (x+size, y)], fill=color, width=width)
        draw.line([(x, y-size), (x, y+size)], fill=color, width=width)
        draw.ellipse([(x-8, y-8), (x+8, y+8)], outline=color, width=3)
        
        return result
    
    def find_element_step_by_step(
        self,
        screenshot: Image.Image,
        description: str,
        context: str = "",
        prefer_primary: bool = True,
        step: int = 0,
        zoom_point: Optional[Tuple[int, int]] = None,
        zoom_offset: Tuple[int, int] = (0, 0),
        zoom_scale: float = 1.0,
        spacing: int = 40,
        zoom_spacing: int = 60
    ) -> Tuple[Optional[Union[ClickTarget, Tuple[str, Tuple[int, int]]]], Optional[Image.Image], dict]:
        """
        Step-by-step element finding for debug mode.
        
        Args:
            step: 0 = show overlay, 1 = analyze, 2 = show zoomed overlay (if zoomed), 3 = analyze zoomed
        
        Returns:
            (result, overlay_image, number_map)
            result can be ClickTarget, ("zoom", (x,y)), or None
        """
        if step == 0:
            # Step 0: Show initial overlay
            overlay, number_map = self.create_numbered_overlay(screenshot, spacing=spacing)
            return (None, overlay, number_map)
        
        elif step == 1:
            # Step 1: Analyze initial overlay
            overlay, number_map = self.create_numbered_overlay(screenshot, spacing=spacing)
            result = self.analyze_overlay(overlay, number_map, description, context, prefer_primary, is_zoomed=False)
            
            if isinstance(result, ClickTarget):
                # Click - draw green crosshair
                overlay_with_crosshair = self.draw_crosshair(overlay, result.x, result.y, "green")
                return (result, overlay_with_crosshair, number_map)
            elif isinstance(result, tuple) and result[0] == "zoom":
                # Zoom - draw red crosshair
                zoom_x, zoom_y = result[1]
                overlay_with_crosshair = self.draw_crosshair(overlay, zoom_x, zoom_y, "red")
                return (result, overlay_with_crosshair, number_map)
            else:
                return (None, overlay, number_map)
        
        elif step == 2 and zoom_point:
            # Step 2: Show zoomed overlay
            zoom_x, zoom_y = zoom_point
            
            # Zoom 170% around that point
            zoom_factor = 1.7
            crop_size = int(min(screenshot.width, screenshot.height) / zoom_factor)
            
            # Calculate crop bounds
            left = max(0, zoom_x - crop_size // 2)
            top = max(0, zoom_y - crop_size // 2)
            right = min(screenshot.width, zoom_x + crop_size // 2)
            bottom = min(screenshot.height, zoom_y + crop_size // 2)
            
            # Crop and resize to zoom
            zoomed_region = screenshot.crop((left, top, right, bottom))
            zoomed_width = int(zoomed_region.width * zoom_factor)
            zoomed_height = int(zoomed_region.height * zoom_factor)
            zoomed_image = zoomed_region.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
            
            # Create overlay on zoomed image with configurable spacing
            overlay, number_map = self.create_numbered_overlay(zoomed_image, spacing=zoom_spacing)
            return (None, overlay, number_map)
        
        elif step == 3 and zoom_point:
            # Step 3: Analyze zoomed overlay
            zoom_x, zoom_y = zoom_point
            
            # Zoom 170% around that point
            zoom_factor = 1.7
            crop_size = int(min(screenshot.width, screenshot.height) / zoom_factor)
            
            # Calculate crop bounds
            left = max(0, zoom_x - crop_size // 2)
            top = max(0, zoom_y - crop_size // 2)
            right = min(screenshot.width, zoom_x + crop_size // 2)
            bottom = min(screenshot.height, zoom_y + crop_size // 2)
            
            # Crop and resize to zoom
            zoomed_region = screenshot.crop((left, top, right, bottom))
            zoomed_width = int(zoomed_region.width * zoom_factor)
            zoomed_height = int(zoomed_region.height * zoom_factor)
            zoomed_image = zoomed_region.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
            
            # Create overlay and analyze with configurable spacing
            overlay, number_map = self.create_numbered_overlay(zoomed_image, spacing=zoom_spacing)
            result = self.analyze_overlay(overlay, number_map, description, context, prefer_primary, is_zoomed=True)
            
            if isinstance(result, ClickTarget):
                # result.x and result.y are in zoomed image coordinates
                # Store zoomed coordinates for drawing
                zoomed_x = result.x
                zoomed_y = result.y
                
                # Convert from zoomed coordinates back to original screenshot coordinates
                abs_x = int((result.x / zoom_factor) + left)
                abs_y = int((result.y / zoom_factor) + top)
                result.x = abs_x
                result.y = abs_y
                
                # Draw green crosshair on zoomed image using zoomed coordinates
                overlay_with_crosshair = self.draw_crosshair(overlay, zoomed_x, zoomed_y, "green")
                return (result, overlay_with_crosshair, number_map)
            else:
                return (result, overlay, number_map)
        
        return (None, None, {})
    
    def find_element(
        self,
        screenshot: Image.Image,
        description: str,
        context: str = "",
        prefer_primary: bool = True,
        return_debug_images: bool = False,
        image_callback: callable = None,
        status_callback: callable = None,
        zoom_level: int = 0
    ) -> Tuple[Optional[ClickTarget], Optional[List[Image.Image]]]:
        """
        Find a UI element using numbered overlay approach with multi-level zoom capability.
        Uses recursive zoom: click if confident, otherwise zoom and try again (up to max_zoom_steps).
        (Non-debug mode - runs all steps automatically)
        
        Args:
            image_callback: Optional callback(image, info_text) to display images in real-time
            status_callback: Optional callback(status_text) to show current status
            zoom_level: Current zoom level (0 = original, 1 = first zoom, etc.)
        """
        debug_images = [] if return_debug_images else None
        max_zoom_steps = config.vision_max_zoom_steps
        
        # Status update
        zoom_text = f" (Zoom {zoom_level})" if zoom_level > 0 else ""
        if status_callback:
            status_callback(f"Smart Click{zoom_text}: Creating numbered overlay...")
        
        # Determine spacing based on zoom level - tighter grid when zoomed but not too dense
        # Level 0: 50px spacing (initial view)
        # Level 1+: 35px spacing (tighter grid for precision, but not overwhelming)
        spacing = 50 if zoom_level == 0 else 40
        
        # First pass: Overlay numbers all over the screen
        overlay, number_map = self.create_numbered_overlay(screenshot, spacing=spacing)
        if return_debug_images:
            debug_images.append(overlay)
        
        # Show overlay image immediately
        if image_callback:
            image_callback(overlay, f"Smart Click{zoom_text}: {len(number_map)} points - Analyzing...")
        
        if status_callback:
            status_callback(f"Smart Click{zoom_text}: Sending to GPT-5.2 ({len(number_map)} points)...")
        
        result = self.analyze_overlay(overlay, number_map, description, context, prefer_primary, is_zoomed=(zoom_level > 0))
        
        if isinstance(result, ClickTarget):
            # AI is confident - click directly
            if status_callback:
                status_callback(f"Smart Click{zoom_text}: Found target at ({result.x}, {result.y})")
            
            # Draw crosshair on overlay to show where we're clicking
            final_overlay = self.draw_crosshair(overlay.copy(), result.x, result.y, "green")
            if image_callback:
                image_callback(final_overlay, f"Smart Click{zoom_text}: CLICK at ({result.x}, {result.y})")
            if return_debug_images:
                debug_images.append(final_overlay)
            
            return (result, debug_images)
        
        if isinstance(result, tuple) and result[0] == "zoom":
            # AI wants to zoom for more precision
            if zoom_level >= max_zoom_steps:
                # Max zoom steps reached, use the zoom point as click target
                zoom_x, zoom_y = result[1]
                if status_callback:
                    status_callback(f"Smart Click{zoom_text}: Max zoom reached, clicking at ({zoom_x}, {zoom_y})")
                
                target = ClickTarget(
                    x=zoom_x,
                    y=zoom_y,
                    confidence=0.7,
                    description=f"{description} (max zoom reached)"
                )
                return (target, debug_images)
            
            zoom_x, zoom_y = result[1]
            
            if status_callback:
                status_callback(f"Smart Click{zoom_text}: Zooming into ({zoom_x}, {zoom_y})...")
            
            # Show zoom point on current overlay
            zoom_overlay_marked = self.draw_crosshair(overlay.copy(), zoom_x, zoom_y, "red")
            if image_callback:
                image_callback(zoom_overlay_marked, f"Smart Click{zoom_text}: ZOOM into ({zoom_x}, {zoom_y})")
            
            # Zoom 170% around that point
            zoom_factor = 1.7
            crop_size = int(min(screenshot.width, screenshot.height) / zoom_factor)
            
            # Calculate crop bounds
            left = max(0, zoom_x - crop_size // 2)
            top = max(0, zoom_y - crop_size // 2)
            right = min(screenshot.width, zoom_x + crop_size // 2)
            bottom = min(screenshot.height, zoom_y + crop_size // 2)
            
            # Crop and resize to zoom
            zoomed_region = screenshot.crop((left, top, right, bottom))
            zoomed_width = int(zoomed_region.width * zoom_factor)
            zoomed_height = int(zoomed_region.height * zoom_factor)
            zoomed_image = zoomed_region.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
            
            # Recursively call find_element on the zoomed image
            zoom_result, zoom_debug_images = self.find_element(
                zoomed_image,
                description,
                context,
                prefer_primary,
                return_debug_images,
                image_callback,
                status_callback,
                zoom_level + 1
            )
            
            if zoom_result:
                # Convert from zoomed coordinates back to original screenshot coordinates
                # Need to account for all zoom levels
                total_zoom_factor = zoom_factor ** (zoom_level + 1)
                abs_x = int((zoom_result.x / zoom_factor) + left)
                abs_y = int((zoom_result.y / zoom_factor) + top)
                zoom_result.x = abs_x
                zoom_result.y = abs_y
                
                if status_callback:
                    status_callback(f"Smart Click: Found target at ({abs_x}, {abs_y})")
                
                # Add zoom debug images to our list
                if return_debug_images and zoom_debug_images:
                    debug_images.extend(zoom_debug_images)
                
                return (zoom_result, debug_images)
            else:
                if status_callback:
                    status_callback(f"Smart Click{zoom_text}: Could not find target in zoomed view")
        else:
            if status_callback:
                status_callback(f"Smart Click{zoom_text}: Could not find target")
        
        return (None, debug_images)
    
    def find_and_click(
        self,
        screenshot: Image.Image,
        description: str,
        context: str = "",
        prefer_primary: bool = True,
        return_debug_images: bool = False
    ) -> Tuple[bool, Optional[ClickTarget], str, Optional[List[Image.Image]]]:
        """
        Find element and return click coordinates.
        
        Returns:
            (success, target, message, debug_images)
        """
        result = self.find_element(screenshot, description, context, prefer_primary, return_debug_images)
        target, debug_images = result
        
        if target and target.confidence >= 0.5:
            return (True, target, f"Found: {target.description} at ({target.x}, {target.y})", debug_images)
        elif target:
            return (True, target, f"Low confidence match at ({target.x}, {target.y})", debug_images)
        else:
            return (False, None, f"Could not find: {description}", debug_images)


# Singleton
_vision_click: Optional[VisionClickEngine] = None


def get_vision_click() -> VisionClickEngine:
    """Get or create the vision click singleton."""
    global _vision_click
    if _vision_click is None:
        _vision_click = VisionClickEngine()
    return _vision_click
