"""
OCR module for text detection and localization.
Uses Tesseract for reliable text extraction with bounding boxes.
"""

import re
import os
import io
from typing import List, Optional, Tuple, Protocol
from dataclasses import dataclass
from difflib import SequenceMatcher

from PIL import Image
import pytesseract

from .config import config
from .screenshot import ScreenRegion


class BaseOCREngine(Protocol):
    def process(
        self,
        image: Image.Image,
        offset: Tuple[int, int] = (0, 0),
        include_phrases: bool = True,
    ) -> "OCRResult":
        ...

    def process_with_preprocessing(
        self,
        image: Image.Image,
        offset: Tuple[int, int] = (0, 0),
        include_phrases: bool = True,
    ) -> "OCRResult":
        ...


@dataclass
class OCRMatch:
    """Represents a text match found by OCR."""
    text: str
    confidence: float
    bbox: ScreenRegion
    source: str = "word"  # "word" | "phrase"
    
    @property
    def center(self) -> Tuple[int, int]:
        return self.bbox.center


@dataclass 
class OCRResult:
    """Full OCR result with all detected text."""
    matches: List[OCRMatch]
    raw_text: str

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _compact_spaced_letters(text: str) -> Optional[str]:
        tokens = [t for t in text.split() if t]
        if len(tokens) >= 2 and all(len(t) == 1 for t in tokens):
            return "".join(tokens)
        return None

    @classmethod
    def _variants_for_match(cls, text: str) -> List[str]:
        norm = cls._normalize_text(text)
        variants = {norm} if norm else set()
        compact = cls._compact_spaced_letters(norm)
        if compact:
            variants.add(compact)
        return list(variants)
    
    def find_text(
        self,
        query: str,
        exact: bool = False,
        occurrence: int = 1,
        min_confidence: float = None
    ) -> Optional[OCRMatch]:
        """
        Find text in the OCR results.
        
        Args:
            query: Text to search for
            exact: If True, require exact match; if False, allow fuzzy matching
            occurrence: Which occurrence to return (1-based)
            min_confidence: Minimum confidence threshold (defaults to config value)
        
        Returns:
            OCRMatch if found, None otherwise
        """
        """
        Find text in the OCR results.
        
        Args:
            query: Text to search for
            exact: If True, require exact match; if False, allow fuzzy matching
            occurrence: Which occurrence to return (1-based)
            min_confidence: Minimum confidence threshold
        
        Returns:
            OCRMatch if found, None otherwise
        """
        min_conf = min_confidence or config.ocr_confidence_threshold
        matches = []
        
        for match in self.matches:
            if match.confidence < min_conf:
                continue
            
            if exact:
                if match.text.strip().lower() == query.strip().lower():
                    matches.append(match)
            else:
                # Fuzzy matching with better phrase detection
                match_text = match.text.strip().lower()
                query_text = query.strip().lower()

                # Normalize and handle spaced-letter logos (e.g., "T E S L A")
                match_variants = self._variants_for_match(match_text)
                query_variants = self._variants_for_match(query_text)
                if match_variants and query_variants:
                    matched = False
                    for qv in query_variants:
                        for mv in match_variants:
                            if not qv or not mv:
                                continue
                            if qv == mv:
                                matches.append((match, 98))
                                matched = True
                                break
                            if len(qv) >= 3 and (qv in mv or mv in qv):
                                matches.append((match, 92))
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        continue
                
                # Skip single character matches when searching for multi-word phrases
                query_word_count = len(query_text.split())
                if query_word_count > 1 and len(match_text) <= 2:
                    continue  # Skip single chars when looking for phrases
                
                # Direct substring match (most reliable) - prioritize longer matches
                if query_text in match_text:
                    matches.append((match, 100))  # High priority
                    continue
                if match_text in query_text and len(match_text) >= 3:
                    matches.append((match, 90))  # High priority
                    continue
                
                # Exact word match (for single words)
                if query_word_count == 1 and match_text == query_text:
                    matches.append((match, 95))
                    continue
                
                # Multi-word phrase matching
                if query_word_count > 1:
                    query_words = set(query_text.split())
                    match_words = set(match_text.split())
                    # Check if all query words are in match (or vice versa)
                    if query_words.issubset(match_words) or match_words.issubset(query_words):
                        # Prefer longer matches
                        score = 80 + min(len(match_text), 20)
                        matches.append((match, score))
                        continue
                    # Check if most words match
                    common_words = query_words.intersection(match_words)
                    if len(common_words) >= len(query_words) * 0.7:  # 70% of words match
                        score = 70 + len(common_words) * 5
                        matches.append((match, score))
                        continue
                
                # Fuzzy ratio matching - only for longer text
                if len(match_text) >= 3 and len(query_text) >= 3:
                    ratio = SequenceMatcher(None, match_text, query_text).ratio()
                    if ratio > 0.7:  # Higher threshold for fuzzy
                        score = int(ratio * 60)  # Lower priority than exact matches
                        matches.append((match, score))
                        continue
                
                # Word-level matching for single words
                if query_word_count == 1:
                    query_word = query_text
                    if len(query_word) > 2 and query_word in match_text:
                        matches.append((match, 50))
                        continue
                    # Fuzzy match for single words
                    ratio = SequenceMatcher(None, match_text, query_word).ratio()
                    if ratio > 0.75:
                        matches.append((match, int(ratio * 50)))
        
        # Sort by score (highest first) and return the requested occurrence
        if matches:
            # Extract matches and sort by score
            scored_matches = sorted(matches, key=lambda x: x[1], reverse=True)
            actual_matches = [m[0] for m in scored_matches]
            
            if occurrence <= len(actual_matches):
                return actual_matches[occurrence - 1]
        
        if matches and occurrence <= len(matches):
            return matches[occurrence - 1]
        return None
    
    def find_all(
        self,
        query: str,
        exact: bool = False,
        min_confidence: float = None
    ) -> List[OCRMatch]:
        """Find all occurrences of text."""
        min_conf = min_confidence or config.ocr_confidence_threshold
        matches = []
        query_variants = self._variants_for_match(query.strip().lower())
        
        for match in self.matches:
            if match.confidence < min_conf:
                continue
            
            if exact:
                if match.text.strip().lower() == query.strip().lower():
                    matches.append(match)
            else:
                match_text = match.text.strip().lower()
                match_variants = self._variants_for_match(match_text)
                direct_hit = False
                if query_variants and match_variants:
                    for qv in query_variants:
                        for mv in match_variants:
                            if not qv or not mv:
                                continue
                            if qv == mv or (len(qv) >= 3 and (qv in mv or mv in qv)):
                                direct_hit = True
                                break
                        if direct_hit:
                            break

                ratio = SequenceMatcher(
                    None,
                    match_text,
                    query.strip().lower()
                ).ratio()
                if direct_hit or ratio > 0.7 or query.lower() in match.text.lower():
                    matches.append(match)
        
        return matches
    
    def find_nearest(
        self,
        query: str,
        reference_point: Tuple[int, int],
        min_confidence: float = None
    ) -> Optional[OCRMatch]:
        """Find the occurrence nearest to a reference point."""
        matches = self.find_all(query, min_confidence=min_confidence)
        if not matches:
            return None
        
        def distance(match: OCRMatch) -> float:
            cx, cy = match.center
            rx, ry = reference_point
            return ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
        
        return min(matches, key=distance)


class OCREngine:
    """Tesseract-based OCR engine."""
    
    def __init__(self):
        # Support multiple languages - add Swedish for better detection
        self.language = config.ocr_language
        # Try to use Swedish + English if available
        self.languages = ["swe", "eng"]  # Swedish + English
        self._verify_tesseract()
    
    def _verify_tesseract(self):
        """Verify Tesseract is installed and accessible."""
        # Try common installation paths on Windows
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\calle\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
            "tesseract.exe"  # In case it's already in PATH
        ]

        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        else:
            # If none of the paths exist, try to use the default and see if it works
            pass

        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                "Tesseract OCR is not installed or not found.\n"
                "Please install Tesseract:\n"
                "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  The agent will automatically find it if installed in standard locations."
            ) from e

    def _select_language(self) -> str:
        """Pick the best available language combo for OCR."""
        lang_to_use = (self.language or "").strip()
        try:
            available = set(pytesseract.get_languages(config=""))
            if lang_to_use:
                parts = [p.strip() for p in lang_to_use.replace(",", "+").split("+") if p.strip()]
                if parts and all(p in available for p in parts):
                    return "+".join(parts)
            if "swe" in available and "eng" in available:
                lang_to_use = "swe+eng"
            elif "swe" in available:
                lang_to_use = "swe"
            elif "eng" in available:
                lang_to_use = "eng"
        except Exception:
            lang_to_use = self.language
        return lang_to_use

    def _binarize_otsu(self, gray: Image.Image, fallback: int = 160) -> Image.Image:
        """Binarize a grayscale image using Otsu thresholding (with fallback)."""
        try:
            import numpy as np
        except Exception:
            return gray.point(lambda p: 255 if p > fallback else 0)
        arr = np.array(gray)
        if arr.size == 0:
            return gray
        hist = np.bincount(arr.ravel(), minlength=256).astype(float)
        total = float(arr.size)
        sum_total = float((np.arange(256) * hist).sum())
        sum_b = 0.0
        w_b = 0.0
        var_max = 0.0
        threshold = fallback
        for i in range(256):
            w_b += hist[i]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += i * hist[i]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > var_max:
                var_max = var_between
                threshold = i
        return gray.point(lambda p: 255 if p > threshold else 0)
    
    def process(
        self,
        image: Image.Image,
        offset: Tuple[int, int] = (0, 0),
        include_phrases: bool = True
    ) -> OCRResult:
        """
        Process an image and extract text with bounding boxes.
        
        Args:
            image: PIL Image to process
            offset: (x, y) offset to add to all coordinates (for region captures)
            include_phrases: If True, add combined multi-word phrase boxes
        
        Returns:
            OCRResult with all detected text and positions
        """
        lang_to_use = self._select_language()
        
        image_rgb = image.convert("RGB")
        # Get detailed OCR data - use word-level detection
        data = pytesseract.image_to_data(
            image_rgb,
            lang=lang_to_use,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6 -c preserve_interword_spaces=1"  # Accurate LSTM + preserve spacing
        )
        
        matches = []
        n_boxes = len(data["level"])

        def looks_like_text(text: str) -> bool:
            if not text:
                return False
            allowed_punct = set(" -_.:,/()[]+'\"&@%")
            swedish_chars = "\u00e5\u00e4\u00f6\u00c5\u00c4\u00d6"
            for char in text:
                if char.isalnum() or char.isspace() or char in allowed_punct or char in swedish_chars:
                    continue
                return False
            return any(c.isalnum() or c in swedish_chars for c in text)

        def min_conf_for_text(text: str, base: float = 0.0) -> float:
            t = text.strip()
            if not t:
                return 100.0
            length = len(t)
            if length <= 2:
                if any(ch.isdigit() for ch in t):
                    return max(25.0, base)
                if len(t) == 1 and t.isalpha() and t.isupper():
                    # Allow single-letter logo glyphs (e.g., T E S L A)
                    return max(20.0, base)
                return max(35.0, base)
            if length <= 4:
                return max(25.0, base)
            return max(15.0, base)

        def extract_word_matches(data_dict, min_conf_base: float = 0.0) -> List[OCRMatch]:
            word_matches_local = []
            for i in range(len(data_dict["level"])):
                text = data_dict["text"][i].strip()
                conf = float(data_dict["conf"][i])

                if text and conf > 0 and len(text) > 0:  # Skip empty strings and invalid confidence
                    bbox = ScreenRegion(
                        left=data_dict["left"][i] + offset[0],
                        top=data_dict["top"][i] + offset[1],
                        width=data_dict["width"][i],
                        height=data_dict["height"][i]
                    )
                    if bbox.width < 4 or bbox.height < 4:
                        continue
                    if not looks_like_text(text):
                        continue
                    if conf < min_conf_for_text(text, base=min_conf_base):
                        continue
                    word_matches_local.append(OCRMatch(text=text, confidence=conf, bbox=bbox, source="word"))
            return word_matches_local

        def merge_matches(primary: List[OCRMatch], extra: List[OCRMatch]) -> List[OCRMatch]:
            merged = {}
            def key_for(m: OCRMatch) -> Tuple[str, int, int, int, int]:
                return (
                    m.text.strip().lower(),
                    m.bbox.left // 3,
                    m.bbox.top // 3,
                    m.bbox.right // 3,
                    m.bbox.bottom // 3,
                )
            for m in primary:
                merged[key_for(m)] = m
            for m in extra:
                k = key_for(m)
                if k not in merged or m.confidence > merged[k].confidence:
                    merged[k] = m
            return list(merged.values())

        # First pass: collect individual word matches
        word_matches = extract_word_matches(data, min_conf_base=0.0)

        # If recall is low, add a couple of stronger-but-cleaner passes
        try:
            from PIL import ImageEnhance, ImageFilter, ImageOps

            if len(word_matches) < 80:
                # Low-contrast boost (white on color backgrounds)
                enhanced = ImageOps.autocontrast(image_rgb, cutoff=1)
                enhanced = ImageEnhance.Contrast(enhanced).enhance(1.6)
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=140, threshold=3))
                data_enhanced = pytesseract.image_to_data(
                    enhanced,
                    lang=lang_to_use,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
                )
                word_matches = merge_matches(word_matches, extract_word_matches(data_enhanced, min_conf_base=10.0))

            if len(word_matches) < 60:
                # Inversion pass for light text on dark-ish backgrounds
                gray = image_rgb.convert("L")
                inv = ImageOps.invert(gray)
                inv = ImageOps.autocontrast(inv, cutoff=1)
                inv = ImageEnhance.Contrast(inv).enhance(1.4)
                inv_rgb = inv.convert("RGB")
                data_inv = pytesseract.image_to_data(
                    inv_rgb,
                    lang=lang_to_use,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
                )
                word_matches = merge_matches(word_matches, extract_word_matches(data_inv, min_conf_base=12.0))

            if len(word_matches) < 50:
                # V-channel binarization for light text on dark/color backgrounds
                hsv = image_rgb.convert("HSV")
                v = hsv.split()[2]
                v = ImageOps.autocontrast(v, cutoff=2)
                v = ImageEnhance.Contrast(v).enhance(1.8)
                v = v.filter(ImageFilter.UnsharpMask(radius=1, percent=130, threshold=3))
                bw_v = self._binarize_otsu(v, fallback=165)
                data_v = pytesseract.image_to_data(
                    bw_v,
                    lang=lang_to_use,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
                )
                word_matches = merge_matches(word_matches, extract_word_matches(data_v, min_conf_base=8.0))

            if len(word_matches) < 45:
                # Low-saturation bright text (e.g., white on blue buttons)
                try:
                    import numpy as np
                except Exception:
                    np = None
                if np is not None:
                    hsv = image_rgb.convert("HSV")
                    h, s, v = hsv.split()
                    s_arr = np.array(s)
                    v_arr = np.array(v)
                    # Bright + low saturation tends to capture white/light text
                    mask = (v_arr > 150) & (s_arr < 90)
                    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
                    # Thicken thin glyphs a bit
                    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
                    bw_ls = ImageOps.invert(mask_img)  # black text on white
                    data_ls = pytesseract.image_to_data(
                        bw_ls,
                        lang=lang_to_use,
                        output_type=pytesseract.Output.DICT,
                        config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
                    )
                    word_matches = merge_matches(word_matches, extract_word_matches(data_ls, min_conf_base=6.0))

            if len(word_matches) < 40:
                # Aggressive inverted-binary pass for very low-contrast light text
                gray = image_rgb.convert("L")
                inv = ImageOps.invert(gray)
                inv = ImageOps.autocontrast(inv, cutoff=1)
                inv = ImageEnhance.Contrast(inv).enhance(1.6)
                bw_inv = self._binarize_otsu(inv, fallback=160)
                bw_inv = bw_inv.filter(ImageFilter.MaxFilter(3))
                data_inv_bw = pytesseract.image_to_data(
                    bw_inv,
                    lang=lang_to_use,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
                )
                word_matches = merge_matches(word_matches, extract_word_matches(data_inv_bw, min_conf_base=6.0))
        except Exception:
            pass
        
        # Second pass: combine nearby words into phrases
        # This helps match multi-word text like "Godkänn alla"
        if word_matches:
            # Sort by position (top to bottom, left to right)
            word_matches.sort(key=lambda m: (m.bbox.top, m.bbox.left))
            
            # Add individual words
            matches.extend(word_matches)
            
            if include_phrases:
                # Combine adjacent words on the same line into phrases
                i = 0
                while i < len(word_matches):
                    current_line = []
                    current_top = word_matches[i].bbox.top
                    line_tolerance = word_matches[i].bbox.height * 0.5  # Words on same line
                    
                    # Collect words on the same line
                    j = i
                    while j < len(word_matches):
                        match = word_matches[j]
                        if abs(match.bbox.top - current_top) <= line_tolerance:
                            current_line.append(match)
                            j += 1
                        else:
                            break
                    
                    # Create phrases from 2+ word combinations
                    if len(current_line) >= 2:
                        max_phrase_words = 3
                        if all(len(w.text.strip()) == 1 for w in current_line):
                            max_phrase_words = min(8, len(current_line))
                        for start in range(len(current_line)):
                            for end in range(start + 1, min(start + 1 + max_phrase_words, len(current_line) + 1)):
                                phrase_words = current_line[start:end]
                                if len(phrase_words) >= 2:
                                    # Combine text
                                    phrase_text = " ".join([w.text for w in phrase_words])
                                    
                                    # Calculate combined bounding box
                                    left = min(w.bbox.left for w in phrase_words)
                                    top = min(w.bbox.top for w in phrase_words)
                                    right = max(w.bbox.right for w in phrase_words)
                                    bottom = max(w.bbox.bottom for w in phrase_words)
                                    
                                    # Average confidence
                                    avg_conf = sum(w.confidence for w in phrase_words) / len(phrase_words)
                                    
                                    # Only add if it's a meaningful phrase (not just spaces)
                                    if len(phrase_text.strip()) > 0:
                                        phrase_bbox = ScreenRegion(
                                            left=left,
                                            top=top,
                                            width=right - left,
                                            height=bottom - top
                                        )
                                        matches.append(OCRMatch(
                                            text=phrase_text,
                                            confidence=avg_conf,
                                            bbox=phrase_bbox,
                                            source="phrase"
                                        ))
                    
                    i = j
        
        # Also get the full text for context
        raw_text = pytesseract.image_to_string(
            image_rgb, lang=lang_to_use, config="--oem 3 --psm 6"
        )
        
        return OCRResult(matches=matches, raw_text=raw_text)
    
    def process_with_preprocessing(
        self,
        image: Image.Image,
        offset: Tuple[int, int] = (0, 0),
        include_phrases: bool = True
    ) -> OCRResult:
        """Process image with preprocessing for better accuracy."""
        from PIL import ImageEnhance, ImageFilter, ImageOps, ImageStat

        # Convert to grayscale and upscale to preserve small UI text
        gray = image.convert("L")
        w, h = gray.size
        scale = 1
        if w < 1200 or h < 800:
            scale = 2
        if w < 800 or h < 600:
            scale = 3
        # Large screenshots still contain small UI text; upscale for better recall.
        if scale == 1 and (w >= 1400 or h >= 900):
            scale = 2
        upscaled = gray.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

        # Boost contrast and normalize
        normalized = ImageOps.autocontrast(upscaled, cutoff=2)
        enhanced = ImageEnhance.Contrast(normalized).enhance(2.0)

        # Sharpen edges for crisp text
        sharpened = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))

        # Light binarization to separate text from background (adaptive)
        bw = self._binarize_otsu(sharpened, fallback=165)
        try:
            avg = ImageStat.Stat(sharpened).mean[0]
            if avg < 120:
                # Dark background -> invert so text becomes dark on light
                bw = ImageOps.invert(bw)
        except Exception:
            pass

        # Run OCR directly so we can scale boxes back down to original coords
        lang_to_use = self._select_language()
        data = pytesseract.image_to_data(
            bw,
            lang=lang_to_use,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6 -c preserve_interword_spaces=1"
        )

        def looks_like_text(text: str) -> bool:
            if not text:
                return False
            allowed_punct = set(" -_.:,/()[]+'\"&@%")
            swedish_chars = "\u00e5\u00e4\u00f6\u00c5\u00c4\u00d6"
            for char in text:
                if char.isalnum() or char.isspace() or char in allowed_punct or char in swedish_chars:
                    continue
                return False
            return any(c.isalnum() or c in swedish_chars for c in text)

        word_matches = []
        n_boxes = len(data["level"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text and conf > 0 and len(text) > 0:
                left = int(data["left"][i] / scale) + offset[0]
                top = int(data["top"][i] / scale) + offset[1]
                width = int(data["width"][i] / scale)
                height = int(data["height"][i] / scale)
                bbox = ScreenRegion(left=left, top=top, width=width, height=height)
                if bbox.width < 4 or bbox.height < 4:
                    continue
                if not looks_like_text(text):
                    continue
                word_matches.append(OCRMatch(text=text, confidence=conf, bbox=bbox, source="word"))

        matches = []
        if word_matches:
            word_matches.sort(key=lambda m: (m.bbox.top, m.bbox.left))
            matches.extend(word_matches)

            if include_phrases:
                i = 0
                while i < len(word_matches):
                    current_line = []
                    current_top = word_matches[i].bbox.top
                    line_tolerance = word_matches[i].bbox.height * 0.5

                    j = i
                    while j < len(word_matches):
                        match = word_matches[j]
                        if abs(match.bbox.top - current_top) <= line_tolerance:
                            current_line.append(match)
                            j += 1
                        else:
                            break

                    if len(current_line) >= 2:
                        max_phrase_words = 3
                        if all(len(w.text.strip()) == 1 for w in current_line):
                            max_phrase_words = min(8, len(current_line))
                        for start in range(len(current_line)):
                            for end in range(start + 1, min(start + 1 + max_phrase_words, len(current_line) + 1)):
                                phrase_words = current_line[start:end]
                                if len(phrase_words) >= 2:
                                    phrase_text = " ".join([w.text for w in phrase_words])
                                    left = min(w.bbox.left for w in phrase_words)
                                    top = min(w.bbox.top for w in phrase_words)
                                    right = max(w.bbox.right for w in phrase_words)
                                    bottom = max(w.bbox.bottom for w in phrase_words)
                                    avg_conf = sum(w.confidence for w in phrase_words) / len(phrase_words)
                                    if len(phrase_text.strip()) > 0:
                                        phrase_bbox = ScreenRegion(
                                            left=left,
                                            top=top,
                                            width=right - left,
                                            height=bottom - top
                                        )
                                        matches.append(OCRMatch(
                                            text=phrase_text,
                                            confidence=avg_conf,
                                            bbox=phrase_bbox,
                                            source="phrase"
                                        ))

                    i = j

        raw_text = pytesseract.image_to_string(
            bw, lang=lang_to_use, config="--oem 3 --psm 6"
        )

        return OCRResult(matches=matches, raw_text=raw_text)


# Singleton instance
class EasyOCREngine:
    """EasyOCR-based OCR engine (free, no Tesseract dependency)."""

    def __init__(self):
        # Avoid Unicode progress bar issues on Windows consoles.
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("PYTHONUTF8", "1")
        self.language = config.ocr_language
        self.languages = self._select_languages()
        self._reader = self._init_reader()

    def _select_languages(self) -> List[str]:
        lang_str = (self.language or "").strip()
        if not lang_str:
            return ["sv", "en"]
        parts = re.split(r"[,+\s]+", lang_str.lower())
        mapping = {
            "swe": "sv",
            "sv": "sv",
            "eng": "en",
            "en": "en",
        }
        langs: List[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            code = mapping.get(p, p)
            if code not in langs:
                langs.append(code)
        if not langs:
            langs = ["sv", "en"]
        return langs

    def _init_reader(self):
        try:
            import easyocr
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "EasyOCR is not installed. Install with: python -m pip install easyocr"
            ) from exc
        gpu_setting = (config.easyocr_gpu or "").strip().lower()
        if gpu_setting in ("1", "true", "yes", "on"):
            use_gpu = True
        elif gpu_setting in ("0", "false", "no", "off"):
            use_gpu = False
        else:
            try:
                import torch
                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False
        return easyocr.Reader(self.languages, gpu=use_gpu, verbose=False)

    @staticmethod
    def _bbox_from_points(points) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        left = int(min(xs))
        top = int(min(ys))
        right = int(max(xs))
        bottom = int(max(ys))
        return left, top, right, bottom

    def _process_image(
        self,
        image: Image.Image,
        offset: Tuple[int, int],
    ) -> OCRResult:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("numpy is required for EasyOCR") from exc

        img_rgb = image.convert("RGB")
        arr = np.array(img_rgb)
        result = self._reader.readtext(arr)

        matches: List[OCRMatch] = []
        for item in result:
            if len(item) < 3:
                continue
            bbox_points, text, conf = item
            if not text:
                continue
            left, top, right, bottom = self._bbox_from_points(bbox_points)
            bbox = ScreenRegion(
                left=left + offset[0],
                top=top + offset[1],
                width=max(1, right - left),
                height=max(1, bottom - top),
            )
            # EasyOCR confidence is 0..1
            conf_pct = float(conf) * 100.0 if float(conf) <= 1.0 else float(conf)
            matches.append(
                OCRMatch(
                    text=text,
                    confidence=conf_pct,
                    bbox=bbox,
                    source="phrase",
                )
            )

        raw_text = " ".join([m.text for m in matches])
        return OCRResult(matches=matches, raw_text=raw_text)

    def process(
        self,
        image: Image.Image,
        offset: Tuple[int, int] = (0, 0),
        include_phrases: bool = True,
    ) -> OCRResult:
        return self._process_image(image, offset)

    def process_with_preprocessing(
        self,
        image: Image.Image,
        offset: Tuple[int, int] = (0, 0),
        include_phrases: bool = True,
    ) -> OCRResult:
        try:
            from PIL import ImageEnhance, ImageFilter, ImageOps, ImageStat

            gray = image.convert("L")
            normalized = ImageOps.autocontrast(gray, cutoff=2)
            enhanced = ImageEnhance.Contrast(normalized).enhance(1.8)
            sharpened = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=140, threshold=3))
            bw = sharpened
            try:
                avg = ImageStat.Stat(sharpened).mean[0]
                if avg < 120:
                    bw = ImageOps.invert(sharpened)
            except Exception:
                pass
            return self._process_image(bw.convert("RGB"), offset)
        except Exception:
            return self._process_image(image, offset)


# Singleton instance
class TextractOCREngine:
    """AWS Textract-based OCR engine (cloud)."""

    def __init__(self):
        try:
            import boto3
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "boto3 is required for Textract. Install with: python -m pip install boto3"
            ) from exc

        self._boto3 = boto3
        self.region = (
            (os.getenv("AWS_REGION") or "").strip()
            or (os.getenv("AWS_DEFAULT_REGION") or "").strip()
            or "us-east-1"
        )
        # Allow explicit credentials via env, otherwise boto3 will use its default chain.
        self.access_key = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
        self.secret_key = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
        self.session_token = (os.getenv("AWS_SESSION_TOKEN") or "").strip()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        kwargs = {"region_name": self.region}
        if self.access_key and self.secret_key:
            kwargs["aws_access_key_id"] = self.access_key
            kwargs["aws_secret_access_key"] = self.secret_key
            if self.session_token:
                kwargs["aws_session_token"] = self.session_token
        self._client = self._boto3.client("textract", **kwargs)
        return self._client

    @staticmethod
    def _bbox_to_pixels(bbox: dict, width: int, height: int) -> Tuple[int, int, int, int]:
        left = int(bbox.get("Left", 0.0) * width)
        top = int(bbox.get("Top", 0.0) * height)
        w = int(bbox.get("Width", 0.0) * width)
        h = int(bbox.get("Height", 0.0) * height)
        return left, top, left + w, top + h

    def _image_bytes(self, image: Image.Image) -> bytes:
        """Encode image to JPEG and keep size under Textract 5MB limit."""
        img = image.convert("RGB")

        def encode_jpeg(quality: int) -> bytes:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()

        data = encode_jpeg(85)
        if len(data) <= 4_700_000:
            return data

        # Downscale progressively if needed
        w, h = img.size
        scale = 0.85
        while len(data) > 4_700_000 and w > 320 and h > 240:
            w = int(w * scale)
            h = int(h * scale)
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            data = encode_jpeg(80)
        return data

    def _process_image(
        self,
        image: Image.Image,
        offset: Tuple[int, int],
        include_phrases: bool,
    ) -> OCRResult:
        client = self._get_client()
        img_bytes = self._image_bytes(image)
        response = client.detect_document_text(Document={"Bytes": img_bytes})
        blocks = response.get("Blocks", [])

        matches: List[OCRMatch] = []
        width, height = image.size

        for block in blocks:
            btype = block.get("BlockType")
            # For Textract, avoid LINE blocks to prevent merged headers (e.g., Excel column letters).
            if btype != "WORD":
                continue
            text = block.get("Text", "")
            if not text:
                continue
            bbox = block.get("Geometry", {}).get("BoundingBox", {})
            l, t, r, b = self._bbox_to_pixels(bbox, width, height)
            if r <= l or b <= t:
                continue
            conf = float(block.get("Confidence", 0.0))
            # Heuristic: split merged Excel header letters (e.g., "DCDEFGHIJ") into per-letter boxes.
            if text.isalpha() and text.isupper() and len(text) >= 2:
                box_w = max(1, r - l)
                box_h = max(1, b - t)
                if box_w / max(1, box_h) >= 3.0:
                    char_w = box_w / len(text)
                    for idx, ch in enumerate(text):
                        cl = int(l + idx * char_w)
                        cr = int(l + (idx + 1) * char_w)
                        if cr <= cl:
                            continue
                        bbox_region = ScreenRegion(
                            left=cl + offset[0],
                            top=t + offset[1],
                            width=max(1, cr - cl),
                            height=box_h,
                        )
                        matches.append(
                            OCRMatch(
                                text=ch,
                                confidence=conf,
                                bbox=bbox_region,
                                source="word",
                            )
                        )
                    continue

            bbox_region = ScreenRegion(
                left=l + offset[0],
                top=t + offset[1],
                width=max(1, r - l),
                height=max(1, b - t),
            )
            matches.append(
                OCRMatch(
                    text=text,
                    confidence=conf,
                    bbox=bbox_region,
                    source="word",
                )
            )

        raw_text = " ".join([m.text for m in matches])
        return OCRResult(matches=matches, raw_text=raw_text)

    def process(
        self,
        image: Image.Image,
        offset: Tuple[int, int] = (0, 0),
        include_phrases: bool = True,
    ) -> OCRResult:
        return self._process_image(image, offset, include_phrases)

    def process_with_preprocessing(
        self,
        image: Image.Image,
        offset: Tuple[int, int] = (0, 0),
        include_phrases: bool = True,
    ) -> OCRResult:
        # Textract does its own preprocessing; just run the standard call.
        return self._process_image(image, offset, include_phrases)


# Singleton instance
_ocr_instance: Optional[BaseOCREngine] = None
_ocr_instance_kind: Optional[str] = None


def get_ocr_engine() -> BaseOCREngine:
    """Get or create the OCR engine singleton."""
    global _ocr_instance, _ocr_instance_kind
    engine_name = (config.ocr_engine or "").strip().lower()
    if engine_name in ("easyocr", "easy", "easy_ocr"):
        if _ocr_instance is None or _ocr_instance_kind != "easyocr":
            try:
                _ocr_instance = EasyOCREngine()
                _ocr_instance_kind = "easyocr"
            except Exception:
                _ocr_instance = OCREngine()
                _ocr_instance_kind = "tesseract"
        return _ocr_instance
    if engine_name in ("textract", "aws_textract", "aws-textract"):
        if _ocr_instance is None or _ocr_instance_kind != "textract":
            try:
                _ocr_instance = TextractOCREngine()
                _ocr_instance_kind = "textract"
            except Exception:
                _ocr_instance = OCREngine()
                _ocr_instance_kind = "tesseract"
        return _ocr_instance

    if _ocr_instance is None or _ocr_instance_kind != "tesseract":
        _ocr_instance = OCREngine()
        _ocr_instance_kind = "tesseract"
    return _ocr_instance
