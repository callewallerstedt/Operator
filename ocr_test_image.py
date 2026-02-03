#!/usr/bin/env python3
"""Small OCR tester for a single image."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

from agent.ocr import get_ocr_engine, OCRMatch, OCRResult
from agent.screenshot import ScreenRegion


def _merge_matches(primary: List[OCRMatch], extra: List[OCRMatch]) -> List[OCRMatch]:
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


def _find_exact_phrase_from_words(words: List[OCRMatch], query: str) -> OCRMatch | None:
    if not words:
        return None
    q_norm = _normalize_exact(query)
    if not q_norm:
        return None
    # Group words by line (looser tolerance than OCR default).
    heights = [w.bbox.height for w in words if w.bbox.height > 0]
    med_h = sorted(heights)[len(heights) // 2] if heights else 12
    tol = max(8, int(med_h * 0.8))
    words_sorted = sorted(words, key=lambda w: (w.bbox.top, w.bbox.left))
    lines: List[List[OCRMatch]] = []
    for w in words_sorted:
        placed = False
        for line in lines:
            if abs(w.bbox.top - line[0].bbox.top) <= tol:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w.bbox.left)
        max_phrase_words = 4
        for start in range(len(line)):
            for end in range(start + 1, min(start + 1 + max_phrase_words, len(line) + 1)):
                phrase_words = line[start:end]
                phrase_text = " ".join([w.text for w in phrase_words]).strip()
                if _exact_match(phrase_text, query):
                    left = min(w.bbox.left for w in phrase_words)
                    top = min(w.bbox.top for w in phrase_words)
                    right = max(w.bbox.right for w in phrase_words)
                    bottom = max(w.bbox.bottom for w in phrase_words)
                    bbox = ScreenRegion(left=left, top=top, width=right - left, height=bottom - top)
                    avg_conf = sum(w.confidence for w in phrase_words) / len(phrase_words)
                    return OCRMatch(text=phrase_text, confidence=avg_conf, bbox=bbox, source="phrase")
    return None


def _find_blue_regions(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # Blue-ish range (tuned for UI buttons; higher saturation to avoid sky/snow)
    mask = (h >= 90) & (h <= 135) & (s >= 100) & (v >= 60)
    mask = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions: List[Tuple[int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 300 or w < 40 or h < 18:
            continue
        regions.append((x, y, x + w, y + h))
    return regions


def _center_in_bbox(match: OCRMatch, bbox: ScreenRegion) -> bool:
    cx, cy = match.center
    return (bbox.left <= cx <= bbox.right) and (bbox.top <= cy <= bbox.bottom)


def _ocr_string_variants(image: Image.Image, lang: str, whitelist: str | None = None) -> List[str]:
    try:
        import pytesseract
        from PIL import ImageEnhance, ImageFilter, ImageOps
        import numpy as np
        import cv2
    except Exception:
        return []
    # Ensure tesseract executable is configured (get_ocr_engine sets tesseract_cmd).
    try:
        get_ocr_engine()
    except Exception:
        pass
    variants: List[Image.Image] = []
    base = image.convert("RGB")
    # Upscale small crops to help OCR
    w, h = base.size
    if w < 250 or h < 80:
        base = base.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        w, h = base.size
    # Extra upscale for button text
    if w < 500 or h < 160:
        base = base.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    variants.append(base)
    gray = base.convert("L")
    variants.append(gray)
    # High-contrast sharpened
    hc = ImageOps.autocontrast(gray, cutoff=2)
    hc = ImageEnhance.Contrast(hc).enhance(2.0)
    hc = hc.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    variants.append(hc)
    # Stronger sharpened variant
    hc2 = ImageOps.autocontrast(gray, cutoff=1)
    hc2 = ImageEnhance.Contrast(hc2).enhance(2.6)
    hc2 = hc2.filter(ImageFilter.UnsharpMask(radius=3, percent=220, threshold=2))
    variants.append(hc2)
    # Inverted high-contrast
    inv = ImageOps.invert(hc)
    variants.append(inv)
    # White-text mask (low saturation, high value) for light text on blue
    try:
        arr = np.array(base.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        text_mask = (s < 80) & (v > 180)
        text_mask = (text_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        text_img = Image.fromarray(text_mask)  # white text on black
        variants.append(text_img)
    except Exception:
        pass
    texts: List[str] = []
    for v in variants:
        for oem in (3, 1):
            for psm in (7, 6, 11, 13):
                try:
                    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
                    if whitelist:
                        config += f" -c tessedit_char_whitelist={whitelist}"
                    txt = pytesseract.image_to_string(
                        v,
                        lang=lang,
                        config=config,
                    )
                    if txt:
                        texts.append(txt)
                except Exception:
                    continue
    return texts


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = (
        t.replace("\u00e5", "a")
        .replace("\u00e4", "a")
        .replace("\u00f6", "o")
        .replace("\u00e9", "e")
        .replace("\u00e8", "e")
        .replace("\u00ea", "e")
    )
    out = []
    for ch in t:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
    return " ".join("".join(out).split())


def _normalize_exact(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    swedish = "\u00e5\u00e4\u00f6"
    out = []
    for ch in t:
        # Keep letters (including åäö) and spaces only; drop punctuation/noise.
        if ch.isalpha() or ch in swedish or ch.isspace():
            out.append(ch)
    return " ".join("".join(out).split())


def _exact_match(text: str, query: str) -> bool:
    # Strict match if any line equals query after normalization.
    q = _normalize_exact(query)
    if not q:
        return False
    lines = [ln.strip() for ln in (text or "").splitlines()]
    for ln in lines:
        if _normalize_exact(ln) == q:
            return True
    return False


def _score_match(text: str, query: str) -> float:
    if not text:
        return 0.0
    t = _normalize_text(text)
    q = _normalize_text(query)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    return SequenceMatcher(None, t, q).ratio()


def _merge_matches(primary: List[OCRMatch], extra: List[OCRMatch]) -> List[OCRMatch]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Test OCR on a single image.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--text", default="", help="Text to search for")
    parser.add_argument("--out", default="", help="Write debug overlay image to this path")
    parser.add_argument("--max", type=int, default=25, help="Max matches to print")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing pass")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    ocr = get_ocr_engine()

    result = ocr.process(img, include_phrases=True)
    matches = list(result.matches)

    if not args.no_preprocess:
        result_pp = ocr.process_with_preprocessing(img, include_phrases=True)
        matches = _merge_matches(matches, result_pp.matches)

    query = args.text.strip()
    found = None
    focus_bbox = None
    if query:
        # 1) Exact match from OCR matches
        exacts = [m for m in matches if _exact_match(m.text, query)]
        if exacts:
            exacts.sort(key=lambda m: m.confidence, reverse=True)
            found = exacts[0]
            focus_bbox = found.bbox

        # 2) Try to reconstruct exact phrase from word matches
        if found is None:
            words = [m for m in matches if getattr(m, "source", "") == "word"]
            found = _find_exact_phrase_from_words(words, query)
            if found is not None:
                focus_bbox = found.bbox

        # 3) If still missing, try focused OCR on blue button-like regions
        if found is None:
            try:
                import pytesseract
                available = set(pytesseract.get_languages(config=""))
                # Prefer Swedish-only for diacritics on button text.
                lang = "swe" if "swe" in available else ocr._select_language()
            except Exception:
                lang = "swe+eng"
            whitelist = None
            blue_regions = _find_blue_regions(img)
            blue_regions.sort(key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]), reverse=True)
            for l, t, r, b in blue_regions[:1]:
                pad_x = max(6, int((r - l) * 0.08))
                pad_y = max(4, int((b - t) * 0.15))
                left = max(0, l - pad_x)
                right = min(img.width, r + pad_x)
                top = max(0, t - pad_y)
                bottom = min(img.height, b + pad_y)
                if right <= left or bottom <= top:
                    continue
                crop = img.crop((left, top, right, bottom))
                for txt in _ocr_string_variants(crop, lang, whitelist=whitelist):
                    if _exact_match(txt, query):
                        bbox = ScreenRegion(left=left, top=top, width=right - left, height=bottom - top)
                        found = OCRMatch(text=query, confidence=90.0, bbox=bbox, source="phrase")
                        focus_bbox = bbox
                        break
                if found is not None:
                    break

        # 4) If still missing, anchor on best matching single-word and expand crop
        if found is None:
            first_word = _normalize_text(query).split()[0] if _normalize_text(query) else ""
            if first_word:
                try:
                    import pytesseract
                    available = set(pytesseract.get_languages(config=""))
                    lang = "swe" if "swe" in available else ocr._select_language()
                except Exception:
                    lang = "swe+eng"
                whitelist = None
                # Use approximate matches to the first word to find likely button area
                approx = []
                for m in matches:
                    if _score_match(m.text, first_word) >= 0.75:
                        approx.append(m)
                approx.sort(key=lambda m: m.confidence, reverse=True)
                for m in approx[:3]:
                    bx = m.bbox
                    pad_left = max(10, int(bx.width * 0.5))
                    pad_right = max(120, int(bx.width * 4.0))
                    pad_y = max(12, int(bx.height * 2.0))
                    left = max(0, bx.left - pad_left)
                    right = min(img.width, bx.right + pad_right)
                    top = max(0, bx.top - pad_y)
                    bottom = min(img.height, bx.bottom + pad_y)
                    if right <= left or bottom <= top:
                        continue
                    crop = img.crop((left, top, right, bottom))
                    # Refine to blue region inside the crop, if possible
                    blue_regions = _find_blue_regions(crop)
                    if blue_regions:
                        # Use the largest blue region inside the crop
                        blue_regions.sort(key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]), reverse=True)
                        bl, bt, br, bb = blue_regions[0]
                        crop = crop.crop((bl, bt, br, bb))
                    for txt in _ocr_string_variants(crop, lang, whitelist=whitelist):
                        if _exact_match(txt, query):
                            bbox = ScreenRegion(left=left, top=top, width=right - left, height=bottom - top)
                            found = OCRMatch(text=query, confidence=90.0, bbox=bbox, source="phrase")
                            focus_bbox = bbox
                            break
                    if found is not None:
                        break

        # 5) Brute-force: exact match from OCR string on the strongest blue region
        if found is None and blue_regions:
            l, t, r, b = blue_regions[0]
            crop = img.crop((l, t, r, b))
            try:
                import pytesseract
                lang = "swe+eng"
                txt = pytesseract.image_to_string(
                    crop,
                    lang=lang,
                    config="--oem 3 --psm 7 -c preserve_interword_spaces=1",
                )
                if _exact_match(txt, query):
                    bbox = ScreenRegion(left=l, top=t, width=r - l, height=b - t)
                    found = OCRMatch(text=query, confidence=90.0, bbox=bbox, source="phrase")
                    focus_bbox = bbox
            except Exception:
                pass

    print(f"Total matches: {len(matches)}")
    if query:
        if found:
            print(
                f"FOUND: '{query}' "
                f"(ocr='{found.text}', sim={_score_match(found.text, query):.2f}, conf={found.confidence:.1f}) "
                f"at {found.bbox}"
            )
        else:
            print("NOT FOUND")

    # Print top matches by confidence or similarity to query
    if query:
        scored = [(m, _score_match(m.text, query)) for m in matches]
        scored.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
        print("Top matches by similarity:")
        for m, s in scored[: args.max]:
            print(f"  sim={s:.2f} conf={m.confidence:.1f} text='{m.text}'")
    else:
        matches.sort(key=lambda m: m.confidence, reverse=True)
        print("Top matches by confidence:")
        for m in matches[: args.max]:
            print(f"  conf={m.confidence:.1f} text='{m.text}'")

    if args.out:
        overlay = img.copy().convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw_matches = matches
        if query:
            if focus_bbox is not None:
                # Reduce noise: only show matches whose center falls inside focus bbox
                draw_matches = [
                    m for m in matches
                    if _center_in_bbox(m, focus_bbox) and (m.confidence >= 40.0) and len(m.text.strip()) >= 2
                ]
            else:
                # If we never found anything, only draw exact matches to reduce noise
                draw_matches = [m for m in matches if _exact_match(m.text, query)]

        for m in draw_matches:
            l, t, r, b = m.bbox.left, m.bbox.top, m.bbox.right, m.bbox.bottom
            draw.rectangle([l, t, r, b], outline=(255, 0, 0, 200), width=2)
            label = m.text[:20]
            draw.text((l, max(0, t - 12)), label, fill=(255, 0, 0, 255), font=font)
        overlay.convert("RGB").save(args.out)
        print(f"Wrote overlay to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
