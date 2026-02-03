from __future__ import annotations

import math
from typing import Any, List, Tuple

from PIL import Image

from .ocr import get_ocr_engine, OCRResult
from .screenshot import ScreenRegion


def run_ocr_in_roi(
    ctx: Any,
    image: Image.Image,
    roi: ScreenRegion,
    *,
    force_full: bool = False,
    include_phrases: bool = False,
    query_text: str = "",
) -> OCRResult:
    """Run OCR inside a region of interest and return OCRResult."""
    ocr_result = None
    include_phrases = include_phrases or force_full
    try:
        def merge_results(base: OCRResult, extra: OCRResult) -> OCRResult:
            all_matches = {}
            for match in base.matches:
                key = (match.bbox.left, match.bbox.top, match.text)
                if key not in all_matches or match.confidence > all_matches[key].confidence:
                    all_matches[key] = match
            for match in extra.matches:
                key = (match.bbox.left, match.bbox.top, match.text)
                if key not in all_matches or match.confidence > all_matches[key].confidence:
                    all_matches[key] = match
            return OCRResult(matches=list(all_matches.values()), raw_text=base.raw_text or extra.raw_text)

        if getattr(ctx, "use_paddleocr", False) and ctx.init_paddleocr():
            ctx.log("OCR: using PaddleOCR (sv/Latin multilingual) for ROI.", "info")
            try:
                paddle_matches = ctx._nm_run_paddleocr_roi(image, roi)
                if paddle_matches:
                    ocr_result = OCRResult(
                        matches=paddle_matches,
                        raw_text=" ".join([m.text for m in paddle_matches]),
                    )
                    ctx.log(
                        f"PaddleOCR in ROI produced {len(ocr_result.matches)} text matches.",
                        "info",
                    )
                else:
                    ctx.log("PaddleOCR in ROI found no matches - falling back to Tesseract.", "info")
            except Exception as exc:
                ctx.log(f"PaddleOCR failed in ROI: {exc} - falling back to Tesseract.", "error")

        ocr_engine = get_ocr_engine()
        is_textract = type(ocr_engine).__name__ == "TextractOCREngine"
        roi_crop = image.crop((roi.left, roi.top, roi.right, roi.bottom))

        # Always run base engine when force_full or PaddleOCR is not used.
        if ocr_result is None or force_full or is_textract:
            tesseract_result = ocr_engine.process(
                roi_crop,
                offset=(roi.left, roi.top),
                include_phrases=include_phrases,
            )
            ctx.log(f"OCR (standard) in ROI produced {len(tesseract_result.matches)} text matches.", "info")
            if ocr_result is None:
                ocr_result = tesseract_result
            else:
                ocr_result = merge_results(ocr_result, tesseract_result)

        # Optional preprocessing pass (skip for Textract and in fast mode)
        if (
            ocr_result is not None
            and not is_textract
            and (force_full or not getattr(ctx, "ocr_fast_mode", True))
        ):
            try:
                ocr_result_preprocessed = ocr_engine.process_with_preprocessing(
                    roi_crop,
                    offset=(roi.left, roi.top),
                    include_phrases=include_phrases,
                )
                ocr_result = merge_results(ocr_result, ocr_result_preprocessed)
                ctx.log(
                    f"OCR (with preprocessing) found {len(ocr_result.matches)} total unique matches.",
                    "info",
                )
            except Exception as exc:
                ctx.log(f"Preprocessed OCR failed: {exc}, using standard OCR only", "info")
    except Exception as exc:
        ctx.log(f"OCR failed in ROI: {exc}", "error")
        ocr_result = OCRResult(matches=[], raw_text="")

    if ocr_result is None:
        ocr_result = OCRResult(matches=[], raw_text="")

    if ocr_result.matches and is_textract and include_phrases:
        try:
            query_words = [w for w in (query_text or "").strip().split() if w]
            target_len = len(query_words)
            if target_len >= 2:
                # Build phrase boxes from adjacent words on the same line.
                words = [m for m in ocr_result.matches if m.source == "word" or m.source == "phrase"]
                # Sort top-to-bottom, left-to-right
                words.sort(key=lambda m: (m.bbox.top, m.bbox.left))
                new_phrases = []
                i = 0
                while i < len(words):
                    line = []
                    base_top = words[i].bbox.top
                    tol = max(4, int(words[i].bbox.height * 0.5))
                    j = i
                    while j < len(words) and abs(words[j].bbox.top - base_top) <= tol:
                        line.append(words[j])
                        j += 1
                    # Build only exact-length phrases to avoid long header merges.
                    if len(line) >= target_len:
                        for start in range(0, len(line) - target_len + 1):
                            chunk = line[start : start + target_len]
                            phrase_text = " ".join([w.text for w in chunk])
                            left = min(w.bbox.left for w in chunk)
                            top = min(w.bbox.top for w in chunk)
                            right = max(w.bbox.right for w in chunk)
                            bottom = max(w.bbox.bottom for w in chunk)
                            avg_conf = sum(w.confidence for w in chunk) / max(1, len(chunk))
                            phrase_bbox = ScreenRegion(
                                left=left,
                                top=top,
                                width=right - left,
                                height=bottom - top,
                            )
                            new_phrases.append(
                                type(chunk[0])(
                                    text=phrase_text,
                                    confidence=avg_conf,
                                    bbox=phrase_bbox,
                                    source="phrase",
                                )
                            )
                    i = j
                if new_phrases:
                    ocr_result.matches.extend(new_phrases)
        except Exception:
            pass

    if ocr_result.matches and is_textract and query_text:
        def _norm(s: str) -> str:
            return " ".join((s or "").strip().lower().split())
        qn = _norm(query_text)
        if qn:
            exact_matches = [m for m in ocr_result.matches if _norm(m.text) == qn]
            if not exact_matches:
                try:
                    from .ocr import EasyOCREngine, OCRResult
                    easy_engine = EasyOCREngine()
                    easy_exact: List[OCRResult] = []
                    max_boxes = 120
                    boxes = ocr_result.matches[:max_boxes]
                    if len(ocr_result.matches) > max_boxes:
                        ctx.log(f"Textract returned {len(ocr_result.matches)} boxes; EasyOCR fallback capped to {max_boxes}.", "info")
                    for m in boxes:
                        b = m.bbox
                        crop = image.crop((b.left, b.top, b.right, b.bottom))
                        try:
                            res = easy_engine.process(crop, offset=(b.left, b.top), include_phrases=False)
                        except Exception:
                            continue
                        for em in res.matches:
                            if _norm(em.text) == qn:
                                exact_matches.append(em)
                    if exact_matches:
                        ocr_result = OCRResult(
                            matches=exact_matches,
                            raw_text=" ".join([m.text for m in exact_matches]),
                        )
                        ctx.log(f"EasyOCR fallback found {len(exact_matches)} exact matches for '{query_text}'.", "info")
                except Exception as exc:
                    ctx.log(f"EasyOCR fallback failed: {exc}", "error")

    if ocr_result.matches and not is_textract:
        try:
            ocr_result.matches = ctx._nm_refine_ocr_matches(image, ocr_result.matches)
        except Exception:
            # Best effort: keep matches even if refinement fails.
            pass

    return ocr_result


def build_ocr_candidates(
    ctx: Any,
    snapshot: Any,
    planner: Any,
    roi: ScreenRegion,
    ocr_result: OCRResult,
    start_id: int,
    candidate_cls: Any,
    min_similarity: float = 0.0,
    similarity_fn: Any = None,
) -> Tuple[List[Any], int]:
    """Build OCR-based CandidatePackage entries."""
    candidates: List[Any] = []
    next_id = start_id
    img = snapshot.image

    for match in ocr_result.matches:
        text = match.text.strip()
        if not text:
            continue

        bx = match.bbox
        # Light padding to avoid clipping antialiased glyph edges
        pad_x = max(2, int(bx.width * 0.08))
        pad_y = max(2, int(bx.height * 0.12))

        left = max(roi.left, bx.left - pad_x)
        top = max(roi.top, bx.top - pad_y)
        right = min(roi.right, bx.right + pad_x)
        bottom = min(roi.bottom, bx.bottom + pad_y)

        click_x, click_y = ctx._nm_text_center_in_bbox(img, (left, top, right, bottom))
        color = ctx._nm_compute_region_color(img, (left, top, right, bottom))

        if similarity_fn is not None:
            try:
                text_sim = float(similarity_fn(text))
            except Exception:
                text_sim = ctx._nm_text_similarity(planner.text_intent, text)
        else:
            text_sim = ctx._nm_text_similarity(planner.text_intent, text)
        if min_similarity > 0.0 and text_sim < min_similarity:
            continue
        size = (right - left, bottom - top)
        aspect = size[0] / max(1, size[1])
        # More lenient shape scoring - accept wider range of aspect ratios
        # Buttons can be square, wide, or even somewhat tall
        if 0.8 <= aspect <= 8.0:
            shape_score = 1.0
        elif 0.5 <= aspect <= 12.0:
            shape_score = 0.7
        else:
            shape_score = 0.5

        # Distance to window center as a weak prior for "center" zone
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

        # Total deterministic score - weights are arbitrary but fixed
        total = (
            scores["text_match"] * 0.6
            + scores["shape_plausibility"] * 0.25
            + scores["location_match"] * 0.15
        )

        candidates.append(
            candidate_cls(
                id=next_id,
                bbox=(left, top, right, bottom),
                click_point=(click_x, click_y),
                text=text,
                color=color,
                scores=scores,
                total_score=total,
                source="ocr",
            )
        )
        next_id += 1

    return candidates, next_id
