#!/usr/bin/env python3
"""PaddleOCR test script for a single image (free OCR)."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
import os

# Reduce Paddle model hoster check and noisy logs.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLEOCR_SHOW_LOG", "False")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_cinn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

try:
    from paddleocr import PaddleOCR
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "paddleocr is required. Install with: python -m pip install paddleocr"
    ) from exc


SWEDISH = "\u00e5\u00e4\u00f6"


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
    out = []
    for ch in t:
        if ch.isalpha() or ch in SWEDISH or ch.isspace():
            out.append(ch)
    return " ".join("".join(out).split())


def _exact_match(text: str, query: str) -> bool:
    q = _normalize_exact(query)
    if not q:
        return False
    return _normalize_exact(text.strip()) == q


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


def _bbox_from_points(points: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))
    return left, top, right, bottom


def _init_ocr() -> PaddleOCR:
    # Try a robust set of args, fallback if PaddleOCR rejects any
    try:
        return PaddleOCR(lang="sv", use_textline_orientation=False)
    except Exception:
        try:
            return PaddleOCR(lang="sv")
        except Exception:
            return PaddleOCR()


def main() -> int:
    parser = argparse.ArgumentParser(description="Test PaddleOCR on a single image.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--text", default="", help="Text to search for (exact match)")
    parser.add_argument("--out", default="", help="Write debug overlay image to this path")
    parser.add_argument("--max", type=int, default=25, help="Max matches to print")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    ocr = _init_ocr()

    # PaddleOCR expects numpy array
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise SystemExit("numpy is required") from exc

    img_arr = np.array(img)
    result = ocr.ocr(img_arr)

    lines = []
    if isinstance(result, list) and result:
        # PaddleOCR v3: result[0] is list of lines
        items = result[0] if isinstance(result[0], list) else result
        for line in items:
            if not line or len(line) < 2:
                continue
            bbox_points = line[0]
            text, conf = line[1]
            if not text:
                continue
            l, t, r, b = _bbox_from_points(bbox_points)
            lines.append({
                "text": text,
                "conf": float(conf) * 100.0,
                "bbox": (l, t, r, b),
            })

    query = args.text.strip()
    found = None
    if query:
        exacts = [l for l in lines if _exact_match(l["text"], query)]
        if exacts:
            exacts.sort(key=lambda x: x["conf"], reverse=True)
            found = exacts[0]

    print(f"Lines: {len(lines)}")
    if query:
        if found:
            print(f"FOUND: '{query}' (ocr='{found['text']}', conf={found['conf']:.1f})")
        else:
            print("NOT FOUND")

    if query:
        scored = [(l, _score_match(l["text"], query)) for l in lines]
        scored.sort(key=lambda x: (x[1], x[0]["conf"]), reverse=True)
        print("Top matches by similarity:")
        for l, s in scored[: args.max]:
            print(f"  sim={s:.2f} conf={l['conf']:.1f} text='{l['text']}'")
    else:
        lines.sort(key=lambda x: x["conf"], reverse=True)
        print("Top lines by confidence:")
        for l in lines[: args.max]:
            print(f"  conf={l['conf']:.1f} text='{l['text']}'")

    if args.out:
        overlay = img.copy().convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        draw_lines = lines
        if query:
            if found is not None:
                draw_lines = [found]
            else:
                draw_lines = [l for l in lines if _exact_match(l["text"], query)]

        for l in draw_lines:
            left, top, right, bottom = l["bbox"]
            draw.rectangle([left, top, right, bottom], outline=(255, 0, 0, 200), width=2)
            label = (l["text"] or "")[:30]
            draw.text((left, max(0, top - 14)), label, fill=(255, 0, 0, 255), font=font)

        overlay.convert("RGB").save(args.out)
        print(f"Wrote overlay to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
