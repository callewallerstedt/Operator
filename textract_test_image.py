#!/usr/bin/env python3
"""Simple Textract test script for a single image."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

try:
    import boto3
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "boto3 is required for Textract. Install with: python -m pip install boto3"
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
    for line in (text or "").splitlines():
        if _normalize_exact(line.strip()) == q:
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


def _bbox_to_pixels(bbox: dict, width: int, height: int) -> Tuple[int, int, int, int]:
    left = int(bbox.get("Left", 0.0) * width)
    top = int(bbox.get("Top", 0.0) * height)
    w = int(bbox.get("Width", 0.0) * width)
    h = int(bbox.get("Height", 0.0) * height)
    return left, top, left + w, top + h


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Test AWS Textract on a single image.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--text", default="", help="Text to search for (exact match)")
    parser.add_argument("--out", default="", help="Write debug overlay image to this path")
    parser.add_argument("--max", type=int, default=25, help="Max matches to print")
    parser.add_argument("--region", default="", help="AWS region (falls back to AWS_REGION/AWS_DEFAULT_REGION)")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    width, height = img.size

    region = (
        args.region
        or "".join((__import__("os").getenv("AWS_REGION", ""),))
        or __import__("os").getenv("AWS_DEFAULT_REGION", "")
        or "us-east-1"
    )

    client = boto3.client("textract", region_name=region)
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    response = client.detect_document_text(Document={"Bytes": image_bytes})
    blocks = response.get("Blocks", [])

    lines = [b for b in blocks if b.get("BlockType") == "LINE"]
    words = [b for b in blocks if b.get("BlockType") == "WORD"]

    query = args.text.strip()
    found = None
    if query:
        exacts = [b for b in lines if _exact_match(b.get("Text", ""), query)]
        if exacts:
            exacts.sort(key=lambda b: b.get("Confidence", 0.0), reverse=True)
            found = exacts[0]

    print(f"Lines: {len(lines)} | Words: {len(words)}")
    if query:
        if found:
            print(
                f"FOUND: '{query}' (textract='{found.get('Text', '')}', conf={found.get('Confidence', 0.0):.1f})"
            )
        else:
            print("NOT FOUND")

    if query:
        scored = [(b, _score_match(b.get("Text", ""), query)) for b in lines]
        scored.sort(key=lambda x: (x[1], x[0].get("Confidence", 0.0)), reverse=True)
        print("Top matches by similarity:")
        for b, s in scored[: args.max]:
            print(f"  sim={s:.2f} conf={b.get('Confidence', 0.0):.1f} text='{b.get('Text', '')}'")
    else:
        lines.sort(key=lambda b: b.get("Confidence", 0.0), reverse=True)
        print("Top lines by confidence:")
        for b in lines[: args.max]:
            print(f"  conf={b.get('Confidence', 0.0):.1f} text='{b.get('Text', '')}'")

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
                draw_lines = [b for b in lines if _exact_match(b.get("Text", ""), query)]

        for b in draw_lines:
            bbox = b.get("Geometry", {}).get("BoundingBox", {})
            l, t, r, btm = _bbox_to_pixels(bbox, width, height)
            draw.rectangle([l, t, r, btm], outline=(255, 0, 0, 200), width=2)
            label = (b.get("Text", "") or "")[:30]
            draw.text((l, max(0, t - 14)), label, fill=(255, 0, 0, 255), font=font)

        overlay.convert("RGB").save(args.out)
        print(f"Wrote overlay to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
