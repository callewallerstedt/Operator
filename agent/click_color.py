from __future__ import annotations

import math
from typing import Any, List, Tuple, Union

from PIL import Image

from .screenshot import ScreenRegion


RegionList = List[Tuple[int, int, int, int]]
RegionReturn = Union[RegionList, Tuple[RegionList, List[Any]]]


def find_color_regions(
    ctx: Any,
    image: Image.Image,
    roi: ScreenRegion,
    target_color: str,
    accent_color_relevant: bool,
    max_area: int,
    return_masks: bool = False,
) -> RegionReturn:
    """Find connected regions matching a target color mask."""
    masks_u8 = ctx._nm_build_color_masks(image, roi, target_color, accent_color_relevant)
    if not masks_u8:
        return ([], []) if return_masks else []

    try:
        import cv2
        import numpy as np
    except Exception:
        ctx.log("OpenCV (cv2) is not available; color-region detection is disabled.", "error")
        return ([], []) if return_masks else []

    # Prefer user-selected splits when provided; otherwise choose best-looking ones.
    masks_for_regions = masks_u8
    selected_indices: List[int] = list(range(1, len(masks_u8) + 1))
    split_override = ctx._nm_get_color_split_override(target_color)
    use_direct_bbox = False
    if split_override:
        chosen = []
        chosen_idx = []
        for idx in split_override:
            if 1 <= idx <= len(masks_u8):
                chosen.append(masks_u8[idx - 1])
                chosen_idx.append(idx)
        if chosen:
            masks_for_regions = chosen
            selected_indices = chosen_idx
            use_direct_bbox = True
            ctx.log(
                f"Color mask using split override for '{target_color}': {selected_indices} of {len(masks_u8)}.",
                "info",
            )
    elif len(masks_u8) > 1:
        # Use all splits by default so every masked area can become a candidate.
        masks_for_regions = list(masks_u8)
        selected_indices = list(range(1, len(masks_u8) + 1))

    regions: RegionList = []
    roi_area = max(1, (roi.right - roi.left) * (roi.bottom - roi.top))
    min_area = max(ctx._nm_get_color_min_area(), int(roi_area * 0.00005))

    # Combine masks for later ratio checks and brightness filtering
    combined_mask = np.zeros_like(masks_for_regions[0], dtype=np.uint8)
    for m in masks_for_regions:
        if m is not None:
            combined_mask = np.maximum(combined_mask, m)

    def _apply_brightness_filter(regs: RegionList) -> RegionList:
        color_key = ctx._nm_normalize_color(target_color)
        if color_key in {"", "unknown", "colorful"}:
            return regs
        try:
            v_settings = ctx.nm_color_settings.get("val", {})

            def _v_limits(name: str):
                vmin, vmax = v_settings.get(name, (0, 255))
                try:
                    vmin = int(float(vmin))
                    vmax = int(float(vmax))
                except Exception:
                    vmin, vmax = (0, 255)
                vmin = max(0, min(255, vmin))
                vmax = max(0, min(255, vmax))
                if vmin > vmax:
                    vmin, vmax = vmax, vmin
                full = (vmin <= 0) and (vmax >= 255)
                return vmin, vmax, full

            if color_key == "red":
                vmin1, vmax1, full1 = _v_limits("red1")
                vmin2, vmax2, full2 = _v_limits("red2")
                if full1 and full2:
                    return regs
            else:
                vmin, vmax, full = _v_limits(color_key)
                if full:
                    return regs

            roi_img = image.crop((roi.left, roi.top, roi.right, roi.bottom)).convert("RGB")
            roi_arr = np.array(roi_img)
            hsv = cv2.cvtColor(roi_arr, cv2.COLOR_RGB2HSV)
            v = hsv[:, :, 2]
            if color_key == "red":
                v_allow = ((v >= vmin1) & (v <= vmax1)) | ((v >= vmin2) & (v <= vmax2))
            else:
                v_allow = (v >= vmin) & (v <= vmax)

            filtered: RegionList = []
            for l, t, r, b in regs:
                rl = max(0, l - roi.left)
                rt = max(0, t - roi.top)
                rr = min(roi.right - roi.left, r - roi.left)
                rb = min(roi.bottom - roi.top, b - roi.top)
                if rr <= rl or rb <= rt:
                    continue
                sub_mask = combined_mask[rt:rb, rl:rr]
                if sub_mask.size == 0:
                    continue
                sub_v = v_allow[rt:rb, rl:rr]
                if np.any((sub_mask > 0) & sub_v):
                    filtered.append((l, t, r, b))
            return filtered
        except Exception:
            return regs

    if use_direct_bbox:
        for mask_u8 in masks_for_regions:
            if mask_u8 is None:
                continue
            work = mask_u8.copy()
            contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area < min_area:
                    continue
                if max_area > 0 and area > max_area:
                    continue
                if w < 4 or h < 4:
                    continue
                pad_out = 2
                l = max(0, x - pad_out)
                t = max(0, y - pad_out)
                r = min(roi.right - roi.left, x + w + pad_out)
                b = min(roi.bottom - roi.top, y + h + pad_out)
                regions.append((roi.left + l, roi.top + t, roi.left + r, roi.top + b))
        regions = _apply_brightness_filter(regions)
        return (regions, masks_u8) if return_masks else regions

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for mask_u8 in masks_for_regions:
        if mask_u8 is None:
            continue
        work = mask_u8.copy()
        try:
            area_ratio = float(np.count_nonzero(work)) / float(work.size)
        except Exception:
            area_ratio = 0.0
        if area_ratio >= 0.15:
            # Large masks should not be expanded; tighten them to avoid merges.
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel_small, iterations=1)
            work = cv2.erode(work, kernel_small, iterations=1)
        else:
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel, iterations=1)
            work = cv2.dilate(work, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue
            if w < 4 or h < 4:
                continue

            pad_out = 2
            l = max(0, x - pad_out)
            t = max(0, y - pad_out)
            r = min(roi.right - roi.left, x + w + pad_out)
            b = min(roi.bottom - roi.top, y + h + pad_out)
            regions.append((roi.left + l, roi.top + t, roi.left + r, roi.top + b))

    # Border-based detection: find rectangular edges and keep those with strong color inside.
    try:
        roi_img = image.crop((roi.left, roi.top, roi.right, roi.bottom)).convert("RGB")
        roi_arr = np.array(roi_img)
        gray = cv2.cvtColor(roi_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue
            if w < 30 or h < 12:
                continue

            # Ratio of target color inside the rectangle
            rect_mask = combined_mask[y:y + h, x:x + w]
            if rect_mask.size == 0:
                continue
            color_ratio = float(np.count_nonzero(rect_mask)) / float(rect_mask.size)
            if color_ratio < 0.12:
                continue

            # Border edge density (approximate "hard border")
            rect = np.ones((h, w), dtype=np.uint8)
            inner = cv2.erode(rect, np.ones((3, 3), dtype=np.uint8), iterations=1)
            border = rect - inner
            border_edges = edges[y:y + h, x:x + w]
            border_count = int(np.count_nonzero(border))
            if border_count == 0:
                continue
            border_ratio = float(np.count_nonzero(border_edges[border == 1])) / float(border_count)
            if border_ratio < 0.15:
                continue

            regions.append((roi.left + x, roi.top + y, roi.left + x + w, roi.top + y + h))
    except Exception:
        # Best-effort only; ignore border detection failures.
        pass

    # Additional border-first rectangles that match target color (handles blue surrounded by blue)
    border_color_regions = ctx._nm_find_border_color_regions(image, roi, target_color, max_area)
    if border_color_regions:
        regions.extend(border_color_regions)

    # Merge very-close regions to handle tight multi-part icons (e.g., Windows logo)
    def merge_close(regs: RegionList, gap: int) -> RegionList:
        if not regs:
            return regs
        merged = regs[:]
        changed = True
        while changed:
            changed = False
            result = []
            used = [False] * len(merged)
            for i, (l1, t1, r1, b1) in enumerate(merged):
                if used[i]:
                    continue
                nl, nt, nr, nb = l1, t1, r1, b1
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    l2, t2, r2, b2 = merged[j]
                    if (nl - gap) <= r2 and (nr + gap) >= l2 and (nt - gap) <= b2 and (nb + gap) >= t2:
                        nl = min(nl, l2)
                        nt = min(nt, t2)
                        nr = max(nr, r2)
                        nb = max(nb, b2)
                        used[j] = True
                        changed = True
                result.append((nl, nt, nr, nb))
            merged = result
        return merged

    regions = merge_close(regions, gap=3)

    # Ensure any remaining masked components become regions (so visible masks yield candidates).
    try:
        comp_mask = (combined_mask > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(comp_mask, connectivity=8)
        if num > 1:
            region_mask = np.zeros_like(comp_mask, dtype=np.uint8)
            for l, t, r, b in regions:
                rl = max(0, l - roi.left)
                rt = max(0, t - roi.top)
                rr = min(roi.right - roi.left, r - roi.left)
                rb = min(roi.bottom - roi.top, b - roi.top)
                if rr > rl and rb > rt:
                    region_mask[rt:rb, rl:rr] = 1

            extras: RegionList = []
            for label in range(1, num):
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                w = int(stats[label, cv2.CC_STAT_WIDTH])
                h = int(stats[label, cv2.CC_STAT_HEIGHT])
                area = max(1, w * h)
                if area < min_area:
                    continue
                if max_area > 0 and area > max_area:
                    continue
                if w < 2 or h < 2:
                    continue
                sub = region_mask[y:y + h, x:x + w]
                if sub.size > 0 and np.any(sub):
                    continue
                extras.append((roi.left + x, roi.top + y, roi.left + x + w, roi.top + y + h))

            if extras:
                regions.extend(extras)
                regions = merge_close(regions, gap=3)
    except Exception:
        pass

    regions = _apply_brightness_filter(regions)
    return (regions, masks_u8) if return_masks else regions


def build_color_candidates(
    ctx: Any,
    snapshot: Any,
    roi: ScreenRegion,
    color_regions: RegionList,
    start_id: int,
    candidate_cls: Any,
) -> Tuple[List[Any], int, int, int, int]:
    """Build color-mask CandidatePackage entries."""
    candidates: List[Any] = []
    next_id = start_id

    color_candidates_added = 0
    color_candidates_skipped = 0
    roi_area = max(1, (roi.right - roi.left) * (roi.bottom - roi.top))
    color_noise_area_min = max(60, int(roi_area * 0.00002))

    for left, top, right, bottom in color_regions:
        area = max(1, (right - left) * (bottom - top))
        if area < color_noise_area_min:
            color_candidates_skipped += 1
            continue
        click_x = (left + right) // 2
        click_y = (top + bottom) // 2
        color = ctx._nm_compute_region_color(snapshot.image, (left, top, right, bottom))

        size = (right - left, bottom - top)
        aspect = size[0] / max(1, size[1])
        if 0.8 <= aspect <= 8.0:
            shape_score = 1.0
        elif 0.5 <= aspect <= 12.0:
            shape_score = 0.7
        else:
            shape_score = 0.5

        wx = (snapshot.window_rect[0] + snapshot.window_rect[2]) / 2.0
        wy = (snapshot.window_rect[1] + snapshot.window_rect[3]) / 2.0
        dx = (click_x - wx) / max(1.0, snapshot.window_rect[2] - snapshot.window_rect[0])
        dy = (click_y - wy) / max(1.0, snapshot.window_rect[3] - snapshot.window_rect[1])
        dist_norm = math.sqrt(dx * dx + dy * dy)
        location_score = max(0.0, 1.0 - dist_norm)

        area_ratio = area / roi_area
        scores = {
            "text_match": 0.0,
            "shape_plausibility": shape_score,
            "location_match": location_score,
        }

        size_penalty = 0.0
        if area_ratio > 0.25:
            size_penalty = min(1.5, (area_ratio - 0.25) * 4.0)

        total = (
            scores["shape_plausibility"] * 0.6
            + scores["location_match"] * 0.4
            - size_penalty
        )

        candidates.append(
            candidate_cls(
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
        color_candidates_added += 1

    return (
        candidates,
        next_id,
        color_candidates_added,
        color_candidates_skipped,
        color_noise_area_min,
    )
