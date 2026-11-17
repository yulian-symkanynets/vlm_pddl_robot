"""Image resolution per request: fewer image tokens, faster perception.

The VLM's cost grows with the number of image tokens it has to read. Qwen-VL models
turn every 32 x 32 pixel block of the (resized) image into one token, so a 1280 x 960
picture is about 1200 tokens and a 448 x 336 picture about 150. This module lets each
request choose how much detail the model gets:

    "resolution": "low" | "medium" | "high" | "full" | <max side in pixels> | "auto"
    "crop": [x0, y0, x1, y1]           optional, fractions 0..1 of the image (the work area)

    low      448 px longest side,  256 tokens budget
    medium   768 px,               512
    high     1280 px,             1024   (the server's default before this module)
    full     no resizing,          the server's --max-visual-tokens
    auto     low first; if that result is unreliable, medium; then high

The resizing happens on the server, so the robot side does not change. Bytes on the
network only go down if the laptop also sends a smaller picture.

"Unreliable" in auto mode means: the model's scene could not be read, an object's
support is unknown, an object is reported hidden, or the instruction names something
the model did not see. A plan that was found, or a real "no plan" / "clarify" about the
task itself, is accepted at the lowest level that produced it.

Cropping to the work area keeps detail where the objects are and drops the rest of
the frame (robot base, cables, floor), so the same token budget sees the objects larger.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

LEVELS = {                         # name: (max side in px, visual-token budget)
    "low": (448, 256),
    "medium": (768, 512),
    "high": (1280, 1024),
}
AUTO_ORDER = ("low", "medium", "high")
TOKEN_PX = 32                      # one visual token per 32 x 32 px after the 2 x 2 merge (Qwen-VL)


def tokens_for(width: int, height: int) -> int:
    """Approximate number of visual tokens for an image of this size."""
    return math.ceil(width / TOKEN_PX) * math.ceil(height / TOKEN_PX)


def parse(resolution, default_tokens: int) -> tuple[str, int | None, int]:
    """-> (label, max side in px or None for no resizing, token budget)."""
    if resolution in (None, "", "full"):
        return "full", None, default_tokens
    if isinstance(resolution, str) and resolution in LEVELS:
        side, tokens = LEVELS[resolution]
        return resolution, side, tokens
    try:
        side = int(resolution)
    except (TypeError, ValueError):
        raise ValueError(f"resolution must be one of {', '.join(LEVELS)}, full, auto, or a number of pixels")
    if not 64 <= side <= 4096:
        raise ValueError("resolution in pixels must be between 64 and 4096")
    # budget = tokens of a 4:3 image with this longest side, rounded up to a cached size
    need = tokens_for(side, side * 3 // 4)
    budget = next((b for b in (256, 512, 768, 1024, 1536, 2048, 4096) if b >= need), 4096)
    return f"{side}px", side, budget


def prepare(paths: list[Path], resolution=None, crop=None, default_tokens: int = 1024,
            quality: int = 90) -> tuple[list[Path], dict]:
    """Crop and resize the request's images (new files next to the originals).

    Returns (paths for the model, info with sizes, approximate tokens and the token budget).
    """
    label, side, budget = parse(resolution, default_tokens)
    if crop is not None:
        if len(crop) != 4 or not all(0 <= float(c) <= 1 for c in crop) or crop[0] >= crop[2] or crop[1] >= crop[3]:
            raise ValueError("crop must be [x0, y0, x1, y1] with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    info = {"resolution": label, "max_visual_tokens": budget, "crop": crop, "images": []}
    if side is None and crop is None:
        for p in paths:
            w, h = _size(p)
            info["images"].append({"file": p.name, "width": w, "height": h, "bytes": p.stat().st_size,
                                   "approx_tokens": min(tokens_for(w, h), budget)})
        return list(paths), info
    from PIL import Image, ImageOps
    out = []
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if crop is not None:
                w, h = im.size
                im = im.crop((round(crop[0] * w), round(crop[1] * h), round(crop[2] * w), round(crop[3] * h)))
            if side is not None and max(im.size) > side:
                scale = side / max(im.size)
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
            q = p.with_name(f"{p.stem}_{label}{'_crop' if crop else ''}.jpg")
            im.save(q, "JPEG", quality=quality)
            info["images"].append({"file": q.name, "width": im.width, "height": im.height,
                                   "bytes": q.stat().st_size, "approx_tokens": min(tokens_for(*im.size), budget)})
            out.append(q)
    return out, info


def _size(p: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def unreliable(res: dict) -> str | None:
    """Why a perception result at this resolution should not be trusted, or None."""
    msg = (res.get("message") or "").lower()
    if res.get("status") == "error" and ("perception" in msg or msg.startswith("scene")):
        return "the model's scene could not be read"
    if res.get("status") == "need_observation":
        return "an object's support is unknown"
    if (res.get("scene") or {}).get("hidden"):
        return "an object is reported hidden"
    if res.get("status") == "clarify" and ("not in the scene" in msg or "do not see" in msg):
        return "the instruction names an object the model did not see"
    return None


def run_with_resolution(run_fn, images: list[Path], resolution=None, crop=None,
                        default_tokens: int = 1024, **kw) -> dict:
    """Run the pipeline (`run_fn`, e.g. openworld.run_until_ok) on prepared images.

    resolution="auto" tries low, medium, high and stops at the first reliable result.
    The response gets a "resolution" field: what was used and every level tried.
    """
    levels = list(AUTO_ORDER) if resolution == "auto" else [resolution]
    tried, res = [], None
    for k, level in enumerate(levels):
        t0 = time.perf_counter()
        prepared, info = prepare(images, level, crop, default_tokens)
        res = run_fn(images=prepared, max_visual_tokens=info["max_visual_tokens"], **kw)
        why = unreliable(res) if resolution == "auto" else None
        tried.append({"resolution": info["resolution"], "max_visual_tokens": info["max_visual_tokens"],
                      "approx_tokens": sum(i["approx_tokens"] for i in info["images"]),
                      "image_bytes": sum(i["bytes"] for i in info["images"]),
                      "status": res.get("status"), "seconds": round(time.perf_counter() - t0, 3),
                      "not_reliable_because": why})
        if why is None or k == len(levels) - 1:
            break
    res["resolution"] = {"requested": resolution if resolution is not None else "full",
                         "used": tried[-1]["resolution"], "crop": crop, "tried": tried}
    return res


def perceive_with_resolution(perceive_fn, images: list[Path], resolution=None, crop=None,
                             default_tokens: int = 1024):
    """Perception only (no planning). `perceive_fn(images, max_visual_tokens)` returns
    (scene or None, error or None, model output). Same levels and auto rule as above.
    Returns (scene, error, model output, resolution info)."""
    levels = list(AUTO_ORDER) if resolution == "auto" else [resolution]
    tried = []
    for k, level in enumerate(levels):
        t0 = time.perf_counter()
        prepared, info = prepare(images, level, crop, default_tokens)
        scene, err, g = perceive_fn(prepared, info["max_visual_tokens"])
        why = None
        if resolution == "auto":
            sup = (scene or {}).get("support") or {}
            if err:
                why = "the model's scene could not be read"
            elif any(str(v).lower() in ("unknown", "?", "none", "") for v in sup.values()):
                why = "an object's support is unknown"
            elif (scene or {}).get("hidden"):
                why = "an object is reported hidden"
        tried.append({"resolution": info["resolution"], "max_visual_tokens": info["max_visual_tokens"],
                      "approx_tokens": sum(i["approx_tokens"] for i in info["images"]),
                      "image_bytes": sum(i["bytes"] for i in info["images"]),
                      "seconds": round(time.perf_counter() - t0, 3), "not_reliable_because": why})
        if why is None or k == len(levels) - 1:
            break
    return scene, err, g, {"requested": resolution if resolution is not None else "full",
                           "used": tried[-1]["resolution"], "crop": crop, "tried": tried}
