#!/usr/bin/env python3
"""
Resolution Snap v2 — AI Outpainting Bucketer for ComfyUI
═══════════════════════════════════════════════════════════════════
Snaps oddly-sized images to the nearest standard resolution,
centers the original on a blue-padded canvas, and queues ComfyUI
to AI-fill the borders.

No more "3824×2150" output files — only clean, standard sizes.

Usage:
    python resolution_snap.py              # full run
    python resolution_snap.py --dry-run    # preview assignments only
    python resolution_snap.py --list-buckets  # dump the bucket table

WORKFLOW REQUIREMENT (API-format JSON):
    • Node 8  — LoadImage   (receives the padded canvas)
    • Node 14 — ImageToMask (extracts blue channel → inpaint mask)
    • Node 6  — InpaintModelConditioning
    • Node 10 — SaveImage   (filename_prefix set by script)
"""

import os
import sys
import json
import time
import uuid
import argparse
import requests
from math import gcd
from PIL import Image
from pathlib import Path
from urllib.parse import urlencode

# ══════════════════════════════ CONFIGURATION ═════════════════════════════════

INPUT_DIR     = r"C:\Path\To\Your\Images"
OUTPUT_DIR    = r"C:\Path\To\Your\Output"
COMFY_URL     = "http://127.0.0.1:8188"
WORKFLOW_PATH = "outpainting_workflow_v2.json"

PADDING_COLOR = (0, 0, 255)       # Pure blue — triggers ImageToMask (blue ch.)
POLL_INTERVAL = 2.0                # Seconds between completion checks
JOB_TIMEOUT   = 600                # Max seconds to wait per image

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}

# ══════════════════════════════ RESOLUTION BUCKETS ════════════════════════════
#
#   All landscape-oriented (width ≥ height).  Portrait is auto-derived.
#   Sorted at startup by pixel area so the matcher always finds the
#   SMALLEST bucket that fully contains the source image.
#
#   ~70 buckets covering:
#     16:9 · 16:10 · 4:3 · 3:2 · 21:9 · 32:9 · 5:4 · 1:1
#     Standard display · Cinema DCI · Camera sensor · Social media
#
RESOLUTION_BUCKETS_RAW = [

    # ── Tiny / Thumbnail / Web ─────────────────────────────────
    (426,  240),       # 240p
    (480,  320),       # HVGA               3:2
    (640,  360),       # 360p  nHD          16:9
    (640,  480),       # VGA                 4:3
    (800,  480),       # WVGA                5:3
    (800,  600),       # SVGA                4:3
    (854,  480),       # 480p  FWVGA        16:9

    # ── Square ─────────────────────────────────────────────────
    (512,  512),       # Common AI / icon     1:1
    (768,  768),       # SD 1.5 sweet spot    1:1
    (1024, 1024),      # SDXL / 1K square     1:1
    (1080, 1080),      # Instagram square     1:1
    (1440, 1440),      # Social hi-res        1:1
    (2048, 2048),      # 2K square            1:1
    (2160, 2160),      # UHD square           1:1
    (4096, 4096),      # 4K square            1:1

    # ── Sub-HD ─────────────────────────────────────────────────
    (960,  540),       # qHD                 16:9
    (960,  640),       # DVGA                 3:2
    (1024, 576),       # WSVGA               16:9
    (1024, 600),       #                     ~16:9
    (1024, 768),       # XGA                  4:3
    (1152, 648),       #                     16:9
    (1152, 720),       #                      8:5
    (1152, 864),       # XGA+                 4:3

    # ── HD tier ────────────────────────────────────────────────
    (1280, 720),       # 720p  HD            16:9
    (1280, 768),       # WXGA                 5:3
    (1280, 800),       # WXGA                16:10
    (1280, 960),       # SXGA⁻                4:3
    (1280, 1024),      # SXGA                  5:4
    (1366, 768),       # HD    laptop        ~16:9
    (1440, 900),       # WXGA+               16:10
    (1440, 960),       #                      3:2
    (1600, 900),       # HD+                 16:9
    (1600, 1024),      #                     25:16
    (1600, 1200),      # UXGA                 4:3
    (1680, 1050),      # WSXGA+              16:10

    # ── Full HD tier ───────────────────────────────────────────
    (1920, 1080),      # 1080p FHD           16:9
    (1920, 1200),      # WUXGA               16:10
    (1920, 1280),      #                      3:2
    (2048, 1080),      # 2K DCI Cinema       ~17:9
    (2048, 1152),      # QWXGA               16:9
    (2048, 1280),      #                     16:10
    (2048, 1536),      # QXGA                 4:3

    # ── Ultrawide FHD ──────────────────────────────────────────
    (2560, 1080),      # UW-FHD              21:9
    (3440, 1080),      #                     ~32:10
    (3840, 1080),      # DFHD dual           32:9

    # ── QHD tier ───────────────────────────────────────────────
    (2560, 1440),      # 1440p QHD           16:9
    (2560, 1600),      # WQXGA               16:10
    (2560, 1700),      #                     ~3:2
    (2880, 1620),      #                     16:9
    (2880, 1800),      # Retina MacBook      16:10
    (2880, 1920),      #                      3:2

    # ── Ultrawide QHD ──────────────────────────────────────────
    (3440, 1440),      # UW-QHD              21:9
    (3840, 1600),      # UW-QHD+            ~21:9
    (5120, 1440),      # DQHD super UW       32:9

    # ── 3K / QHD+ ─────────────────────────────────────────────
    (3200, 1800),      # QHD+                16:9
    (3200, 2000),      #                     16:10
    (3200, 2400),      # QUXGA                4:3
    (3000, 2000),      # 6 MP                 3:2

    # ── Camera / Photo Sensors ─────────────────────────────────
    (4000, 3000),      # 12 MP                4:3
    (4032, 3024),      # 12 MP  iPhone         4:3
    (4500, 3000),      # ~13 MP               3:2
    (4624, 3468),      # 16 MP                 4:3
    (6000, 4000),      # 24 MP                3:2
    (6016, 4016),      # 24 MP                ~3:2
    (8000, 6000),      # 48 MP                 4:3

    # ── 4K tier ────────────────────────────────────────────────
    (3840, 2160),      # 4K UHD              16:9
    (3840, 2400),      # WQUXGA              16:10
    (4096, 2160),      # 4K DCI Cinema       ~17:9
    (4096, 2304),      #                     16:9
    (4096, 2560),      #                     16:10
    (4096, 3072),      # HXGA                  4:3

    # ── Ultrawide 4K ───────────────────────────────────────────
    (5120, 2160),      # UW-4K               21:9
    (5120, 2880),      # 5K                  16:9
    (5120, 3200),      # WHXGA               16:10
    (5760, 2400),      # UW                  ~21:9

    # ── 6K tier ────────────────────────────────────────────────
    (6016, 3384),      # Apple Pro Display 6K 16:9
    (6144, 3456),      # 6K                  16:9
    (6144, 4608),      # 6K                    4:3

    # ── 8K tier ────────────────────────────────────────────────
    (7680, 4320),      # 8K UHD              16:9
    (7680, 4800),      # 8K                  16:10
    (8192, 4320),      # 8K DCI              ~17:9
    (8192, 4608),      #                     16:9
    (8192, 6144),      #                       4:3
]

# Sort by total pixel area — matcher walks this list and picks the first fit
RESOLUTION_BUCKETS = sorted(
    RESOLUTION_BUCKETS_RAW,
    key=lambda b: (b[0] * b[1], b[0])
)

# ── Well-known shorthand names ────────────────────────────────────────────────

_BUCKET_NAMES = {
    (426,  240):  "240p",        (640,  360):  "360p",
    (854,  480):  "480p",        (640,  480):  "VGA",
    (800,  600):  "SVGA",        (1024, 768):  "XGA",
    (1280, 720):  "720p",        (1280, 800):  "WXGA",
    (1280, 1024): "SXGA",        (1366, 768):  "HD",
    (1600, 900):  "HD+",         (1600, 1200): "UXGA",
    (1680, 1050): "WSXGA+",      (1920, 1080): "1080p",
    (1920, 1200): "WUXGA",       (2048, 1080): "2K DCI",
    (2560, 1080): "UW-FHD",      (2560, 1440): "1440p",
    (2560, 1600): "WQXGA",       (2880, 1800): "Retina",
    (3440, 1440): "UW-QHD",      (3840, 1600): "UW-QHD+",
    (3200, 1800): "QHD+",        (3200, 2400): "QUXGA",
    (3840, 2160): "4K",          (3840, 2400): "WQUXGA",
    (4096, 2160): "4K DCI",      (5120, 1440): "DQHD",
    (5120, 2880): "5K",          (6016, 3384): "Apple 6K",
    (7680, 4320): "8K",          (8192, 4320): "8K DCI",
}


def _aspect_label(w: int, h: int) -> str:
    """Simplify an aspect ratio, e.g. 1920×1080 → '16:9'."""
    g = gcd(w, h)
    aw, ah = w // g, h // g
    # Collapse common near-ratios
    known = {
        (16, 9): "16:9",   (16, 10): "16:10",  (4, 3): "4:3",
        (3, 2): "3:2",     (5, 4): "5:4",      (5, 3): "5:3",
        (1, 1): "1:1",     (21, 9): "21:9",    (32, 9): "32:9",
        (8, 5): "16:10",   (683, 384): "~16:9",
        (85, 48): "16:9",  (128, 75): "~16:9",
    }
    return known.get((aw, ah), f"{aw}:{ah}")


def bucket_label(w: int, h: int) -> str:
    """Human-readable bucket name: '4K 16:9' or '1080p portrait 16:9'."""
    lw, lh = (max(w, h), min(w, h))
    name = _BUCKET_NAMES.get((lw, lh), f"{w}×{h}")
    ar = _aspect_label(lw, lh)
    suffix = " portrait" if h > w else ""
    return f"{name}{suffix} ({ar})"

# ══════════════════════════════ BUCKET MATCHER ════════════════════════════════

def get_nearest_bucket(width: int, height: int) -> tuple[int, int, str]:
    """
    Find the smallest standard bucket that fully contains the image.

    Handles landscape, portrait, and square automatically.
    If no bucket is large enough, returns the largest bucket and
    the canvas builder will downscale to fit.

    Returns:
        (target_width, target_height, action)
        action is one of: "exact", "pad", "downscale+pad"
    """
    # Exact match shortcut
    for bw, bh in RESOLUTION_BUCKETS:
        if (width == bw and height == bh) or (width == bh and height == bw):
            return width, height, "exact"

    is_portrait = height > width
    is_square   = height == width

    # Normalize to landscape for comparison
    w = height if is_portrait else width
    h = width  if is_portrait else height

    # Walk sorted buckets — first match is smallest containing bucket
    for bw, bh in RESOLUTION_BUCKETS:
        if w <= bw and h <= bh:
            if is_square and bw == bh:
                return bw, bh, "pad"
            if is_portrait:
                return bh, bw, "pad"
            return bw, bh, "pad"

    # Exceeds every bucket → downscale into the largest one
    bw, bh = RESOLUTION_BUCKETS[-1]
    if is_portrait:
        return bh, bw, "downscale+pad"
    return bw, bh, "downscale+pad"

# ═════════════════════════════ CANVAS BUILDER ═════════════════════════════════

def build_canvas(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Center the source image on a blue canvas at the target resolution.

    If the source overflows the target on either axis, it is
    proportionally downscaled (LANCZOS) to fit first.

    The blue border becomes the inpainting mask via ImageToMask.
    """
    src = img.copy()

    if src.width > target_w or src.height > target_h:
        ratio = min(target_w / src.width, target_h / src.height)
        src = src.resize(
            (round(src.width * ratio), round(src.height * ratio)),
            Image.LANCZOS
        )

    canvas = Image.new('RGB', (target_w, target_h), PADDING_COLOR)
    offset_x = (target_w  - src.width)  // 2
    offset_y = (target_h  - src.height) // 2
    canvas.paste(src, (offset_x, offset_y))
    return canvas

# ══════════════════════════════ COMFYUI API ═══════════════════════════════════

def comfy_is_reachable() -> bool:
    try:
        return requests.get(f"{COMFY_URL}/system_stats", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def comfy_upload_image(filepath: str, upload_name: str) -> dict:
    with open(filepath, 'rb') as f:
        resp = requests.post(
            f"{COMFY_URL}/upload/image",
            files={'image': (upload_name, f, 'image/png')},
            data={'overwrite': 'true'},
        )
    resp.raise_for_status()
    return resp.json()


def comfy_queue(workflow: dict, client_id: str) -> str:
    resp = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
    )
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"ComfyUI rejected prompt: {body['error']}")
    return body["prompt_id"]


def comfy_wait(prompt_id: str) -> dict:
    url = f"{COMFY_URL}/history/{prompt_id}"
    t0  = time.time()
    while time.time() - t0 < JOB_TIMEOUT:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if prompt_id in data:
                    entry = data[prompt_id]
                    outputs = entry.get("outputs", {})
                    for node_out in outputs.values():
                        if node_out.get("images"):
                            return entry
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(
                            f"ComfyUI job failed:\n{json.dumps(status, indent=2)}"
                        )
        except requests.ConnectionError:
            pass
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {prompt_id} did not finish within {JOB_TIMEOUT}s")


def comfy_download_output(job_result: dict, save_path: str) -> str | None:
    for _nid, node_out in job_result.get("outputs", {}).items():
        for img_info in node_out.get("images", []):
            qs = urlencode({
                "filename":  img_info["filename"],
                "subfolder": img_info.get("subfolder", ""),
                "type":      img_info.get("type", "output"),
            })
            resp = requests.get(f"{COMFY_URL}/view?{qs}")
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(resp.content)
            return save_path
    return None

# ═══════════════════════════ BUCKET TABLE PRINTER ════════════════════════════

def print_bucket_table():
    """Pretty-print every available bucket, grouped by area tier."""

    tiers = [
        ("Tiny / Web",       0,         500_000),
        ("Square",           0,         99_999_999,  True),  # special flag
        ("Sub-HD",           500_000,   1_000_000),
        ("HD",               900_000,   1_800_000),
        ("Full HD",          1_800_000, 3_200_000),
        ("Ultrawide FHD",    2_700_000, 4_200_000),
        ("QHD",              3_600_000, 5_500_000),
        ("Ultrawide QHD",    4_900_000, 7_500_000),
        ("3K / QHD+",        5_500_000, 8_000_000),
        ("Camera / Photo",   10_000_000, 50_000_000),
        ("4K",               8_000_000, 13_000_000),
        ("5K",               11_000_000, 19_000_000),
        ("6K",               19_000_000, 30_000_000),
        ("8K",               30_000_000, 99_999_999),
    ]

    printed = set()
    print(f"\n  {'Bucket':>14}   {'Label':<20}  {'Aspect':<8}  {'Pixels':>12}")
    print(f"  {'─'*14}   {'─'*20}  {'─'*8}  {'─'*12}")

    for bw, bh in RESOLUTION_BUCKETS:
        if (bw, bh) in printed:
            continue
        lw, lh = max(bw, bh), min(bw, bh)
        name = _BUCKET_NAMES.get((lw, lh), "")
        ar   = _aspect_label(lw, lh)
        px   = bw * bh
        sq   = "■" if bw == bh else " "
        print(f"  {bw:>6} × {bh:<5}  {name:<20}  {ar:<8}  {px:>12,}")
        printed.add((bw, bh))

    print(f"\n  Total: {len(RESOLUTION_BUCKETS)} buckets")
    print(f"  (Portrait orientations are auto-derived by swapping W↔H)\n")

# ═══════════════════════════ MAIN PROCESSING LOOP ════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Resolution Snap — AI Outpainting Bucketer for ComfyUI"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview bucket assignments without processing"
    )
    parser.add_argument(
        "--list-buckets", action="store_true",
        help="Print the full bucket table and exit"
    )
    args = parser.parse_args()

    # ── Header ──
    print("=" * 66)
    print("   Resolution Snap v2 — AI Outpainting Bucketer for ComfyUI")
    print("=" * 66)

    if args.list_buckets:
        print_bucket_table()
        return

    dry_run = args.dry_run

    # ── Validate paths ──
    if not os.path.isdir(INPUT_DIR):
        print(f"\n  [ERROR] Input directory not found:\n    {INPUT_DIR}")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not dry_run and not os.path.isfile(WORKFLOW_PATH):
        print(f"\n  [ERROR] Workflow JSON not found:\n    {WORKFLOW_PATH}")
        sys.exit(1)

    base_workflow = None
    if not dry_run:
        with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
            base_workflow = json.load(f)

    # ── ComfyUI connectivity ──
    if not dry_run:
        print(f"\n  Connecting to ComfyUI at {COMFY_URL} …", end=" ")
        if not comfy_is_reachable():
            print("FAILED")
            print("  Make sure ComfyUI is running, then retry.")
            sys.exit(1)
        print("OK")
    else:
        print("\n  ▸ DRY RUN — previewing bucket assignments only")

    # ── Gather files ──
    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    )
    if not files:
        print(f"\n  No image files found in {INPUT_DIR}")
        return

    print(f"\n  {len(files)} image(s) to process")
    print(f"  {len(RESOLUTION_BUCKETS)} resolution buckets loaded\n")

    client_id = str(uuid.uuid4())
    stats     = {"filled": 0, "exact": 0, "failed": 0}

    # ── Per-bucket counters for the summary ──
    bucket_usage: dict[tuple[int, int], int] = {}

    for idx, filename in enumerate(files, 1):
        img_path = os.path.join(INPUT_DIR, filename)
        stem     = Path(filename).stem
        tag      = f"[{idx}/{len(files)}]"

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as exc:
            print(f"  ✗ {tag} {filename} — cannot open ({exc})")
            stats["failed"] += 1
            continue

        ow, oh = img.size
        tw, th, action = get_nearest_bucket(ow, oh)
        label = bucket_label(tw, th)
        bucket_usage[(tw, th)] = bucket_usage.get((tw, th), 0) + 1

        # ── Already exact standard size ──
        if action == "exact":
            print(f"  = {tag} {filename} ({ow}×{oh}) — already {label}, copying")
            img.save(os.path.join(OUTPUT_DIR, f"{stem}.png"))
            stats["exact"] += 1
            continue

        # ── Report planned action ──
        action_str = "downscale → center" if "downscale" in action else "center + pad"
        pad_w = tw - min(ow, tw)
        pad_h = th - min(oh, th)
        print(f"  ▸ {tag} {filename}")
        print(f"         {ow}×{oh}  →  {tw}×{th}  {label}")
        print(f"         [{action_str}]  padding: {pad_w}px × {pad_h}px")

        if dry_run:
            stats["filled"] += 1
            continue

        # ── Build blue-padded canvas ──
        canvas    = build_canvas(img, tw, th)
        temp_name = f"_resnap_{uuid.uuid4().hex[:8]}.png"
        temp_path = os.path.join(OUTPUT_DIR, temp_name)
        canvas.save(temp_path, "PNG")

        try:
            comfy_upload_image(temp_path, temp_name)

            workflow = json.loads(json.dumps(base_workflow))
            workflow["8"]["inputs"]["image"]            = temp_name
            workflow["10"]["inputs"]["filename_prefix"]  = f"ResSnap/{stem}"

            prompt_id = comfy_queue(workflow, client_id)
            print(f"         queued ({prompt_id[:12]}…) ", end="", flush=True)

            result   = comfy_wait(prompt_id)
            out_path = os.path.join(OUTPUT_DIR, f"{stem}_{tw}x{th}.png")
            saved    = comfy_download_output(result, out_path)

            if saved:
                print(f" →  {Path(saved).name}")
                stats["filled"] += 1
            else:
                print("  [warning] no output image in job result")
                stats["failed"] += 1

        except Exception as exc:
            print(f"\n         [ERROR] {exc}")
            stats["failed"] += 1

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ── Summary ──
    print()
    print("=" * 66)
    print("   SUMMARY")
    print("=" * 66)
    print(f"   AI-filled:       {stats['filled']}")
    print(f"   Already standard: {stats['exact']}")
    print(f"   Failed:          {stats['failed']}")
    print()

    if bucket_usage:
        print("   Bucket distribution:")
        for (bw, bh), count in sorted(bucket_usage.items(),
                                       key=lambda x: x[1], reverse=True):
            lbl = bucket_label(bw, bh)
            bar = "█" * count
            print(f"     {bw:>5}×{bh:<5}  {lbl:<28}  {count:>3}  {bar}")
        print()

    print(f"   Output folder:   {OUTPUT_DIR}")
    print("=" * 66)


if __name__ == "__main__":
    main()