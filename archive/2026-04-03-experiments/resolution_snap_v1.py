#!/usr/bin/env python3
"""
Resolution Snap — AI Outpainting Bucketer for ComfyUI
═════════════════════════════════════════════════════════════════
Snaps oddly-sized images to the nearest standard resolution
(720p / 1080p / 1440p / 4K / 8K), centers the original on a
blue-padded canvas, and queues ComfyUI to AI-fill the borders.

Result: no more "3824×2150" files — only clean, standard sizes.

WORKFLOW REQUIREMENT:
  Uses a companion ComfyUI workflow (API-format JSON) with:
    • Node 8  — LoadImage       (receives the padded canvas)
    • Node 14 — ImageToMask     (extracts blue channel → mask)
    • Node 6  — InpaintModelConditioning
    • Node 10 — SaveImage       (filename_prefix is set by script)
  The mask is auto-generated from the pure-blue padding.
"""

import os
import sys
import json
import time
import uuid
import requests
from PIL import Image
from pathlib import Path
from urllib.parse import urlencode

# ══════════════════════════════ CONFIGURATION ═════════════════════════════════

INPUT_DIR     = r"C:\Path\To\Your\Images"
OUTPUT_DIR    = r"C:\Path\To\Your\Output"
COMFY_URL     = "http://127.0.0.1:8188"
WORKFLOW_PATH = "workflow1.json"

PADDING_COLOR = (0, 0, 255)       # Pure Blue — triggers ImageToMask (blue channel)
DRY_RUN       = False              # True = preview bucketing without touching ComfyUI
POLL_INTERVAL = 2.0                # Seconds between completion checks
JOB_TIMEOUT   = 600                # Max seconds to wait per image

# Landscape resolution buckets (width, height).
# Portrait is auto-derived by swapping w↔h.
# Sorted smallest → largest — the script picks the SMALLEST that fits.
RESOLUTION_BUCKETS = [
    (1280,   720),   #  720p   HD
    (1920,  1080),   # 1080p   Full HD
    (2560,  1440),   # 1440p   QHD
    (3840,  2160),   # 4K      UHD
    (7680,  4320),   # 8K      UHD
]

# Human-readable labels for console output
BUCKET_LABELS = {
    (1280,   720): "720p",
    (1920,  1080): "1080p",
    (2560,  1440): "1440p",
    (3840,  2160): "4K",
    (7680,  4320): "8K",
}

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}

# ══════════════════════════════ BUCKET LOGIC ══════════════════════════════════

def get_nearest_bucket(width: int, height: int) -> tuple[int, int, str]:
    """
    Find the smallest standard bucket that fully contains the image.

    Handles landscape, portrait, and square orientations automatically.
    If no bucket is large enough, returns the largest bucket and the
    canvas builder will downscale to fit.

    Returns:
        (target_width, target_height, action_string)
    """
    is_portrait = height > width

    # Normalize to landscape for uniform comparison
    w = height if is_portrait else width
    h = width  if is_portrait else height

    # Pass 1 — smallest bucket that fully contains the image
    for bw, bh in RESOLUTION_BUCKETS:
        if w <= bw and h <= bh:
            if is_portrait:
                return bh, bw, "pad"
            return bw, bh, "pad"

    # Pass 2 — image exceeds every bucket → use largest + downscale
    bw, bh = RESOLUTION_BUCKETS[-1]
    if is_portrait:
        return bh, bw, "downscale+pad"
    return bw, bh, "downscale+pad"


def bucket_label(w: int, h: int) -> str:
    """Human-readable bucket name, e.g. '4K' or '1080p portrait'."""
    # Normalize to landscape to look up
    lw, lh = (max(w, h), min(w, h))
    name = BUCKET_LABELS.get((lw, lh), f"{w}×{h}")
    if h > w:
        name += " portrait"
    return name

# ═════════════════════════════ CANVAS BUILDER ═════════════════════════════════

def build_canvas(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Center the source image on a blue canvas at the target resolution.

    If the source is larger than the target on either axis, it is
    proportionally downscaled (LANCZOS) to fit first.  The surrounding
    blue area becomes the inpainting mask via the workflow's
    ImageToMask node.

    NOTE: The source image should not contain large regions of pure
    blue (0, 0, 255) or those pixels will also be masked/filled.
    """
    src = img.copy()

    # Downscale to fit if the image overflows the bucket
    if src.width > target_w or src.height > target_h:
        ratio = min(target_w / src.width, target_h / src.height)
        new_w = round(src.width  * ratio)
        new_h = round(src.height * ratio)
        src = src.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new('RGB', (target_w, target_h), PADDING_COLOR)
    offset_x = (target_w  - src.width)  // 2
    offset_y = (target_h  - src.height) // 2
    canvas.paste(src, (offset_x, offset_y))
    return canvas

# ══════════════════════════════ COMFYUI API ═══════════════════════════════════

def comfy_is_reachable() -> bool:
    """Quick connectivity check."""
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def comfy_upload_image(filepath: str, upload_name: str) -> dict:
    """
    Upload an image to ComfyUI's /input directory via the REST API.
    The returned name is what you pass into a LoadImage node.
    """
    with open(filepath, 'rb') as f:
        resp = requests.post(
            f"{COMFY_URL}/upload/image",
            files={'image': (upload_name, f, 'image/png')},
            data={'overwrite': 'true'},
        )
    resp.raise_for_status()
    return resp.json()


def comfy_queue(workflow: dict, client_id: str) -> str:
    """Queue a prompt and return its prompt_id."""
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
    """
    Poll /history/{prompt_id} until the job finishes.
    Returns the full history entry (including 'outputs').
    """
    url = f"{COMFY_URL}/history/{prompt_id}"
    t0 = time.time()
    while time.time() - t0 < JOB_TIMEOUT:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if prompt_id in data:
                    entry = data[prompt_id]
                    # Check for completed output images
                    outputs = entry.get("outputs", {})
                    for node_out in outputs.values():
                        if node_out.get("images"):
                            return entry
                    # Check for explicit error status
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(
                            f"ComfyUI job failed:\n{json.dumps(status, indent=2)}"
                        )
        except requests.ConnectionError:
            pass   # ComfyUI might briefly drop during heavy loads
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {prompt_id} did not finish within {JOB_TIMEOUT}s")


def comfy_download_output(job_result: dict, save_path: str) -> str | None:
    """Download the first output image from a completed job to disk."""
    for _node_id, node_out in job_result.get("outputs", {}).items():
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

# ═══════════════════════════ MAIN PROCESSING LOOP ════════════════════════════

def main():
    # ── Header ──
    print("=" * 62)
    print("   Resolution Snap — AI Outpainting Bucketer for ComfyUI")
    print("=" * 62)

    # ── Validate paths ──
    if not os.path.isdir(INPUT_DIR):
        print(f"\n  [ERROR] Input directory not found:\n    {INPUT_DIR}")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(WORKFLOW_PATH):
        print(f"\n  [ERROR] Workflow JSON not found:\n    {WORKFLOW_PATH}")
        sys.exit(1)
    with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
        base_workflow = json.load(f)

    # ── ComfyUI connectivity ──
    if not DRY_RUN:
        print(f"\n  Connecting to ComfyUI at {COMFY_URL} …", end=" ")
        if not comfy_is_reachable():
            print("FAILED")
            print("  Make sure ComfyUI is running, then retry.")
            sys.exit(1)
        print("OK")
    else:
        print("\n  ** DRY RUN — previewing bucket assignments only **")

    # ── Gather files ──
    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    )
    if not files:
        print(f"\n  No image files found in {INPUT_DIR}")
        return

    # ── Print bucket table ──
    print(f"\n  Available buckets:")
    for bw, bh in RESOLUTION_BUCKETS:
        print(f"    {bw:>5} x {bh:<5}  {BUCKET_LABELS.get((bw,bh),'')}")
    print(f"\n  {len(files)} image(s) to process\n")

    client_id = str(uuid.uuid4())
    stats = {"filled": 0, "copied": 0, "failed": 0}

    for idx, filename in enumerate(files, 1):
        img_path = os.path.join(INPUT_DIR, filename)
        stem = Path(filename).stem
        tag  = f"[{idx}/{len(files)}]"

        # ── Open image ──
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as exc:
            print(f"  X {tag} {filename} — cannot open ({exc})")
            stats["failed"] += 1
            continue

        ow, oh = img.size
        tw, th, action = get_nearest_bucket(ow, oh)
        label = bucket_label(tw, th)

        # ── Already at target? ──
        if ow == tw and oh == th:
            print(f"  = {tag} {filename} ({ow}x{oh}) — already {label}, copying")
            img.save(os.path.join(OUTPUT_DIR, f"{stem}.png"))
            stats["copied"] += 1
            continue

        # ── Report planned action ──
        arrow = "downscale+center" if "downscale" in action else "center+pad"
        print(f"  > {tag} {filename}")
        print(f"         {ow}x{oh}  -->  {tw}x{th}  ({label})  [{arrow}]")

        if DRY_RUN:
            stats["filled"] += 1
            continue

        # ── Build blue-padded canvas ──
        canvas = build_canvas(img, tw, th)

        temp_name = f"_resnap_{uuid.uuid4().hex[:8]}.png"
        temp_path = os.path.join(OUTPUT_DIR, temp_name)
        canvas.save(temp_path, "PNG")

        try:
            # Upload canvas to ComfyUI
            comfy_upload_image(temp_path, temp_name)

            # Prepare a fresh copy of the workflow
            workflow = json.loads(json.dumps(base_workflow))
            workflow["8"]["inputs"]["image"]           = temp_name
            workflow["10"]["inputs"]["filename_prefix"] = f"ResSnap/{stem}"

            # Queue and wait
            prompt_id = comfy_queue(workflow, client_id)
            print(f"         queued ({prompt_id[:12]}…) ", end="", flush=True)

            result = comfy_wait(prompt_id)

            # Download finished image
            out_path = os.path.join(OUTPUT_DIR, f"{stem}_{tw}x{th}.png")
            saved = comfy_download_output(result, out_path)

            if saved:
                print(f" -->  {Path(saved).name}")
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
    print("\n" + "=" * 62)
    print("   SUMMARY")
    print("=" * 62)
    print(f"   AI-filled:       {stats['filled']}")
    print(f"   Copied as-is:    {stats['copied']}")
    print(f"   Failed:          {stats['failed']}")
    print(f"   Output folder:   {OUTPUT_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()