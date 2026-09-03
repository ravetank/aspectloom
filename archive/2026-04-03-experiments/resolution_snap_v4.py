#!/usr/bin/env python3
"""
Resolution Snap v2.1 — AI Outpainting Bucketer for ComfyUI
═══════════════════════════════════════════════════════════════════
Snaps oddly-sized images to the nearest standard resolution,
centers the original on an RGBA canvas (transparent padding),
and queues ComfyUI to AI-fill the borders via inpainting.

The transparent padding becomes the inpaint mask automatically
via LoadImage's built-in alpha→mask conversion. No more
ImageToMask color-channel hacks.

Usage:
    python resolution_snap.py                 # full run
    python resolution_snap.py --dry-run       # preview assignments
    python resolution_snap.py --list-buckets  # dump bucket table
    python resolution_snap.py --verbose       # show WebSocket chatter
"""

import os
import sys
import json
import time
import uuid
import argparse
import threading
import requests
import websocket
from math import gcd
from PIL import Image
from pathlib import Path
from urllib.parse import urlencode

# ══════════════════════════════ CONFIGURATION ═════════════════════════════════

INPUT_DIR     = r"C:\Path\To\Your\Images"
OUTPUT_DIR    = r"C:\Path\To\Your\Output"
COMFY_URL     = "http://127.0.0.1:8188"
WORKFLOW_PATH = "outpainting_workflow_v2.json"

JOB_TIMEOUT   = 600

# Fill color for the padding area (visible in the RGB channels but
# the alpha channel is what actually controls the mask).
# Gray blends more naturally if the inpaint model peeks at pixel values.
PADDING_RGB   = (128, 128, 128)

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}

# ══════════════════════════════ RESOLUTION BUCKETS ════════════════════════════

RESOLUTION_BUCKETS_RAW = [
    # ── Tiny / Web ─────────────────────────────────────────────
    (426,  240),    (480,  320),    (640,  360),    (640,  480),
    (800,  480),    (800,  600),    (854,  480),

    # ── Square ─────────────────────────────────────────────────
    (512,  512),    (768,  768),    (1024, 1024),   (1080, 1080),
    (1440, 1440),   (2048, 2048),   (2160, 2160),   (4096, 4096),

    # ── Sub-HD ─────────────────────────────────────────────────
    (960,  540),    (960,  640),    (1024, 576),    (1024, 600),
    (1024, 768),    (1152, 648),    (1152, 720),    (1152, 864),

    # ── HD ─────────────────────────────────────────────────────
    (1280, 720),    (1280, 768),    (1280, 800),    (1280, 960),
    (1280, 1024),   (1366, 768),    (1440, 900),    (1440, 960),
    (1600, 900),    (1600, 1024),   (1600, 1200),   (1680, 1050),

    # ── Full HD ────────────────────────────────────────────────
    (1920, 1080),   (1920, 1200),   (1920, 1280),   (2048, 1080),
    (2048, 1152),   (2048, 1280),   (2048, 1536),

    # ── Ultrawide FHD ──────────────────────────────────────────
    (2560, 1080),   (3440, 1080),   (3840, 1080),

    # ── QHD ────────────────────────────────────────────────────
    (2560, 1440),   (2560, 1600),   (2560, 1700),   (2880, 1620),
    (2880, 1800),   (2880, 1920),

    # ── Ultrawide QHD ──────────────────────────────────────────
    (3440, 1440),   (3840, 1600),   (5120, 1440),

    # ── 3K / QHD+ ─────────────────────────────────────────────
    (3200, 1800),   (3200, 2000),   (3200, 2400),   (3000, 2000),

    # ── Camera / Photo ─────────────────────────────────────────
    (4000, 3000),   (4032, 3024),   (4500, 3000),   (4624, 3468),
    (6000, 4000),   (6016, 4016),   (8000, 6000),

    # ── 4K ─────────────────────────────────────────────────────
    (3840, 2160),   (3840, 2400),   (4096, 2160),   (4096, 2304),
    (4096, 2560),   (4096, 3072),

    # ── Ultrawide 4K / 5K ──────────────────────────────────────
    (5120, 2160),   (5120, 2880),   (5120, 3200),   (5760, 2400),

    # ── 6K ─────────────────────────────────────────────────────
    (6016, 3384),   (6144, 3456),   (6144, 4608),

    # ── 8K ─────────────────────────────────────────────────────
    (7680, 4320),   (7680, 4800),   (8192, 4320),   (8192, 4608),
    (8192, 6144),
]

RESOLUTION_BUCKETS = sorted(
    RESOLUTION_BUCKETS_RAW,
    key=lambda b: (b[0] * b[1], b[0])
)

_BUCKET_NAMES = {
    (426,240):"240p",       (640,360):"360p",       (854,480):"480p",
    (640,480):"VGA",        (800,600):"SVGA",       (1024,768):"XGA",
    (1280,720):"720p",      (1280,800):"WXGA",      (1280,1024):"SXGA",
    (1366,768):"HD",        (1600,900):"HD+",       (1600,1200):"UXGA",
    (1680,1050):"WSXGA+",   (1920,1080):"1080p",    (1920,1200):"WUXGA",
    (2048,1080):"2K DCI",   (2560,1080):"UW-FHD",   (2560,1440):"1440p",
    (2560,1600):"WQXGA",    (2880,1800):"Retina",   (3440,1440):"UW-QHD",
    (3840,1600):"UW-QHD+",  (3200,1800):"QHD+",     (3200,2400):"QUXGA",
    (3840,2160):"4K",       (3840,2400):"WQUXGA",   (4096,2160):"4K DCI",
    (5120,1440):"DQHD",     (5120,2880):"5K",       (6016,3384):"Apple 6K",
    (7680,4320):"8K",       (8192,4320):"8K DCI",
}


def _aspect_label(w, h):
    g = gcd(w, h); aw, ah = w // g, h // g
    known = {
        (16,9):"16:9",(16,10):"16:10",(4,3):"4:3",(3,2):"3:2",
        (5,4):"5:4",(5,3):"5:3",(1,1):"1:1",(21,9):"21:9",
        (32,9):"32:9",(8,5):"16:10",(683,384):"~16:9",
        (85,48):"16:9",(128,75):"~16:9",
    }
    return known.get((aw, ah), f"{aw}:{ah}")


def bucket_label(w, h):
    lw, lh = max(w, h), min(w, h)
    name = _BUCKET_NAMES.get((lw, lh), f"{w}×{h}")
    ar = _aspect_label(lw, lh)
    port = " portrait" if h > w else ""
    return f"{name}{port} ({ar})"

# ══════════════════════════════ BUCKET MATCHER ════════════════════════════════

def get_nearest_bucket(width, height):
    for bw, bh in RESOLUTION_BUCKETS:
        if (width == bw and height == bh) or (width == bh and height == bw):
            return width, height, "exact"

    is_portrait = height > width
    w = height if is_portrait else width
    h = width  if is_portrait else height

    for bw, bh in RESOLUTION_BUCKETS:
        if w <= bw and h <= bh:
            return (bh, bw, "pad") if is_portrait else (bw, bh, "pad")

    bw, bh = RESOLUTION_BUCKETS[-1]
    return (bh, bw, "downscale+pad") if is_portrait else (bw, bh, "downscale+pad")

# ═════════════════════════════ CANVAS BUILDER ═════════════════════════════════

def build_canvas(img: Image.Image, tw: int, th: int) -> Image.Image:
    """
    Center the source on an RGBA canvas.

    ┌─────────────────────────────────────┐
    │  alpha=0  (transparent padding)     │
    │  ┌───────────────────────────────┐  │
    │  │  alpha=255  (original image)  │  │
    │  │                               │  │
    │  └───────────────────────────────┘  │
    │                                     │
    └─────────────────────────────────────┘

    ComfyUI LoadImage reads alpha=0 as mask=1.0 (inpaint here).
    No color-channel tricks needed.
    """
    src = img.convert('RGB')

    # Downscale if source overflows target
    if src.width > tw or src.height > th:
        ratio = min(tw / src.width, th / src.height)
        src = src.resize(
            (round(src.width * ratio), round(src.height * ratio)),
            Image.LANCZOS,
        )

    # Create transparent RGBA canvas
    # RGB = neutral gray (less likely to bleed through than bright colors)
    # Alpha = 0 (transparent → mask = "inpaint this")
    pad_r, pad_g, pad_b = PADDING_RGB
    canvas = Image.new('RGBA', (tw, th), (pad_r, pad_g, pad_b, 0))

    # Convert source to RGBA with full opacity
    src_rgba = src.copy()
    src_rgba.putalpha(Image.new('L', src.size, 255))  # alpha = 255 everywhere

    ox = (tw - src.width)  // 2
    oy = (th - src.height) // 2
    canvas.paste(src_rgba, (ox, oy))

    return canvas

# ══════════════════════════ COMFYUI WEBSOCKET CLIENT ═════════════════════════

class ComfyConnection:
    def __init__(self, server_url, verbose=False):
        self.server_url = server_url.rstrip("/")
        self.client_id  = str(uuid.uuid4())
        self.verbose    = verbose
        self._ws        = None
        self._thread    = None
        self._connected = threading.Event()
        self._job_done  = threading.Event()
        self._lock      = threading.Lock()
        self._current_pid = None
        self._error       = None

    def connect(self):
        ws_url = (self.server_url
                  .replace("http://", "ws://")
                  .replace("https://", "wss://"))
        url = f"{ws_url}/ws?clientId={self.client_id}"
        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 30, "ping_timeout": 10},
            daemon=True,
        )
        self._thread.start()
        if not self._connected.wait(timeout=15):
            raise ConnectionError(
                f"WebSocket timed out. Is ComfyUI running at {self.server_url}?"
            )

    def disconnect(self):
        if self._ws:
            self._ws.close()

    def _on_open(self, ws):
        self._connected.set()
        if self.verbose:
            print("         [ws] connected")

    def _on_message(self, ws, message):
        if isinstance(message, bytes):
            return
        try:
            msg = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return
        msg_type = msg.get("type", "")
        data     = msg.get("data", {})
        with self._lock:
            pid = data.get("prompt_id")
            if pid and pid != self._current_pid:
                return
            if msg_type == "executing":
                node = data.get("node")
                if node is None:
                    self._job_done.set()
                elif self.verbose:
                    print(f"         [ws] executing node {node}")
            elif msg_type == "execution_error":
                self._error = data
                self._job_done.set()
            elif msg_type == "progress" and self.verbose:
                mx  = data.get("max", 1)
                val = data.get("value", 0)
                pct = int(val / mx * 100) if mx else 0
                print(f"         [ws] progress {pct}%")

    def _on_error(self, ws, error):
        if self.verbose:
            print(f"         [ws] error: {error}")

    def _on_close(self, ws, code, msg):
        self._connected.clear()
        if self.verbose:
            print(f"         [ws] closed ({code})")

    def upload_image(self, filepath, upload_name):
        with open(filepath, 'rb') as f:
            resp = requests.post(
                f"{self.server_url}/upload/image",
                files={"image": (upload_name, f, "image/png")},
                data={"overwrite": "true"},
            )
        resp.raise_for_status()
        return resp.json().get("name", upload_name)

    def queue_and_wait(self, workflow, timeout=JOB_TIMEOUT):
        if not self._connected.is_set():
            raise ConnectionError("WebSocket not connected")
        self._job_done.clear()
        self._error = None
        resp = requests.post(
            f"{self.server_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"ComfyUI rejected prompt:\n  {body['error']}")
        prompt_id = body["prompt_id"]
        with self._lock:
            self._current_pid = prompt_id
        if not self._job_done.wait(timeout=timeout):
            raise TimeoutError(f"Job {prompt_id} timed out after {timeout}s")
        if self._error:
            raise RuntimeError(
                f"Execution error:\n{json.dumps(self._error, indent=2)}"
            )
        time.sleep(0.5)
        resp = requests.get(f"{self.server_url}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()
        if prompt_id not in history:
            raise RuntimeError(f"Job {prompt_id} missing from /history")
        return prompt_id, history[prompt_id]

    def download_output(self, job_result, save_path):
        for _nid, node_out in job_result.get("outputs", {}).items():
            for img_info in node_out.get("images", []):
                qs = urlencode({
                    "filename":  img_info["filename"],
                    "subfolder": img_info.get("subfolder", ""),
                    "type":      img_info.get("type", "output"),
                })
                resp = requests.get(f"{self.server_url}/view?{qs}")
                resp.raise_for_status()
                with open(save_path, 'wb') as f:
                    f.write(resp.content)
                return save_path
        return None

# ═══════════════════════════ BUCKET TABLE PRINTER ════════════════════════════

def print_bucket_table():
    printed = set()
    print(f"\n  {'Bucket':>14}   {'Label':<20}  {'Aspect':<8}  {'Pixels':>12}")
    print(f"  {'─'*14}   {'─'*20}  {'─'*8}  {'─'*12}")
    for bw, bh in RESOLUTION_BUCKETS:
        if (bw, bh) in printed:
            continue
        lw, lh = max(bw, bh), min(bw, bh)
        name = _BUCKET_NAMES.get((lw, lh), "")
        ar   = _aspect_label(lw, lh)
        print(f"  {bw:>6} × {bh:<5}  {name:<20}  {ar:<8}  {bw*bh:>12,}")
        printed.add((bw, bh))
    print(f"\n  Total: {len(RESOLUTION_BUCKETS)} buckets")
    print(f"  (Portrait auto-derived by swapping W↔H)\n")

# ═══════════════════════════ MAIN ════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Resolution Snap — AI Outpainting Bucketer"
    )
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--list-buckets", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print("=" * 66)
    print("   Resolution Snap v2.1 — Alpha-Mask Outpainting Bucketer")
    print("=" * 66)

    if args.list_buckets:
        print_bucket_table()
        return

    if not os.path.isdir(INPUT_DIR):
        print(f"\n  [ERROR] Input dir not found: {INPUT_DIR}")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.dry_run and not os.path.isfile(WORKFLOW_PATH):
        print(f"\n  [ERROR] Workflow not found: {WORKFLOW_PATH}")
        sys.exit(1)

    base_workflow = None
    if not args.dry_run:
        with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
            base_workflow = json.load(f)

    comfy = None
    if not args.dry_run:
        print(f"\n  Connecting to ComfyUI at {COMFY_URL} …", end=" ", flush=True)
        try:
            comfy = ComfyConnection(COMFY_URL, verbose=args.verbose)
            comfy.connect()
            print("OK  (WebSocket)")
        except Exception as exc:
            print(f"FAILED\n  {exc}")
            sys.exit(1)
    else:
        print("\n  ▸ DRY RUN — preview only")

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    )
    if not files:
        print(f"\n  No images in {INPUT_DIR}")
        return

    print(f"\n  {len(files)} image(s)  •  {len(RESOLUTION_BUCKETS)} buckets")
    print(f"  Mask method: alpha channel (binary)\n")

    stats = {"filled": 0, "exact": 0, "failed": 0}
    bucket_usage = {}

    try:
        for idx, filename in enumerate(files, 1):
            img_path = os.path.join(INPUT_DIR, filename)
            stem = Path(filename).stem
            tag  = f"[{idx}/{len(files)}]"

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

            if action == "exact":
                print(f"  = {tag} {filename} ({ow}×{oh}) — already {label}, copying")
                img.save(os.path.join(OUTPUT_DIR, f"{stem}.png"))
                stats["exact"] += 1
                continue

            act_str = "downscale → center" if "downscale" in action else "center + pad"
            pad_w = tw - min(ow, tw)
            pad_h = th - min(oh, th)
            print(f"  ▸ {tag} {filename}")
            print(f"         {ow}×{oh}  →  {tw}×{th}  {label}")
            print(f"         [{act_str}]  padding: {pad_w}px × {pad_h}px")

            if args.dry_run:
                stats["filled"] += 1
                continue

            # Build RGBA canvas (transparent padding = mask)
            canvas = build_canvas(img, tw, th)
            temp_name = f"_resnap_{uuid.uuid4().hex[:8]}.png"
            temp_path = os.path.join(OUTPUT_DIR, temp_name)
            canvas.save(temp_path, "PNG")   # PNG preserves alpha

            try:
                actual_name = comfy.upload_image(temp_path, temp_name)

                workflow = json.loads(json.dumps(base_workflow))
                workflow["8"]["inputs"]["image"]            = actual_name
                workflow["10"]["inputs"]["filename_prefix"]  = f"ResSnap/{stem}"

                print(f"         queued … ", end="", flush=True)
                pid, result = comfy.queue_and_wait(workflow)
                print(f"done ({pid[:12]}…)")

                out_path = os.path.join(OUTPUT_DIR, f"{stem}_{tw}x{th}.png")
                saved = comfy.download_output(result, out_path)

                if saved:
                    print(f"         saved → {Path(saved).name}")
                    stats["filled"] += 1
                else:
                    print(f"         [warning] no output image")
                    stats["failed"] += 1

            except Exception as exc:
                print(f"\n         [ERROR] {exc}")
                stats["failed"] += 1

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    finally:
        if comfy:
            comfy.disconnect()

    print()
    print("=" * 66)
    print("   SUMMARY")
    print("=" * 66)
    print(f"   AI-filled:        {stats['filled']}")
    print(f"   Already standard: {stats['exact']}")
    print(f"   Failed:           {stats['failed']}")
    if bucket_usage:
        print("\n   Bucket distribution:")
        for (bw, bh), count in sorted(
            bucket_usage.items(), key=lambda x: x[1], reverse=True
        ):
            lbl = bucket_label(bw, bh)
            bar = "█" * count
            print(f"     {bw:>5}×{bh:<5}  {lbl:<28}  {count:>3}  {bar}")
    print(f"\n   Output folder: {OUTPUT_DIR}")
    print("=" * 66)


if __name__ == "__main__":
    main()