#!/usr/bin/env python3
"""
Resolution Snap v2.1 — AI Outpainting Bucketer for ComfyUI
═══════════════════════════════════════════════════════════════════
Snaps oddly-sized images to the nearest standard resolution,
centers the original on a blue-padded canvas, and queues ComfyUI
to AI-fill the borders via inpainting.

No more "3824×2150" output files — only clean, standard sizes.

Uses ComfyUI's WebSocket API for rock-solid job tracking
(no more polling races or phantom interrupts).

Usage:
    python resolution_snap.py                 # full run
    python resolution_snap.py --dry-run       # preview assignments
    python resolution_snap.py --list-buckets  # dump bucket table
    python resolution_snap.py --verbose       # show WebSocket chatter

WORKFLOW REQUIREMENT (API-format JSON):
    Node 8  — LoadImage              (receives the padded canvas)
    Node 14 — ImageToMask channel=blue (blue padding → inpaint mask)
    Node 6  — InpaintModelConditioning
    Node 10 — SaveImage              (filename_prefix set by script)
"""

import os
import sys
import json
import time
import uuid
import argparse
import threading
import requests
import websocket                     # from websocket-client
from math import gcd
from PIL import Image
from pathlib import Path
from urllib.parse import urlencode

# ══════════════════════════════ CONFIGURATION ═════════════════════════════════

INPUT_DIR     = r"C:\Path\To\Your\Images"
OUTPUT_DIR    = r"C:\Path\To\Your\Output"
COMFY_URL     = "http://127.0.0.1:8188"
WORKFLOW_PATH = "outpainting_workflow_v2.json"

PADDING_COLOR = (0, 0, 255)        # Pure blue → ImageToMask (blue channel)
JOB_TIMEOUT   = 600                # Max seconds per image before giving up

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}

# ══════════════════════════════ RESOLUTION BUCKETS ════════════════════════════
#
#   All landscape-oriented (width ≥ height).  Portrait auto-derived.
#   Sorted at startup by pixel area — matcher picks the SMALLEST
#   bucket that fully contains the source.
#
RESOLUTION_BUCKETS_RAW = [
    # ── Tiny / Thumbnail / Web ─────────────────────────────────
    (426,  240),       # 240p                16:9
    (480,  320),       # HVGA                 3:2
    (640,  360),       # 360p  nHD           16:9
    (640,  480),       # VGA                  4:3
    (800,  480),       # WVGA                 5:3
    (800,  600),       # SVGA                 4:3
    (854,  480),       # 480p  FWVGA         16:9

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
    (1280, 1024),      # SXGA                 5:4
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

# Sort by total pixel area (then width as tiebreaker)
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
    g = gcd(w, h)
    aw, ah = w // g, h // g
    known = {
        (16,9):"16:9", (16,10):"16:10", (4,3):"4:3",
        (3,2):"3:2", (5,4):"5:4", (5,3):"5:3",
        (1,1):"1:1", (21,9):"21:9", (32,9):"32:9",
        (8,5):"16:10", (683,384):"~16:9", (85,48):"16:9",
        (128,75):"~16:9",
    }
    return known.get((aw, ah), f"{aw}:{ah}")


def bucket_label(w: int, h: int) -> str:
    lw, lh = (max(w, h), min(w, h))
    name = _BUCKET_NAMES.get((lw, lh), f"{w}×{h}")
    ar   = _aspect_label(lw, lh)
    port = " portrait" if h > w else ""
    return f"{name}{port} ({ar})"

# ══════════════════════════════ BUCKET MATCHER ════════════════════════════════

def get_nearest_bucket(width: int, height: int) -> tuple[int, int, str]:
    """
    Smallest standard bucket that fully contains the image.
    Returns (target_w, target_h, action).
    action: "exact" | "pad" | "downscale+pad"
    """
    # Exact match shortcut
    for bw, bh in RESOLUTION_BUCKETS:
        if (width == bw and height == bh) or (width == bh and height == bw):
            return width, height, "exact"

    is_portrait = height > width
    # Normalize to landscape
    w = height if is_portrait else width
    h = width  if is_portrait else height

    for bw, bh in RESOLUTION_BUCKETS:
        if w <= bw and h <= bh:
            return (bh, bw, "pad") if is_portrait else (bw, bh, "pad")

    # Exceeds all buckets → downscale into largest
    bw, bh = RESOLUTION_BUCKETS[-1]
    return (bh, bw, "downscale+pad") if is_portrait else (bw, bh, "downscale+pad")

# ═════════════════════════════ CANVAS BUILDER ═════════════════════════════════

def build_canvas(img: Image.Image, tw: int, th: int) -> Image.Image:
    src = img.copy()
    if src.width > tw or src.height > th:
        ratio = min(tw / src.width, th / src.height)
        src = src.resize(
            (round(src.width * ratio), round(src.height * ratio)),
            Image.LANCZOS
        )
    canvas = Image.new('RGB', (tw, th), PADDING_COLOR)
    ox = (tw - src.width)  // 2
    oy = (th - src.height) // 2
    canvas.paste(src, (ox, oy))
    return canvas

# ══════════════════════════ COMFYUI WEBSOCKET CLIENT ═════════════════════════
#
#  Uses the same WebSocket protocol as the ComfyUI frontend.
#  This eliminates the polling race that caused the
#  "Interrupting prompt" errors.
#

class ComfyConnection:
    """
    Manages a persistent WebSocket to ComfyUI for reliable job lifecycle:
      connect → upload → queue → wait (via WS signal) → download → repeat
    """

    def __init__(self, server_url: str, verbose: bool = False):
        self.server_url = server_url.rstrip("/")
        self.client_id  = str(uuid.uuid4())
        self.verbose    = verbose

        self._ws: websocket.WebSocketApp | None = None
        self._thread:   threading.Thread | None = None
        self._connected = threading.Event()
        self._job_done  = threading.Event()
        self._lock      = threading.Lock()

        self._current_pid: str | None = None
        self._error:   dict | None    = None
        self._progress_pct: int       = 0

    # ── Connection lifecycle ──────────────────────────────────

    def connect(self):
        ws_url = (self.server_url
                  .replace("http://",  "ws://")
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
                f"WebSocket to {url} timed out.\n"
                f"Is ComfyUI running at {self.server_url}?"
            )

    def disconnect(self):
        if self._ws:
            self._ws.close()

    # ── WebSocket callbacks ───────────────────────────────────

    def _on_open(self, ws):
        self._connected.set()
        if self.verbose:
            print("         [ws] connected")

    def _on_message(self, ws, message):
        # ComfyUI can send binary (preview images) — skip those
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
            if pid is not None and pid != self._current_pid:
                return  # message for a different prompt

            if msg_type == "progress":
                mx  = data.get("max", 1)
                val = data.get("value", 0)
                self._progress_pct = int(val / mx * 100) if mx else 0

            elif msg_type == "executing":
                node = data.get("node")
                if node is None:
                    # ▸ All nodes finished — this is the definitive signal
                    self._job_done.set()
                elif self.verbose:
                    print(f"         [ws] node {node}")

            elif msg_type == "execution_error":
                self._error = data
                self._job_done.set()

            elif msg_type == "execution_cached" and self.verbose:
                print(f"         [ws] cached {len(data.get('nodes',[]))} nodes")

    def _on_error(self, ws, error):
        if self.verbose:
            print(f"         [ws] error: {error}")

    def _on_close(self, ws, code, msg):
        self._connected.clear()
        if self.verbose:
            print(f"         [ws] closed (code={code})")

    # ── Image upload ──────────────────────────────────────────

    def upload_image(self, filepath: str, upload_name: str) -> str:
        """Upload to ComfyUI /input, return the name it was saved as."""
        with open(filepath, 'rb') as f:
            resp = requests.post(
                f"{self.server_url}/upload/image",
                files={"image": (upload_name, f, "image/png")},
                data={"overwrite": "true"},
            )
        resp.raise_for_status()
        return resp.json().get("name", upload_name)

    # ── Queue + Wait ──────────────────────────────────────────

    def queue_and_wait(
        self, workflow: dict, timeout: float = JOB_TIMEOUT
    ) -> tuple[str, dict]:
        """
        Queue a prompt and block until ComfyUI signals completion
        via WebSocket.  Returns (prompt_id, history_entry).
        """
        if not self._connected.is_set():
            raise ConnectionError("WebSocket not connected to ComfyUI")

        # Reset per-job state
        self._job_done.clear()
        self._error        = None
        self._progress_pct = 0

        # ── Queue via REST ──
        resp = requests.post(
            f"{self.server_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
        )
        resp.raise_for_status()
        body = resp.json()

        if "error" in body:
            detail = body.get("node_errors", "")
            raise RuntimeError(
                f"ComfyUI rejected prompt:\n  {body['error']}\n  {detail}"
            )

        prompt_id = body["prompt_id"]

        with self._lock:
            self._current_pid = prompt_id

        # ── Wait for WebSocket "executing: null" ──
        if not self._job_done.wait(timeout=timeout):
            raise TimeoutError(
                f"Job {prompt_id} did not finish within {timeout}s"
            )

        if self._error:
            raise RuntimeError(
                f"ComfyUI execution error:\n"
                f"{json.dumps(self._error, indent=2)}"
            )

        # Small grace period for history to be fully written
        time.sleep(0.5)

        # ── Fetch outputs from history ──
        resp = requests.get(f"{self.server_url}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()

        if prompt_id not in history:
            raise RuntimeError(
                f"Job {prompt_id} signaled complete but missing from /history"
            )

        return prompt_id, history[prompt_id]

    # ── Download result ───────────────────────────────────────

    def download_output(self, job_result: dict, save_path: str) -> str | None:
        """Download the first output image to disk."""
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
    print(f"  {'─' * 14}   {'─' * 20}  {'─' * 8}  {'─' * 12}")
    for bw, bh in RESOLUTION_BUCKETS:
        if (bw, bh) in printed:
            continue
        lw, lh = max(bw, bh), min(bw, bh)
        name = _BUCKET_NAMES.get((lw, lh), "")
        ar   = _aspect_label(lw, lh)
        print(f"  {bw:>6} × {bh:<5}  {name:<20}  {ar:<8}  {bw * bh:>12,}")
        printed.add((bw, bh))
    print(f"\n  Total: {len(RESOLUTION_BUCKETS)} buckets")
    print(f"  (Portrait auto-derived by swapping W↔H)\n")

# ═══════════════════════════ MAIN PROCESSING LOOP ════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Resolution Snap — AI Outpainting Bucketer for ComfyUI"
    )
    ap.add_argument("--dry-run",      action="store_true",
                    help="Preview bucket assignments without processing")
    ap.add_argument("--list-buckets", action="store_true",
                    help="Print the full bucket table and exit")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show WebSocket messages and debug info")
    args = ap.parse_args()

    print("=" * 66)
    print("   Resolution Snap v2.1 — AI Outpainting Bucketer")
    print("=" * 66)

    if args.list_buckets:
        print_bucket_table()
        return

    # ── Validate paths ──
    if not os.path.isdir(INPUT_DIR):
        print(f"\n  [ERROR] Input directory not found:\n    {INPUT_DIR}")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.dry_run and not os.path.isfile(WORKFLOW_PATH):
        print(f"\n  [ERROR] Workflow JSON not found:\n    {WORKFLOW_PATH}")
        sys.exit(1)

    base_workflow = None
    if not args.dry_run:
        with open(WORKFLOW_PATH, 'r', encoding='utf-8') as f:
            base_workflow = json.load(f)

    # ── Connect to ComfyUI ──
    comfy: ComfyConnection | None = None
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
        print("\n  ▸ DRY RUN — previewing bucket assignments only")

    # ── Gather image files ──
    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    )
    if not files:
        print(f"\n  No image files found in {INPUT_DIR}")
        return

    print(f"\n  {len(files)} image(s) to process")
    print(f"  {len(RESOLUTION_BUCKETS)} resolution buckets loaded\n")

    stats = {"filled": 0, "exact": 0, "failed": 0}
    bucket_usage: dict[tuple[int, int], int] = {}

    try:
        for idx, filename in enumerate(files, 1):
            img_path = os.path.join(INPUT_DIR, filename)
            stem     = Path(filename).stem
            tag      = f"[{idx}/{len(files)}]"

            # ── Open image ──
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

            # ── Already exact ──
            if action == "exact":
                print(f"  = {tag} {filename} ({ow}×{oh}) — already {label}, copying")
                img.save(os.path.join(OUTPUT_DIR, f"{stem}.png"))
                stats["exact"] += 1
                continue

            # ── Report ──
            act_str = "downscale → center" if "downscale" in action else "center + pad"
            pad_w = tw - min(ow, tw)
            pad_h = th - min(oh, th)
            print(f"  ▸ {tag} {filename}")
            print(f"         {ow}×{oh}  →  {tw}×{th}  {label}")
            print(f"         [{act_str}]  padding: {pad_w}px × {pad_h}px")

            if args.dry_run:
                stats["filled"] += 1
                continue

            # ── Build canvas ──
            canvas    = build_canvas(img, tw, th)
            temp_name = f"_resnap_{uuid.uuid4().hex[:8]}.png"
            temp_path = os.path.join(OUTPUT_DIR, temp_name)
            canvas.save(temp_path, "PNG")

            try:
                # Upload — use the name ComfyUI actually saved
                actual_name = comfy.upload_image(temp_path, temp_name)

                # Prepare workflow copy
                workflow = json.loads(json.dumps(base_workflow))
                workflow["8"]["inputs"]["image"]            = actual_name
                workflow["10"]["inputs"]["filename_prefix"]  = f"ResSnap/{stem}"

                # Queue & wait (WebSocket-based — no polling race)
                print(f"         queued … ", end="", flush=True)
                pid, result = comfy.queue_and_wait(workflow)
                print(f"done ({pid[:12]}…)")

                # Download
                out_path = os.path.join(OUTPUT_DIR, f"{stem}_{tw}x{th}.png")
                saved    = comfy.download_output(result, out_path)

                if saved:
                    print(f"         saved → {Path(saved).name}")
                    stats["filled"] += 1
                else:
                    print(f"         [warning] no output image in result")
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

    # ── Summary ──
    print()
    print("=" * 66)
    print("   SUMMARY")
    print("=" * 66)
    print(f"   AI-filled:        {stats['filled']}")
    print(f"   Already standard: {stats['exact']}")
    print(f"   Failed:           {stats['failed']}")
    print()

    if bucket_usage:
        print("   Bucket distribution:")
        for (bw, bh), count in sorted(
            bucket_usage.items(), key=lambda x: x[1], reverse=True
        ):
            lbl = bucket_label(bw, bh)
            bar = "█" * count
            print(f"     {bw:>5}×{bh:<5}  {lbl:<28}  {count:>3}  {bar}")
        print()

    print(f"   Output folder:   {OUTPUT_DIR}")
    print("=" * 66)


if __name__ == "__main__":
    main()