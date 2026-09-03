#!/usr/bin/env python3
"""
Resolution Snap v3.3 — AI Outpainting Bucketer for ComfyUI
═══════════════════════════════════════════════════════════════════════
Snaps oddly-sized images to the nearest standard resolution,
centers the original on an RGBA canvas (edge-extended padding),
and queues ComfyUI to AI-fill the borders via inpainting.

v3.3: Two root-cause fixes for border bars:
  1. Clean alpha mask — no gray RGB payload under transparent pixels
  2. Aspect-aware bucket matcher — minimizes padding by preferring
     buckets with similar aspect ratios, not just "first fit"

Usage:
    python resolution_snap.py                 # full run
    python resolution_snap.py --dry-run       # preview assignments
    python resolution_snap.py --list-buckets  # dump bucket table
    python resolution_snap.py --verbose       # debug output + save mask PNGs
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
from PIL import Image, ImageFilter
from pathlib import Path
from urllib.parse import urlencode

# ══════════════════════════════════ CONFIGURATION ═════════════════════════════════

INPUT_DIR  = r"C:\Path\To\Your\Images"
OUTPUT_DIR = r"C:\Path\To\Your\Output"
COMFY_URL  = "http://127.0.0.1:8188"

# ── Model ─────────────────────────────────────────────────────────────────────────
INPAINT_MODEL = r"inpaint\juggernautXL_versionXInpaint.safetensors"

# ── Sampler ───────────────────────────────────────────────────────────────────────
SAMPLER_STEPS     = 16           # 12-20 at 1024px is a good range
SAMPLER_CFG       = 7.0          # 6-8; too low = no guidance = flat fills
SAMPLER_NAME      = "dpmpp_2m"
SAMPLER_SCHEDULER = "karras"
SAMPLER_DENOISE   = 1.0          # 1.0 = full generation in masked area
SAMPLER_SEED      = 1982

# ── Prompts ───────────────────────────────────────────────────────────────────────
POSITIVE_PROMPT = (
    "highly detailed background, seamless extension, photorealistic, "
    "landscape, ultra quality, matching lighting, masterpiece, 8k"
)
NEGATIVE_PROMPT = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, "
    "sketch, cartoon, drawing, anime), text, cropped, out of frame, "
    "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, "
    "mutilated, extra fingers, mutated hands, poorly drawn hands, "
    "poorly drawn face, mutation, deformed, blurry, dehydrated, "
    "bad anatomy, bad proportions, extra limbs, cloned face, disfigured, "
    "gross proportions, malformed limbs, missing arms, missing legs, "
    "extra arms, extra legs, fused fingers, too many fingers, long neck, "
    "UnrealisticDream"
)

# ── Canvas / Mask ─────────────────────────────────────────────────────────────────
# Edge extension: stretch source edge pixels into padding, then blur.
# Gives the model color context without flat fills that survive weak inpainting.
EDGE_EXTEND          = True      # Extend source edges into padding area
EDGE_EXTEND_BLUR_PX  = 24       # Blur the extended edges (0 = sharp repeating edge)
# Mask feathering: blur the alpha boundary between keep/inpaint.
# START AT 0 — only add back after confirming bars are gone.
MASK_FEATHER_PX      = 0
# Resolution cap: inpaint at this max, upscale result after.
# SD1.5 native = 512. 1024 is a good speed/quality tradeoff on a 3090.
MAX_INPAINT_LONG_EDGE = 1024

# ── Bucket strategy ──────────────────────────────────────────────────────────────
# "aspect"    — score by aspect-ratio closeness + padding waste (recommended)
# "first_fit" — pick the first bucket where both dims fit (original behavior)
BUCKET_STRATEGY   = "aspect"
MAX_PADDING_WARN  = 0.30         # Warn when padding exceeds 30% of canvas area

# ── Timing / Memory ──────────────────────────────────────────────────────────────
JOB_TIMEOUT       = 600
POLL_FALLBACK_SEC = 5
FLUSH_EVERY_N     = 1
FULL_UNLOAD_EVERY = 5

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


# ══════════════════════════════ EMBEDDED WORKFLOW ═════════════════════════════════

def build_workflow(image_name: str, filename_prefix: str) -> dict:
    return {
        "2": {
            "inputs": {"ckpt_name": INPAINT_MODEL},
            "class_type": "CheckpointLoaderSimple",
        },
        "3": {
            "inputs": {"text": POSITIVE_PROMPT, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
        },
        "4": {
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
        },
        "6": {
            "inputs": {
                "noise_mask": True,
                "positive": ["3", 0], "negative": ["4", 0],
                "vae": ["2", 2], "pixels": ["8", 0], "mask": ["8", 1],
            },
            "class_type": "InpaintModelConditioning",
        },
        "1": {
            "inputs": {
                "seed": SAMPLER_SEED, "control_after_generate": "fixed",
                "steps": SAMPLER_STEPS, "cfg": SAMPLER_CFG,
                "sampler_name": SAMPLER_NAME, "scheduler": SAMPLER_SCHEDULER,
                "denoise": SAMPLER_DENOISE,
                "model": ["2", 0],
                "positive": ["6", 0], "negative": ["6", 1],
                "latent_image": ["6", 2],
            },
            "class_type": "KSampler",
        },
        "7": {
            "inputs": {"samples": ["1", 0], "vae": ["2", 2]},
            "class_type": "VAEDecode",
        },
        "8": {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
        },
        "10": {
            "inputs": {"filename_prefix": filename_prefix, "images": ["7", 0]},
            "class_type": "SaveImage",
        },
    }


# ══════════════════════════════ RESOLUTION BUCKETS ════════════════════════════════

RESOLUTION_BUCKETS_RAW = [
    (426,240),(480,320),(640,360),(640,480),(800,480),(800,600),(854,480),
    (512,512),(768,768),(1024,1024),(1080,1080),(1440,1440),
    (2048,2048),(2160,2160),(4096,4096),
    (960,540),(960,640),(1024,576),(1024,600),(1024,768),
    (1152,648),(1152,720),(1152,864),
    (1280,720),(1280,768),(1280,800),(1280,960),(1280,1024),
    (1366,768),(1440,900),(1440,960),(1600,900),(1600,1024),
    (1600,1200),(1680,1050),
    (1920,1080),(1920,1200),(1920,1280),(2048,1080),(2048,1152),
    (2048,1280),(2048,1536),
    (2560,1080),(3440,1080),(3840,1080),
    (2560,1440),(2560,1600),(2560,1700),(2880,1620),(2880,1800),(2880,1920),
    (3440,1440),(3840,1600),(5120,1440),
    (3200,1800),(3200,2000),(3200,2400),(3000,2000),
    (4000,3000),(4032,3024),(4500,3000),(4624,3468),
    (6000,4000),(6016,4016),(8000,6000),
    (3840,2160),(3840,2400),(4096,2160),(4096,2304),(4096,2560),(4096,3072),
    (5120,2160),(5120,2880),(5120,3200),(5760,2400),
    (6016,3384),(6144,3456),(6144,4608),
    (7680,4320),(7680,4800),(8192,4320),(8192,4608),(8192,6144),
]

RESOLUTION_BUCKETS = sorted(
    RESOLUTION_BUCKETS_RAW,
    key=lambda b: (b[0] * b[1], b[0]),
)

_BUCKET_NAMES = {
    (426,240):"240p",(640,360):"360p",(854,480):"480p",
    (640,480):"VGA",(800,600):"SVGA",(1024,768):"XGA",
    (1280,720):"720p",(1280,800):"WXGA",(1280,1024):"SXGA",
    (1366,768):"HD",(1600,900):"HD+",(1600,1200):"UXGA",
    (1680,1050):"WSXGA+",(1920,1080):"1080p",(1920,1200):"WUXGA",
    (2048,1080):"2K DCI",(2560,1080):"UW-FHD",(2560,1440):"1440p",
    (2560,1600):"WQXGA",(2880,1800):"Retina",(3440,1440):"UW-QHD",
    (3840,1600):"UW-QHD+",(3200,1800):"QHD+",(3200,2400):"QUXGA",
    (3840,2160):"4K",(3840,2400):"WQUXGA",(4096,2160):"4K DCI",
    (5120,1440):"DQHD",(5120,2880):"5K",(6016,3384):"Apple 6K",
    (7680,4320):"8K",(8192,4320):"8K DCI",
}

_ASPECT_KNOWN = {
    (16,9):"16:9",(16,10):"16:10",(4,3):"4:3",(3,2):"3:2",
    (5,4):"5:4",(5,3):"5:3",(1,1):"1:1",(21,9):"21:9",
    (32,9):"32:9",(8,5):"16:10",(683,384):"~16:9",
    (85,48):"16:9",(128,75):"~16:9",
}


def _aspect_label(w, h):
    g = gcd(w, h)
    return _ASPECT_KNOWN.get((w // g, h // g), f"{w // g}:{h // g}")


def bucket_label(w, h):
    lw, lh = max(w, h), min(w, h)
    name = _BUCKET_NAMES.get((lw, lh), f"{w}x{h}")
    ar = _aspect_label(lw, lh)
    port = " portrait" if h > w else ""
    return f"{name}{port} ({ar})"


# ═══════════════════════════════ BUCKET MATCHER ═══════════════════════════════════

def _orient_bucket(bw, bh, is_portrait):
    """Orient a bucket to match source orientation."""
    if is_portrait:
        return min(bw, bh), max(bw, bh)
    return max(bw, bh), min(bw, bh)


def _score_bucket(src_w, src_h, cw, ch):
    """
    Score a candidate bucket. Lower = better match.

    Weights:
      aspect ratio closeness × 5   (dominant factor)
      padding waste          × 2   (secondary)

    Returns (score, waste, fits_without_downscale).
    """
    src_ar  = src_w / src_h
    cand_ar = cw / ch
    ar_diff = abs(src_ar - cand_ar) / max(src_ar, cand_ar)

    fits = src_w <= cw and src_h <= ch

    if fits:
        waste = 1.0 - (src_w * src_h) / (cw * ch)
    else:
        scale = min(cw / src_w, ch / src_h)
        waste = 1.0 - (src_w * scale * src_h * scale) / (cw * ch)

    score = ar_diff * 5.0 + waste * 2.0
    return score, waste, fits


def _first_fit_bucket(width, height):
    """Original strategy: first bucket where both dims fit."""
    is_portrait = height > width
    w = height if is_portrait else width
    h = width  if is_portrait else height

    for bw, bh in RESOLUTION_BUCKETS:
        if w <= bw and h <= bh:
            cw, ch = _orient_bucket(bw, bh, is_portrait)
            waste = 1.0 - (width * height) / (cw * ch)
            return cw, ch, "pad", waste

    bw, bh = RESOLUTION_BUCKETS[-1]
    cw, ch = _orient_bucket(bw, bh, is_portrait)
    scale = min(cw / width, ch / height)
    waste = 1.0 - (width * scale * height * scale) / (cw * ch)
    return cw, ch, "downscale+pad", waste


def _aspect_fit_bucket(width, height):
    """
    Aspect-aware strategy: score every bucket by AR closeness + padding,
    prefer buckets that don't need downscaling.
    """
    is_portrait = height > width

    fit_list  = []   # source fits inside bucket
    down_list = []   # source needs downscaling

    seen = set()
    for bw, bh in RESOLUTION_BUCKETS:
        cw, ch = _orient_bucket(bw, bh, is_portrait)
        if (cw, ch) in seen:
            continue
        seen.add((cw, ch))

        score, waste, fits = _score_bucket(width, height, cw, ch)

        if fits:
            fit_list.append((score, cw, ch, waste))
        else:
            down_list.append((score, cw, ch, waste))

    # Prefer fit (no downscale) over downscale
    if fit_list:
        fit_list.sort()
        score, cw, ch, waste = fit_list[0]
        return cw, ch, "pad", waste

    if down_list:
        down_list.sort()
        score, cw, ch, waste = down_list[0]
        return cw, ch, "downscale+pad", waste

    # Should never reach here
    bw, bh = RESOLUTION_BUCKETS[-1]
    cw, ch = _orient_bucket(bw, bh, is_portrait)
    return cw, ch, "downscale+pad", 0.5


def get_nearest_bucket(width, height):
    """
    Find the best standard resolution bucket for a source image.
    Returns (target_w, target_h, action, padding_waste_ratio).
    """
    # Exact match shortcut
    for bw, bh in RESOLUTION_BUCKETS:
        if (width == bw and height == bh) or (width == bh and height == bw):
            return width, height, "exact", 0.0

    if BUCKET_STRATEGY == "first_fit":
        return _first_fit_bucket(width, height)
    return _aspect_fit_bucket(width, height)


# ═══════════════════════════════ CANVAS BUILDER ═══════════════════════════════════

def _fit_source(src: Image.Image, tw: int, th: int) -> Image.Image:
    """Downscale source if larger than target, preserving aspect ratio."""
    if src.width <= tw and src.height <= th:
        return src
    ratio = min(tw / src.width, th / src.height)
    return src.resize(
        (round(src.width * ratio), round(src.height * ratio)),
        Image.LANCZOS,
    )


def _center_offset(src_dim: int, canvas_dim: int) -> int:
    return (canvas_dim - src_dim) // 2


def _snap8(val: int) -> int:
    """Round to nearest multiple of 8 (SD latent alignment)."""
    return max(8, round(val / 8) * 8)


def _extend_edges(canvas: Image.Image, src: Image.Image, ox: int, oy: int):
    """
    Stretch source edge pixels into padding area.

    Before (black padding):         After (edge-extended):
    ┌──────────────────────┐        ┌──────────────────────┐
    │                      │        │ top edge stretched ↕  │
    │   ┌────────────┐     │        │←←┌────────────┐→→→→│
    │   │   source   │     │        │←←│   source   │→→→→│
    │   └────────────┘     │        │←←└────────────┘→→→→│
    │                      │        │ bot edge stretched ↕  │
    └──────────────────────┘        └──────────────────────┘
    """
    tw, th = canvas.size
    sw, sh = src.size

    # Left strip
    if ox > 0:
        left_col = src.crop((0, 0, 1, sh))
        canvas.paste(left_col.resize((ox, sh), Image.NEAREST), (0, oy))

    # Right strip
    right_x = ox + sw
    if right_x < tw:
        right_col = src.crop((sw - 1, 0, sw, sh))
        canvas.paste(right_col.resize((tw - right_x, sh), Image.NEAREST), (right_x, oy))

    # Top strip (full width, after left/right are filled)
    if oy > 0:
        top_row = canvas.crop((0, oy, tw, oy + 1))
        canvas.paste(top_row.resize((tw, oy), Image.NEAREST), (0, 0))

    # Bottom strip (full width)
    bot_y = oy + sh
    if bot_y < th:
        bot_row = canvas.crop((0, bot_y - 1, tw, bot_y))
        canvas.paste(bot_row.resize((tw, th - bot_y), Image.NEAREST), (0, bot_y))


def build_canvas(img: Image.Image, tw: int, th: int) -> Image.Image:
    """
    RGBA canvas for inpainting.

    RGB layer:  source centered + edge-extended into padding (optional blur)
    Alpha layer: 255 = keep (source area), 0 = inpaint (padding area)

    No hidden color payload under the transparent region — the RGB data
    in the padding area is meaningful edge content, not flat gray.
    """
    src = _fit_source(img.convert('RGB'), tw, th)
    ox  = _center_offset(src.width, tw)
    oy  = _center_offset(src.height, th)

    # ── RGB layer ──
    canvas_rgb = Image.new('RGB', (tw, th), (0, 0, 0))
    canvas_rgb.paste(src, (ox, oy))

    if EDGE_EXTEND:
        _extend_edges(canvas_rgb, src, ox, oy)
        if EDGE_EXTEND_BLUR_PX > 0:
            blurred = canvas_rgb.filter(
                ImageFilter.GaussianBlur(radius=EDGE_EXTEND_BLUR_PX)
            )
            # Keep source sharp, blur only the padding
            sharp_mask = Image.new('L', (tw, th), 0)
            sharp_mask.paste(Image.new('L', src.size, 255), (ox, oy))
            canvas_rgb = Image.composite(canvas_rgb, blurred, sharp_mask)

    # ── Alpha layer (clean, no feather by default) ──
    alpha = Image.new('L', (tw, th), 0)
    alpha.paste(Image.new('L', src.size, 255), (ox, oy))

    if MASK_FEATHER_PX > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=MASK_FEATHER_PX))

    canvas_rgb.putalpha(alpha)
    return canvas_rgb


def compute_inpaint_size(tw: int, th: int) -> tuple[int, int]:
    """Cap resolution for fast inpainting, snap to 8px grid."""
    long_edge = max(tw, th)
    if long_edge <= MAX_INPAINT_LONG_EDGE:
        return _snap8(tw), _snap8(th)
    scale = MAX_INPAINT_LONG_EDGE / long_edge
    return _snap8(round(tw * scale)), _snap8(round(th * scale))


def composite_result(
    ai_result: Image.Image,
    original: Image.Image,
    full_tw: int,
    full_th: int,
) -> Image.Image:
    """Upscale AI result to full size, paste original pixels back on top."""
    if ai_result.size != (full_tw, full_th):
        ai_result = ai_result.resize((full_tw, full_th), Image.LANCZOS)

    final = ai_result.convert('RGB')
    src = _fit_source(original.convert('RGB'), full_tw, full_th)
    ox = _center_offset(src.width, full_tw)
    oy = _center_offset(src.height, full_th)
    final.paste(src, (ox, oy))
    return final


# ═══════════════════════════════ COMFYUI CONNECTION ═══════════════════════════════

class ComfyConnection:

    def __init__(self, server_url, verbose=False):
        self.server_url = server_url.rstrip("/")
        self.client_id  = str(uuid.uuid4())
        self.verbose    = verbose

        self._ws              = None
        self._thread          = None
        self._connected       = threading.Event()
        self._job_done        = threading.Event()
        self._lock            = threading.Lock()
        self._current_pid     = None
        self._error           = None
        self._completed_pids  = set()
        self._ws_alive        = False

    def connect(self):
        ws_url = (self.server_url
                  .replace("http://", "ws://")
                  .replace("https://", "wss://"))
        self._ws = websocket.WebSocketApp(
            f"{ws_url}/ws?clientId={self.client_id}",
            on_open=self._on_open, on_message=self._on_message,
            on_error=self._on_error, on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 15, "ping_timeout": 10},
            daemon=True,
        )
        self._thread.start()
        if not self._connected.wait(timeout=15):
            raise ConnectionError(
                f"WebSocket timed out. Is ComfyUI at {self.server_url}?"
            )

    def disconnect(self):
        if self._ws:
            self._ws.close()

    def _on_open(self, ws):
        self._connected.set()
        self._ws_alive = True
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
        pid      = data.get("prompt_id")

        if self.verbose:
            node = data.get("node", "")
            extra = f" node={node}" if node else ""
            print(f"         [ws] {msg_type} pid={str(pid)[:12]}{extra}")

        with self._lock:
            if msg_type == "executing" and data.get("node") is None and pid:
                self._completed_pids.add(pid)
                if self._current_pid and pid == self._current_pid:
                    self._job_done.set()
            elif msg_type == "execution_error":
                self._error = data
                self._job_done.set()

    def _on_error(self, ws, error):
        self._ws_alive = False
        if self.verbose:
            print(f"         [ws] error: {error}")

    def _on_close(self, ws, code, msg):
        self._ws_alive = False
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

    def free_memory(self, unload_models=False):
        try:
            resp = requests.post(
                f"{self.server_url}/free",
                json={"unload_models": unload_models, "free_memory": True},
                timeout=10,
            )
            resp.raise_for_status()
            if self.verbose:
                tag = "models + cache" if unload_models else "cache"
                print(f"         [mem] flushed {tag}")
        except requests.RequestException as exc:
            if self.verbose:
                print(f"         [mem] flush failed: {exc}")

    def queue_and_wait(self, workflow, timeout=JOB_TIMEOUT):
        if not self._connected.is_set():
            raise ConnectionError("WebSocket not connected")

        self._job_done.clear()
        self._error = None
        with self._lock:
            self._completed_pids.clear()
            self._current_pid = None

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
            if prompt_id in self._completed_pids:
                self._job_done.set()

        deadline = time.time() + timeout
        while not self._job_done.is_set():
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Job {prompt_id} timed out")
            if self._job_done.wait(timeout=min(POLL_FALLBACK_SEC, remaining)):
                break
            self._poll_history(prompt_id)

        if self._error:
            raise RuntimeError(
                f"Execution error:\n{json.dumps(self._error, indent=2)}"
            )

        time.sleep(0.3)
        resp = requests.get(f"{self.server_url}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()
        if prompt_id not in history:
            raise RuntimeError(f"Job {prompt_id} not in /history")
        return prompt_id, history[prompt_id]

    def _poll_history(self, prompt_id):
        try:
            hr = requests.get(
                f"{self.server_url}/history/{prompt_id}", timeout=5
            )
            if hr.status_code != 200:
                return
            hdata = hr.json()
            if prompt_id not in hdata:
                return
            entry = hdata[prompt_id]
            for nout in entry.get("outputs", {}).values():
                if nout.get("images"):
                    self._job_done.set()
                    return
            if entry.get("status", {}).get("status_str") == "error":
                self._error = entry["status"]
                self._job_done.set()
        except requests.RequestException:
            pass

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


# ═══════════════════════════════ DISPLAY ══════════════════════════════════════════

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
        print(f"  {bw:>6} x {bh:<5}  {name:<20}  {ar:<8}  {bw*bh:>12,}")
        printed.add((bw, bh))
    print(f"\n  Total: {len(RESOLUTION_BUCKETS)} buckets\n")


# ═══════════════════════════════ JOB PROCESSOR ════════════════════════════════════

class VRAMTracker:
    def __init__(self, comfy: ComfyConnection):
        self.comfy = comfy
        self._since_flush  = 0
        self._since_unload = 0

    def tick(self):
        self._since_flush  += 1
        self._since_unload += 1
        if self._since_unload >= FULL_UNLOAD_EVERY:
            print(f"         [mem] full VRAM flush + model unload")
            self.comfy.free_memory(unload_models=True)
            self._since_unload = 0
            self._since_flush  = 0
        elif self._since_flush >= FLUSH_EVERY_N:
            self.comfy.free_memory(unload_models=False)
            self._since_flush = 0

    def reset(self):
        self.comfy.free_memory(unload_models=True)
        self._since_flush  = 0
        self._since_unload = 0


def process_image(
    comfy: ComfyConnection,
    img: Image.Image,
    tw: int, th: int,
    stem: str,
    out_dir: str,
    verbose: bool = False,
) -> str | None:
    """Build canvas, inpaint at capped res, upscale, composite original back."""

    inp_w, inp_h = compute_inpaint_size(tw, th)
    capped = (inp_w != _snap8(tw) or inp_h != _snap8(th))
    if capped:
        print(f"         inpaint at {inp_w}x{inp_h} -> upscale to {tw}x{th}")

    canvas = build_canvas(img, inp_w, inp_h)

    # Debug: save what the model will see
    if verbose:
        dbg_dir = os.path.join(out_dir, "_debug")
        os.makedirs(dbg_dir, exist_ok=True)
        canvas.convert('RGB').save(os.path.join(dbg_dir, f"{stem}_rgb.png"))
        canvas.getchannel('A').save(os.path.join(dbg_dir, f"{stem}_mask.png"))
        print(f"         [debug] saved rgb + mask to _debug/")

    temp_name = f"_resnap_{uuid.uuid4().hex[:8]}.png"
    temp_path = os.path.join(out_dir, temp_name)
    canvas.save(temp_path, "PNG")

    try:
        actual_name = comfy.upload_image(temp_path, temp_name)
        workflow = build_workflow(actual_name, f"ResSnap/{stem}")

        t0 = time.time()
        print(f"         queued ... ", end="", flush=True)
        pid, result = comfy.queue_and_wait(workflow)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s ({pid[:12]}...)")

        raw_path = os.path.join(out_dir, f"_raw_{uuid.uuid4().hex[:8]}.png")
        downloaded = comfy.download_output(result, raw_path)

        if not downloaded:
            print(f"         [warning] no output image from ComfyUI")
            return None

        try:
            ai_result = Image.open(raw_path)
            final = composite_result(ai_result, img, tw, th)
            out_path = os.path.join(out_dir, f"{stem}_{tw}x{th}.png")
            final.save(out_path, "PNG")
            print(f"         saved -> {Path(out_path).name}")
            return out_path
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ═══════════════════════════════════ MAIN ═════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Resolution Snap — AI Outpainting Bucketer"
    )
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--list-buckets", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print("=" * 66)
    print("   Resolution Snap v3.3 — Aspect-Aware Outpainting")
    print("=" * 66)

    if args.list_buckets:
        print_bucket_table()
        return

    if not os.path.isdir(INPUT_DIR):
        print(f"\n  [ERROR] Input dir not found: {INPUT_DIR}")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    comfy = None
    if not args.dry_run:
        print(f"\n  Connecting to ComfyUI at {COMFY_URL} ...", end=" ", flush=True)
        try:
            comfy = ComfyConnection(COMFY_URL, verbose=args.verbose)
            comfy.connect()
            print("OK")
        except Exception as exc:
            print(f"FAILED\n  {exc}")
            sys.exit(1)
    else:
        print("\n  >> DRY RUN — preview only")

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if Path(f).suffix.lower() in VALID_EXTENSIONS
    )
    if not files:
        print(f"\n  No images in {INPUT_DIR}")
        return

    print(f"\n  Model:       {INPAINT_MODEL}")
    print(f"  Sampler:     {SAMPLER_NAME} / {SAMPLER_SCHEDULER}")
    print(f"  Steps:       {SAMPLER_STEPS}  |  CFG: {SAMPLER_CFG}  |  Denoise: {SAMPLER_DENOISE}")
    print(f"  Max inpaint: {MAX_INPAINT_LONG_EDGE}px  |  Feather: {MASK_FEATHER_PX}px")
    print(f"  Bucket mode: {BUCKET_STRATEGY}")
    edge_str = f"edge-extend + {EDGE_EXTEND_BLUR_PX}px blur" if EDGE_EXTEND else "black"
    print(f"  Padding:     {edge_str}")
    print(f"\n  {len(files)} image(s)  |  {len(RESOLUTION_BUCKETS)} buckets\n")

    stats = {"filled": 0, "exact": 0, "skipped": 0, "failed": 0}
    bucket_usage = {}
    vram = VRAMTracker(comfy) if comfy else None
    t_start = time.time()

    try:
        for idx, filename in enumerate(files, 1):
            img_path = os.path.join(INPUT_DIR, filename)
            stem = Path(filename).stem
            tag  = f"[{idx}/{len(files)}]"

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as exc:
                print(f"  X {tag} {filename} — cannot open ({exc})")
                stats["failed"] += 1
                continue

            ow, oh = img.size
            tw, th, action, waste = get_nearest_bucket(ow, oh)
            label = bucket_label(tw, th)
            bucket_usage[(tw, th)] = bucket_usage.get((tw, th), 0) + 1

            if action == "exact":
                print(f"  = {tag} {filename} ({ow}x{oh}) — already {label}, copying")
                img.save(os.path.join(OUTPUT_DIR, f"{stem}.png"))
                stats["exact"] += 1
                continue

            act_str = "downscale -> center" if "downscale" in action else "center + pad"
            waste_pct = f"{waste * 100:.1f}%"
            warn = " ⚠ HIGH" if waste > MAX_PADDING_WARN else ""

            print(f"  > {tag} {filename}")
            print(f"         {ow}x{oh}  ->  {tw}x{th}  {label}")
            print(f"         [{act_str}]  padding: {waste_pct}{warn}")

            if args.dry_run:
                stats["filled"] += 1
                continue

            try:
                result = process_image(
                    comfy, img, tw, th, stem, OUTPUT_DIR,
                    verbose=args.verbose,
                )
                if result:
                    stats["filled"] += 1
                else:
                    stats["failed"] += 1
                vram.tick()

            except Exception as exc:
                print(f"\n         [ERROR] {exc}")
                stats["failed"] += 1
                vram.reset()

    finally:
        if comfy:
            comfy.free_memory(unload_models=True)
            comfy.disconnect()

    total_time = time.time() - t_start
    processed  = stats['filled'] + stats['exact']

    print()
    print("=" * 66)
    print("   SUMMARY")
    print("=" * 66)
    print(f"   AI-filled:        {stats['filled']}")
    print(f"   Already standard: {stats['exact']}")
    print(f"   Failed:           {stats['failed']}")
    print(f"   Total time:       {total_time:.0f}s"
          f"  ({total_time / max(1, processed):.1f}s avg)")

    if bucket_usage:
        print("\n   Bucket distribution:")
        for (bw, bh), count in sorted(
            bucket_usage.items(), key=lambda x: x[1], reverse=True
        ):
            lbl = bucket_label(bw, bh)
            bar = "#" * min(count, 50)
            print(f"     {bw:>5}x{bh:<5}  {lbl:<28}  {count:>3}  {bar}")

    print(f"\n   Output folder: {OUTPUT_DIR}")
    print("=" * 66)


if __name__ == "__main__":
    main()