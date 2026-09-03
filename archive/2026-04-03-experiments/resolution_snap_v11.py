#!/usr/bin/env python3
"""
Resolution Snap v4.0 — AI Outpainting Bucketer for ComfyUI
═══════════════════════════════════════════════════════════════════════
Standardizes images into resolution buckets, builds a context-aware
conditioning canvas, and outpaints the padding so the result fills the
bucket cleanly without flat border bars.

Key changes vs v3.1:
- smarter bucket scoring (less gratuitous padding)
- no hidden gray RGB in transparent regions
- context-aware padding via edge extension + blur
- explicit soft transition between preserved source and AI outpaint
- optional debug exports for conditioning inputs
- CLI overrides for input/output/comfy URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from math import gcd
from pathlib import Path
from urllib.parse import urlencode

import requests
import websocket
from PIL import Image, ImageFilter

# ══════════════════════════════════ CONFIGURATION ═════════════════════════════════

INPUT_DIR = r"C:\Path\To\Your\Images"
OUTPUT_DIR = r"C:\Path\To\Your\Output"
COMFY_URL = "http://127.0.0.1:8188"

# ── Model ─────────────────────────────────────────────────────────────────────────
INPAINT_MODEL = r"inpaint\juggernautXL_versionXInpaint.safetensors"

# ── Sampler ───────────────────────────────────────────────────────────────────────
SAMPLER_STEPS = 20
SAMPLER_CFG = 7
SAMPLER_NAME = "dpmpp_2m"
SAMPLER_SCHEDULER = "karras"
SAMPLER_DENOISE = 1.0
SAMPLER_SEED = 1982

# ── Prompts ───────────────────────────────────────────────────────────────────────
POSITIVE_PROMPT = (
    "highly detailed background, seamless extension, photorealistic, "
    "landscape, ultra quality, matching lighting, coherent continuation, "
    "clean edges, masterpiece, 8k"
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
    "harsh frame, visible border, obvious edge, letterbox bars, pillarbox bars, "
    "UnrealisticDream"
)

# ── Canvas / Blend ────────────────────────────────────────────────────────────────
MASK_FEATHER_PX = 16
SOURCE_BLEND_PX = 24
CONTEXT_BLUR_PX = 40
MAX_INPAINT_LONG_EDGE = 1536

# ── Timing / Memory ──────────────────────────────────────────────────────────────
JOB_TIMEOUT = 600
POLL_FALLBACK_SEC = 5
FLUSH_EVERY_N = 1
FULL_UNLOAD_EVERY = 5

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


# ══════════════════════════════ EMBEDDED WORKFLOW ═════════════════════════════════

def build_workflow(image_name: str, filename_prefix: str) -> dict:
    return {
        "2": {
            "inputs": {"ckpt_name": INPAINT_MODEL},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint (Inpainting)"},
        },
        "3": {
            "inputs": {"text": POSITIVE_PROMPT, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"},
        },
        "4": {
            "inputs": {"text": NEGATIVE_PROMPT, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"},
        },
        "6": {
            "inputs": {
                "noise_mask": True,
                "positive": ["3", 0],
                "negative": ["4", 0],
                "vae": ["2", 2],
                "pixels": ["8", 0],
                "mask": ["8", 1],
            },
            "class_type": "InpaintModelConditioning",
            "_meta": {"title": "Inpaint Model Conditioning"},
        },
        "1": {
            "inputs": {
                "seed": SAMPLER_SEED,
                "control_after_generate": "fixed",
                "steps": SAMPLER_STEPS,
                "cfg": SAMPLER_CFG,
                "sampler_name": SAMPLER_NAME,
                "scheduler": SAMPLER_SCHEDULER,
                "denoise": SAMPLER_DENOISE,
                "model": ["2", 0],
                "positive": ["6", 0],
                "negative": ["6", 1],
                "latent_image": ["6", 2],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "7": {
            "inputs": {"samples": ["1", 0], "vae": ["2", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "8": {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "10": {
            "inputs": {"filename_prefix": filename_prefix, "images": ["7", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
    }


# ══════════════════════════════ RESOLUTION BUCKETS ════════════════════════════════

RESOLUTION_BUCKETS_RAW = [
    (426, 240), (480, 320), (640, 360), (640, 480), (800, 480), (800, 600), (854, 480),
    (512, 512), (768, 768), (1024, 1024), (1080, 1080), (1440, 1440),
    (2048, 2048), (2160, 2160), (4096, 4096),
    (960, 540), (960, 640), (1024, 576), (1024, 600), (1024, 768),
    (1152, 648), (1152, 720), (1152, 864),
    (1280, 720), (1280, 768), (1280, 800), (1280, 960), (1280, 1024),
    (1366, 768), (1440, 900), (1440, 960), (1600, 900), (1600, 1024),
    (1600, 1200), (1680, 1050),
    (1920, 1080), (1920, 1200), (1920, 1280), (2048, 1080), (2048, 1152),
    (2048, 1280), (2048, 1536),
    (2560, 1080), (3440, 1080), (3840, 1080),
    (2560, 1440), (2560, 1600), (2560, 1700), (2880, 1620), (2880, 1800), (2880, 1920),
    (3440, 1440), (3840, 1600), (5120, 1440),
    (3200, 1800), (3200, 2000), (3200, 2400), (3000, 2000),
    (4000, 3000), (4032, 3024), (4500, 3000), (4624, 3468),
    (6000, 4000), (6016, 4016), (8000, 6000),
    (3840, 2160), (3840, 2400), (4096, 2160), (4096, 2304), (4096, 2560), (4096, 3072),
    (5120, 2160), (5120, 2880), (5120, 3200), (5760, 2400),
    (6016, 3384), (6144, 3456), (6144, 4608),
    (7680, 4320), (7680, 4800), (8192, 4320), (8192, 4608), (8192, 6144),
]

RESOLUTION_BUCKETS = sorted(RESOLUTION_BUCKETS_RAW, key=lambda b: (b[0] * b[1], b[0]))

_BUCKET_NAMES = {
    (426, 240): "240p", (640, 360): "360p", (854, 480): "480p",
    (640, 480): "VGA", (800, 600): "SVGA", (1024, 768): "XGA",
    (1280, 720): "720p", (1280, 800): "WXGA", (1280, 1024): "SXGA",
    (1366, 768): "HD", (1600, 900): "HD+", (1600, 1200): "UXGA",
    (1680, 1050): "WSXGA+", (1920, 1080): "1080p", (1920, 1200): "WUXGA",
    (2048, 1080): "2K DCI", (2560, 1080): "UW-FHD", (2560, 1440): "1440p",
    (2560, 1600): "WQXGA", (2880, 1800): "Retina", (3440, 1440): "UW-QHD",
    (3840, 1600): "UW-QHD+", (3200, 1800): "QHD+", (3200, 2400): "QUXGA",
    (3840, 2160): "4K", (3840, 2400): "WQUXGA", (4096, 2160): "4K DCI",
    (5120, 1440): "DQHD", (5120, 2880): "5K", (6016, 3384): "Apple 6K",
    (7680, 4320): "8K", (8192, 4320): "8K DCI",
}

_ASPECT_KNOWN = {
    (16, 9): "16:9", (16, 10): "16:10", (4, 3): "4:3", (3, 2): "3:2",
    (5, 4): "5:4", (5, 3): "5:3", (1, 1): "1:1", (21, 9): "21:9",
    (32, 9): "32:9", (8, 5): "16:10", (683, 384): "~16:9",
    (85, 48): "16:9", (128, 75): "~16:9",
}


def _aspect_label(w: int, h: int) -> str:
    g = gcd(w, h)
    return _ASPECT_KNOWN.get((w // g, h // g), f"{w // g}:{h // g}")


def bucket_label(w: int, h: int) -> str:
    lw, lh = max(w, h), min(w, h)
    name = _BUCKET_NAMES.get((lw, lh), f"{w}x{h}")
    ar = _aspect_label(lw, lh)
    port = " portrait" if h > w else ""
    return f"{name}{port} ({ar})"


# ═══════════════════════════════ BUCKET MATCHER ═══════════════════════════════════

def _candidate_buckets(width: int, height: int) -> list[tuple[int, int]]:
    is_portrait = height > width
    w = height if is_portrait else width
    h = width if is_portrait else height

    candidates: list[tuple[int, int]] = []
    for bw, bh in RESOLUTION_BUCKETS:
        lw, lh = max(bw, bh), min(bw, bh)
        if lw >= w and lh >= h:
            candidates.append((lh, lw) if is_portrait else (lw, lh))
    return candidates


def choose_bucket(width: int, height: int) -> tuple[int, int, str]:
    for bw, bh in RESOLUTION_BUCKETS:
        if (width == bw and height == bh) or (width == bh and height == bw):
            return width, height, "exact"

    candidates = _candidate_buckets(width, height)
    if candidates:
        src_ar = width / height

        def score(bucket: tuple[int, int]) -> tuple[float, float, int, int]:
            bw, bh = bucket
            area_growth = (bw * bh) - (width * height)
            aspect_delta = abs((bw / bh) - src_ar)
            pad_w = bw - min(width, bw)
            pad_h = bh - min(height, bh)
            return (aspect_delta, area_growth / max(1, bw * bh), pad_w + pad_h, bw * bh)

        best = min(candidates, key=score)
        return best[0], best[1], "pad"

    # no bucket can contain the image as-is; choose the biggest and downscale into it
    bw, bh = RESOLUTION_BUCKETS[-1]
    is_portrait = height > width
    return ((bh, bw, "downscale+pad") if is_portrait else (bw, bh, "downscale+pad"))


# ═══════════════════════════════ CANVAS BUILDING ══════════════════════════════════

def _snap8(val: int) -> int:
    return max(8, round(val / 8) * 8)


def _fit_source(src: Image.Image, tw: int, th: int) -> Image.Image:
    if src.width <= tw and src.height <= th:
        return src
    ratio = min(tw / src.width, th / src.height)
    return src.resize((round(src.width * ratio), round(src.height * ratio)), Image.LANCZOS)


def _center_offset(src_dim: int, canvas_dim: int) -> int:
    return (canvas_dim - src_dim) // 2


def compute_inpaint_size(tw: int, th: int) -> tuple[int, int]:
    long_edge = max(tw, th)
    if long_edge <= MAX_INPAINT_LONG_EDGE:
        return _snap8(tw), _snap8(th)
    scale = MAX_INPAINT_LONG_EDGE / long_edge
    return _snap8(round(tw * scale)), _snap8(round(th * scale))


def _make_soft_alpha(width: int, height: int, rect: tuple[int, int, int, int], feather_px: int) -> Image.Image:
    alpha = Image.new("L", (width, height), 0)
    x0, y0, x1, y1 = rect
    alpha.paste(255, (x0, y0, x1, y1))
    if feather_px > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return alpha


def _make_preserve_mask(width: int, height: int, rect: tuple[int, int, int, int], blend_px: int) -> Image.Image:
    x0, y0, x1, y1 = rect
    if blend_px <= 0 or (x1 - x0) <= blend_px * 2 or (y1 - y0) <= blend_px * 2:
        return _make_soft_alpha(width, height, rect, 0)

    inner = (x0 + blend_px, y0 + blend_px, x1 - blend_px, y1 - blend_px)
    return _make_soft_alpha(width, height, inner, blend_px)


def _make_edge_extended_background(src: Image.Image, tw: int, th: int) -> Image.Image:
    src = src.convert("RGB")
    bg = Image.new("RGB", (tw, th))
    ox = _center_offset(src.width, tw)
    oy = _center_offset(src.height, th)
    rw = tw - ox - src.width
    bh = th - oy - src.height

    bg.paste(src, (ox, oy))

    # Sides
    if ox > 0:
        left = src.crop((0, 0, 1, src.height)).resize((ox, src.height), Image.BILINEAR)
        bg.paste(left, (0, oy))
    if rw > 0:
        right = src.crop((src.width - 1, 0, src.width, src.height)).resize((rw, src.height), Image.BILINEAR)
        bg.paste(right, (ox + src.width, oy))
    if oy > 0:
        top = src.crop((0, 0, src.width, 1)).resize((src.width, oy), Image.BILINEAR)
        bg.paste(top, (ox, 0))
    if bh > 0:
        bottom = src.crop((0, src.height - 1, src.width, src.height)).resize((src.width, bh), Image.BILINEAR)
        bg.paste(bottom, (ox, oy + src.height))

    # Corners
    if ox > 0 and oy > 0:
        bg.paste(src.crop((0, 0, 1, 1)).resize((ox, oy), Image.NEAREST), (0, 0))
    if rw > 0 and oy > 0:
        bg.paste(src.crop((src.width - 1, 0, src.width, 1)).resize((rw, oy), Image.NEAREST), (ox + src.width, 0))
    if ox > 0 and bh > 0:
        bg.paste(src.crop((0, src.height - 1, 1, src.height)).resize((ox, bh), Image.NEAREST), (0, oy + src.height))
    if rw > 0 and bh > 0:
        bg.paste(
            src.crop((src.width - 1, src.height - 1, src.width, src.height)).resize((rw, bh), Image.NEAREST),
            (ox + src.width, oy + src.height),
        )

    if CONTEXT_BLUR_PX > 0:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=CONTEXT_BLUR_PX))
        bg.paste(src, (ox, oy))

    return bg


def build_conditioning_rgba(img: Image.Image, tw: int, th: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    src = _fit_source(img.convert("RGB"), tw, th)
    ox = _center_offset(src.width, tw)
    oy = _center_offset(src.height, th)
    rect = (ox, oy, ox + src.width, oy + src.height)

    bg = _make_edge_extended_background(src, tw, th)
    alpha = _make_soft_alpha(tw, th, rect, MASK_FEATHER_PX)

    rgba = bg.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba, rect


def blend_ai_with_source(
    ai_result: Image.Image,
    original: Image.Image,
    full_tw: int,
    full_th: int,
) -> Image.Image:
    if ai_result.size != (full_tw, full_th):
        ai_result = ai_result.resize((full_tw, full_th), Image.LANCZOS)

    ai_rgb = ai_result.convert("RGB")
    src = _fit_source(original.convert("RGB"), full_tw, full_th)
    ox = _center_offset(src.width, full_tw)
    oy = _center_offset(src.height, full_th)
    rect = (ox, oy, ox + src.width, oy + src.height)

    source_canvas = ai_rgb.copy()
    source_canvas.paste(src, (ox, oy))

    preserve_mask = _make_preserve_mask(full_tw, full_th, rect, SOURCE_BLEND_PX)
    return Image.composite(source_canvas, ai_rgb, preserve_mask)


# ═══════════════════════════════ COMFYUI CONNECTION ═══════════════════════════════

class ComfyConnection:
    def __init__(self, server_url: str, verbose: bool = False):
        self.server_url = server_url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self.verbose = verbose

        self._ws = None
        self._thread = None
        self._connected = threading.Event()
        self._job_done = threading.Event()
        self._lock = threading.Lock()
        self._current_pid = None
        self._error = None
        self._completed_pids: set[str] = set()
        self._ws_alive = False

    def connect(self):
        ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws = websocket.WebSocketApp(
            f"{ws_url}/ws?clientId={self.client_id}",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 15, "ping_timeout": 10},
            daemon=True,
        )
        self._thread.start()
        if not self._connected.wait(timeout=15):
            raise ConnectionError(f"WebSocket timed out. Is ComfyUI at {self.server_url}?")

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
        data = msg.get("data", {})
        pid = data.get("prompt_id")

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

    def upload_image(self, filepath: str, upload_name: str):
        with open(filepath, "rb") as f:
            resp = requests.post(
                f"{self.server_url}/upload/image",
                files={"image": (upload_name, f, "image/png")},
                data={"overwrite": "true"},
            )
        resp.raise_for_status()
        return resp.json().get("name", upload_name)

    def free_memory(self, unload_models: bool = False):
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

    def queue_and_wait(self, workflow: dict, timeout: int = JOB_TIMEOUT):
        if not self._connected.is_set():
            raise ConnectionError("WebSocket not connected")

        self._job_done.clear()
        self._error = None
        with self._lock:
            self._completed_pids.clear()
            self._current_pid = None

        resp = requests.post(f"{self.server_url}/prompt", json={"prompt": workflow, "client_id": self.client_id})
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
            raise RuntimeError(f"Execution error:\n{json.dumps(self._error, indent=2)}")

        time.sleep(0.3)
        resp = requests.get(f"{self.server_url}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()
        if prompt_id not in history:
            raise RuntimeError(f"Job {prompt_id} not in /history")
        return prompt_id, history[prompt_id]

    def _poll_history(self, prompt_id: str):
        try:
            hr = requests.get(f"{self.server_url}/history/{prompt_id}", timeout=5)
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
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                self._error = status
                self._job_done.set()
        except requests.RequestException:
            pass

    def download_output(self, job_result: dict, save_path: str):
        for _nid, node_out in job_result.get("outputs", {}).items():
            for img_info in node_out.get("images", []):
                qs = urlencode(
                    {
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info.get("type", "output"),
                    }
                )
                resp = requests.get(f"{self.server_url}/view?{qs}")
                resp.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(resp.content)
                return save_path
        return None


# ═══════════════════════════════ BUCKET TABLE ═════════════════════════════════════

def print_bucket_table():
    printed = set()
    print(f"\n  {'Bucket':>14}   {'Label':<20}  {'Aspect':<8}  {'Pixels':>12}")
    print(f"  {'─'*14}   {'─'*20}  {'─'*8}  {'─'*12}")
    for bw, bh in RESOLUTION_BUCKETS:
        if (bw, bh) in printed:
            continue
        lw, lh = max(bw, bh), min(bw, bh)
        name = _BUCKET_NAMES.get((lw, lh), "")
        ar = _aspect_label(lw, lh)
        print(f"  {bw:>6} x {bh:<5}  {name:<20}  {ar:<8}  {bw*bh:>12,}")
        printed.add((bw, bh))
    print(f"\n  Total: {len(RESOLUTION_BUCKETS)} buckets\n")


# ═══════════════════════════════ JOB PROCESSOR ════════════════════════════════════

class VRAMTracker:
    def __init__(self, comfy: ComfyConnection):
        self.comfy = comfy
        self._since_flush = 0
        self._since_unload = 0

    def tick(self):
        self._since_flush += 1
        self._since_unload += 1

        if self._since_unload >= FULL_UNLOAD_EVERY:
            print("         [mem] full VRAM flush + model unload")
            self.comfy.free_memory(unload_models=True)
            self._since_unload = 0
            self._since_flush = 0
        elif self._since_flush >= FLUSH_EVERY_N:
            self.comfy.free_memory(unload_models=False)
            self._since_flush = 0

    def reset(self):
        self.comfy.free_memory(unload_models=True)
        self._since_flush = 0
        self._since_unload = 0


def process_image(
    comfy: ComfyConnection,
    img: Image.Image,
    tw: int,
    th: int,
    stem: str,
    out_dir: str,
    save_debug: bool = False,
) -> str | None:
    inp_w, inp_h = compute_inpaint_size(tw, th)
    capped = (inp_w != _snap8(tw) or inp_h != _snap8(th))

    if capped:
        print(f"         inpaint at {inp_w}x{inp_h} -> upscale to {tw}x{th}")

    conditioning, _ = build_conditioning_rgba(img, inp_w, inp_h)

    temp_name = f"_resnap_{uuid.uuid4().hex[:8]}.png"
    temp_path = os.path.join(out_dir, temp_name)
    raw_path = os.path.join(out_dir, f"_raw_{uuid.uuid4().hex[:8]}.png")
    conditioning.save(temp_path, "PNG")

    if save_debug:
        debug_path = os.path.join(out_dir, f"{stem}_debug_conditioning_{inp_w}x{inp_h}.png")
        conditioning.save(debug_path, "PNG")

    try:
        actual_name = comfy.upload_image(temp_path, temp_name)
        workflow = build_workflow(actual_name, f"ResSnap/{stem}")

        t0 = time.time()
        print("         queued ... ", end="", flush=True)
        pid, result = comfy.queue_and_wait(workflow)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s ({pid[:12]}...)")

        downloaded = comfy.download_output(result, raw_path)
        if not downloaded:
            print("         [warning] no output image from ComfyUI")
            return None

        ai_result = Image.open(raw_path)
        final = blend_ai_with_source(ai_result, img, tw, th)
        out_path = os.path.join(out_dir, f"{stem}_{tw}x{th}.png")
        final.save(out_path, "PNG")
        print(f"         saved -> {Path(out_path).name}")
        return out_path

    finally:
        for path in (temp_path, raw_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ═══════════════════════════════════ MAIN ═════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resolution Snap — AI Outpainting Bucketer")
    ap.add_argument("--input-dir", default=INPUT_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--comfy-url", default=COMFY_URL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-buckets", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    print("=" * 66)
    print("   Resolution Snap v4.0 — Seamless Outpaint Bucketer")
    print("=" * 66)

    if args.list_buckets:
        print_bucket_table()
        return

    if not os.path.isdir(args.input_dir):
        print(f"\n  [ERROR] Input dir not found: {args.input_dir}")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    comfy = None
    if not args.dry_run:
        print(f"\n  Connecting to ComfyUI at {args.comfy_url} ...", end=" ", flush=True)
        try:
            comfy = ComfyConnection(args.comfy_url, verbose=args.verbose)
            comfy.connect()
            print("OK")
        except Exception as exc:
            print(f"FAILED\n  {exc}")
            sys.exit(1)
    else:
        print("\n  >> DRY RUN — preview only")

    files = sorted(
        f for f in os.listdir(args.input_dir) if Path(f).suffix.lower() in VALID_EXTENSIONS
    )
    if not files:
        print(f"\n  No images in {args.input_dir}")
        return

    print(f"\n  Model:        {INPAINT_MODEL}")
    print(f"  Sampler:      {SAMPLER_NAME} / {SAMPLER_SCHEDULER}")
    print(f"  Steps:        {SAMPLER_STEPS}  |  CFG: {SAMPLER_CFG}  |  Denoise: {SAMPLER_DENOISE}")
    print(f"  Max inpaint:  {MAX_INPAINT_LONG_EDGE}px long edge")
    print(f"  Mask feather: {MASK_FEATHER_PX}px")
    print(f"  Source blend: {SOURCE_BLEND_PX}px")
    print(f"  Context blur: {CONTEXT_BLUR_PX}px")
    print(f"\n  {len(files)} image(s)  |  {len(RESOLUTION_BUCKETS)} buckets\n")

    stats = {"filled": 0, "exact": 0, "failed": 0}
    bucket_usage: dict[tuple[int, int], int] = {}
    vram = VRAMTracker(comfy) if comfy else None
    t_start = time.time()

    try:
        for idx, filename in enumerate(files, 1):
            img_path = os.path.join(args.input_dir, filename)
            stem = Path(filename).stem
            tag = f"[{idx}/{len(files)}]"

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"  X {tag} {filename} — cannot open ({exc})")
                stats["failed"] += 1
                continue

            ow, oh = img.size
            tw, th, action = choose_bucket(ow, oh)
            label = bucket_label(tw, th)
            bucket_usage[(tw, th)] = bucket_usage.get((tw, th), 0) + 1

            if action == "exact":
                print(f"  = {tag} {filename} ({ow}x{oh}) — already {label}, copying")
                img.save(os.path.join(args.output_dir, f"{stem}.png"))
                stats["exact"] += 1
                continue

            act_str = "downscale -> center" if "downscale" in action else "center + outpaint"
            pad_w = tw - min(ow, tw)
            pad_h = th - min(oh, th)
            print(f"  > {tag} {filename}")
            print(f"         {ow}x{oh}  ->  {tw}x{th}  {label}")
            print(f"         [{act_str}]  extra canvas: {pad_w}px x {pad_h}px")

            if args.dry_run:
                stats["filled"] += 1
                continue

            try:
                result = process_image(comfy, img, tw, th, stem, args.output_dir, save_debug=args.save_debug)
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
    processed = stats["filled"] + stats["exact"]

    print()
    print("=" * 66)
    print("   SUMMARY")
    print("=" * 66)
    print(f"   AI-filled:        {stats['filled']}")
    print(f"   Already standard: {stats['exact']}")
    print(f"   Failed:           {stats['failed']}")
    print(f"   Total time:       {total_time:.0f}s  ({total_time / max(1, processed):.1f}s avg)")

    if bucket_usage:
        print("\n   Bucket distribution:")
        for (bw, bh), count in sorted(bucket_usage.items(), key=lambda x: x[1], reverse=True):
            lbl = bucket_label(bw, bh)
            bar = "#" * min(count, 50)
            print(f"     {bw:>5}x{bh:<5}  {lbl:<28}  {count:>3}  {bar}")

    print(f"\n   Output folder: {args.output_dir}")
    print("=" * 66)


if __name__ == "__main__":
    main()
