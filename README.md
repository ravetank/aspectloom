# AI Sizing Normalizer

AI Sizing Normalizer takes a folder of images, chooses a conventional target
resolution for each one, and uses a local
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) server to outpaint the
missing canvas area. The original center is blended back into the result so
the source image stays sharp while the generated border heals naturally.

This is a personal batch-processing utility, first built on April 3, 2026.
Version 1.0.0 is the cleaned, documented baseline.

## What it does

For every supported image in one input folder, the app:

1. Selects a standard landscape, portrait, or square resolution.
2. Centers the source on an RGBA canvas.
3. Exposes the empty area plus a narrow overlap ring as an inpainting mask.
4. Sends an embedded workflow to ComfyUI.
5. Waits through WebSocket events, with HTTP history polling as a fallback.
6. Downloads the generated image and softly restores the untouched center.
7. Saves a PNG in the output folder.

It supports PNG, JPEG, WebP, BMP, and TIFF input. Input subfolders are not
scanned.

## Before you run it

You need:

- [Pixi](https://pixi.sh/latest/) for the reproducible Python environment.
- A running ComfyUI instance, normally at `http://127.0.0.1:8188`.
- An SDXL-compatible inpainting checkpoint available to ComfyUI.
- Enough local compute and VRAM for the checkpoint and chosen inpaint size.

The default checkpoint is:

```text
inpaint\juggernautXL_versionXInpaint.safetensors
```

That path is relative to ComfyUI's checkpoint directory. Use `--model` if
your filename or subfolder differs. The embedded workflow uses only standard
ComfyUI nodes.

## Quick start

Open PowerShell in this repository and install the locked environment:

```powershell
pixi install
```

Preview how the images will be assigned without starting ComfyUI or writing
anything:

```powershell
pixi run dry-run --input-dir "D:\Pictures\To Normalize"
```

Start ComfyUI, then perform the full run:

```powershell
pixi run run --input-dir "D:\Pictures\To Normalize"
```

By default, output goes to an `ai-sized` folder inside the input folder. To
put it elsewhere:

```powershell
pixi run run --input-dir "D:\Pictures\To Normalize" --output-dir "D:\Pictures\Normalized"
```

If your checkpoint has a different name:

```powershell
pixi run run --input-dir "D:\Pictures\To Normalize" --model "inpaint\my_model.safetensors"
```

Existing output files are skipped. Add `--overwrite` only when you intend to
replace them.

## Useful commands

| Goal | Command |
| --- | --- |
| Show help | `pixi run run --help` |
| Show version | `pixi run run --version` |
| List all resolution buckets | `pixi run list-buckets` |
| Preview a folder | `pixi run dry-run --input-dir "PATH"` |
| Process a folder | `pixi run run --input-dir "PATH"` |
| Use another ComfyUI server | Add `--comfy-url "http://HOST:8188"` |
| Keep diagnostic canvases/masks | Add `--save-debug` |
| Show WebSocket diagnostics | Add `--verbose` |
| Run tests | `pixi run test` |
| Compile-check the source | `pixi run check` |

### Bucket strategies

The default `--bucket-strategy aspect` balances aspect-ratio similarity and
unused canvas area. It usually gives the most natural outpainting shape.

`--bucket-strategy first_fit` instead selects the smallest-area bucket that
fully contains the source. It can reduce output size, but may add more border
on one axis.

Images larger than every available bucket are proportionally downscaled into
the best matching bucket. Images already at a bucket resolution are copied to
PNG without contacting ComfyUI.

### Padding modes

The default `--padding-mode transparent` leaves the generated area empty and
uses alpha as the inpaint mask.

`--padding-mode edge_extend` stretches and blurs edge pixels into the
generated area. This can help with some models, but transparent padding is the
safer general default.

## Output and safety

- The input folder is never modified.
- The output folder is required to be different from the input folder.
- Existing outputs are skipped unless `--overwrite` is supplied.
- Temporary `_resnap_*.png` and `_raw_*.png` files are removed after each
  image, including after most failures.
- ComfyUI also writes its own workflow output below
  `ComfyUI/output/AI-Sizing-Normalizer/`.
- Exact bucket matches are named `NAME.png`.
- Outpainted results are named `NAME_WIDTHxHEIGHT.png`.
- Two source files with the same stem, such as `photo.jpg` and `photo.webp`,
  target the same output name. The second one is skipped unless overwrite is
  enabled.

Run `--dry-run` first when using a new folder or a new set of bucket
settings.

## Tuning

The most useful defaults are grouped at the top of
`resolution_snap.py`. Command-line options cover paths, model, server,
bucket strategy, and padding mode. Edit these constants for deeper tuning:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `SAMPLER_STEPS` | 18 | Diffusion steps per image |
| `SAMPLER_CFG` | 6.5 | Prompt guidance strength |
| `SAMPLER_SEED` | 1982 | Fixed seed for repeatable runs |
| `OUTPAINT_OVERLAP_PX` | 40 | Source-edge band the model may repaint |
| `MASK_FEATHER_PX` | 8 | Softens the inpaint-mask boundary |
| `PRESERVE_CENTER_BLEND_PX` | 28 | Softly restores the original center |
| `MAX_INPAINT_LONG_EDGE` | 1536 | Normal maximum inference edge |
| `HIGH_WASTE_LONG_EDGE` | 2048 | Inference edge used when padding is large |
| `JOB_TIMEOUT` | 900 s | Maximum wait per ComfyUI job |
| `FULL_UNLOAD_EVERY` | 5 | Fully unload the model every N jobs |

`POSITIVE_PROMPT` and `NEGATIVE_PROMPT` are intentionally style-neutral.
Change them when a batch needs a specific visual treatment.

## Troubleshooting

### “Input dir not found”

Check the quoted path after `--input-dir`. PowerShell paths containing spaces
must stay inside quotes.

### “WebSocket timed out”

Confirm ComfyUI is running and its address matches `--comfy-url`. Open
`http://127.0.0.1:8188` in a browser on the same machine as a quick check.

### ComfyUI rejects the prompt

The most common cause is a missing checkpoint. Confirm the `--model` value
matches the filename shown by ComfyUI's checkpoint loader, including any
subfolder.

### Visible seams or an inner rectangle

Try increasing `OUTPAINT_OVERLAP_PX`,
`PRESERVE_CENTER_BLEND_PX`, or `MASK_FEATHER_PX` in small increments.
Different inpainting models respond differently.

### Results are slow or run out of VRAM

Lower `MAX_INPAINT_LONG_EDGE` and `HIGH_WASTE_LONG_EDGE`. The final image
is still resized to the selected bucket; these values control only the
inference canvas.

### A completed image was skipped

That is the default protection against overwriting. Use `--overwrite` if the
existing file should be regenerated.

## Project layout

```text
resolution_snap.py   Maintained application and embedded ComfyUI workflow
pixi.toml            Environment, dependencies, and convenience tasks
pixi.lock            Reproducible dependency lock
tests/               Offline unit tests for buckets, masks, compositing, workflow
archive/             April 3 development experiments; not supported entry points
dist/                Locally generated source ZIPs; ignored by Git
```

The local `.pixi` environment is intentionally excluded from Git and release
archives. Recreate it at any time with `pixi install`.

## Development and maintenance

Before committing a change:

```powershell
pixi run check
pixi run test
pixi run python resolution_snap.py --list-buckets
```

For a real end-to-end check, put one expendable image in a temporary folder,
run `--dry-run`, then process it through a running ComfyUI instance.

When revisiting this project months from now, check these four things first:

1. Does `pixi install` still solve the locked environment?
2. Does the configured checkpoint still exist in ComfyUI?
3. Do `pixi run check` and `pixi run test` pass?
4. Does one dry run choose the bucket you expect before processing a batch?

## Historical note

The files under `archive/2026-04-03-experiments/` preserve the iterations
that led to the maintained overlap-ring implementation. They are kept for
reference and archaeology; fixes should go into the root-level
`resolution_snap.py`.
