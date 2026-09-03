# Aspectloom

**Standard frames. Seamless edges.**

Aspectloom turns a folder of mixed-size images into a predictable set of
standard resolutions. When an image does not fill its target aspect ratio, it
asks a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server to
outpaint the missing edges, then blends source-derived pixels back into the
center so detail there stays sharp. Oversized inputs are proportionally
downscaled first.

It is a small, open-source companion in spirit to
[Lenscribe](https://github.com/ravetank/lenscribe): Lenscribe gives image files
useful names; Aspectloom gives images consistent frames. They are separate
applications and neither one is required by the other.

Aspectloom was first built as a personal utility on April 3, 2026. Version
1.0.0 is the cleaned, documented baseline.

## What it does

For each supported image directly inside an input folder, Aspectloom:

1. Chooses a standard landscape, portrait, or square resolution.
2. Centers the source on a larger RGBA canvas when necessary.
3. Marks the empty area and a narrow overlap ring as an inpainting mask.
4. Sends the included workflow to ComfyUI.
5. Waits for completion, using WebSocket events with HTTP polling as a
   fallback.
6. Downloads the generated image and softly restores the source-derived
   center.
7. Saves the finished image as a PNG in a separate output folder.

Supported inputs are PNG, JPEG, WebP, BMP, and TIFF. Subfolders are not
scanned. Images that already match a configured bucket are converted directly
to PNG without contacting ComfyUI.

## First-time setup

You need:

- [Git](https://git-scm.com/) to clone the repository.
- [Pixi](https://pixi.sh/latest/) to create the reproducible Python
  environment.
- A local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installation.
- An SDXL-compatible inpainting checkpoint available to ComfyUI.
- Enough local compute and VRAM for that checkpoint and the selected inpaint
  size.

### 1. Get Aspectloom

```powershell
git clone https://github.com/ravetank/aspectloom.git
cd aspectloom
pixi install
```

### 2. Check the model name

The default checkpoint is:

```text
inpaint\juggernautXL_versionXInpaint.safetensors
```

That path is relative to ComfyUI's `models/checkpoints` folder. If your model
has a different filename or subfolder, pass it with `--model`. Aspectloom does
not include or download a checkpoint.

### 3. Start ComfyUI

Start ComfyUI normally and leave it running. Aspectloom expects it at
`http://127.0.0.1:8188` unless you provide another address with `--comfy-url`.
The embedded workflow uses only standard ComfyUI nodes.

### 4. Preview the batch

Dry-run mode reads image dimensions and shows the selected resolution for each
file. It does not contact ComfyUI or write output.

```powershell
pixi run dry-run --input-dir "D:\Pictures\To Normalize"
```

Check the proposed targets before continuing.

### 5. Process the batch

```powershell
pixi run run --input-dir "D:\Pictures\To Normalize"
```

Finished files appear in `D:\Pictures\To Normalize\aspectloom-output` by
default. The source files are left alone.

To choose a different output folder and checkpoint:

```powershell
pixi run run --input-dir "D:\Pictures\To Normalize" `
  --output-dir "D:\Pictures\Normalized" `
  --model "inpaint\my_model.safetensors"
```

Existing output files are skipped. Add `--overwrite` only when you intend to
replace them.

## Command reference

| Goal | Command |
| --- | --- |
| Show all options | `pixi run run --help` |
| Show the version | `pixi run run --version` |
| List resolution buckets | `pixi run list-buckets` |
| Preview a folder | `pixi run dry-run --input-dir "PATH"` |
| Process a folder | `pixi run run --input-dir "PATH"` |
| Choose an output folder | Add `--output-dir "PATH"` |
| Choose a checkpoint | Add `--model "SUBFOLDER\MODEL.safetensors"` |
| Use another ComfyUI server | Add `--comfy-url "http://HOST:8188"` |
| Replace existing results | Add `--overwrite` |
| Keep diagnostic canvases and masks | Add `--save-debug` |
| Show connection diagnostics | Add `--verbose` |
| Run the offline tests | `pixi run test` |
| Compile-check the source | `pixi run check` |

You can also run `python aspectloom.py ...` from any Python environment that
contains the packages listed in `pixi.toml`. The old `resolution_snap.py`
filename remains as a compatibility launcher.

## How target sizes are chosen

The built-in buckets cover common landscape, portrait, and square resolutions.
Run `pixi run list-buckets` to see the complete current list.

The default `--bucket-strategy aspect` balances aspect-ratio similarity with
the amount of empty canvas to generate. It usually produces the most natural
outpainting shape.

`--bucket-strategy first_fit` chooses the smallest-area bucket that fully
contains the source. It can reduce output dimensions but may add more border
on one axis.

If an image is larger than every bucket, Aspectloom proportionally scales it
down to fit the best-matching target. The selected target is always shown in a
dry run.

## Padding and compositing

The default `--padding-mode transparent` leaves the new canvas area empty and
uses alpha to define the inpainting mask.

`--padding-mode edge_extend` stretches and blurs edge pixels into the new area.
Some models respond well to that extra context, but transparent padding is the
safer general default.

After generation, Aspectloom softly restores the source image over the center
of the result. This preserves original detail while retaining a blended band
where the generated border meets the source.

## Output and safety

- The input folder and source files are never modified.
- The output folder must be different from the input folder.
- Existing outputs are skipped unless `--overwrite` is supplied.
- Temporary `_resnap_*.png` and `_raw_*.png` files are removed after each
  image, including after most failures.
- ComfyUI also keeps its workflow output under
  `ComfyUI/output/Aspectloom/`.
- Exact bucket matches are named `NAME.png`.
- Outpainted results are named `NAME_WIDTHxHEIGHT.png`.
- Two source files with the same stem, such as `photo.jpg` and `photo.webp`,
  target the same output name. The second is skipped unless overwrite is
  enabled.

By default, image data stays on the computer running Aspectloom and ComfyUI.
If `--comfy-url` points to another machine, images are sent to that server.
Only use a remote server you trust, and do not expose an unauthenticated
ComfyUI instance directly to the public internet.

ComfyUI and any checkpoint you use are separate projects with their own
licenses and terms. Confirm that your chosen model permits your intended use.

## Tuning

Common paths and behavior are available as command-line options. Deeper image
generation settings live near the top of `aspectloom.py`:

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

`POSITIVE_PROMPT` and `NEGATIVE_PROMPT` are intentionally style-neutral. Edit
them when a batch needs a particular visual treatment.

## Troubleshooting

### “Input dir not found”

Check the quoted path after `--input-dir`. PowerShell paths containing spaces
must remain inside quotes.

### “WebSocket timed out” or a connection error

Confirm ComfyUI is running and its address matches `--comfy-url`. Opening
`http://127.0.0.1:8188` in a browser on the same machine is a quick local
check. With a remote server, also verify its firewall and network path.

### ComfyUI rejects the prompt

The most common cause is a missing checkpoint. Confirm that `--model` exactly
matches the filename shown in ComfyUI's checkpoint loader, including any
subfolder.

### Visible seams or an inner rectangle

Try changing `OUTPAINT_OVERLAP_PX`, `PRESERVE_CENTER_BLEND_PX`, or
`MASK_FEATHER_PX` in small increments. Different inpainting models respond
differently.

### Results are slow or run out of VRAM

Lower `MAX_INPAINT_LONG_EDGE` and `HIGH_WASTE_LONG_EDGE`. The final image is
still resized to the selected bucket; these values control only the inference
canvas.

### A completed image was skipped

That is the default overwrite protection. Use `--overwrite` if the existing
file should be regenerated.

## Project layout

```text
aspectloom.py        Maintained application and embedded ComfyUI workflow
resolution_snap.py   Compatibility launcher for the original filename
pixi.toml             Environment, dependencies, and convenience tasks
pixi.lock             Reproducible dependency lock
tests/                Offline tests for buckets, masks, compositing, and workflow
archive/              April 3 experiments; not supported entry points
dist/                 Locally generated source ZIPs; ignored by Git
```

The local `.pixi` environment is excluded from Git and release archives.
Recreate it at any time with `pixi install`.

## Development and maintenance

Before committing a change:

```powershell
pixi run check
pixi run test
pixi run list-buckets
```

For a real end-to-end check, put one expendable image in a temporary folder,
run a dry run, then process it through a running ComfyUI instance.

When returning to this project months from now, check these four things first:

1. Does `pixi install` still solve the locked environment?
2. Does the configured checkpoint still exist in ComfyUI?
3. Do `pixi run check` and `pixi run test` pass?
4. Does a dry run choose the targets you expect before processing a batch?

The files under `archive/2026-04-03-experiments/` preserve the development
iterations that led to the maintained overlap-ring implementation. They are
for reference and archaeology; fixes belong in `aspectloom.py`.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Aspectloom is licensed under the [Apache License 2.0](LICENSE). ComfyUI and
model files are not part of Aspectloom and are governed by their own licenses.
