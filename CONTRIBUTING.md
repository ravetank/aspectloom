# Contributing to Aspectloom

Aspectloom is a small personal utility, but clear bug reports and focused
improvements are welcome.

## Before opening an issue

- Run the latest `main` branch.
- Run `pixi install`, then `pixi run check` and `pixi run test`.
- Try the same input with `--dry-run` and note the selected bucket.
- Check whether the problem comes from Aspectloom, ComfyUI, or the selected
  checkpoint.

For a bug, include the operating system, Python/Pixi details, ComfyUI version,
checkpoint name, complete command, relevant console output, and source image
dimensions. Do not upload private images or secrets. A synthetic replacement
image that reproduces the issue is ideal.

## Making a change

1. Create a focused branch.
2. Keep the command-line interface backward compatible when practical.
3. Add or update offline tests for behavior changes.
4. Run:

   ```powershell
   pixi run check
   pixi run test
   pixi run list-buckets
   ```

5. If generation behavior changed, test one expendable image with a local
   ComfyUI instance and describe the result in the pull request.

Keep generated images, checkpoints, the `.pixi` environment, and personal
input folders out of commits. Contributions are accepted under the repository's
[Apache License 2.0](LICENSE).
