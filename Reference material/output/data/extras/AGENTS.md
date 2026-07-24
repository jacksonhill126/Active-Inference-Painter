# `output/data/extras/` - agent guide

Regenerable NPZ+JSON sidecars for extras topic scripts.

## Rules

1. Generated `.npz` and `.json` files are ignored by git; do not hand-edit
   them.
2. Topic READMEs are hand-maintained and should name the producing extras
   script.
3. Extras scripts should write here via `save_extra_data` or figure/animation
   helpers that infer `output/data/extras/<topic>/` from the figure path.
4. Validate after regenerating extras:

   ```bash
   uv run python scripts/validate_raw_data_exports.py --root output/data
   ```
