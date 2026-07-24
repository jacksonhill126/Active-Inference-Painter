# `output/figures/extras/` - agent guide

Regenerable PNG/GIF media for extras topic scripts.

## Rules

1. Generated media here is ephemeral and ignored by git.
2. Topic README files are hand-maintained and must stay aligned with
   `extras/<topic>/README.md`.
3. Re-render extras through the shared menu runner:

   ```bash
   uv run python -m active_inference.menu --extras
   ```

4. Validate rendered extras together with chapter artifacts:

   ```bash
   uv run python scripts/validate_rendered_figures.py --root output/figures
   ```
