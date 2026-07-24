# `extras/` - agent guide

Cross-cutting topic orchestrators that sit beside the chapter reproduction
spine. Use this folder for small deterministic demos that clarify a concept
shared by multiple chapters, such as entropy, temperature, or the
thermodynamic bridge.

## Rules

1. **Keep extras thin.** Each topic script imports from `active_inference` and
   the Python standard library only; reusable math belongs in
   `src/active_inference/`.
2. **One topic, one folder.** A topic lives under `extras/<topic>/` with a
   `README.md`, a required `visualize_<topic>.py` script, and optional
   `simulate_<topic>.py` / `animation_<topic>.py` wrappers declared by
   `active_inference.extra_topics`.
3. **`--save` is mandatory.** Saved runs write a PNG to
   `output/figures/extras/<topic>/` and paired NPZ+JSON raw-data sidecars to
   `output/data/extras/<topic>/`.
4. **Use `save_extra_data`.** For bespoke arrays, call
   `save_extra_data(topic, stem, arrays, metadata, figures=...)`; figure
   helpers can export plotted data automatically when the path is under
   `output/figures/extras/<topic>/`.
5. **Document at the point of use.** The topic README must name the script,
   run command, figure path, and raw-data path.

## Verification

```bash
uv run pytest tests/extras -v
uv run python scripts/validate_raw_data_exports.py --root output/data
```

## Do Not Put

- Chapter-specific examples that belong in `chapters/chapter_NN/`.
- Shared algorithms or plotting helpers; move those to `src/active_inference/`.
- Generated figures or raw data; those belong under `output/`.
