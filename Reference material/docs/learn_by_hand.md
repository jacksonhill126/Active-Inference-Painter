# Learning by hand — no LLM, deliberate practice

This repository is built for you to *do the work*, not to watch a model do it
for you. There is no LLM review stage, no code-generation step, no autopilot:
every figure, number, and test in `output/` is produced by a script you run and
a library you can read. The understanding comes from the run → observe → fix
loop, not from a summary someone else wrote.

Use this page as the companion to [`reading_order.md`](reading_order.md). Where
that page tells you *what to read*, this page tells you *how to practice* so the
material actually sticks.

## The three surfaces you actually touch

Everything you learn by doing happens across three places. Keep them mentally
separate — each has a different job.

| Surface | What it is | What you do there |
|---|---|---|
| `src/active_inference/` | The library — every algorithm (distributions, inference, estimators, visualizations). Pure logic, fully tested. | **Read it** to understand *how* a method works. Occasionally **edit it** when extending. |
| `chapters/`, `extras/`, `demo/` | Thin orchestrators (≤ ~120 LOC) that wire library components together to produce a figure or raw-data file. No domain logic. | **Run them** to see the method in action. **Copy/tweak one** to run your own experiment. |
| `tests/` | pytest suite mirroring `src/` one-for-one; 90%+ coverage, no mocks. | **Run them** to confirm your change is correct. **Add a small test** to lock in what you learned. |

That's the whole map. The `scripts/` validators and `run.sh` menu are just
drivers around these three — they never contain the understanding.

## Run each piece by hand

You don't need to launch anything big. Run the smallest thing that will give
you feedback:

```bash
# One example, headless, saving figure + raw data
uv run python chapters/chapter_02/example_2_2_linear_probabilistic.py --save

# One whole chapter through the text menu
./run.sh --chapter 2
uv run python -m active_inference.menu --chapter 2

# Unit tests for the logic you just read (fast, no subprocess)
uv run pytest tests/core tests/estimators

# Smoke tests: every chapter script run with --save
uv run pytest tests/chapters -v

# Check the generated artifacts are real (not hand-faked)
uv run python scripts/validate_raw_data_exports.py --root output/data
uv run python scripts/validate_orchestrator_provenance.py
uv run python scripts/validate_source_spine.py --require-pdf

# Export Jupyter notebooks (one per chapter/extras/demo)
uv run python scripts/export_notebooks.py
```

Each command reports real pass/fail against real artifacts. Read the output,
change something, run it again. That loop is the practice.

## No LLM to skip — the loop *is* the teacher

Unlike some research pipelines that bolt an optional "LLM scientific review"
stage onto the end, this repo has none. There is nothing to turn off. The
intended workflow is already the manual one:

1. You read the source in `src/`.
2. You run an orchestrator and inspect `output/figures/` and `output/data/`.
3. You form a hypothesis ("if I widen the prior, the posterior should shift
   toward the likelihood").
4. You change a parameter and re-run — and *see* whether you were right.

The correction signal is the rendered figure and the numeric sidecar, not a
chat reply. That is what makes it stick.

## The interactive menu is also by hand

`./run.sh` is a stdlib text menu — no model behind it. Use it to browse and
run chapters without remembering filenames:

```bash
./run.sh                 # interactive text menu
./run.sh --all           # render every chapter to output/figures/
./run.sh --chapter 3     # one chapter
./run.sh --script example_2_2   # one orchestrator by filename fragment
./run.sh --list          # print the discovered menu and exit

# Local browser UI — one tab per chapter, render buttons, inline docs
./run.sh --web
```

`./run.sh --web` is the same content with a clickable gallery; nothing is
generated for you that you couldn't generate from the terminal.

## A good learning loop

Pick one concept and run the full cycle. Example: Chapter 2, Example 2.2
(Gaussian likelihood × Gaussian prior).

1. **Read the thin orchestrator.**
   `chapters/chapter_02/example_2_2_linear_probabilistic.py` — see how it
   builds a process + model and calls the inference routine.
2. **Read the logic it calls.**
   `src/active_inference/core/inference.py` (`GridBayesianInference`) and
   `src/active_inference/estimators/mle.py` — the actual math.
3. **Run it.**
   `uv run python chapters/chapter_02/example_2_2_linear_probabilistic.py --save`
   Then open `output/figures/chapter_02/...` and
   `output/data/chapter_02/<stem>.npz` + `<stem>.json`.
4. **Change one thing and re-run.** Copy the script to a scratch file (or edit
   a parameter), e.g. set the prior precision very low, and observe the
   posterior mean slide toward the MLE. This is the feedback that teaches.
5. **Connect to the book.** Open [`topics/bayesian_inference.md`](topics/bayesian_inference.md)
   and [`notation.md`](notation.md) to map the code identifier to the book
   symbol (e.g. `m_x`, `s2_x` ↔ prior mean/variance).
6. **Lock it in with a test.** Write a small assertion about what you observed
   (e.g. posterior mean → MLE as prior precision → 0) in `tests/`. Run
   `uv run pytest tests/chapters -v` to confirm nothing regressed.

Steps 3–4 are the part an LLM summary would skip. They are the part that builds
the intuition.

## Where to learn the rules

- [`reading_order.md`](reading_order.md) — Path A (book follower), Path D
  (extender), Path E (claim verifier).
- [`architecture.md`](architecture.md) — the layered design; which subpackage
  owns which concern.
- [`notation.md`](notation.md) — every book symbol ↔ Python identifier.
- [`AGENTS.md`](../AGENTS.md) — the contributor contract and "How to Add a New
  Example".
- [`reference/`](reference/) — per-subpackage API catalogue (`core.md`,
  `estimators.md`, `utils.md`, `visualizations.md`).

## The one hard rule

`output/` (figures, data, notebooks) is git-ignored and regenerated from source.
Never hand-edit an artifact under `output/` to make a validator pass or to make
a figure "look right." Fix the source in `src/` or the orchestrator.

The validators — `validate_raw_data_exports.py`,
`validate_orchestrator_provenance.py`, `validate_source_spine.py` — exist to
catch real gaps (missing raw-data sidecars, domain logic leaking into a
wrapper, source-status drift). Patching the output to satisfy them defeats the
discipline that makes this repo teach you something. If a validator fails, the
bug is upstream in your code, not in the artifact.

## Quick orientation

If you only ever read one page, make it [`cookbook.md`](cookbook.md) — then run
every recipe and change a parameter in each. If you read a second, make it
[`architecture.md`](architecture.md) so you know which layer to open when
something surprises you.
