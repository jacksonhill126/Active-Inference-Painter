# Validation Summary

## Historical full S0 baseline

The accepted T-104 record is `docs/BASELINE_TEST_RESULT_2026-07-24.md`:
`python -m pytest -q` completed with 252 passes, exit status 0, and duration
349.09 seconds on Python 3.14.3.

## 2026-08-11 capture-focused regression

The focused selection covered the default fail-closed sensor boundary, runtime
state, command/canvas response, white/black canvas rendering, telemetry CSV and
rolling retention, code version, frontend contract, and motor telemetry schema.

Clean rerun result: `11 passed, 6 warnings in 227.05s`, process exit status 0.
The six warnings are the expected deprecation warning emitted by fixtures that
construct the obsolete `planner_state_kind='summary'` compatibility path; the
captured runtime itself used `spatial_material`.

An earlier identical invocation reached `11 passed` in 186.74 seconds but the
outer command wrapper timed out immediately afterward and returned 124. It is
recorded here rather than hidden; the clean rerun supersedes it as acceptance
evidence.

## AI-106 focused acceptance

Terminal coverage/stopping validation completed separately with `48 passed in
3.60s`. See
`docs/TERMINAL_COVERAGE_STOPPING_ACCEPTANCE_2026-08-11.md`.
