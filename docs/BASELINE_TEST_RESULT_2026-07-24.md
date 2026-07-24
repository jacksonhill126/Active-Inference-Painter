# Baseline Test Result: 2026-07-24

Task: T-104
Result: pass
Observation boundary: `baseline-oracle-v0`
Plant contract: `native-abstract-v0`
Interface contract: `plant-interface-v1`

## Command

Run from the repository root:

```powershell
python -m pytest -q
```

## Result

```text
252 passed in 349.09s (0:05:49)
```

There were no failures, errors, skips, or expected failures. Pytest exited with
status code `0`.

## Environment

| Item | Value |
| --- | --- |
| Operating system | Windows 11 `10.0.26200` |
| Python | `3.14.3` |
| pytest | `9.1.1` |
| NumPy | `2.4.3` |
| PyTorch | `2.11.0+cu126` |
| CUDA available to PyTorch | `True` |
| Pillow | `12.1.1` |
| Package/source build | `0.1.0+code.66` |
| Source fingerprint | `714ba7ecd71678c2efe018761c5e7fe697ec96e3be71b43a8439701e7e2f456e` |
| Base Git commit | `7c0aedbd7c8b819271413999352849531b9916d0` |

The suite ran against the working tree containing the M0, AI-101 through
AI-103, and T-101 through T-103 changes. The source build fingerprint was
captured after the run without intervening source changes. The tree had not
yet been committed, so the base Git commit alone is not the tested revision;
the source fingerprint is the exact runtime source identity.

## Interpretation

This result establishes that the current baseline implementation and contract
tests are mutually consistent. It does not establish:

- sensor-equivalent inference;
- calibrated VFE, EFE, transition, hierarchy, or motor uncertainty;
- removal of copied simulator/RNG state from motor forecasts;
- MuJoCo parity;
- physical safety or sim-to-real validity;
- emergent composition.

Those remain gated by the corresponding M1-M3 and T-109 tasks.
