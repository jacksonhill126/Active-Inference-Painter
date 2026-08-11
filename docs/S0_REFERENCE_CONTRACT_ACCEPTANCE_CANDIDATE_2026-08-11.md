# S0 Reference-Contract Acceptance Candidate

Date: 2026-08-11
Decision owner: Jackson
Tracker task: T-108 (`Ready`, not yet accepted)

## Recommendation

Accept S0 as a **versioned abstract comparison reference**, with no hardware or
calibration claim. The current evidence satisfies T-101 through T-107. The
remaining T-108 action is the owner’s explicit lock decision.

This decision would authorize S1/S2 and later backends to compare against or
explicitly retarget from `native-abstract-v0`. It would not require MuJoCo to
copy the native co-located shoulder geometry, representative actuator
parameters, or oracle data access. In fact, those differences must remain
named.

## Evidence now present

- T-101: native arm/material reference recorded in
  `docs/NATIVE_PLANT_REFERENCE.md`.
- T-102: thickness-derived coverage, including white-on-white, is test-backed.
- T-103: `plant-interface-v1` separates painting intent, low-level
  realization, sensor packets, beliefs, counterfactuals, evaluation truth, and
  external safety.
- T-104: the accepted 2026-07-24 full baseline recorded 252 passing tests.
- T-105: the versioned 2026-08-11 default web capture contains finite state,
  256 x 256 canvas PNG, 56 x 96 telemetry table, frontend assets, startup logs,
  resolved config, and exact hashes. A current focused runtime selection passes
  11 tests with exit status 0.
- T-106: one consolidated table classifies every shortcut as S0 reference-only,
  a MuJoCo calibration need, or a physical validation need.
- T-107: bundle location/content/manifest rules are documented and instantiated.

## Stable interfaces proposed for lock

- joint order and logical native frame under the `native-abstract-v0` label;
- normalized Cartesian/contact `StrokeAction` painting intent;
- conventional IK/trajectory/control below the painting-policy decision;
- material thickness, wetness, pigment, tone, and thickness-derived coverage
  semantics;
- the command/sensor/belief/counterfactual/evaluation-truth separation; and
- versioned backend/model/likelihood/approximation provenance.

## Explicitly provisional details

- native geometry, gains, dynamics, friction, backlash, compliance, contact,
  and noise values;
- selected RobStride plant parameters until physical system identification;
- brush anisotropy, deposition, ferrule stick/release, and load-history models;
- camera intrinsics, noise, rates, timing, synchronization, housings, and mounts;
- the moment-matched terminal coverage forecast rejected for M2 by AI-106;
- finite candidate-set posterior semantics from AI-111; and
- all sensor-equivalent and sim-to-real claims.

## Open issue that does not need to be hidden

T-109 remains blocked: native execution has no conforming `PlantBackend` sensor
adapter, so the default sensor boundary correctly fails closed. This prevents
the native runtime from serving as sensor-equivalent cognition but does not
prevent it from serving as the S0 abstract reference. The capture demonstrates
that distinction directly rather than using oracle mode to make the baseline
look more complete.

## Owner decision requested

The recommended decision text is:

> Accept S0 / `native-abstract-v0` as the reproducible abstract reference under
> the stated interface and material contracts. Its geometry, plant dynamics,
> contact, sensors, and controller parameters remain provisional; T-109 and the
> classified MuJoCo/hardware work remain open. S1/S2 may compare against or
> explicitly retarget from this reference without treating it as physical
> truth.
