# Active-Inference Painter

[![Smoke tests](https://github.com/jacksonhill126/Active-Inference-Painter/actions/workflows/ci.yml/badge.svg)](https://github.com/jacksonhill126/Active-Inference-Painter/actions/workflows/ci.yml)

An embodied active-inference research project built around a simulated robotic
abstract painter.

The project asks whether a physically situated agent can develop temporally
extended spatial organization through perception, prediction, action, and
belief revision without reference images, aesthetic rewards, demonstrated
painting policies, or fine-tuning on a painting corpus.

> **Status:** M1 formal-baseline research prototype. Paint material remains in the custom
> Python process; realized arm dynamics and contact can use either the native
> reference plant or the hardware-oriented MuJoCo model. It is not validated
> for physical hardware, its default sensor path remains fail-closed, and its
> present visual output should not be treated as evidence of learned
> composition.

## Quick Start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m active_painter.web_server
```

Open `http://127.0.0.1:8017`.

The live default is deliberately fail-closed: the viewer runs, but active
policy inference remains disabled. To run the bounded, simulation-only sensor
loop that actually paints through MuJoCo cameras and encoder/contact packets:

```bash
python -m pip install -e ".[dev,mujoco]"
python -m active_painter.web_server --plant-backend mujoco --enable-provisional-sensor-policy --stroke-tone-prior black
```

This opt-in `provisional-sensor-simulation-v0` profile waits for an initial
registered camera likelihood and body posterior, selects painting policies,
forecasts them in a separately constructed MuJoCo model, executes a stroke,
waits for a causally later camera correction, and repeats. Policy inference
does not copy the live canvas, brush, or plant state. The fixed forecast model,
independent substrate/brush priors, compact brush history, and zero-history
compliance prior are explicit simulation approximations. The low-level
execution/safety controller may still use exact simulator pose and contact
below the selected painting policy. This is an integration demonstration, not
a hardware-calibrated or painting-quality result.

The former hidden-state baseline is retained only as an explicitly labelled
diagnostic comparator:

```bash
python -m active_painter.web_server --observation-mode oracle_material_state
```

That mode is not sensor-equivalent and must not support physical-agent claims.

To use MuJoCo for scripted/viewer dynamics without enabling the provisional
painting-policy loop:

```bash
python -m pip install -e ".[dev,mujoco]"
python -m active_painter.web_server --plant-backend mujoco
```

For accelerated shared pretraining, collect independent headless
MuJoCo/camera-posterior trajectories and then train centrally:

```bash
python -m active_painter.parallel_collect --workers 6 --trajectories 60 --output-dir runs/corpus
python -m active_painter.offline_train --manifest runs/corpus/split_manifest.json --output-checkpoint runs/checkpoints/shared-pretraining.pt --device cuda
```

Workers keep their posteriors and histories isolated. Dataset assignment occurs
by whole trajectory before local-patch extraction, and the output is labelled
shared pretraining rather than one agent's individual development. See
[`docs/PARALLEL_TRAINING_PIPELINE_2026-08-10.md`](docs/PARALLEL_TRAINING_PIPELINE_2026-08-10.md).
The collector recycles the full runtime after each trajectory by default and
retains/audits completed shards if another job fails. The accepted AI-108
baseline combines bounded fixed-roll and opt-in `research_full_roll` batches;
see its [technical record](docs/AI108_CORPUS_TECHNICAL_2026-08-11.md) and
[owner brief](docs/AI108_CORPUS_OWNER_BRIEF_2026-08-11.md).
AI-107's held-out calibration on that corpus is also complete, with a negative
M2 result; see the
[technical calibration record](docs/AI107_UNCERTAINTY_CALIBRATION_TECHNICAL_2026-08-11.md)
and [owner brief](docs/AI107_UNCERTAINTY_CALIBRATION_OWNER_BRIEF_2026-08-11.md).

An experimental conditional patch-transition VAE can be trained against the
same whole-trajectory splits in shadow mode:

```bash
python -m active_painter.conditional_vae_train --manifest runs/corpus/split_manifest.json --output-checkpoint runs/checkpoints/conditional-patch-cvae.pt --device cuda
```

It has no policy influence. It predicts local material consequences and keeps
latent outcome variation separate from ensemble disagreement; it is not the
project's composition/order model. See the
[technical record](docs/CONDITIONAL_PATCH_VAE_SHADOW_BASELINE_2026-08-11.md)
and [owner brief](docs/CONDITIONAL_PATCH_VAE_OWNER_BRIEF_2026-08-11.md).

Run the deterministic smoke suite:

```bash
python -m pytest -q \
  tests/test_action_encoding.py \
  tests/test_canvas.py \
  tests/test_motion_manifold.py \
  tests/test_motor_reliability.py \
  tests/test_modality_units.py \
  tests/test_native_contract.py \
  tests/test_observation_factorization.py \
  tests/test_plant_interface.py \
  tests/test_precision_beliefs.py \
  tests/test_preferences.py \
  tests/test_proposal.py \
  tests/test_proposal_convergence.py \
  tests/test_reference_oracle.py \
  tests/test_sensor_access_contract.py \
  tests/test_spatial_state.py \
  tests/test_stop_policy.py \
  tests/test_stop_prior.py
```

PowerShell accepts the same file list on one line. The complete suite is
`python -m pytest -q`; it currently includes expensive and timing-sensitive
integration tests. The latest observed results and environmental caveats are
recorded in [`docs/PROGRESS.md`](docs/PROGRESS.md).

## System Overview

```mermaid
flowchart LR
    GP["Generative process<br/>arm, contact, wet paint"] --> O["Agent-accessible<br/>observations"]
    O --> Q["State inference<br/>minimize VFE"]
    Q --> PI["Policy inference<br/>expected free energy"]
    PI --> A["Selected painting policy<br/>mark, passage, or stop"]
    A --> C["IK, trajectory, motor control<br/>and external safety"]
    C --> GP
    GP --> T["Observed transitions"]
    T --> L["Online transition<br/>model learning"]
    L --> Q
    L --> PI
```

Painting decisions are intended to remain inside an explicit probabilistic
model. Conventional inverse kinematics, trajectory generation, motor control,
collision checks, and hard safety constraints realize or veto a selected
policy below that boundary.

## Plant And Motor Terminology

Four related terms refer to different decisions and models:

| Term | Current meaning |
| --- | --- |
| Hardware actuator assignment | Fixed in the current hardware-oriented draft: RobStride 03 for `yaw`/`pitch`, RobStride 02 for `roll`/`elbow`. It is not selected dynamically. |
| Native plant | `native-abstract-v0` / `JointPlant`, a representative model not identified from those motors. |
| MuJoCo plant | `mujoco-robstride-electromechanical-v4`, a selectable vendor-grounded realized-execution backend, not yet a calibrated hardware twin. |
| Motor realization | An inferred controller/trajectory latent; it chooses how to realize a painting policy, not which motor product is installed. |

Counterfactual motor forecasts preserve the selected plant: native execution
uses native forecasts, while MuJoCo execution uses independent MuJoCo rollout
data under the same immutable model. In the MuJoCo runtime, encoder/contact
samples feed a versioned body estimator and forecast particles initialize
joint position/velocity from that posterior with independent future-noise
seeds. Its likelihood precision is provisional simulation-only, contact belief
is not yet mapped into brush compliance, and substrate grain, brush
history, and model parameters remain unresolved. Material fields are
posterior-conditioned when a `SpatialCanvasState` is supplied. Brush load and
pigment are sampled from the frozen `BrushLoadBelief`, while bristle-scale mark
variation uses an independent versioned prior instead of the live brush RNG.
The whole forecast remains oracle-conditioned. Forecast-driven policy inference
therefore remains confined to the explicit `baseline-oracle-v0` diagnostic;
MuJoCo parameter uncertainty is not yet sampled. See the canonical
[model record](models/README.md), the [native reference](docs/NATIVE_PLANT_REFERENCE.md),
and the [control/plant/policy boundary](docs/CONTROL_PLANT_POLICY_BOUNDARY.md).

## Implemented Prototype

- Stochastic four-degree-of-freedom arm, encoder, actuator, compliance, and
  contact simulation.
- Persistent wet oil-paint material state with thickness, pigment, surface
  tone, and material coverage.
- Sparse pixel-local learned transition ensembles with aleatoric and ensemble
  uncertainty.
- Variational state inference and separately logged VFE diagnostics.
- EFE-based policy inference with an immediate-stop policy and terminal
  material-coverage preference.
- Hierarchical mark, polyline, passage, and passage-plan proposals.
- An in-progress amortized mark/passage proposal network conditioned on canvas
  and relational beliefs. Its emission mixture defaults to zero; a dedicated
  unit suite covers density normalization/support, sampler parity, training,
  checkpoint continuation, and separation from EFE. The proposal-budget audit
  found that posterior mass and deep-horizon winning geometry do not converge
  under the tested budgets, so the result is explicitly conditional on the
  sampled candidate set. It is not a policy prior or a correction for
  finite-candidate bias.
- Registered global/requested-foveal camera observations, a provisional camera
  likelihood, and isolated compact body/brush posterior components. These are
  not yet connected into a complete sensor-only painting loop.
- Motor-realization forecasts over Cartesian, joint-space, elbow-pivot, and
  upper-arm-roll alternatives.
- Checkpointing, replay, telemetry export, and a Python-backed Three.js viewer
  that builds its robot hierarchy from the MuJoCo XML while retaining the
  Python material-canvas display.
- Selectable native and MuJoCo realized-execution backends behind the SI-unit
  plant contract. The MuJoCo backend uses output-equivalent RobStride dcmotors
  with voltage, current-lag, back-EMF, and peak-torque saturation. MuJoCo motor
  forecasts now use independent instances of that same plant, with explicit
  exact-state-oracle initialization and approximation provenance.
- Broad deterministic and integration test coverage across material,
  inference, hierarchy, arm, execution, and runtime behavior.

Several higher-level elements remain provisional. The sensor-equivalent path
fails closed before painting policy inference; the explicit oracle comparator
uses simulator information unavailable to a physical robot. Global and
relational latents are not yet validated as predictively necessary, policy
inference is proposal-limited, and the custom mechanical model has not been
calibrated against hardware.

## Research Discipline

The governing constraint is that painting-level decision terms must be
identifiable as likelihoods, transition priors, prior preferences, precision
beliefs, VFE/EFE terms, or policy priors/posteriors. Approximations are named as
approximations. Ordinary rewards are not relabeled as active inference.

The project deliberately separates:

- hidden generative-process truth;
- observations physically available to the agent;
- prior and posterior beliefs;
- predicted policy consequences;
- controller targets and realized motion;
- evaluation-only measurements.

That separation is incomplete in the current prototype and is a central target
of M1-M3.

## Roadmap

The work is organized as a research spine and a supporting embodiment spine:

| Sequence | Purpose |
| --- | --- |
| M0 | Project operations, manifests, versions, failures, and validation gates |
| M1 | Formal baseline, sensor-access audit, VFE/EFE verification, calibration |
| M2 | Sensor-equivalent multiscale perception and uncertain latent state |
| M3 | Foveated hierarchical policy inference and mechanism ablations |
| S0-S2 | Native reference contract, MuJoCo clone, and backend adapter |
| M4-M8 | Observatory, geometry, CAD, hardware bring-up, and experiments |

See [`planning/MILESTONE_INDEX.md`](planning/MILESTONE_INDEX.md) for status and
dependencies. Milestones are capability-gated rather than treated as a promise
that every planned feature will be built.

## Documentation

- [`docs/PROGRESS.md`](docs/PROGRESS.md): current evidence, known failures, and public progress log.
- [`docs/BASELINE_TEST_RESULT_2026-07-24.md`](docs/BASELINE_TEST_RESULT_2026-07-24.md): latest complete-suite environment and result.
- [`docs/RESEARCH_CHARTER.md`](docs/RESEARCH_CHARTER.md): scientific intent and scope.
- [`docs/OWNER_CONTRIBUTION_BRIEF_2026-08-11.md`](docs/OWNER_CONTRIBUTION_BRIEF_2026-08-11.md): owner-facing account of the project's conceptual, architectural, and embodiment contributions.
- [`docs/OWNER_STEERING_CATALOG_2026-08-11.md`](docs/OWNER_STEERING_CATALOG_2026-08-11.md): chronological catalog of 145 recoverable owner decisions, corrections, selections, deferrals, and meaningful steering actions since June 29.
- [`docs/PROJECT_DECISION_PROVENANCE_2026-08-11.md`](docs/PROJECT_DECISION_PROVENANCE_2026-08-11.md): evidence-qualified owner/agent decision ledger for future project instances.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): long-form architecture brief.
- [`docs/GENERATIVE_MODEL_SPEC.md`](docs/GENERATIVE_MODEL_SPEC.md): implemented probabilistic factorization, VFE/EFE map, and approximation register.
- [`docs/PROPOSAL_CONVERGENCE_RESULT_2026-08-04.md`](docs/PROPOSAL_CONVERGENCE_RESULT_2026-08-04.md): AI-111 candidate-budget, horizon, seed, and mixture result.
- [`docs/OBSERVATION_FACTOR_AUDIT.md`](docs/OBSERVATION_FACTOR_AUDIT.md): independent material factors, units, normalization, and provisional likelihoods.
- [`docs/NATIVE_PLANT_REFERENCE.md`](docs/NATIVE_PLANT_REFERENCE.md): versioned native arm, canvas, contact, and material contract.
- [`docs/CONTROL_PLANT_POLICY_BOUNDARY.md`](docs/CONTROL_PLANT_POLICY_BOUNDARY.md): backend-neutral command, sensor, belief, forecast, and evaluation interfaces.
- [`docs/VARIABLE_SENSOR_ACCESS_LEDGER.md`](docs/VARIABLE_SENSOR_ACCESS_LEDGER.md): simulator-to-agent values, physical accessibility, and research blockers.
- [`docs/CURRENT_IMPLEMENTATION.md`](docs/CURRENT_IMPLEMENTATION.md): detailed prototype implementation notes.
- [`docs/DEVELOPMENT_AUDIT.md`](docs/DEVELOPMENT_AUDIT.md): historical engineering audit.
- [`planning/README.md`](planning/README.md): milestone planning system.

## Reproducibility And Safety

The project is developing version and experiment manifests for code,
configuration, learned state, simulator, calibration, hardware, random seeds,
and output artifacts. Failed and interrupted runs are retained as evidence.

Hard joint, current, force, workspace, watchdog, and non-finite-state limits
remain external to painting-policy inference. The present simulator is not a
safety case for a physical robot.
