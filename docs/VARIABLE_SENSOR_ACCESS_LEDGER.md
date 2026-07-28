# Variable And Sensor-Access Ledger

Ledger version: `sensor-access-ledger-v0`

Baseline: `baseline-oracle-v0`

Status: accepted implementation audit for `AI-102` on 2026-07-24. The
machine-readable inventory is
[`planning/variable-sensor-access-ledger.json`](../planning/variable-sensor-access-ledger.json).

## 1. Purpose

This ledger answers a deliberately strict question:

> What information crosses from the generative process into inference,
> planning, learning, control, and diagnostics, and could a physical robot
> actually obtain it?

The answer matters because adding uncertainty after reading exact simulator
state does not create a physical observation model. A Gaussian centered on the
true paint-thickness array is still an oracle-state likelihood. Likewise, a
counterfactual rollout initialized by deep-copying the complete simulator is
not equivalent to prediction from an inferred bodily state.

This document classifies the current implementation. It does not yet replace
the privileged paths. That replacement is the core of M2.

## 2. Classification Vocabulary

Every inventoried value has one primary classification:

| Class | Meaning |
| --- | --- |
| Physical sensor observation | A simulated value with a direct planned hardware sensor analogue. It may still be unused by the current agent. |
| Derived observation | A deterministic or statistical transform. It is sensor-equivalent only when all of its inputs are permitted observations or posterior beliefs. |
| Hidden state | Process state or a process parameter that a physical agent cannot read directly. |
| Simulator-only evaluation label | Truth retained for rendering, debugging, calibration assessment, or failure analysis, not agent inference. |
| Execution-only state | Commands, calibrated geometry, low-level controller state, or hard-safety state below painting-policy selection. |

Classification describes what a value *is*. Access status describes how it is
currently used. For example, Cartesian tip position is a derived quantity, but
the current controller derives it from true process pose, making that access
privileged.

## 3. Current Boundary

The intended physical boundary is:

```text
hidden world and body state
        |
        v
sensor generative process
        |
        v
camera + encoder + current + contact observations
        |
        v
posterior beliefs -> EFE -> painting policy
        |
        v
commands -> conventional controller -> hard safety -> plant
```

The current baseline is instead:

```text
VerticalCanvas material arrays --------------------+
                                                     |
true ArmPainterSim pose/contact ---------------------+--> agent and controller
                                                     |
deep copy of full simulator, parameters, and RNG ----+--> motor-policy rollout

simulated encoder/current packet ------------------------> diagnostics only
perfect canvas render -----------------------------------> browser only
```

The second diagram is useful for development, but it is not a sensor-equivalent
active-inference agent.

## 4. Temporary Oracle Baseline

`baseline-oracle-v0` is the explicit name for the current observation
condition.

Its runtime observation mode is `oracle_material_state`. It may be used for:

- native process and controller development;
- transition-learning smoke tests;
- VFE/EFE equation tests under a declared oracle likelihood;
- a matched upper-bound comparator for later sensor-equivalent experiments;
- visualization and simulator-only evaluation.

It may not support claims of:

- sensor-only state inference;
- physical deployability;
- calibrated camera, proprioceptive, or contact perception;
- learning exclusively from physically available sensory history;
- embodied prediction without simulator-state leakage.

Every experiment manifest using the current runtime must record:

```text
observation_baseline = baseline-oracle-v0
observation_mode = oracle_material_state
sensor_equivalent = false
```

An interesting painting does not relax this declaration.

## 5. Canvas And Material Variables

### 5.1 Hidden process fields

`VerticalCanvas` stores:

- `thickness[N,N]`;
- `wetness[N,N]`;
- `black_mass[N,N]`;
- `surface_tone[N,N]`;
- fixed substrate `grain[N,N]`.

These are hidden material states. A physical camera can observe reflected
light from the surface, not bulk pigment mass, true thickness, wetness, or the
simulator's internal surface-tone variable.

The current spatial path reads these arrays in
`spatial_state.material_grid_from_canvas()`. The summary path reads them in
`arm_agent_driver.canvas_summary_state()`. They then become:

- posterior means;
- replay-buffer inputs and targets;
- pixel and coarse hierarchy inputs;
- EFE rollout initial states;
- passage-observation evidence.

This is the largest direct oracle path.

### 5.2 Deterministic derived fields

The process derives:

- material occupancy from thresholded thickness;
- opacity from thickness;
- visible tone from surface tone;
- observed tone from ground tone, opacity, and surface tone;
- ground contrast from observed tone;
- scalar material coverage from occupancy.

These transforms are useful definitions, but most are not independent sensor
channels. Coverage and contrast inherit the same hidden material information
as their parents. Treating the parent fields and their deterministic
transforms as independent Gaussian evidence can double-count information.
`AI-103` owns that decision.

### 5.3 Summary observation

Summary mode receives six exact aggregates:

1. material coverage mean;
2. mean thickness;
3. maximum thickness;
4. mean wetness;
5. overlap fraction;
6. painted-ground contrast.

All six are derived from hidden process arrays. None is currently inferred
from an image.

### 5.4 Spatial observation

Spatial mode receives native-pixel material means, configured log variances,
and deterministic coarse-grained copies. Its six channels are thickness,
wetness, black mass, surface tone, ground contrast, and material coverage.

The word "observation" here means an observation of simulator state, not a
camera observation. The configured variance changes posterior precision but
does not make the mean sensor-equivalent.

### 5.5 Browser canvas

The browser PNG is produced from perfect orthographic `observed_tone`. It is
the closest current variable to a camera observation, but it has no camera
pose, optics, occlusion, glare, latency, quantization, or sensor noise, and it
is not fed into inference.

## 6. Bodily Variables

### 6.1 True pose and Cartesian geometry

`ArmPainterSim.actual_pose` is the true link-side joint pose. The driver and
stroke controllers use it directly for:

- forward kinematics and tip position;
- approach timing;
- contact release and retraction;
- stroke tracking error;
- motor rollout initialization;
- visualization and telemetry.

The simulated encoder packet is not used for these operations. On hardware,
the controller should consume encoder samples or an estimator posterior, and
painting inference should receive a declared proprioceptive likelihood.

Cartesian tip, joint points, and arm axes are derived values. They are
permitted below policy selection when computed from calibrated geometry and
sensor estimates. They are privileged in the present implementation because
they are computed from true pose.

### 6.2 Plant state and parameters

The native `JointPlant` contains exact:

- link and motor velocities;
- motor-side angle;
- thermal state;
- inertia, friction, stiffness, backlash, coupling, and gravity parameters;
- process-noise parameters and RNG state.

The motor planner receives these through a deep copy of `ArmPainterSim`.
Nominal plant parameters may legitimately become part of the agent's
generative model, but reading the process instance's exact parameter and
dynamic state is not parameter inference. Future work must distinguish:

- versioned nominal parameters;
- calibrated parameter uncertainty;
- online parameter beliefs;
- hidden true process values retained only for evaluation.

### 6.3 Simulated physical sensors

The process already emits plausible sensor analogues for:

- joint encoder position;
- joint encoder velocity;
- motor current;
- applied or reported voltage.

These are currently exported to the web UI and telemetry CSV but are not
assimilated by the painting posterior or used by the controller. In other
words, the most physically legitimate bodily observations exist, while the
live code mostly uses the hidden states they were meant to obscure.

Computed torque and position error are derived signals. Exact friction,
backlash, elastic deflection, load decomposition, process torque, and encoder
noise standard deviation are simulator-only labels unless a future estimator
produces beliefs over them.

### 6.4 Contact

`ContactState` contains exact on-canvas status, bushing deflection, force,
pressure, brush width, and contact point. It is computed from geometry and the
intended pressure command. There is no contact sensor likelihood.

The exact contact state currently affects:

- contact-aware control;
- retraction and contact release;
- forecast feasibility and motor EFE;
- realized path and pressure residuals;
- motor-reliability learning.

This means the reliability learner is also oracle-conditioned. A future
physical path must derive its residuals from camera/encoder estimates and
force, deflection, current, or contact-switch observations.

## 7. Brush State

The brush process holds load, pigment contamination, carried tone, path
distance, bristle gains, streak phases, edge-wobble phases, and RNG state.
These are hidden causes of visible marks.

The brush is intentionally loaded before each stroke action, which removes
cross-stroke fresh-load memory from the current transition model. It then
remains loaded continuously, and contact/pressure alone determine deposition;
tracking error cannot toggle paint. Even so, the motor planner deep-copies the
exact brush and its RNG. The rollout therefore knows a random bristle
realization that a physical agent could not know before contact.

Longer term, commanded paint load and nominal brush identity may be known.
Actual loading, contamination, bristle state, and tip lag should remain latent
and be inferred from sensory consequences.

## 8. Counterfactual Forecast Leakage

Global and local planning currently call `copy.deepcopy(sim)`. The resulting
snapshot includes:

- exact material fields;
- true pose and plant dynamic state;
- exact contact state;
- exact brush state;
- exact process parameters;
- plant RNG state;
- brush RNG state.

Each motor alternative then deep-copies that snapshot again. Plant parameter
jitter adds uncertainty, but it does not remove the privileged initial state.
Rollout particle zero retains the copied plant RNG continuation, and the brush
RNG is copied without an independent-noise boundary. Depending on intervening
live steps, this can correlate counterfactual and process noise in a way no
physical agent could exploit.

This does **not** mean simulation-based prediction is illegitimate. The
principled replacement is:

1. infer a posterior over bodily, brush, contact, and material state from
   permitted observations;
2. initialize forecast particles from that posterior;
3. sample independent future process noise;
4. propagate parameter uncertainty separately;
5. compare predicted sensory outcomes with later sensor observations.

Until that path exists, embodied EFE results must be labeled
oracle-conditioned.

## 9. Passage And Hierarchy Leakage

The passage posterior combines executed action geometry with an exact
before/after thickness-delta centroid and post-stroke surface tone. This is a
mixed action/outcome pseudo-likelihood. Its geometry is partly commanded and
its visual evidence is materially privileged.

The canvas and relational hierarchies inherit exact spatial material fields.
Therefore, evidence that those latents retain or organize structure is valid
only under the oracle-observation baseline. It is not yet evidence that a
sensor-driven agent can infer the same structure.

## 10. Diagnostics Boundary

The browser and telemetry log intentionally expose process truth:

- true joint pose, points, and tip;
- exact contact state;
- exact coverage;
- true link velocity;
- transmission and load decomposition;
- controller phase and target state.

These are useful evaluation labels. They are acceptable so long as:

- the diagnostics path never feeds painting inference;
- exported columns retain truth-versus-sensor labels;
- experiment analysis does not call process truth an agent observation;
- hardware comparisons use matched sensor estimates where required.

Commands, target pose, controller phase, and hard safety checks are
execution-only. Hard joint, force, current, workspace, and watchdog limits
remain external to active-inference preferences.

## 11. Blocking Matrix

| Privileged path | Claims blocked | Required tasks |
| --- | --- | --- |
| Exact material arrays and aggregates | Sensor-only perception, physical deployment, sensor-driven hierarchy | `AI-103`, `AI-201` through `AI-204`, `AI-216` |
| True pose and Cartesian tip | Sensor-driven embodiment and controller portability | `T-109`, `AI-201`, `AI-203`, `AI-204`, `AI-216` |
| Exact contact state in control and learning | Calibrated contact inference and physical reliability learning | `T-109`, `AI-201`, `AI-203`, `AI-204`, `AI-216` |
| Full simulator snapshot and copied RNG | Non-oracle embodied prediction | `T-109`, `AI-202`, `AI-203`, `AI-204`, `AI-216` |
| Exact plant parameters | Identified dynamics or calibrated energetic prediction | `AI-107`, `T-109`, `T-106`, later calibration work |
| Exact material delta in passage update | Sensor-driven passage belief | `AI-103`, `AI-203`, `AI-204`, `AI-207`, `AI-210`, `AI-216` |
| Mixed truth and sensor telemetry | Reproducible hardware-comparable diagnostics | `T-105`, `T-106` |

`AI-216` must not accept M2 while any research-critical live path depends on
the oracle material state, true pose/contact, or copied process RNG.

## 12. Immediate Decisions Handed To AI-103

This ledger deliberately does not decide the observation factorization. It
makes the next questions concrete:

1. Which visual quantity is actually emitted by the fixed-camera process?
2. Which material variables remain hidden causes?
3. Are coverage and contrast posterior summaries or likelihood channels?
4. Which deterministic transforms must be excluded from independent evidence?
5. What units and normalizations define each retained likelihood?
6. Which proprioceptive channels are independently observed versus derived?
7. How are contact and force represented when no direct force sensor exists?

## 13. Maintenance Contract

The JSON ledger is the machine-readable source for field coverage. Tests
require every dataclass field in `ArmPainterSim`, `VerticalCanvas`,
`JointPlant`, `MotorTelemetry`, `ContactState`, `Brush`, `ArmKinematics`, and
`ExecutionForecast` to appear in a classified entry.

When a field or access path is added:

1. classify it before using it in inference, planning, or learning;
2. state whether a physical platform can obtain it;
3. name its likelihood or explicitly mark that none exists;
4. assign blockers for privileged research-critical access;
5. update the runtime observation-boundary diagnostic when the baseline mode
   changes.

Static field coverage cannot detect every indirect leak. Gate reviews must
still trace copied objects, closures, callbacks, replay targets, and
diagnostic feedback paths.
