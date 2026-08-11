# Simulator Shortcut And Validation Classification

Date: 2026-08-11
Closes: T-106
Reference scope: `native-abstract-v0`, with MuJoCo and hardware transfer seams

## Reading this table

“Acceptable baseline” means acceptable for reproducing and comparing the S0
abstract simulator. It does **not** mean physically accurate. “MuJoCo
calibration” means the mechanism belongs in a hardware-oriented realized plant
and needs parameter identification or richer simulation evidence. “Hardware
validation” means simulation cannot settle the claim: physical sensors,
fixtures, paint, brush, cameras, timing, or assembled mechanics must be
measured.

The selected hardware draft is RobStride 03 at yaw/pitch and RobStride 02 at
upper-arm roll/elbow. The native plant is not identified from those motors.
The selected camera draft is exactly two not-yet-purchased Raspberry Pi Global
Shutter Cameras (Sony IMX296). Neither selection changes the classification
below into calibration evidence.

## Consolidated acceptance table

| Shortcut or unavailable real-world quantity | Current treatment | S0 baseline | MuJoCo calibration need | Hardware validation need |
| --- | --- | --- | --- | --- |
| Native link geometry and co-located shoulder frame | Two 13 in abstract links, native canvas at `y=17`, logical retarget to the separated MuJoCo body | Accepted as `native-abstract-v0` comparison coordinates | Identify offsets, inertias, joint-axis errors, compliance, collision geometry | Measure assembled frames, link datums, backlash, repeatability, and canvas registration |
| Native joint actuator dynamics | Representative `JointPlant` electrical, elastic, friction, backlash, thermal, coupling, and noise parameters | Accepted only as a reproducible abstract actuator hypothesis | Fit the selected-plant dynamics; current `mujoco-robstride-electromechanical-v4` values include vendor-derived and approximate fields | Joint-rig current/torque/speed/thermal/system-identification data |
| Exact simulated joint/link state | Native process directly owns motor/link state; oracle diagnostic can expose it | Allowed for process evaluation and `baseline-oracle-v0` only | MuJoCo runtime must expose sensor packets and infer body state, which the provisional path now does | Encoder/current/temperature/contact channels need measured noise, delay, bias, dropout, and calibration |
| Copied process/RNG state in legacy forecasts | Default/oracle forecast container can copy substrate grain/model context and native process state | Kept only as named nonconforming oracle reference | Independent MuJoCo counterfactuals now initialize q/qvel, material, and compact brush slices from frozen beliefs; contact/model-parameter uncertainty remains | Prove the hardware estimator/forecast boundary cannot access process truth |
| Canvas material arrays | `VerticalCanvas` owns exact thickness, wetness, black mass, and surface tone; coverage/contrast are derived | Material semantics and white-on-white coverage are accepted process truth | Python paint remains the declared MuJoCo material-process boundary; counterfactual initialization is posterior-conditioned but approximate | Camera likelihood must infer material consequences; physical paint thickness, wetness, opacity, mixing, drying, and substrate tooth need measurement |
| Visible tone as a material proxy | Surface rendering derives tone from ground, opacity, and surface pigment | Accepted only as a superficial rendering/material fixture; never a substitute for coverage | Calibrate simulated photometry only for a declared camera/material process | Measure illumination, exposure, spectral response, glare, occlusion, paint opacity, and color response |
| Native contact and intended pressure | Canvas penetration plus representative spring; force is max of geometric and intended-pressure paths | Accepted as an execution/contact fixture, not a pressure likelihood | Identify contact stiffness/damping, compliance state, friction, and map inferred contact into MuJoCo brush state | Force/contact sensing, brush deflection, ferrule collision, chatter/stick-slip, and safe force envelope require physical fixtures |
| Round-brush footprint and anisotropy | Angle-dependent ellipse, pressure splay, taper; handle-leading friction advantage and ferrule-leading stick/release | Accepted only as the named Baxter-inspired provisional process consequence | Calibrate geometry/friction/compliance and learn an uncertain brush transition model instead of reusing process equations | High-speed/force/image brush tests across angle, load, direction, paint load, canvas tooth, and wear |
| Brush load and pigment microstructure | Compact load/average-pigment belief; forecast particles use a fixed representative or independently sampled provisional microstructure | Acceptable as a compact integration approximation, with collapsed history declared | Infer/load a stochastic compliance and deposition state; validate held-paint/bristle-history approximation | Measure reload, depletion, retention, mixing, bristle history, cleaning, and brush-to-brush variation |
| Camera geometry and image formation | Two provisional IMX296 oblique views in MJCF, analytic pinhole/noise/latency assumptions, identity appearance in the smoke profile | Not part of native S0 authority; acceptable only for simulation integration | Preserve geometry/occlusion and independent camera noise without leaking segmentation/material truth | Purchase, calibrate, synchronize, mount, and measure the two cameras/lenses; validate exposure, blur, rolling/global timing, latency, noise, dropout, distortion, and occlusion |
| Exact segmentation, visibility, pose, and contact | Available inside the generative process for rendering/evaluation | Prohibited from sensor-equivalent policy inference; allowed in oracle/evaluation ledgers | Keep internal truth off agent-facing records; use registered camera/body likelihoods | Audit real runtime data paths and estimator performance against calibration fixtures |
| Summary six-state planner | Historical coverage/wetness/thickness/contrast/edge/activity aggregate | Retained only as `obsolete_compatibility_fixture` | None; do not tune it into an embodiment model | None; replace with sensor-conditioned multiscale inference rather than validate the shortcut |
| Terminal coverage forecast | Single moment-matched Beta with boundary/concentration restrictions | Provisional integration only; AI-106 verifies algebra but rejects M2 approval | Replace with converged particle/sample risk or calibrated richer bounded family | Validate terminal material posterior and uncertainty from physical observations |
| Local material transition likelihood | CNN baseline plus shadow conditional patch cVAE trained from camera-derived trajectory patches | Tooling only; no live-scale evidence yet | Live and corpus support/pixel scale must match; held-out calibration and multi-step uncertainty required | Transfer/calibrate on physical camera and paint data without exact-state labels in live inference |
| Counterfactual brush/material physics | Independent forecasts currently reuse the programmed process equations under independent state | Acceptable only as simulation-only integration baseline | Replace with learned uncertain generative approximations once corpus/calibration support exists | Test predictive coverage on real strokes, OOD loads, and changing materials |
| Finite policy candidate set | Hand-written and optionally learned computational proposals; posterior is `Q(pi | sampled set S)` | Accepted only with AI-111’s negative convergence limitation | AI-306 must distinguish proposal density, base measure, and policy prior and quantify finite-budget bias | Hardware does not resolve the inference approximation; only latency feasibility transfers |
| Painting-policy horizon and one-particle smoke profile | Bounded simulation-throughput profile: 8 candidates, depth 1, one particle | Acceptable only as a smoke/integration profile | Benchmark deeper/more-particle inference separately from physics and rendering | Validate compute/latency/watchdog feasibility on the deployed controller |
| IK, joint splines, pivots, and roll realizations | Conditional low-level realizations beneath selected Cartesian painting policy | Conventional engineering boundary accepted | Compare predicted physical consequences under the selected plant; do not let IK choose painting intent | Validate tracking, singularity margin, current, force, and safety on hardware |
| Hard limits and watchdogs | External safety constraints rather than rewards/preferences | Correct architectural placement | Exercise in backend fault/safety tests | Independent safety architecture, limits, E-stop, containment, supervision, and hazard validation |
| Random substrate grain and process seeds | Programmed process variation; independent forecast seeds in the provisional sensor smoke path | Acceptable for reproducibility when recorded | Learn or calibrate distributions; sample body/material parameter uncertainty | Measure actual canvas/paint variation and test seed-distribution adequacy |

## Stable S0 contracts

The following are stable reference interfaces, not claims of physical fidelity:

- native joint order `yaw, pitch, roll, elbow` and the `native-abstract-v0`
  coordinate/material contract;
- `StrokeAction` as Cartesian/contact painting intent;
- the separation of painting policy from IK, trajectories, motor control, and
  hard safety;
- material coverage derived from thickness rather than visible tone;
- distinct command, physical-sensor, posterior-belief, counterfactual, and
  simulator-evaluation records in `plant-interface-v1`; and
- explicit backend/model/likelihood/approximation provenance.

Changing a plant parameter does not violate S0 if a new version is recorded and
the stable interfaces remain explicit. Treating a representative parameter as
measured hardware truth does violate the contract.

## Evidence map

- Native constants and material semantics: `docs/NATIVE_PLANT_REFERENCE.md`
- Policy/control/plant boundary: `docs/CONTROL_PLANT_POLICY_BOUNDARY.md`
- Sensor-access permissions and remaining leaks:
  `docs/VARIABLE_SENSOR_ACCESS_LEDGER.md`
- Current generative model and approximations: `docs/GENERATIVE_MODEL_SPEC.md`
- MuJoCo and selected hardware draft: `models/README.md`
- Camera geometry evidence: `docs/CAMERA_OBSERVABILITY_BRIEF.md`
- Brush process research: `docs/BRUSH_ANISOTROPY_RESEARCH_2026-08-10.md`
- Terminal forecast decision:
  `docs/TERMINAL_COVERAGE_STOPPING_ACCEPTANCE_2026-08-11.md`

## T-106 conclusion

The shortcut inventory is now consolidated and each item is classified. S0 may
use `native-abstract-v0` as a reproducible comparison reference, but no row in
the baseline column authorizes sensor-equivalent, calibrated-digital-twin, or
hardware-performance claims.
