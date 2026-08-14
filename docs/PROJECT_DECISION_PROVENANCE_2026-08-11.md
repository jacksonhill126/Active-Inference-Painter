# Project Decision Provenance Record

Date: 2026-08-11
Purpose: durable attribution record for future project agents

The explanatory companion for the project owner is
[What You Have Contributed To Active-Inference Painter](OWNER_CONTRIBUTION_BRIEF_2026-08-11.md).

## Scope and method

This record identifies consequential architectural, conceptual, embodiment,
and development decisions for which the project owner's contribution is
visible in accessible evidence. It is not a claim about copyright, commit
authorship, or a percentage of total effort.

Sources reviewed:

- locally stored Codex project sessions beginning 2026-06-29, including the
  long sessions `019f18a7-823b-7eb0-8219-8622c84d9363` and
  `019f3cbd-bd7f-7f53-a52a-a1c7334eba90`;
- locally stored Claude project sessions
  `8ad2a6a6-a7c4-4562-a682-0533f05c67b0`,
  `e655b4e4-3c9d-42df-ba8b-c994be9d2285`, and
  `7ccc29fa-acb4-47b6-b8ff-3a5910bea9b7`;
- current Codex project task `019fcd79-6504-7960-b99e-11677a30bd72`;
- earlier Codex project task `019fa461-8f55-7f93-99ea-ddbb0064b324`;
- textbook-study task `019fd2d2-b2ad-7fd3-8780-9fa81b49ae61` where relevant to
  the owner's requested mathematical understanding;
- current repository documentation, source layout, and Git history;
- the standing owner-supplied `AGENTS.md` instructions.

Two visible ChatGPT conversations titled `Robot painter feasibility` and
`Isaac Sim Paint Simulation` contained relevant feasibility questions but no
additional recoverable architectural decisions. A ChatGPT project named `Arm`
is visible but its earlier constituent chats were not enumerated, and the
initial repository predates the first local Codex log. The detailed recovered
history is compiled in
`docs/OWNER_STEERING_CATALOG_2026-08-11.md`.

Attribution labels used below:

- **Owner-originated:** the owner stated the mechanism, constraint, or
  rationale directly.
- **Owner-corrected:** an implementation or project description changed after
  a specific owner diagnosis.
- **Owner-selected:** the owner selected among proposed alternatives or set a
  priority.
- **Collaborative:** owner framing and agent formalization are both material.
- **Agent-formalized:** the owner supplied intent or physical insight; the
  equations, numerical values, or software design were primarily produced by
  an agent.
- **Candidate:** a stated direction that is not yet a canonical, validated
  design.

## Decision ledger

### P-001 — Active inference is the complete painting-level cognitive boundary

- **Attribution:** owner-set governing constraint; agent-formalized boundary.
- **Decision:** no aesthetic reward, vague score mixture, or conventional
  planner may select a painting policy. Painting-level quantities must be
  explicit generative-model, preference, precision, VFE, EFE, or policy
  factors. Conventional IK, dynamics, interpolation, motor control, collision
  checking, and hard safety may realize a selected policy below that boundary.
- **Evidence:** owner-supplied standing instructions in `AGENTS.md` and the
  historical continuation brief.
- **Technical consequence:** separate painting policy, motor realization,
  plant, inference, evaluation, and safety layers; explicit VFE/EFE logging.
- **Durable artifacts:** `AGENTS.md`, `docs/RESEARCH_CHARTER.md`,
  `docs/CONTROL_PLANT_POLICY_BOUNDARY.md`, `docs/GENERATIVE_MODEL_SPEC.md`.
- **Status:** canonical.

### P-002 — Online inference may not consume hidden simulator state

- **Attribution:** owner-originated constraint; agent-formalized sensor path.
- **Decision:** exact pose, contact, material state, visibility, segmentation,
  and similar process truth are forbidden as online observations unless a
  declared physical sensor supplies them. Hidden truth remains available for
  evaluation and explicitly named oracle modes only.
- **Direct evidence:** earlier task turn
  `019faa56-bdfb-7483-9ce5-6dc3d36738df` ("make sure the model doesnt get
  access to any hidden environment states") and current-task turn
  `019fd776-b036-72b1-abd1-257d34b8df9e` (runnable model not using exact sim
  states).
- **Technical consequence:** sensor-access ledger, camera likelihood,
  action-conditioned capture clock, fail-closed runtime, causal rejection of
  stale frames, and separation of process labels from observations.
- **Durable artifacts:** `docs/VARIABLE_SENSOR_ACCESS_LEDGER.md`,
  `docs/ACTION_CONDITIONED_CAMERA_LOOP_2026-08-06.md`,
  `docs/PROVISIONAL_SENSOR_SIMULATION_2026-08-06.md`,
  `planning/variable-sensor-access-ledger.json`.
- **Status:** canonical; physical calibration and several likelihoods remain
  open.

### P-003 — Obsolete summary state is not the intended painting hierarchy

- **Attribution:** owner-originated rejection and replacement criterion;
  agent-formalized learned hierarchy.
- **Decision:** six hand-selected global variables are not acceptable as the
  highest-level painting representation. Higher layers should learn flexible
  abstract latent causes because they improve multiscale prediction. The
  summary implementation remains only as a compatibility/test fixture.
- **Direct evidence:** earlier task turns
  `019faf8f-29fa-7da2-9047-f0aedf04e7e6` and
  `019faf95-925d-7360-9af0-615c61d79078`.
- **Technical consequence:** explicit deprecation of summary mode and work on
  canvas, relational, passage, and plan latents with slower dynamics.
- **Durable artifacts:** `docs/ARCHITECTURE.md` section 4,
  `docs/GENERATIVE_MODEL_SPEC.md` sections 3.3 and 6.1,
  `src/active_painter/canvas_hierarchy.py`,
  `src/active_painter/spatial_hierarchy.py`,
  `src/active_painter/passage_inference.py`.
- **Status:** canonical direction; current hierarchy is still only partly
  learned and not complete bidirectional message passing.

### P-004 — Material ontology includes pigment, brush loading, pickup, and local wet behavior

- **Attribution:** owner-originated material requirements; agent-formalized
  process and belief updates.
- **Decision:** represent black and white material explicitly; treat brush load
  and average pigment mixture as persistent inferred state; let the brush pick
  up canvas pigment; permit an initially instantaneous selectable reload;
  defer cleaning; model wet mobility where brush-local/foveal evidence can
  support it rather than using coarse wetness/thickness as global compositional
  variables.
- **Direct evidence:** earlier task turns
  `019faf1c-f7a8-7240-8cfc-220b5cf6efe2`,
  `019faf23-7eba-7622-aca6-7971151d4233`,
  `019faf25-c302-7031-8282-798d2963c20c`, and
  `019faf3b-c93b-7810-8caa-33cff6987793`.
- **Technical consequence:** spatial thickness/wetness/black-mass/tone state;
  compact per-brush `q(load, black_fraction)`; depletion, pickup, and reload
  transitions; coverage separated from visible tone.
- **Durable artifacts:** `src/active_painter/brush_loading.py`,
  `src/active_painter/arm_sim.py`, `src/active_painter/spatial_state.py`,
  `docs/BRUSH_FORECAST_INITIALIZATION_2026-08-05.md`,
  `docs/ARCHITECTURE.md` sections 6.4 and 7.
- **Status:** partially implemented. The local camera-derived brush deposition
  statistic and calibrated wet-paint likelihood remain open.

### P-005 — Deposition follows physical contact, not an arbitrary paint gate

- **Attribution:** owner-corrected process semantics; agent implementation.
- **Decision:** painting should follow pressure/contact. Any positive-pressure
  canvas contact may deposit paint, including approach, press, lift, or
  unintended contact; a controller flag must not erase physical consequences.
- **Direct evidence:** earlier task turn
  `019fa9f3-9a2c-77e0-89fb-82135e0e8be4`.
- **Technical consequence:** contact-driven deposition and later correction of
  exit/lift unloading instead of paint gating.
- **Durable artifacts:** `src/active_painter/arm_sim.py`,
  `src/active_painter/stroke_execution.py`,
  `docs/CURRENT_IMPLEMENTATION.md` brush-contact description.
- **Status:** canonical process behavior; not hardware calibrated.

### P-006 — The camera system is active, foveated, sensor-only, and observable under contact

- **Attribution:** owner-originated operational criteria; agent-formalized
  geometry and acquisition interface.
- **Decision:** use high-resolution native images for electronically selected
  foveae rather than mechanically aiming a camera; model-facing input begins in
  grayscale; camera geometry must be tested for arm/end-effector visibility,
  especially during canvas contact; use pose sweeps and publishable evidence;
  perspective normalization and calibration are explicit problems.
- **Direct evidence:** earlier task turns
  `019fb86c-8231-79f1-baf2-df4bed91bf08`,
  `019fafc4-a994-7450-b270-44827a57d5d0`,
  `019faa90-8ee3-7be3-b2f4-7400665082c7`, and
  `019faa86-a9b0-7a30-99ce-25c1244424f1`.
- **Technical consequence:** native/global/foveal products, explicit fovea
  requests, camera timing/calibration metadata, observability sweeps, and the
  fading fovea trace in the viewer.
- **Durable artifacts:** `src/active_painter/camera_observation.py`,
  `src/active_painter/camera_pose_sweep.py`,
  `docs/CAMERA_OBSERVABILITY_BRIEF.md`,
  `docs/VARIABLE_SENSOR_ACCESS_LEDGER.md`.
- **Status:** simulation/design baseline; not sensor-equivalent hardware.

### P-007 — The current physical camera baseline is compact dual IMX296

- **Attribution:** owner-selected hardware configuration; agent-selected exact
  geometry and provisional optics.
- **Decision:** prefer a compact, self-contained rig and use exactly two
  Raspberry Pi Global Shutter Camera modules (Sony IMX296), left and right,
  without separate inspection, overhead, or profile cameras in the initial
  baseline.
- **Direct evidence:** current task owner requests to update the setup "with
  the 2 raspberry pi globbal shutter cams" after rejecting the spatial envelope
  of larger-bodied cameras.
- **Technical consequence:** four-view/full-size proposal replaced by matched
  two-view crossbar; canonical XML/docs/tests updated together.
- **Durable artifacts:** `docs/COMPACT_DUAL_IMX296_CAMERA_RIG_2026-08-07.md`,
  `docs/CAMERA_OBSERVABILITY_BRIEF.md`, `models/README.md`,
  `models/active_inference_painter.xml`.
- **Status:** selected but not purchased; all optics, timing, and geometry are
  provisional until physical calibration.

### P-008 — Direct-drive RobStride hardware must remain distinct from plant and motor-realization beliefs

- **Attribution:** owner-selected/reaffirmed hardware direction and
  owner-corrected documentation; exact axis assignment has incomplete early
  chat provenance.
- **Decision:** direct drive with RobStride motors and no belts; current
  hardware draft fixes RS03 at yaw/pitch and RS02 at roll/elbow. That hardware
  assignment is not the native abstract plant and is not a motor-realization
  latent. Native execution and MuJoCo forecasts must each preserve their own
  selected plant.
- **Direct evidence:** earlier task turn
  `019fa461-a3be-7531-a0fe-f03ad0b1f9d4`; current task turns
  `019fce84-43e5-7072-8402-a16c2b619798` and
  `019fce88-725a-7d90-8776-7f0e9c212ac2`.
- **Technical consequence:** plant vocabulary and documentation were corrected;
  MuJoCo motor forecasts were aligned with the MuJoCo backend rather than the
  native abstract plant.
- **Durable artifacts:** `AGENTS.md`, `models/README.md`,
  `models/active_inference_painter.xml`,
  `docs/CONTROL_PLANT_POLICY_BOUNDARY.md`.
- **Status:** canonical hardware-oriented draft; not calibrated against
  assembled hardware.

### P-009 — MJCF is the embodiment source of truth; MuJoCo is not the paint renderer

- **Attribution:** owner-originated integration direction; agent implementation.
- **Decision:** the MuJoCo XML should interface with the generative-model and
  control stack, and the web visualizer should derive geometry/variables from
  the same XML so geometry does not drift. MuJoCo supplies robot dynamics and
  contact, while the existing paint process/view remains responsible for paint
  material rendering.
- **Direct evidence:** earlier task turns
  `019fa559-2468-7002-9be0-44577580bac1` and
  `019fa461-a3be-7531-a0fe-f03ad0b1f9d4`.
- **Technical consequence:** XML-driven web robot model, MuJoCo backend adapter,
  and shared joint ordering/metadata.
- **Durable artifacts:** `src/active_painter/web_robot_model.py`,
  `src/active_painter/mujoco_backend.py`, `models/active_inference_painter.xml`.
- **Status:** implemented simulation baseline.

### P-010 — Roll must have a painting function, while curvature remains a separate policy variable

- **Attribution:** owner-originated functional requirement and diagnosis;
  agent-formalized semantic split.
- **Decision:** upper-arm roll must be decision-relevant because it can prevent
  ferrule-first/knife-like approaches on upward marks and can change the
  dynamics of natural curves. A repertoire of only straight marks or one
  identical arc is inadequate. Mark curvature and bodily roll realization
  must not be conflated.
- **Direct evidence:** current task turns
  `019fdce4-63ec-7c01-80cd-e4247c3a801d` and
  `019fec46-2247-77b3-a016-dc57c0a44848`.
- **Technical consequence:** continuous signed quadratic curvature support;
  neutral and symmetric fixed-roll realizations; separate proposal densities
  and motor priors; roll/curve validation.
- **Durable artifacts:** `docs/CURVED_MARK_AND_ROLL_REALIZATION_2026-08-07.md`,
  `src/active_painter/policies.py`, `src/active_painter/motor_planning.py`,
  `tests/test_curved_mark_realization.py`.
- **Status:** implemented in the bounded simulation profile; dynamic roll
  during a mark remains research work.

### P-011 — Angled wrist roll was a deliberate exploration, not a selected redesign

- **Attribution:** owner-originated alternative and owner-selected deferral;
  agent-generated comparison branch.
- **Decision:** explore a distal wrist-roll axis with an angled brush as a
  branch, then place the detour on the back burner. Do not replace the accepted
  upper-arm-roll design without a later explicit decision.
- **Direct evidence:** current task turn
  `019fdddf-56ab-7be0-9e2c-ee95085d4d68` and the later owner instruction to put
  the wrist detour on the back burner.
- **Technical consequence:** isolated generated MJCF comparison, reachability
  tests, and explicit non-canonical status.
- **Durable artifacts:** `docs/ANGLED_WRIST_ROLL_EXPLORATION_2026-08-07.md`,
  `src/active_painter/wrist_roll_design.py`,
  `tests/test_wrist_roll_design.py`.
- **Status:** non-canonical exploration.

### P-012 — Brush contact is axisymmetric, angle-dependent, and directionally stick/slip

- **Attribution:** owner-originated physical requirements and owner-corrected
  footprint; agent literature synthesis, equations, coefficients, and code.
- **Decision:** a round brush should not become a fixed-aspect flat/chisel
  stamp. Its footprint is circular at normal incidence and elongates along the
  handle projection as the acute end-effector-axis/canvas-plane angle becomes
  shallower. Ferrule-leading push motion may catch/stick/release against canvas
  tooth; handle-leading pull motion is easier. This changes predicted physical
  consequences and is not a direct preference for pulling.
- **Direct evidence:** current task turns
  `019fec03-1254-7cc1-af3b-5650af34a0a4`,
  `019fec9e-482a-7c83-9ec4-851d9e7b861e`, and
  `019fecb2-1d6a-77c0-856a-e9dc0de9002b`.
- **Technical consequence:** Baxter/LuGre-inspired aggregate tuft deflection;
  frozen-tooth breakaway; separate normal/tangential force; corrected round
  footprint and 12.7 mm envelope.
- **Durable artifacts:** `docs/BRUSH_ANISOTROPY_RESEARCH_2026-08-10.md`,
  `src/active_painter/arm_sim.py`, `tests/test_arm_sim.py`.
- **Status:** provisional generative-process baseline; the corresponding
  learned uncertain agent likelihood is not yet implemented or calibrated.

### P-013 — Stroke exit must unload instead of stamping a terminal disc

- **Attribution:** owner visual diagnosis; agent cause analysis and correction.
- **Decision:** the circular node after each taper is an artifact, not a desired
  round-brush consequence. The realized path must unload contact and taper flow
  before lift without installing an artificial painting gate.
- **Direct evidence:** current task turn
  `019fecc9-e7fc-7d11-b85f-9a5f931043fa`.
- **Technical consequence:** coordinated depth/pressure taper and zero-flow lift
  initialization.
- **Durable artifacts:** `src/active_painter/stroke_execution.py`,
  `docs/BRUSH_ANISOTROPY_RESEARCH_2026-08-10.md`.
- **Status:** implemented and tested in simulation.

### P-014 — Counterfactual brush physics should become a learned generative approximation

- **Attribution:** owner-originated long-term architectural correction; agent
  confirmation and approximation ledger.
- **Decision:** independent copies of the actual process simulator are an
  integration baseline, not the desired counterfactual generative model. The
  agent should ultimately infer a compact stochastic brush/contact model from
  permitted camera and proprioceptive evidence.
- **Direct evidence:** current task turn
  `019fec03-1254-7cc1-af3b-5650af34a0a4` containing the separate statement that
  counterfactual rollouts should be generative approximations rather than
  forward runs of actual physics.
- **Technical consequence:** current process-equation reuse is explicitly
  labeled provisional; telemetry fields were added as training/evaluation
  labels for the later learned split.
- **Durable artifacts:** `docs/BRUSH_ANISOTROPY_RESEARCH_2026-08-10.md` section
  "Active-inference boundary", `docs/GENERATIVE_MODEL_SPEC.md` approximation
  register.
- **Status:** canonical destination, not yet implemented.

### P-015 — Parallel experience collection is necessary for practical learning

- **Attribution:** owner-selected performance priority; agent architecture and
  benchmark.
- **Decision:** do not rely on one real-time visible painter to generate the
  training history. Run isolated simulation environments in parallel, retain
  leakage-resistant observation/action records, and train centrally while
  preserving distinct agent beliefs and random streams.
- **Direct evidence:** current task turn
  `019fecfa-3351-7980-bd6c-7f47cd33f616`.
- **Technical consequence:** headless worker collection, corpus merging,
  centralized offline training, benchmarked worker scaling, and resumable
  checkpoints.
- **Durable artifacts:** `docs/PARALLEL_TRAINING_PIPELINE_2026-08-10.md`,
  `src/active_painter/parallel_collect.py`,
  `src/active_painter/offline_train.py`,
  `src/active_painter/parallel_benchmark.py`.
- **Status:** implemented baseline; composition training remains a separate
  bottleneck and was not established by the first full run.

### P-016 — Support documentation is part of the architecture's control surface

- **Attribution:** owner-originated governance requirement; agent documentation
  work.
- **Decision:** canonical support documents must remain aligned with code and
  hardware decisions because fresh Codex and Claude instances commonly consult
  those records rather than reconstructing the entire codebase.
- **Direct evidence:** current task turn
  `019fce88-725a-7d90-8776-7f0e9c212ac2`.
- **Technical consequence:** plant/motor vocabulary, camera baseline, wrist
  branch status, brush contact rules, and simulation limitations are repeated
  in `AGENTS.md` and linked to canonical records.
- **Durable artifacts:** `AGENTS.md`, `docs/CURRENT_IMPLEMENTATION.md`,
  `docs/PROGRESS.md`, `models/README.md`, `planning/PROJECT_TRACKER.md`.
- **Status:** canonical development practice. Drift remains a continuing risk.

### P-017 — Painterly order is mesoscopic compressibility, not pixel neatness

- **Attribution:** owner-originated conceptual proposal; formalization pending.
- **Decision candidate:** a patch containing a coherent brushmark or
  recognizable brushmark sequence should admit a shorter/more probable
  explanation than overlapping broken fragments with inconsistent edge
  directions and irregular nonrepeating shapes. This property is above raw
  pixel noise and need not be implemented as a hand-coded contour score.
  Locally plausible structure may still conflict with slower, larger-scale
  painting structure.
- **Direct evidence:** current task turn
  `019ff126-ae42-7e22-bee8-029121b5e8d4`.
- **Required active-inference interpretation:** any implementation must declare
  whether this is observation likelihood/evidence under a learned mark model,
  a prior preference over latent trajectories, or a combination with no
  double-counting. It must not become a scalar "messiness reward."
- **Related current artifacts:** `src/active_painter/composition.py`,
  `src/active_painter/canvas_hierarchy.py`, `docs/ARCHITECTURE.md` section 11.
  These implement an earlier compression-gap baseline and do not yet establish
  the newly articulated mesoscopic mark-order model.
- **Status:** candidate; high-priority conceptual input, not yet accepted as a
  complete mathematical design.

### P-018 — Proposed painting loop: local/global model mismatch, then consequence-aware repair

- **Attribution:** owner-originated architectural proposal; formalization
  pending.
- **Decision candidate:** select patches that are either locally
  disordered/unpredictable or incompatible with larger structure, roll out
  candidate marks, and infer a mark expected to reduce that unpredictability.
  A material-consequence predictor is necessary for credible rollouts but
  should not itself be the primary painting-selection objective.
- **Direct evidence:** current task turn
  `019ff129-402a-7d83-9a78-d54fe653cb5d`.
- **Technical caution:** reducing expected prediction error can suppress
  observations rather than improve a model, and active inference normally
  separates epistemic value, ambiguity, and prior preference. The proposal
  needs a joint factorization that distinguishes reducible structural mismatch,
  irreducible noise, local model evidence, and compatibility with slow latent
  predictions.
- **Status:** candidate. No claim should be made that the current planner
  already performs this loop.

### P-019 — Reconsider a pretrained VAE as an enabling low-level mark model

- **Attribution:** owner-supplied project history and candidate direction.
- **Decision candidate, clarified 2026-08-12:** the earliest project used a
  pretrained VAE as the low-level mark predictor. The intended modern use is a
  stochastic, action-conditioned **visual mark-consequence model** trained
  from fresh registered pre/post image patches, action, and brush context. It
  is distinct from both explicit coarse material-state reconstruction and the
  slower mesoscopic composition hierarchy.
- **Direct evidence:** current task turn
  `019ff12d-e547-7e20-88c5-7b2944e3b43e`.
- **Technical caution:** a VAE is a model family, not an active-inference term
  by itself. Pretraining the transition likelihood is compatible with the
  architecture; using latent distance as an undeclared aesthetic reward is
  not. Training data must avoid hidden-state leakage if the runtime model is
  claimed to be sensor-based.
- **Status, corrected 2026-08-12:** the visual proposal remains selected but
  unimplemented. `conditional-local-material-transition-cvae-v0` is a separate
  measured shadow experiment over coarse material posteriors. It has no policy
  influence, and its negative AI-107/AI-109 result does not test the visual
  proposal. The canonical factorization, corpus contract, and admission gates
  are in `VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

## Evolving decisions that should not be mistaken for contradictions

1. **Camera count evolved.** Earlier discussion entertained head-on,
   inspection, and tangent/profile cameras to satisfy visibility. Later spatial
   constraints produced the compact two-IMX296 baseline. The earlier
   alternatives are exploration history, not additional selected cameras.
2. **Brush anisotropy has two meanings.** The owner earlier asked that the
   MuJoCo tangential compliance not contain an arbitrary preferred chisel axis.
   Later directional push/pull behavior was intentionally introduced through
   handle geometry and canvas tooth. An axisymmetric round tuft can have
   isotropic base compliance while its interaction is direction-dependent.
3. **White-on-white observability was temporarily deferred, not removed from
   material state.** White still increases thickness and coverage. The current
   superficial grayscale camera likelihood simply cannot infer that thickness
   from white-on-white appearance.
4. **Wrist roll was explored, then deferred.** Its existence in source and docs
   is not evidence of a selected hardware change.
5. **Process simulation and learned prediction are staged.** Reusing physics in
   independent counterfactual worlds was accepted to get an end-to-end system
   running; the owner subsequently identified the need to replace that with an
   uncertain learned model.

## Important project decisions with incomplete attribution

The following are consequential but cannot be responsibly attributed solely
from the accessible conversations:

- the exact 80--90 percent terminal coverage distribution and its numerical
  concentration;
- the present best-of-family compression-gap equation;
- the exact VFE/EFE normalizers and decomposition conventions;
- exact RS03/RS02 axis assignment if it predates the accessible Codex tasks;
- exact camera coordinates, 4 mm provisional lens, noise, and timing values;
- exact fixed-roll angles, curvature mixture probabilities, process
  coefficients, and rollout budgets;
- most neural layer sizes, optimizer schedules, test fixtures, and checkpoint
  formats.

Some may have originated in the unenumerated ChatGPT `Arm` project or other
pre-2026-06-29 work. In the current evidence, they should be described as
agent-formalized or collaborative rather than silently assigned to either
party.

## Attribution rule for future agents

When adding future entries:

1. Record the owner's actual rationale, not only the final implementation.
2. Distinguish a direct proposal from an approval of an agent proposal.
3. Distinguish a selected design from an exploration or back-burnered branch.
4. Separate physical/process insight from the equations and parameters used to
   approximate it.
5. Treat visual/painterly diagnoses as technical evidence when they expose a
   wrong causal model, while still validating the resulting correction with
   tests and measurements.
6. Do not infer intellectual provenance from the Git author field alone.
7. Mark ideas that have not passed the governing active-inference boundary as
   candidates rather than canonical decisions.
