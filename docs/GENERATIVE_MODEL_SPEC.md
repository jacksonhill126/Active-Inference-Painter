# Executable Generative-Model Specification

Model specification: `baseline-oracle-v0`

Status: implemented-model description for M1 audit, accepted for `AI-101` on
2026-07-24. This is not the M1 lock. Later M1 tasks must independently verify
the equations, sensor boundary, calibration, proposal semantics, and
preference choices described here.

## 1. Purpose And Scope

This document states the probability model that the current code actually
implements, including approximations and ordinary engineering outside the
active-inference model. It covers:

- the native arm, brush, canvas, and material generative process;
- the obsolete summary compatibility fixture and provisional spatial planner;
- local material transitions and their learned parameter ensemble;
- pixel, canvas, relational, and passage beliefs;
- variational state inference;
- expected-free-energy evaluation;
- painting and motor policy inference;
- online learning.

This specification describes the retained **oracle-observation comparator**.
When explicitly enabled, the spatial planner receives arrays deterministically
derived from hidden simulator material state and the summary planner receives
exact aggregate material statistics. Assigned observation variances make
inference probabilistic, but the inputs are not measurements a physical robot
could obtain.

The live runtime now defaults to the fail-closed `sensor-boundary-v0`: it
does not run policy inference, learning, or oracle bootstrap until M2 supplies
fixed-camera, encoder, current, and contact likelihoods and a
sensor-conditioned posterior. `oracle_material_state` remains an explicit
diagnostic-only opt-in. The complete access audit is maintained in
`docs/VARIABLE_SENSOR_ACCESS_LEDGER.md`.

A first non-oracle M2 body-inference factorization is implemented but is not
yet connected to painting policy inference. For each joint it uses

```text
p(q_t | q_(t-1), dq_(t-1))  diagonal constant-velocity Gaussian prior
p(dq_t | dq_(t-1))           diagonal Gaussian random-walk prior
p(o_encoder_q | q_t)         Gaussian likelihood
p(o_encoder_dq | dq_t)       Gaussian likelihood
```

and, when the corresponding physical samples are present,

```text
p(o_switch | c_t)            Bernoulli likelihood
p(o_force | f_t)             Gaussian likelihood
```

The Gaussian and Bernoulli posteriors are conjugate mean-field updates, so the
reported body VFE is the exact factorized
`KL[q(x_t)||p(x_t)] - E_q[log p(o_t|x_t)]` under those assumptions. Its
transition and observation precisions must be supplied in a versioned
`BodyLikelihoodSpec`; they are not inferred from simulator truth. Current,
voltage, temperature, deflection, and fault samples do not enter this
likelihood yet. This is a declared incomplete factorization, not evidence that
the full sensor-equivalent posterior exists.

This specification is "executable" in the limited engineering sense that every
factor and free-energy term maps to a named implementation location. It does
not claim that SGD is exact Bayesian learning or that every current
approximation follows uniquely from the free-energy principle.

## 2. Boundary And Clocks

The implementation has five relevant clocks.

| Clock | Index | Current event | Main implementation |
| --- | --- | --- | --- |
| Physics/control | `tau` | one native plant integration step, normally 1/240 s in the web runtime | `arm_sim.py`, `web_runtime.py` |
| Mark | `t` | one completed brush mark and resulting canvas transition | `arm_agent_driver.py` |
| Passage step | `r` | one mark within a persistent passage belief | `passage_inference.py` |
| Passage boundary | `j` | completion or interruption of a passage | `arm_agent_driver.py`, `canvas_hierarchy.py` |
| Painting/global plan | `k` | global candidate generation, stopping inference, or a completed painting | `arm_agent_driver.py` |

The Markov blanket used by the eventual physical agent should contain sensor
observations and motor commands. The present implementation does not yet
enforce that boundary:

- process states include exact joint, motor, contact, brush, and material
  variables;
- controller and hard-safety code may use process state below policy
  selection;
- painting inference currently receives exact derived material state;
- visualization and evaluation receive additional process truth.

The active-inference painting boundary begins at candidate painting policies
and their predicted sensory/material/proprioceptive consequences. Forward
kinematics, IK, trajectory interpolation, servo dynamics, collision checks,
and hard limits are conventional engineering below the selected policy.

## 3. Variables

### 3.1 Generative-process variables

At physics time `tau`, the native process contains:

```text
x_body_tau = {
    q, q_dot,
    motor_angle, motor_velocity,
    motor_temperature,
    transmission_deflection,
    encoder samples,
    current, torque, voltage,
    contact state
}

x_material_tau[p] = {
    thickness,
    wetness,
    black_mass,
    surface_tone
}

x_brush_tau = {
    selected dedicated brush,
    persistent normalized fresh load and tone for white/black brushes,
    held wet-paint volume and black mass,
    bristle and edge realization,
    path distance
}
```

Here `p` indexes native canvas pixels. Material coverage, observed tone, and
ground contrast are deterministic transforms of material state:

```text
coverage[p] = 1[thickness[p] >= presence_threshold]
opacity[p] = 1 - exp(-thickness[p] / thickness_scale)
observed_tone[p] =
    (1 - opacity[p]) * ground_tone + opacity[p] * surface_tone[p]
ground_contrast[p] = abs(observed_tone[p] - ground_tone)
```

The native process is implemented by `JointPlant`, `ArmPainterSim`,
`VerticalCanvas`, and `Brush` in `src/active_painter/arm_sim.py`.

### 3.2 Painting actions and policies

A mark action is:

```text
a_t = (x0, y0, x1, y1, width, amount, tone, stop)
```

with normalized canvas endpoints and scalar brush variables. The encoded
transition action additionally includes motor-realization channels. See
`StrokeAction` in `env.py` and `action_encoding.py`.

A painting policy is a finite sequence:

```text
pi = (a_t, ..., a_T=stop)
```

Every policy must terminate in `stop`, and the immediate-stop policy is always
present. A policy may carry:

- a `PassageLatent` generating several related marks;
- a `PassagePlanLatent` generating several passages;
- a first-stroke `MotorPrimitiveLatent`;
- an instantaneous first-stroke `BrushPreparationPolicy` in
  `{preserve, reload}`.

These are defined in `policies.py`.

### 3.2.1 Brush-loading belief and preparation policy

For each dedicated color brush, the compact generative model maintains:

```text
q(b_t) = q(load_t) q(black_fraction_t)
```

as bounded Gaussian moments. The preserve transition depletes expected load
after a mark and increases load/mixture uncertainty. The reload transition is
instantaneous:

```text
load' = 1
black_fraction' = selected_tone
```

with small declared reload variance. Preparation inference compares preserve
and reload using conditional mark-outcome EFE:

```text
G_brush =
    precision_material * risk(deposited amount | selected mark)
  + precision_pigment  * risk(black fraction | selected tone)
  + precision_ambiguity * ambiguity

q(pi_brush) proportional to
    p(pi_brush) exp(-precision_policy * G_brush)
```

The reload prior is explicit (`brush_reload_policy_prior`); there is no hard
load threshold. Current motor planning marginalizes preparation evidence but
uses the modal preparation in the expensive physical rollout, a named
approximation. `BrushLoadingModel.infer_load_from_mark` supplies a scalar
Gaussian deposition likelihood and separately logged VFE for a future
camera-derived local-patch observation. No exact process brush state is passed
to this update.

### 3.3 Obsolete summary compatibility belief

Summary mode uses a six-dimensional diagonal-Gaussian belief:

```text
s_summary = (
    material_coverage_mean,
    mean_thickness,
    maximum_thickness,
    mean_wetness,
    overlap_fraction,
    painted_ground_contrast
)
```

The exact process summary is produced by `canvas_summary_state()` in
`arm_agent_driver.py`. The state dimension is configured by
`PainterConfig.state_dim`.

This representation is formally deprecated. It is non-spatial, not
predictively sufficient for image-making, and must not be interpreted as a
highest-level painting latent. It remains only for regression, analytic
reference tests, and old checkpoints. The runtime default is the provisional
spatial-material baseline while a camera-conditioned learned perceptual
hierarchy is implemented.

### 3.4 Provisional spatial material belief

Spatial mode uses six channels:

```text
s_pixel[p] = (
    thickness,
    wetness,
    black_mass,
    surface_tone,
    ground_contrast,
    material_coverage
)
```

The first four channels are the independent process material factors used by
the spatial likelihood. Ground contrast and coverage are deterministic derived
views. The implementation stores all six for planning and diagnostics, projects
the derived views from the primary factors, and excludes them from independent
likelihood evidence.

At native size `N`, the pixel mean has `6*N*N` scalars and the diagonal
log-variance has the same size. The default `PainterConfig` canvas is 48 by 48.
The web process and live spatial observation use a 256 by 256 canvas, while
the driver/bootstrap configuration still carries `canvas_size=64` and a 16 by
16 planner level. This bootstrap/live scale mismatch is a known baseline
limitation.

The material pyramid is a deterministic set of mean-pooled fields at pixel,
tile, and planner scales. Material coverage is pooled from binary pixel
occupancy, not re-thresholded from mean thickness. See `spatial_state.py`.

These material fields are hand-defined physical factors for local transition
modeling, not the desired learned abstract feature hierarchy. The intended
higher layers must infer flexible latent causes from permitted observations and
be retained by held-out predictive necessity rather than by a hand-authored
feature list.

### 3.5 Slower beliefs

When the hierarchy is enabled, the model maintains:

- `z_canvas_j`: an `8 x 4 x 4 = 128` dimensional default canvas latent;
- `z_relation_j`: a 24-dimensional default relational latent;
- `z_passage_r`: a seven-dimensional diagonal-Gaussian geometry belief plus a
  beta-Bernoulli tone factor;
- `theta`: learned local dynamics, hierarchy, and transition parameters.

The canvas and relational posteriors update only at explicit passage
boundaries. The passage posterior may update after each executed subordinate
mark. See `canvas_hierarchy.py` and `passage_inference.py`.

The ensemble members and neural-network weights are not represented by an
explicit normalized variational density. Ensemble disagreement is used as an
approximate parameter-posterior uncertainty.

## 4. Current Joint Density

For one mark-scale episode under painting policy `pi` and motor realization
`m`, the low-level model is treated as:

```math
p(o_{1:T},s_{0:T},\pi,m,\theta)
=p(\theta)p(s_0)p(\pi)p(m\mid\pi)
 \prod_t p_\theta(s_{t+1}\mid s_t,a_t,m_t)
             p(o_{t+1}\mid s_{t+1}).
```

At passage boundaries, the implemented hierarchy adds:

```math
p(z^C_{j+1}\mid z^C_j,z^R_j,d(\pi_j),\theta_C)
p(z^R_{j+1}\mid z^R_j,c(z^C_j),d(\pi_j),\theta_R)
p(o^C_{j+1}\mid z^C_{j+1})
p(o^R_{j+1}\mid z^R_{j+1})
```

where `d(pi)` is a deterministic policy descriptor and `c(z_canvas)` is a
coarse canvas context. For structured passages with learned support, a
passage-conditioned Markov transition is evaluated after each predicted mark.

The slow passage belief uses:

```math
p(z^P_{r+1}\mid z^P_r)
p(o^P_{r+1}\mid z^P_{r+1},a_r,\Delta s_r)
```

implemented as diagonal Gaussian precision fusion for geometry and
beta-Bernoulli updating for tone. Its observation extracts some geometry from
the executed action and some from the material delta. This is a mixed
action/outcome pseudo-likelihood, not yet a calibrated physical likelihood.

Prior preferences are biased generative-model factors used during policy
inference:

```math
p^*(C_T\mid stop)
p^*_{comp}(s_T)
p^*_{motor}(o^{motor}_{1:T})
```

They are not transition rewards.

## 5. Transition Likelihoods

### 5.1 Obsolete summary-fixture transition

`DynamicsEnsemble` implements a diagonal Gaussian:

```math
p_{\theta_e}(s_{t+1}\mid s_t,a_t,m_t)
=N(\mu_{\theta_e}(s_t,a_t,m_t),
   diag(\sigma^2_{\theta_e}(s_t,a_t,m_t))).
```

Each ensemble member is trained by Gaussian negative log likelihood with a
Bernoulli bootstrap mask. Member means approximate parameter uncertainty;
predicted log variances approximate conditional or aleatoric uncertainty.

The support projection enforces nondecreasing material coverage, overlap, and
wetness-compatible state constraints. These are structural transition
assumptions, not learned consequences.

### 5.2 Spatial transition

`LocalSpatialDynamicsEnsemble` uses a convolutional diagonal-Gaussian
transition on the pixel patch touched by the action:

```math
p_{\theta_e}(s^{P}_{t+1}\mid s^{P}_t,a^{P}_t,m_t).
```

The patch is the rasterized stroke support plus a configured margin. Padded
cells are masked out of training likelihoods. Predictions are pasted over an
immutable base canvas.

Outside active support, planning uses an identity transition:

```math
s_{t+1}[outside] = s_t[outside]
```

with configured `local_identity_logvar`. Policy-independent outside-support
entropy constants are omitted and reported by
`identity_transition_approximation`. This is a sparse-computation
approximation.

`dense_grid` remains a compatibility mode that evaluates the learned spatial
transition over the planner grid.

### 5.3 Material support projection

After each learned spatial transition:

- thickness cannot decrease;
- wetness cannot decrease;
- black mass cannot decrease in the learned planning model;
- surface tone is clamped to `[0,1]`;
- coverage and ground contrast are recomputed.

The native wet-paint process can transport black mass locally through brush
pickup and release. The learned projection's pointwise nondecreasing
black-mass constraint therefore cannot represent all process transitions. This
is a known model mismatch requiring a later decision and test.

## 6. Observation Likelihoods

### 6.1 Summary likelihood

`ObservationModel` defines:

```math
p(o_t\mid s_t)=N(s_t,diag(\sigma_o^2(s_t))).
```

Its standard deviation grows with wetness, thickness-related state, and
overlap-related state. Observation ambiguity is differential entropy above a
dry-canvas baseline.

In the arm runtime, `o_t` is the exact six-value simulator summary, not a
physical measurement. The likelihood is therefore provisional.

### 6.2 Spatial likelihood

`SpatialVariationalStateEstimator` uses:

```math
p(o^{pixel}_t\mid s^{pixel}_t)
=N(s^{pixel}_t,diag(\sigma^2_{pixel}(o_t))).
```

The observation is constructed directly from exact `VerticalCanvas` arrays.
The variance is a configured material-dependent function. This is an
oracle-state likelihood and must not support sensor-only claims.

Derived ground-contrast and coverage channels are deterministically projected
after Gaussian fusion and do not contribute separate VFE, transition NLL, or
EFE uncertainty terms. This AI-103 decision is detailed in
`docs/OBSERVATION_FACTOR_AUDIT.md`.

### 6.3 Hierarchy likelihoods

The canvas encoder/decoder defines an approximate latent-variable likelihood
over coarse material fields. The relational observation is a deterministic
region-slot extraction followed by a learned latent encoder/decoder.

These models are trained online on the same developmental stream used for
policy evaluation. Their uncertainty is diagonal Gaussian and their posterior
calibration has not passed a held-out gate.

## 7. Variational Family And VFE

### 7.1 Summary state inference

The variational family is:

```math
q(s_t)=N(\mu_q,diag(\sigma_q^2)).
```

For transition prior `p(s_t|s_{t-1},a_{t-1})`, the optimized free energy is:

```math
F_t =
KL[q(s_t)||p(s_t|s_{t-1},a_{t-1})]
- E_{q(s_t)}[log p(o_t|s_t)].
```

`VariationalStateEstimator` optimizes mean and log variance with Adam. The
expectation uses 8 reparameterized samples per optimization step and 32
samples for the logged decomposition. Log variance is clamped. This is
stochastic variational optimization, not an analytic posterior.

### 7.2 Spatial state inference

The spatial family is a factorized pixel/channel Gaussian:

```math
q(s^{pixel}_t)
=\prod_{p,c}N(\mu_{p,c},\sigma^2_{p,c}).
```

Because both prior and provisional likelihood are diagonal Gaussian,
`SpatialVariationalStateEstimator` uses analytic precision fusion. It logs
mean KL and expected negative log likelihood over the four primary factors in
`nats_per_independent_cell_channel`, then deterministically projects material
support and derived channels.

The transition moments are evaluated only at the previous posterior mean.
Cross-pixel, cross-channel, and nonlinear posterior covariance are omitted.

### 7.3 Slow hierarchy

Canvas, relational, and passage beliefs are diagonal Gaussian, with a
beta-Bernoulli factor for passage tone. Persistent canvas and relational
posteriors use encoder moments and precision fusion at passage boundaries.
Neural parameters are trained by SGD rather than variational message passing.

## 8. Prior Preferences

### 8.1 Terminal material coverage

Only policies ending in `stop` are admitted. Terminal material coverage has:

```math
p^*(C_T\mid stop)=Beta(alpha^*,beta^*)
```

with mean `target_coverage` and concentration
`terminal_concentration`. The default mean is 0.87. The forecast distribution
`q(C_T|pi)` is moment-matched to a Beta distribution.

Terminal risk is:

```math
R_C(pi)=KL[q(C_T|pi)||p^*(C_T|stop)]
       =-H[q(C_T|pi)]-E_q[log p^*(C_T|stop)].
```

It is applied once at the terminal state, never at intermediate marks. White
paint on white ground contributes to coverage through thickness.

### 8.2 Composition preference

When enabled:

```math
gap(s)=ELBO_hierarchy(s)-log p_flat(s)
p^*_{comp}(s) proportional to exp(lambda_comp * gap(s))
R_comp(pi)=-lambda_comp E_q[gap(s_T)].
```

The flat code is fit to the evaluated field, and the hierarchy pays a latent
KL. This is an unnormalized energy-based preference. Because the hierarchy is
trained online on the same stream it evaluates, this creates a self-referential
closed loop. It is not accepted as a principled fixed preference until
`AI-110` approves frozen, cross-fitted, or alternative semantics. Current
composition results are exploratory diagnostics.

### 8.3 Motor homeostatic outcomes

The motor forecast produces normalized current, torque, velocity,
acceleration, target-error, limit-proximity, contact-loss, pressure-error, and
path-error outcomes. Zero-centered diagonal Gaussian preference scales are
declared in `PainterConfig`.

Motor risk is expected negative log preference with policy-independent
Gaussian normalizers omitted. Likelihood ambiguity and process-observation
mutual information are analytic under a diagonal-Gaussian approximation.
Hard current, force, joint, and workspace limits remain external.

## 9. Expected Free Energy

For spatial policy `pi`, the implemented total is:

```math
G(pi) =
    R_C
  + A_o
  + R_transition
  + A_transition
  + R_comp
  + R_canvas
  + R_relation
  + R_passage_canvas
  + R_passage_relation
  + R_motor
  + A_motor
  - I_motor.
```

All active terms are recorded separately in `SpatialEFEComponents`.

### 9.1 Observation ambiguity

`A_o` is expected observation entropy above the dry-canvas baseline. In sparse
mode it is computed on active support and divided by full-canvas area so patch
size is not an implicit preference.

### 9.2 Transition information gain

For the learned ensemble:

```math
R_transition = -H[q(s_{t+1}|pi)]
A_transition = E_{q(theta)} H[p(s_{t+1}|s_t,a_t,theta)]
I_theta = H[q(s_{t+1}|pi)] - E H[p(s_{t+1}|...,theta)].
```

Therefore:

```math
R_transition + A_transition = -I_theta.
```

`epistemic_value` logs this identity and is not added a second time. The
entropy is a diagonal-Gaussian, moment-matched approximation to ensemble
information gain.

### 9.3 Higher-level transition risk

The canvas, relational, and passage trajectory terms are precision-weighted
KL divergences between encoded terminal/step posteriors and learned transition
priors. They remain zero until the relevant likelihood has received training
updates. Immediate stop uses an identity prior.

### 9.4 Motor EFE

Motor risk and ambiguity are added, and motor epistemic value is subtracted.
The learned per-kind inverse-gamma reliability belief inflates expected
squared execution error and likelihood variance for fidelity channels.

### 9.5 Omitted constants and units

The implementation omits several policy-independent differential-entropy and
preference normalizers. Sparse local terms are area-scaled. Summary VFE is
reported in nats; spatial VFE is reported in mean nats per independent
cell-channel; EFE mixes full-policy scalar modality terms after declared
precision scaling.

AI-103 resolved the spatial factorization and normalization decision. AI-104
and AI-105 must independently verify the remaining VFE/EFE reductions,
omissions, and signs.

## 10. Policy Priors, Proposals, And Posterior

### 10.1 Painting policy prior

The only painting-policy prior explicitly included in the global policy
posterior is the immediate-stop prior:

```math
log p_stop(pi) =
    log sigmoid(kappa * (believed_coverage - coverage_midpoint))
```

for immediate stop, and zero log weight for continuation candidates.

### 10.2 Candidate proposal distribution

`PolicySampler` constructs a finite candidate set using:

- uniform/random stroke geometry;
- low-coverage start-point oversampling;
- fixed fractions of passage and passage-plan proposals;
- explicit black/white tone alternatives;
- finite planning-depth sampling.

These are computational proposal distributions. Their log densities are not
included in the policy posterior and no importance correction is applied.
Consequently, proposal frequency affects the finite-budget result. Comments
that call low-coverage or passage proposal frequency an "empirical policy
prior" are not mathematically realized as normalized `p(pi)` terms in the
current posterior. `AI-111` must either define and include the intended policy
prior or retain these strictly as proposal mechanisms and measure convergence.

### 10.3 Passage-local prior

During receding-horizon execution, sampled passage continuations receive the
current `PassageBelief.transition_log_prior`. This is an explicit local
Gaussian/Bernoulli transition prior over the persistent passage latent.

### 10.4 Motor policy prior and marginalization

Enabled motor realization kinds are equiprobable:

```math
p(m|pi)=1/K.
```

For each painting candidate:

```math
log evidence(pi)
=log sum_m exp(log p(m|pi)-gamma G(pi,m)).
```

The conditional motor posterior is normalized over feasible forecasted
realizations. Candidates outside the finite motor-forecast budget retain
base-EFE treatment, so global inference remains budget-truncated.

### 10.5 Painting policy posterior

Without embodied refinement, the active candidate set uses:

```math
q(pi) proportional to
exp(log p_stop(pi) - gamma G_base(pi)).
```

For candidates with embodied refinements, the marginal motor evidence
replaces the base-EFE term:

```math
q(pi) proportional to
exp(log p_stop(pi) + log_evidence_motor(pi)).
```

The code subtracts the minimum EFE before softmax where convenient; this does
not change the normalized posterior. The selected policy is sampled from
`q(pi)`, not chosen by minimum scalar score.

## 11. Learning

Online learning currently consists of:

- Gaussian NLL training of local/summary transition ensembles;
- variational-autoencoder-style hierarchy training;
- Gaussian NLL training of aggregate and passage-step latent transitions;
- inverse-gamma updates for per-motor-kind execution reliability;
- diagonal Gaussian/beta-Bernoulli passage belief updates.

Neural optimization uses Adam, replay buffers, gradient clipping, and
Bernoulli bootstrap masks. These are conventional ML approximations. The
ensemble is treated as an approximate posterior over parameters but is not
derived from an explicit ELBO over `q(theta)`.

Checkpointed learned parameters, optimizer state, replay, calibration beliefs,
episodic beliefs, and current canvas have different persistence semantics.
`AI-112` must finalize and test those semantics before inherited runs support
research claims.

## 12. Factor-To-Code Map

| ID | Role | Mathematical object | Implementation | Current status |
| --- | --- | --- | --- | --- |
| GP-BODY | Generative process | `p(x_body_tau+1|x_body_tau,u_tau)` | `arm_sim.JointPlant`, `ArmPainterSim` | Representative, uncalibrated |
| GP-MATERIAL | Generative process | wet material and brush transfer | `arm_sim.VerticalCanvas.paint_at`, `Brush` | Hand-designed process |
| GM-SUM-TRANS | Transition likelihood | `p_theta(s_t+1|s_t,a_t,m_t)` | `models.DynamicsEnsemble` | Obsolete compatibility fixture |
| GM-PIX-TRANS | Transition likelihood | local `p_theta(s^P_t+1|s^P_t,a^P_t,m_t)` | `models.LocalSpatialDynamicsEnsemble`, `local_spatial.py` | Learned sparse approximation |
| GM-SUM-OBS | Observation likelihood | `p(o_t|s_t)` | `models.ObservationModel` | Obsolete oracle compatibility fixture |
| GM-PIX-OBS | Observation likelihood | `p(o^pixel_t|s^pixel_t)` | `spatial_inference.py` | Oracle material, provisional |
| Q-SUM | Posterior | diagonal `q(s_t)` | `inference.VariationalStateEstimator` | Obsolete compatibility fixture |
| Q-PIX | Posterior | factorized `q(s^pixel_t)` | `spatial_inference.SpatialVariationalStateEstimator` | Analytic diagonal fusion |
| Q-CANVAS | Posterior/likelihood | `q(z_canvas)`, decoder likelihood | `canvas_hierarchy.HierarchicalCanvasModel` | Online diagonal latent |
| Q-REL | Posterior/likelihood | `q(z_relation)`, decoder likelihood | `canvas_hierarchy.py` | Deterministic slots plus latent |
| Q-PASSAGE | Slow posterior | `q(z_passage)` | `passage_inference.PassageBelief` | Mixed pseudo-likelihood |
| Q-BRUSH | Material posterior | `q(load_t)q(black_fraction_t)` per dedicated brush | `brush_loading.BrushLoadingModel` | Compact bounded Gaussian moments; camera update not wired |
| TRANS-BRUSH | Transition likelihood | preserve depletion/uncertainty or pure full reload | `brush_loading.py`, `arm_sim.Brush` | Explicit provisional approximation |
| PREF-COVERAGE | Prior preference | `p*(C_T|stop)` | `preferences.py`, `efe_common.py` | Explicit terminal Beta |
| PREF-COMP | Prior preference | energy from compression gap | `canvas_hierarchy.py`, `spatial_efe.py` | Unresolved closed loop |
| PREF-MOTOR | Prior preference | homeostatic outcome densities | `motor_planning.motor_efe_terms` | Explicit diagonal approximation |
| PRIOR-STOP | Policy prior | `p(stop-first|coverage belief)` | `policies.policy_stop_log_prior` | Explicit |
| PRIOR-PASSAGE | Transition prior | `p(z_passage_r+1|z_passage_r)` | `PassageBelief.transition_log_prior` | Explicit local prior |
| PRIOR-MOTOR | Policy prior | uniform `p(m|pi)` | `motor_planning.py` | Explicit |
| PRIOR-BRUSH | Policy prior | `p(preserve/reload)` | `brush_loading.py` | Explicit |
| PROP-PAINT | Proposal only | finite `q_proposal(pi)` | `policies.PolicySampler` | Uncorrected finite proposal |
| EFE-PIX | Expected free energy | risk/ambiguity hierarchy | `spatial_efe.py` | Approximate, decomposed |
| EFE-MOTOR | Expected free energy | proprioceptive EFE | `motor_planning.py`, driver | Approximate, decomposed |
| EFE-BRUSH | Expected free energy | conditional material/pigment risk and ambiguity | `brush_loading.py`, driver | Approximate, decomposed |
| SAFE | External constraint | hard feasibility and stop | plant, controller, future hardware safety | Not active inference |

## 13. Approximation Register

The following are explicit blockers or limitations, not hidden implementation
details:

1. Exact simulator material state is used as observation.
2. Summary and pixel posteriors are diagonal Gaussian.
3. Derived channels may be double-counted before deterministic projection.
4. Transition moments are propagated at posterior means.
5. Ensemble disagreement is only an approximate parameter posterior.
6. Local sparse rollouts omit outside-support entropy constants.
7. Pointwise nondecreasing black mass conflicts with wet-paint transport.
8. Composition preference is trained on the developmental stream it evaluates.
9. Candidate proposal frequency is not corrected in policy inference.
10. Hierarchy likelihoods activate after small online sample counts and are not
    yet calibrated.
11. Passage observations mix executed action geometry with material evidence.
12. Motor outcome density is diagonal and forecasted by the same broad process
    family used for execution.
13. Motor refinement covers only a finite subset of base painting candidates.
14. Neural parameter learning is SGD, not exact variational parameter
    inference.
15. The web process and planner can use different canvas resolutions.
16. Continuous-density constants and modality reductions require independent
    unit/sign verification.
17. Brush preparation is analytically marginalized, but only its modal policy
    is sent through the expensive motor/material rollout.

## 14. Required Next Decisions

This specification unblocks, but does not answer:

- `AI-102`: which current values cross the future physical sensor boundary;
- `AI-103`: which channels are observations versus deterministic transforms,
  and what their units are;
- `AI-104`: whether VFE matches analytic references;
- `AI-105`: whether EFE signs, identities, and policy posteriors match
  independent references;
- `AI-107`: whether predictive variances are calibrated;
- `AI-110`: whether the composition preference can be retained;
- `AI-111`: whether proposal distributions require explicit correction;
- `AI-112`: what learned and episodic state may persist between runs.

No claim of principled sensor-based active inference should pass the M1 gate
until those decisions have evidence.
