# Executable Generative-Model Specification

Model specification: `baseline-oracle-v0`

Status: implemented-model description for M1 audit, accepted for `AI-101` on
2026-07-24. This is not the M1 lock. Later M1 tasks must independently verify
the equations, sensor boundary, calibration, proposal semantics, and
preference choices described here.

## 0.1 Implemented model versus accepted perceptual target

This document remains the executable specification of what the code currently
does. It is not approval of the six-channel material posterior as the target
painting representation. The owner-confirmed 2026-08-12 direction is
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`: detailed material variables stay in
the process, while the target model predicts registered visual appearance,
oriented boundaries, continuous masses, and their relations. Persistent
canvas-wide wetness is not a target latent. Local material ambiguity may be an
ephemeral action-conditioned interaction latent inferred from a fresh image
crop.

Accordingly, factors below involving 16x16 `SpatialCanvasState` fields describe
the compatibility/integration baseline only. The material cVAE results concern
that target and do not test the proposed visual mark-consequence VAE.

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

This specification describes both the retained **oracle-observation
comparator** and the bounded sensor-path factors implemented so far. When the
oracle comparator is explicitly enabled, the spatial planner receives arrays
deterministically derived from hidden simulator material state and the summary
planner receives exact aggregate material statistics. Assigned observation
variances make that comparator probabilistic, but its inputs are not
measurements a physical robot could obtain.

The live runtime now defaults to the fail-closed `sensor-boundary-v0`: it
does not run policy inference, learning, or oracle bootstrap. A fixed-camera
likelihood and spatial posterior now exist, as does a separate body estimator;
the MuJoCo runtime now uses the latter to replace exact joint-position/velocity
initialization in motor forecasts. Material/brush/contact forecast context and
current/deflection factors remain incomplete.
`oracle_material_state` remains an explicit diagnostic-only opt-in. The
complete access audit is maintained in
`docs/VARIABLE_SENSOR_ACCESS_LEDGER.md`.

A first non-oracle M2 body-inference factorization is implemented and connected
to MuJoCo motor-forecast joint-state initialization, but not to live painting
policy execution. For each joint it uses

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

At runtime, `mujoco-ideal-sensor-body-likelihood-v0` supplies explicit
simulation-only precision/process floors. Its calibration status is
`provisional_simulation_only_not_hardware_calibrated`. Each planning pass
freezes one posterior revision; rollout particle zero uses its mean and later
particles sample the diagonal joint-state variance. Future native process noise
is seeded independently of the live process. Contact probability/force is not
yet mapped into the MuJoCo brush-compliance latent. A frozen
`SpatialCanvasState` now supplies the initial thickness, wetness, black-mass,
and surface-tone distribution: particle zero uses its mean and later particles
sample its diagonal variance at posterior-cell resolution. Nonnegative support,
`black_mass <= thickness`, and tone bounds are physically projected; coverage
and ground contrast are recomputed rather than sampled. Substrate grain, brush
history, and model parameters remain unresolved. Forecast brush load and black
fraction use a frozen diagonal `BrushLoadBelief`; particle zero uses its means
and later particles sample bounded Gaussian moments. Bristle offsets/gains and
mark phases use `brush-microstructure-prior-v0` under an independent forecast
seed, with a deterministic representative particle zero. The approximation
collapses held paint and persistent bristle history into load/average pigment.

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
observations and motor commands. The present implementation enforces it at the
new camera posterior but not throughout the live controller:

- process states include exact joint, motor, contact, brush, and material
  variables;
- controller and hard-safety code may use process state below policy
  selection;
- oracle painting inference receives exact derived material state; sensor mode
  may assimilate registered camera pixels and initialize MuJoCo joint forecasts
  from the body posterior, but remains blocked from policy execution by copied
  material/brush/contact forecast context and incomplete continuous scheduling;
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

The sensor path now adds the analytic likelihood

```math
p(o_p \mid s_p,z_p=\mathrm{inlier})
=\mathcal{N}(o_p;g(s_p),\sigma_c^2),
```

where `g` maps thickness and surface tone to predicted superficial grayscale.
`z_p` is a Bernoulli inlier/outlier latent with a broad state-independent
outlier density. Its posterior responsibility is inferred from image
residuals; no simulator segmentation or visibility mask is observed. A first-
order per-cell update projects this likelihood into the diagonal spatial
posterior. Wetness and bulk black pigment have zero direct image Jacobian.
Global and foveal products derived from one native exposure are mosaicked
before one likelihood factor, avoiding an independence claim for correlated
pixels. This is `camera-spatial-likelihood-v0`, a provisional analytic low-
level observation model rather than the desired learned perceptual hierarchy.

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

### 5.2.1 Shadow conditional latent transition

`ConditionalPatchVAEEnsemble` implements the experimental likelihood

```math
p_{\theta_e}(s^P_{t+1}\mid s^P_t,\log\sigma^{2,P}_t,
  a^P_t,b_t,m_t,z_t),\qquad z_t\sim N(0,I),
```

with recognition density conditioned on both current and next posterior mean
and log variance. Training minimizes the beta-one negative ELBO and logs
reconstruction NLL separately from latent KL. Its bootstrap prior predictive
separates decoder likelihood variance, within-member latent variation, and
between-member disagreement. This factor is currently shadow/offline only: it
is not called by `SpatialExpectedFreeEnergy`, does not alter policy inference,
and must not be described as the implemented composition model. Full boundary
and admission tests are recorded in
`CONDITIONAL_PATCH_VAE_SHADOW_BASELINE_2026-08-11.md`.

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
gap(s)=ELBO_hierarchy(s)-max_m log p_m(s)
p^*_{comp}(s) proportional to exp(lambda_comp * gap(s))
R_comp(pi)=-lambda_comp E_q[gap(s_T)].
```

The opponent is a two-member hand-written baseline family, and the gap is
measured against the best member, so it only credits structure that no member
can explain:

- `p_iid`: the best per-image per-channel iid-cell Gaussian;
- `p_local`: a parameter-free 3x3 hollow-neighbourhood local Markov code. Each
  cell is predicted by the mean of its eight neighbours excluding itself, with
  replicate padding and a per-image per-channel residual variance.

The family is fixed and is never fit to outcomes; only the residual variance
depends on the evaluated field, which is the same licence the iid member has
always had. All three codes (both members and the hierarchical decoder) share
one quantization floor `SIGMA_FLOOR`, so none earns free nats from
continuous-density resolution, and the hierarchy pays a latent KL. Because
`p_local` has a fixed 3x3 spatial scale, gap magnitudes are grid-size
dependent and are not comparable across different `spatial_grid_size` values.
The member selection is declared by `composition_local_baseline_enabled`;
setting it false restores the iid-only baseline exactly.

This is an unnormalized energy-based preference. Because the hierarchy is
trained online on the same stream it evaluates, this creates a self-referential
closed loop. It is not accepted as a principled fixed preference until
`AI-110` approves frozen, cross-fitted, or alternative semantics. Current
composition results are exploratory diagnostics.

### 8.3 Motor homeostatic outcomes

The motor forecast produces normalized current, torque, velocity,
acceleration, target-error, limit-proximity, contact-loss, pressure-error, and
path-error outcomes. Zero-centered diagonal Gaussian preference scales are
declared in `PainterConfig`.

Motor risk is the PRAGMATIC term `E_q[-log p*(o)]` with policy-independent
Gaussian normalizers omitted. It is not a KL: the `-H[q(o)]` term is absent.
The motor contribution to expected free energy is therefore

```math
G_motor = pragmatic - I(s;o) - N_reliability.
```

Because `-I(s;o) = E_{q(s)} H[p(o|s)] - H[q(o)]`, that expression is
algebraically identical to the canonical `KL risk + ambiguity - novelty`.
Subtracting the mutual information once therefore already accounts for the
canonical ambiguity contribution, and the separately logged
`MotorEFETerms.ambiguity` — likelihood entropy measured in excess of the
*preference* scale rather than in absolute nats — must never be added on top of
it. `MotorEFETerms.forecast_outcome_entropy` records `H[q(o)]` so the canonical
KL form of risk stays derivable from telemetry.

`N_reliability` is parameter novelty over the learned per-kind inverse-gamma
motion-reliability belief, the direct analogue of the reference implementation's
Dirichlet novelty term. Likelihood ambiguity, forecast outcome entropy, and
process-observation mutual information are analytic under a diagonal-Gaussian
approximation. Hard current, force, joint, and workspace limits remain external.

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
  - I_motor
  - N_motor.
```

`R_motor` is the motor *pragmatic* term only. `A_motor` (motor ambiguity in
excess of the preference scale) is recorded in `MotorEFETerms` and mirrored into
the components dataclasses, but it is not a summand: `- I_motor` already carries
the canonical ambiguity contribution (see §8.3).

All active terms are recorded separately in `SpatialEFEComponents`.

Every modality contributes `gamma_m * normalizer_m * (raw term)`. `gamma_m` is
the posterior mean of a declared Gamma precision belief
(`precision_beliefs.GammaPrecisionBelief`), not a hand-tuned constant, and
`normalizer_m` reduces the modality to nats per observation channel (§9.5).
`SpatialEFEComponents` and `EFEComponents` record both alongside the declared
normalizer name, so the raw term is recoverable from telemetry
(`raw * gamma_m * normalizer_m == stored`). Modality fields stay POST-weighted,
which is what keeps `total` an exact sum of its summands.

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

The canvas, relational, and passage trajectory terms are BELIEF-weighted KL
divergences between encoded terminal/step posteriors and learned transition
priors: each is scaled by the posterior mean of a Gamma precision belief whose
prior mean is the declared config constant
(`canvas_latent_transition_precision`, `relational_transition_precision`). The
canvas and passage-canvas terms share one belief; the relational and
passage-relational terms share another. They remain zero until the relevant
likelihood has received training updates, and a structurally-absent term stays
exactly `0.0` rather than becoming a NaN. Immediate stop uses an identity prior.

Because the KLs are already averaged over their latent dimension by
`_kl_diagonal`, their declared unit is nats per latent dimension
(`nats_per_canvas_latent_dim`, `nats_per_relational_latent_dim`) and no further
normalizer is applied. That denominator differs in kind from the material
modalities' per-cell-channel densities, which is why every modality records its
own normalizer name rather than a single global unit claim.

### 9.4 Motor EFE

Motor pragmatic value is added. State/observation mutual information and the
learned-reliability parameter novelty are each subtracted exactly once, under
separately declared precisions (`motor_proprioceptive_ambiguity_precision` and
`motor_reliability_novelty_precision`); `MotorEFETerms.epistemic_value` is the
logged sum of the two. Motor ambiguity in excess of the preference scale is
recorded for telemetry and is not a summand. The learned per-kind inverse-gamma
reliability belief inflates expected squared execution error and likelihood
variance for fidelity channels.

All three motor terms are additionally scaled by a MODALITY-level weight
`gamma_motor * (1 / channel_count)`. `gamma_motor` is the posterior mean of a
Gamma precision belief seeded at the declared identity constant
`motor_modality_precision = 1.0`, so an unobserved belief reproduces the previous
arithmetic exactly. The divisor is the only genuinely new normalizer in the
model: the three summed terms are raw sums over all 27 named proprioceptive
outcomes, so before it the modality was stated in a different unit from every
material modality. Dividing by 27 is a >order-of-magnitude reweighting relative
to the material modalities, gated by `modality_normalization_enabled`, and it
changes motor realization selection. It is not a tidy-up.

MEASURED. On one representative stroke across the five declared realization
kinds, `G_motor` goes from `-4.4085, 1.4664, 1.0571, -4.4085, -4.4085` to
exactly one twenty-seventh of each, so the modality's cross-kind SPREAD falls
from 5.875 nats to 0.2176 nats -- comparable to the material modalities' measured
per-candidate spreads of 0.19 to 0.33 nats rather than 20x larger. Inside
`motor_realization_log_evidence` at the web runtime's 0.35 policy precision, the
modal realization is UNCHANGED but the conditional motor posterior flattens from
`[0.305, 0.039, 0.045, 0.305, 0.305]` (entropy 1.353 nats) to
`[0.206, 0.191, 0.192, 0.206, 0.206]` (entropy 1.609 nats, against a uniform
maximum of `log 5 = 1.609`). Per-channel normalization therefore makes motor
realization selection nearly indiscriminate at that precision. That is a genuine
cost of the unit convention, not a defect in it, and it must NOT be papered over
by raising `motor_modality_precision` to taste: doing so would be exactly the
hand-tuning this feature exists to remove. The clean response is either a
declared, separately measured motor modality precision or a per-channel
proprioceptive preference set, both out of scope here.

`MotorEFETerms` records the normalizer, its declared name, and the modality
precision it applied. `motor_realization_log_evidence` keeps a plain-float
policy-precision argument; the driver passes the belief posterior mean there.
The no-candidate-count-bonus property of that logsumexp is a normalization
property and is gamma-independent, re-pinned under a swept precision in
`tests/test_stroke_execution.py`.

`tests/test_reference_oracle.py` pins this arithmetic at all six EFE total sites
(`efe._evaluate_mixture`, `efe._evaluate_ensemble_batch`, and the four spatial
evaluators), so a site that keeps the old `+ A_motor` term fails.

### 9.5 Omitted constants and units

The implementation omits several policy-independent differential-entropy and
preference normalizers. Sparse local terms are area-scaled. Summary VFE is
reported in nats; spatial VFE is reported in mean nats per independent
cell-channel.

EFE modalities are reduced to nats PER OBSERVATION CHANNEL, and each modality
records the name of the normalizer it was reduced by. All normalizers are
MULTIPLICATIVE, never additive: an additive offset would be an unowned constant
inside `G` and would break the standing guarantee that a zero-amount extra
stroke costs exactly zero.

| modality | normalizer | declared name |
| --- | --- | --- |
| `terminal_coverage` | 1 (a scalar Beta on one aggregate coverage channel; bounded by the forecast-family restriction below, not by a divisor) | `nats_per_aggregate_coverage_channel` |
| `observation_ambiguity` | 1 (already divided by `independent_material_channel_count * full_area`) | `nats_per_independent_cell_channel` |
| `transition` | 1 (same helpers) | `nats_per_independent_cell_channel` |
| `composition_gap` | 1 (already averaged over `channels * grid * grid`) | `nats_per_cell_channel_all_material_channels` |
| `canvas_latent_transition` | 1 (`_kl_diagonal` already means over the latent dim) | `nats_per_canvas_latent_dim` |
| `relational_transition` | 1 | `nats_per_relational_latent_dim` |
| `motor_proprioceptive` | 1/27 | `nats_per_proprioceptive_channel` |

The composition normalizer's name deliberately says `all_material_channels`.
The compression gap averages over all six material channels, including the two
deterministic ones (register item 3). That is a PRE-EXISTING deviation from
`spatial_state.independent_material_channel_count`, and the gap is earned mainly
on channels 3-5, so restricting it would gut the signal the composition features
are built on. It is named here so the deviation is visible rather than hidden;
scoping a `composition_gap_independent_channels_only` flag is a separate change.

The normalizer choice is declared but NOT neutral. Measured, the precision
learning rule only responds inside a narrow intermediate band of a modality's own
absolute scale (see register item 26), so the normalizer determines which
precisions are learnable at all. No choice of normalizer makes all seven
modalities simultaneously responsive.

Silent domination is guarded by a tripwire in `tests/test_modality_units.py`: on
a 24-candidate blank-canvas set no modality's mean absolute contribution may
exceed 50x the median active modality's, and the tail is recorded separately
(worst single candidate bounded at 400x, measured 329x). The tripwire is not a
tautology: removing the terminal-forecast concentration floor takes the worst mean
ratio from 20.8x to 1592x and the worst single candidate from 329x to 3.8e4x.

AI-103 resolved the spatial factorization and normalization decision. AI-104
and AI-105 are addressed in §14.

## 10. Policy Priors, Proposals, And Posterior

### 10.1 Painting policy prior

The only painting-policy prior explicitly included in the global policy
posterior is the immediate-stop prior. It is now a PRODUCT of two declared
prior factors, hence a sum of two log-sigmoids:

```math
log p_stop(pi) =
    log sigmoid(kappa * (believed_coverage - coverage_midpoint))
  + log sigmoid(-s * E[dGap] / sd[dGap])
```

for immediate stop, and zero log weight for continuation candidates.

The second factor is the gap-progress term. `dGap` is the compression-gap
increment per completed mark, carried as a Gaussian random-walk belief
(`precision_beliefs.GapIncrementBelief`) updated by a scalar Kalman step from
observed gaps; both its posterior mean and its posterior precision enter, through
the standardized mean. As the believed increment approaches zero -- further marks
are no longer buying structure -- the factor rises toward zero and stopping
becomes a priori less unlikely.

Four properties make it a prior rather than a reward, and all four are pinned by
tests:

1. It is bounded above by zero, so it can only make stopping LESS UNLIKELY and
   can never manufacture positive value for any candidate.
2. It is exactly `0.0` for every continuation policy, for an absent belief, for a
   disabled flag, and before the belief's first observation. The pre-existing
   `log(0.5)` identity at the coverage midpoint is therefore preserved verbatim,
   and is additionally re-asserted by decomposition with an observed belief.
3. `dGap` never enters `G`. Its only consumer is `policy_stop_log_prior`, guarded
   by a test that asserts every field of every EFE component is bit-identical
   before and after feeding the belief a large increment.
4. The terminal coverage PREFERENCE is untouched. `target_coverage = 0.87` and
   `terminal_concentration = 110.0` remain declared and un-learned; only their
   PRECISION is a belief, and only the STOP POLICY PRIOR gained a progress term.
   `tests/test_preferences.py` pins that the Beta concentrations are bit-identical
   after enough precision updates to move every gamma.

DECLARED SCOPE ASYMMETRY. In the receding-horizon local-passage scope,
`_local_passage_candidates` already supplies `log1p(-passage_continuation_probability)`
as the stop candidate's `policy_log_priors[0]`, and that is ADDED to
`policy_stop_log_prior`. The local-passage stop prior is therefore a product of
THREE declared factors while the global stop prior is a product of two. This
overlap pre-dates the gap-progress term; it is recorded here rather than left
implicit, and consolidating it is a candidate follow-up.

### 10.2 Candidate proposal distribution

`PolicySampler` constructs a finite candidate set using:

- uniform/random stroke geometry;
- low-coverage start-point oversampling;
- fixed fractions of passage and passage-plan proposals;
- explicit black/white tone alternatives;
- finite planning-depth sampling.

The working tree also contains an in-progress amortized proposal
`q_proposal(z_pi | q(z_canvas), q(z_relational))` implemented by
`proposal.PolicyProposalNetwork`. It defines normalized categorical,
logit-normal, log-logit-normal, and wrapped-normal factors over the declared
mark/passage latent support. After a planning round it is trained by
self-normalized posterior-weighted maximum likelihood toward the existing base
painting-policy posterior

```math
w_i proportional to exp(log p_stop(pi_i) - gamma G_base(pi_i)).
```

This is amortized inference over candidate latents, not a reward and not a new
outcome preference. It is mixed with the hand-written proposal only when
`learned_proposal_mix > 0`; the default is `0.0`. Immediate stop and
passage-plan compounds are outside its learned support, and the hand-written
branch remains present as the same-round control whenever the learned branch is
enabled.

Both the hand-written and learned mechanisms are computational proposal
distributions. Their log densities are not included in the policy posterior,
and no importance correction is applied.
Consequently, proposal frequency affects the finite-budget result. Comments
that call low-coverage or passage proposal frequency an "empirical policy
prior" are not mathematically realized as normalized `p(pi)` terms in the
current posterior. The learned proposal may improve which hypotheses enter the
finite set, but it does not correct that bias. The dedicated unit suite now
covers density normalization/support, empirical sampler agreement, exact
zero-mixture parity, training, checkpoint continuation, and EFE separation.
The 360-cell candidate-count/horizon/seed/mixture audit recorded in
`docs/PROPOSAL_CONVERGENCE_RESULT_2026-08-04.md` produced a negative convergence
result. The accepted M1 interpretation is therefore explicitly
`Q(pi | sampled candidate set S)`: mixture weights are computational budget
splits, and posterior mass is not proposal invariant. Learned emission remains
zero by default. A complete mixed discrete/continuous `P(pi)`, normalized
`r(pi | belief)`, and tested `log P - log r` correction (or a permanently
set-conditional claim boundary) is required before M3.

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
`q(pi)`, not chosen by minimum scalar score. `policies.policy_posterior_from_efe`
is the single implementation of that softmax and is called by both bare-agent
paths and by `tests/test_epistemic_policy_selection.py`, so a test can no longer
pass while exercising a formula production does not use.

POLICY PRECISION IS A BELIEF. `gamma` is the posterior mean of a Gamma precision
belief seeded at `config.policy_precision` (3.0 in the library default, 0.35 in
the web runtime). Those constants are now prior means, not temperatures. The
driver updates the belief after each planning round from the realized `(G, F)`
pair over the non-stop candidate set, using the reference's Chapter 10 rule.

`F_i = -brush_preparation_log_evidence_i`, the negative log marginal evidence of
realizing candidate `i`'s intended mark amount and pigment under the current
brush-load posterior, with the preserve/reload preparation policy exactly
marginalized over its declared prior. Because that marginalization is exact,
`-log Z` is a variational free energy at its optimal variational posterior. Three
properties make it admissible:

- it is built from `brush_policy_precision`, NOT `policy_precision`, so the
  learned gamma never appears on both sides of its own update;
- it is computed for every non-stop candidate BEFORE the forecast-budget sort, so
  it is defined on the full candidate set and does not depend on gamma-dependent
  pruning;
- it enters only the precision gradient, never any candidate's `G` or logit.

`painting_log_evidence`, which folds `-gamma G` in for stop candidates, is
deliberately NOT used as `F` for this reason.

Stop candidates are excluded from the update: they realize no mark, so they have
no realization free energy, and excluding them keeps policy-precision learning
disentangled from the gap-progress stop prior.

FROZEN FORECAST ORDERING. The forecast-budget sort key stays pinned to the
declared constant `config.policy_precision`, never the belief mean. That ordering
is a declared fixed heuristic below the painting-policy boundary; a learned
precision must not choose which candidates receive an expensive execution
forecast, or it would select the evidence set from which it is itself estimated.

BARE-AGENT MODE CANNOT LEARN. `agent.infer_policy` and
`spatial_agent.infer_policy` have no per-candidate policy-dependent `F`
(`last_vfe` is a state-inference VFE for the single realized observation), so they
do not call `observe_policy` and gamma stays bit-identically at the declared prior
mean. Passing `F = 0` there would make the reference report `converged=True` with
gradient exactly `0.0` -- a degenerate no-op dressed as a fixed point.

## 11. Learning

Online learning currently consists of:

- Gaussian NLL training of local/summary transition ensembles;
- offline/shadow beta-one negative-ELBO training of the conditional local
  patch VAE ensemble; it is not used by the online policy loop;
- offline/shadow exact NLL training of a normalized identity-plus-consequence
  local Gaussian mixture ensemble; it is not used by the online policy loop;
- variational-autoencoder-style hierarchy training;
- Gaussian NLL training of aggregate and passage-step latent transitions;
- posterior-weighted maximum-likelihood training of the provisional amortized
  mark/passage proposal toward the existing base-EFE policy posterior;
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
| GP-BRUSH-CONTACT | Generative process | round footprint conditioned on handle incidence plus `p(z_brush,t+1, f_tangent | z_brush,t, delta_x, N, body_pose, tooth)` | `arm_sim.VerticalCanvas.paint_at`, `Brush.update_contact_dynamics`, `ArmKinematics.brush_axis_world` | Axisymmetric angle-dependent footprint plus Baxter-inspired aggregate stop/play approximation; uncalibrated |
| GP-CAMERA | Generative process | native/global/requested-foveal grayscale products | `camera_observation.CameraObservationProcess` | Simulated, uncalibrated |
| GM-SUM-TRANS | Transition likelihood | `p_theta(s_t+1|s_t,a_t,m_t)` | `models.DynamicsEnsemble` | Obsolete compatibility fixture |
| GM-PIX-TRANS | Transition likelihood | local `p_theta(s^P_t+1|s^P_t,a^P_t,m_t)` | `models.LocalSpatialDynamicsEnsemble`, `local_spatial.py` | Learned sparse approximation; bootstrap evidence from `motion_manifold` (declared flag) or the retained iid source |
| GM-PIX-CVAE-SHADOW | Candidate transition likelihood | local `p_theta(s^P_t+1|s^P_t,logvar^P_t,a^P_t,b_t,m_t,z_t)` with `z_t~N(0,I)` | `conditional_patch_vae.ConditionalPatchVAEEnsemble`, `conditional_vae_train.py` | Implemented offline/shadow only; no policy influence; live-scale likelihood, conditioning, calibration, and sequential-rollout gates not yet passed |
| GM-PIX-MIXTURE-SHADOW | Candidate transition likelihood | normalized local identity/consequence mixture `sum_k pi_k(s^P_t,a^P_t,b_t,m_t) N(s^P_t+1; mu_k, Sigma_k)` | `mixture_transition.LocalMixtureDynamicsEnsemble`, `mixture_transition.py` | Leading AI-109 shadow family; improves exact held-out NLL and multistep error, but exact mixture-CDF calibration fails and it has no policy influence |
| GM-SUM-OBS | Observation likelihood | `p(o_t|s_t)` | `models.ObservationModel` | Obsolete oracle compatibility fixture |
| GM-PIX-OBS | Observation likelihood | `p(o^pixel_t|s^pixel_t)` | `spatial_inference.py` | Oracle material, provisional |
| GM-CAMERA-OBS | Observation likelihood | `p(o^gray_t,z^inlier_t|s^pixel_t)` | `camera_inference.CameraSpatialLikelihood` | Analytic occlusion mixture, uncalibrated |
| Q-SUM | Posterior | diagonal `q(s_t)` | `inference.VariationalStateEstimator` | Obsolete compatibility fixture |
| Q-PIX | Posterior | factorized `q(s^pixel_t)` | `spatial_inference.SpatialVariationalStateEstimator` | Analytic diagonal fusion |
| Q-CANVAS | Posterior/likelihood | `q(z_canvas)`, decoder likelihood | `canvas_hierarchy.HierarchicalCanvasModel` | Online diagonal latent; bootstrap evidence is whole episode canvases via `spatial_agent.add_composition_canvas` |
| Q-REL | Posterior/likelihood | `q(z_relation)`, decoder likelihood | `canvas_hierarchy.py` | Deterministic slots plus latent; same whole-canvas bootstrap evidence |
| Q-PASSAGE | Slow posterior | `q(z_passage)` | `passage_inference.PassageBelief` | Mixed pseudo-likelihood |
| Q-BRUSH | Material posterior | `q(load_t)q(black_fraction_t)` per dedicated brush | `brush_loading.BrushLoadingModel` | Compact bounded Gaussian moments; image-derived mark statistic not wired |
| TRANS-BRUSH | Transition likelihood | preserve depletion/uncertainty or pure full reload | `brush_loading.py`, `arm_sim.Brush` | Explicit provisional approximation |
| GM-BRUSH-CONTACT | Required transition/observation likelihood | stochastic posterior and counterfactual over latent tuft deflection, slip, tangential load, and footprint | not yet separated from process | Independent rollouts reuse GP-BRUSH-CONTACT equations; logged process fields are training/evaluation labels, not observations |
| TRANS-ACTION-CAMERA | Inference schedule | `q^-(s_t+1)=Integral p_theta(s_t+1|s_t,a_t,m_t)q(s_t) ds_t`; then `q(s_t+1)` from a causally later camera likelihood | `spatial_inference.predict`, `arm_agent_driver.PendingActionCameraUpdate`, `web_runtime.py` | Mean-evaluated diagonal transition; post-physics capture gate rejects older frames; camera VFE logged separately |
| MODE-PROVISIONAL-SENSOR | Integration profile | registered `q(s_t)`, `q(x_body,t)`, and `q(b_t)` initialize an independently instantiated fixed-prior MuJoCo/material model before EFE rescoring | `arm_agent_driver._planning_context`, `web_runtime.py` | Opt-in simulation-only: 8 candidates, depth 1, Cartesian IK support, 1 particle; exact live planner state denied; fixed grain/model/compliance priors are uncalibrated |
| PREF-COVERAGE | Prior preference | `p*(C_T|stop)` | `preferences.py`, `efe_common.py` | Explicit terminal Beta |
| PREF-COMP | Prior preference | energy from compression gap | `canvas_hierarchy.py`, `composition.py`, `spatial_efe.py` | Closed loop relocated, not removed: the initial stream is now the agent's own embodiment (flag-gated); best-of-family baseline. Definition unchanged |
| PROP-BOOT | Proposal only (generative process) | body-feasible bootstrap mark proposals | `motion_manifold.MotionManifoldSampler` | Below the painting-policy boundary; declared flag `bootstrap_generator`; supplies no preference and selects no policy |
| PREF-MOTOR | Prior preference | homeostatic outcome densities | `motor_planning.motor_efe_terms` | Explicit diagonal approximation; read-only, never learned from outcomes |
| PRIOR-STOP | Policy prior | `p(stop-first|coverage belief)` | `policies.policy_stop_log_prior` | Explicit |
| PRIOR-PASSAGE | Transition prior | `p(z_passage_r+1|z_passage_r)` | `PassageBelief.transition_log_prior` | Explicit local prior |
| PRIOR-MOTOR | Policy prior | uniform `p(m|pi)` | `motor_planning.py` | Explicit |
| PRIOR-BRUSH | Policy prior | `p(preserve/reload)` | `brush_loading.py` | Explicit |
| PROP-PAINT | Proposal only | finite hand-written/learned `q_proposal(z_pi|belief)` | `policies.PolicySampler`, `proposal.PolicyProposalNetwork`, `policy_ranges.py` | Uncorrected finite proposal; learned emission defaults to zero; AI-111 accepts only `Q(pi | sampled set S)` after negative convergence evidence |
| EFE-PIX | Expected free energy | risk/ambiguity hierarchy | `spatial_efe.py` | Approximate, decomposed |
| EFE-MOTOR | Expected free energy | proprioceptive EFE: pragmatic `-` `I(s;o)` `-` `N_reliability` | `motor_planning.py`, `efe.py`, `spatial_efe.py`, driver | Approximate, decomposed; three components separately logged, excess-entropy ambiguity logged but not summed |
| EFE-BRUSH | Expected free energy | conditional material/pigment risk and ambiguity | `brush_loading.py`, driver | Approximate, decomposed |
| GAMMA-EFE | Precision belief | `q(gamma_m) = Gamma(alpha, beta)` per EFE modality | `precision_beliefs.GammaPrecisionBelief`, `PrecisionLedger` | Reference Ch.10 rule, byte-parity verified; declared bounded support; measured nearly inert (register 26) |
| GAMMA-POLICY | Precision belief | `q(gamma_pi)` scaling `G` inside `Q(pi)` | `precision_beliefs.PrecisionLedger`, `policies.policy_posterior_from_efe` | Prior mean is the previous constant exactly; learned only on the driver path |
| VFE-BRUSH-F | VFE term | `F_i = -log p(mark amount, pigment | pi_i)` marginalized over preparation | `brush_loading.infer_preparation`, driver `_observe_precision_beliefs` | The single policy-dependent free energy driving every precision update |
| UNIT-MODALITY | Unit declaration | per-observation-channel normalizers | `precision_beliefs.NORMALIZER_NAMES`, `ModalityWeights` | Declared, multiplicative only; recorded per component |
| PRIOR-GAP | Transition prior + policy prior | Gaussian random walk on per-mark `dGap`; `logsigmoid(-s z)` factor on `p(stop)` | `precision_beliefs.GapIncrementBelief`, `policies.policy_stop_log_prior` | Explicit; structurally barred from `G` |
| SAFE | External constraint | hard feasibility and stop | plant, controller, future hardware safety | Not active inference |

## 13. Approximation Register

The following are explicit blockers or limitations, not hidden implementation
details:

1. Oracle diagnostic mode still uses exact simulator material state as its
   observation; the default sensor mode uses the analytic camera likelihood but
   remains blocked from live control by copied substrate-grain/contact/model forecast
   context and collapsed brush history. The action-conditioned transition
   prior/camera schedule is implemented, including capture-time gating and
   automatic MuJoCo delivery polling. MuJoCo q/qvel, independent material fields, and
   compact brush initialization are posterior-conditioned when their respective
   snapshots are supplied; brush microstructure uses independent prior noise.
   The opt-in `provisional-sensor-simulation-v0` profile removes the live
   process container from policy forecasts by using a fresh MuJoCo/material
   template with independent grain/brush seeds. It is a bounded integration
   approximation, while the default remains fail-closed.
2. Summary and pixel posteriors are diagonal Gaussian.
3. Derived channels may be double-counted before deterministic projection. The
   compression gap and both baseline members are computed over all six material
   channels, so `ground_contrast` and `material_coverage` carry independent
   likelihood evidence there even though they are deterministic functions of
   channels 0-3; the local member's advantage concentrates on exactly those
   high-amplitude derived channels (measured per-channel log-likelihood on band
   canvases: channels 0-2 pinned near the 2.99 ceiling, channels 3/4/5 at
   2.02/2.26/1.91).
4. Transition moments are propagated at posterior means.
5. Ensemble disagreement is only an approximate parameter posterior.
6. Local sparse rollouts omit outside-support entropy constants.
7. Pointwise nondecreasing black mass conflicts with wet-paint transport.
8. Composition preference is trained on the developmental stream it evaluates.
   The stream's origin is now the agent's own embodiment rather than an arbitrary
   iid mark source (`motion_manifold`, behind `config.bootstrap_generator`), which
   RELOCATES the circularity but does not remove it: the structural prior is still
   evaluated by a code fitted on the agent's own history.
9. Candidate proposal frequency is not corrected in policy inference.
10. Hierarchy likelihoods activate after small online sample counts and are not
    yet calibrated.
11. Passage observations mix executed action geometry with material evidence.
12. Motor outcome density is diagonal and forecasted by the same broad process
    family used for execution. `MotorEFETerms.ambiguity` is likelihood entropy
    measured against the declared preference scale, not in absolute nats, so it
    is not the reference implementation's ambiguity term and is logged rather
    than summed into `G`.
    Brush-contact prediction has the same limitation: independent rollout
    state prevents live-state/RNG leakage, but the rollout still invokes the
    same reduced contact equations as the process. A learned/calibrated
    stochastic `GM-BRUSH-CONTACT` must replace that reuse; exact deflection,
    stick/slip, tooth, and tangential-force fields remain labels rather than
    agent observations.
13. Motor refinement covers only a finite subset of base painting candidates.
14. Neural parameter learning is SGD, not exact variational parameter
    inference.
15. The web process and planner can use different canvas resolutions.
16. Continuous-density constants and modality reductions require independent
    unit/sign verification.
17. The camera update linearizes the nonlinear grayscale likelihood and drops
    posterior cross-covariance between thickness and surface tone.
18. The camera occlusion factor is a mean-field inlier/outlier mixture with
    provisional XML-declared precisions, not a learned occlusion model.
17. Brush preparation is analytically marginalized, but only its modal policy
    is sent through the expensive motor/material rollout.
19. Motor risk is declared to be the PRAGMATIC term `E_q[-log p*(o)]` rather
    than a KL. This is a deliberate choice, not an oversight: the omitted
    `-H[q(o)]` term is logged separately as
    `MotorEFETerms.forecast_outcome_entropy`, so the canonical KL form is
    derivable from telemetry, and `G_motor = pragmatic - I(s;o) - N_reliability`
    is exactly equivalent to `KL risk + ambiguity - novelty`. The field keeps the
    name `risk` because it is part of the driver's `asdict` telemetry wire
    format; a rename would be a separate coordinated change across the
    dataclasses, driver payload, viewer, and tests.
20. `brush_loading.py` computes its material EFE with the same non-canonical
    *shape* (risk as the pragmatic term, ambiguity as `0.5 * sum(log1p(var /
    preference_var))`) but subtracts no mutual information, so it carries NO
    double count and is deliberately unchanged. Do not "harmonize" it by
    symmetry with the motor fix: doing so would introduce a defect where none
    exists.
21. The reference-oracle harness (`tests/test_reference_oracle.py`) discretizes
    continuous Beta densities onto uniform coverage bins by the midpoint rule in
    order to compare against the discrete-only reference. The reference floors
    both logs at `1e-16`, so the discretized KL diverges from the continuous KL
    where the forecast puts mass on numerically-zero preference density, and the
    error GROWS with bin count. The oracle is therefore only valid for coverage
    means in `[0.70, 0.90]` (measured: 5.8e-5 at mean 0.87, but 7.7e-2 at 0.60
    and 4.8e+1 at 0.30). Terminal risk in the low-coverage cold-start regime is
    NOT certified by that test.
22. The summary VFE `total` reported by `inference.VariationalStateEstimator`
    remains a Monte Carlo expectation whose error does not shrink with more
    optimizer steps. Its reporting-only budget is now the declared
    `summary_vfe_report_samples=4096`, raised from the previous implicit 32.
    Measured per-seed deviations from fine-grid integration over seeds 0-4 are
    +0.00613, -0.02316, -0.01272, -0.00398, and +0.01796 nats (maximum absolute
    error 0.02317); the accepted band is +-0.05 nats. Its `complexity` term is
    closed form and verified to 1e-16. The diagnostic draw restores the global
    CPU/CUDA RNG state on exit; this budget therefore cannot change q(s), later
    stochastic learning, EFE, preferences, or policy selection.
23. The compression-gap baseline is an unnormalized best-of-family maximum over
    a two-member hand-written code family, not a normalized Bayesian model
    average. It omits the `log(family size)` model-index cost (`log 2 /
    (channels*grid*grid)` = 0.00045 nats per cell-channel at 6x16x16) and is
    non-differentiable at member ties, which is safe only because the gap is
    evaluated exclusively under `torch.no_grad`. Blank canvases sit exactly on a
    tie (both members measured at 2.993084 nats per cell-channel). A normalized
    `logsumexp(log p_iid, log p_local) - log 2` with a declared uniform model
    prior is the strictly more principled formulation and differs by at most
    that same 0.00045 nats.
24. GAMMA-POLICY / GAMMA-TERMINAL / GAMMA-AMBIG / GAMMA-TRANS / GAMMA-COMP /
    GAMMA-CANVAS / GAMMA-REL / GAMMA-MOTOR. The eight precisions
    (`policy_precision`, `terminal_risk_precision`, `ambiguity_precision`,
    `transition_precision`, `composition_gap_precision`,
    `canvas_latent_transition_precision`, `relational_transition_precision`, and
    the new identity-valued `motor_modality_precision`) are the posterior means of
    Gamma(alpha, beta) beliefs updated by the reference Chapter 10 rule
    `dF/dgamma = (alpha/gamma - beta0) + (pi - pi0).(-G)`, `beta <- max(beta -
    kappa dF/dgamma, eps)`. `precision_beliefs.precision_gradient` and
    `learn_precision` are verified byte-parity against
    `active_inference.core.pomdp` at `alpha = 1` in
    `tests/test_precision_beliefs.py`. Each belief's rate is seeded at
    `beta0 = alpha0 / (declared constant)`, and an unobserved belief returns that
    constant BIT-IDENTICALLY, so `precision_beliefs_enabled=False` and "belief with
    no observations" are the same arithmetic and attribution is exact rather than
    approximate. Three orthogonal declared flags keep the hand-written mechanism
    available: `precision_beliefs_enabled`,
    `modality_precision_beliefs_enabled`, `modality_normalization_enabled`.
    Approximations inside the rule: the Gamma SHAPE `alpha` is declared and never
    updated (so `resolvable_uncertainty()` is a constant and is forbidden from
    every EFE path by contract and by test); the rate is WARM-STARTED at the
    current posterior rate rather than re-initialized at the prior rate each
    round, so evidence accumulates across planning rounds; and the posterior mean
    is clamped to a DECLARED BOUNDED SUPPORT of `[0.1x, 10x]` the prior mean.
25. BOUNDED SUPPORT IS A CHARTER REQUIREMENT, NOT A CONVENIENCE. Measured on a
    24-candidate contribution vector with std 6.9 nats, disagreeing evidence
    (`F = -0.5 G`) drives the unbounded precision to 0.0679 -- a 15x attenuation.
    Applied to `terminal_coverage` that is a 15x attenuation of the declared C
    matrix inside `G`, i.e. outcome data switching a declared preference most of
    the way off. Unbounded it also breaks the JSON contract the viewer depends on
    (`1/eps` is ~1e300, `1/0.0` is inf). The clamp is what stops a precision belief
    becoming a backdoor around "preferences are never learned from outcomes".
26. MEASURED NEGATIVE RESULT: PRECISION ORDERING DOES NOT TRACK
    DISCRIMINATIVENESS. The stated hope was that a composition modality whose
    contributions become discriminative across candidates would gain precision
    while the arrangement-blind terminal-coverage modality lost it, out of the same
    rule. It does not happen, and no `beta0` was tuned to manufacture it. Measured
    over six real planning rounds (spatial driver, 16 candidates, `F` = the
    brush-preparation VFE): `terminal_coverage` gamma 1.0003 with by far the
    LARGEST contribution spread (std 196.3); `composition_gap` gamma 0.9859 with
    nearly the smallest spread (0.190); `transition` gamma 0.8805 (spread 0.331);
    `observation_ambiguity` gamma 1.0106 (spread 0.0314); `motor_proprioceptive`
    gamma 0.9973 (spread 0.0301); `policy` gamma 0.3513 against a 0.35 prior.
    Cause, measured directly: the rule responds to F/G AGREEMENT only inside a
    narrow intermediate band of the modality's OWN ABSOLUTE SCALE, and saturates to
    exactly the prior mean outside it. Sweeping one fixed 24-element shape across
    scale with fixed agreeing `F = 0.5 G`, gamma is 1.0006, 1.0112, 1.0814, 1.4602,
    1.4823, 1.0880, 1.00000, 1.00000 at contribution std 0.034, 0.144, 0.357,
    0.721, 1.44, 3.56, 14.4, 144. The response is NON-MONOTONE in scale, peaking
    near std ~1. A flat or extremely large modality's precision is PINNED at its
    prior mean, never driven down. Consequence: this item and the unit
    normalization of §9.5 are in tension -- moving terminal coverage into the
    responsive band would give it a HIGHER gamma than composition and reverse the
    hoped-for ordering. A mechanism that genuinely rewarded discriminativeness
    would be a different inference problem (e.g. a belief over each modality's
    LIKELIHOOD precision fitted to observed per-modality prediction error), not
    Chapter 10's policy-precision rule, and should be scoped as its own feature.
    The honest deliverable here is the mechanism plus this measurement.
27. THE RULE IS PROVABLY INERT ON A FLAT `F`. `precision_gradient(G, 0, 1.0,
    beta0=1.0)` is exactly `0.0`, and `learn_precision(G, 0, beta0=b0)` returns
    `gamma = 1/b0` with `converged=True` -- a degenerate no-op that LOOKS like a
    learned fixed point. `PrecisionLedger` therefore reports such rounds with
    status `degenerate_flat_F`, never `updated`, and the bare-agent
    `infer_policy` paths (which have no per-candidate `F`) do not call
    `observe_policy` at all rather than passing `F = 0`.
28. F = -BRUSH-PREPARATION LOG EVIDENCE. The policy-dependent free energy driving
    every precision update is
    `F_i = -brush_inference[i].log_evidence`. It is an exact marginalization over
    the two-element preserve/reload preparation prior, hence a VFE at its optimal
    variational posterior. It is built from `brush_policy_precision`, not
    `policy_precision`, and is computed for every non-stop candidate BEFORE the
    forecast-budget sort, so no learned gamma appears on both sides of its own
    update and no gamma-dependent pruning selects the evidence set.
    `painting_log_evidence` was rejected as `F` for exactly that reason. This `F`
    is largely a function of a candidate's amount and tone, so it is only weakly
    correlated with the canvas-level modality rankings -- which is the second
    reason the measured drift in item 26 is small.
29. FROZEN FORECAST-ORDERING KEY. `_infer_policy_with_execution_forecasts` and
    `_infer_spatial_policy_with_execution_forecasts` sort candidates for the motor
    forecast budget by `config.policy_precision * G - brush_log_evidence`, reading
    the DECLARED CONSTANT and never the belief mean. The ordering is a declared
    fixed heuristic below the painting-policy boundary; a learned precision
    selecting which candidates get evaluated would be a precision belief reaching
    above that boundary.
30. GAP-PROGRESS. The per-mark compression-gap increment is a Gaussian
    random-walk belief updated by a scalar Kalman step, consumed ONLY by the stop
    policy prior (§10.1). Named approximation: the increment is AMORTIZED over the
    exact number of marks elapsed between planning-cadence gap readings, because
    `belief_composition_gap` runs a model forward and reading it at mark completion
    would put that forward on the polling thread. This is a deliberate deviation
    from a literal per-mark observation: the sampling is coarse but the denominator
    is exact. A second reading at the same mark index is composition-model TRAINING
    drift, not a per-mark increment; it advances the anchor but is not counted, so
    learning progress is never attributed to mark-making.
31. COV-BETA-FLOOR. `coverage_beta_approximation` restricts the moment-matched
    terminal forecast `q(C_T | pi)` to the INTERIOR-UNIMODAL Beta family
    (`terminal_forecast_concentration_floor = 1.0`, both concentrations >= the
    floor, rescaled by a COMMON factor so the forecast mean is preserved exactly
    and the term stays an exact Beta-Beta KL). It is a declared forecast-family
    restriction, not a clamp on risk, and it is measurably inert on every
    well-conditioned forecast: (mean 0.05, var 1e-4) 246.6508 -> 246.6508;
    (0.05, 1e-8) 250.5407 -> 250.5407; (0.5, 1e-6) 37.2720 -> 37.2720;
    (0.87, 1e-8) 4.4849 -> 4.4849; (0.87, 1e-4) 0.7181 -> 0.7181. It bites only
    where `alpha < 1` or `beta < 1`, i.e. where the moment match implied a boundary
    spike the Beta family cannot honestly represent: at (mean 1e-4, std 2.29e-3)
    the `digamma(alpha -> 0)` singularity gives 53248.18 nats and the floored
    forecast gives 892.24. It therefore CHANGES BEHAVIOUR exactly in the
    low-coverage cold-start regime, so which policy wins at low coverage can
    change; `terminal_forecast_concentration_floor = 0.0` restores the previous
    family exactly and the difference must be A/B measured before being treated as
    settled. Note also that with the floor active on a fully blank canvas every
    candidate's forecast saturates at `alpha = 1`, so terminal risk becomes
    scale-invariant there (measured identical at belief logvar -4, -6, -8, -10).
32. Learned precision and gap-increment state are persisted in the driver
    checkpoint PAYLOAD (`precision_ledger`, `gap_increment_belief`) and NOT in
    `_checkpoint_architecture_metadata`, which is compared with strict inequality
    -- a learned quantity there would discard every trained model and replay buffer
    on disk the moment it moved. A pre-Feature-C checkpoint loads with status
    `loaded` and fresh beliefs. `composition_enabled` was split out of
    `composition_gap_precision` for the same reason: the architecture key must stay
    a declared constant, and a learned precision must never construct or destroy a
    model. The split evaluates identically for every pre-existing config, so no
    checkpoint on disk is invalidated. `PrecisionLedger.__post_init__` creates all
    eight keys EAGERLY because updates run on the planner thread while
    `summary()` runs on the HTTP thread.
33. MANIFOLD SWEEP INTEGRATION IS FIRST-ORDER, AND THE LATENT FIT IS LOSSY.
    `motion_manifold.MotionManifoldSampler._integrate` holds the brush tip on the
    canvas plane by numerically projecting a fixed joint-velocity direction onto
    the null space of the tip-depth gradient. It is a first-order projection
    re-evaluated per integration step, not an exact constrained integration, and
    the resulting arc is then re-expressed as a constant-turn equal-segment
    polyline `PassageLatent`, which is a lossy fit of the true FK path
    (`policies._polyline_relative_vertices` clips total length to `[0.04, 0.86]`
    and turn to `+-1.2` rad). The emitted marks are therefore the LATENT's arc,
    not the body's exact arc. That is the correct trade -- it keeps the latent
    self-consistent for `infer_passage_observation` and `PassageBelief`, which a
    best-fit latent glued onto raw FK segments would not -- but it does attenuate
    the very curvature the generator exists to inject.
34. SWEEP SPEED DOES NOT REACH THE CANVAS. Emitted marks are re-decoded from the
    latent and re-timed by `stroke_execution.adaptive_stroke_timing` inside
    `execute_stroke_action`, so a sweep's velocity profile is discarded.
    `bootstrap_manifold_step_degrees` is therefore an exploration/discretization
    range, not a realized mark speed: it changes which sweeps survive the contact
    band, hence the emitted length distribution, and nothing else. Folding speed
    into `amount` would be illegal -- `amount` is a deposition quantity feeding the
    brush material preference, not a kinematic one -- so the limitation is named
    rather than faked.
35. BOOTSTRAP COMPRESSION-GAP PROBES ARE MEASURED OFF-SCALE AND ON A PARTIAL
    CHANNEL BUDGET. `diagnostics()["compositionBootstrap"]` measures the gap on
    16x16x6 fields taken from a `canvas_size=64` bootstrap simulator while the
    live simulator paints at 256, so mark-to-cell scale differs between the
    bootstrapped structure and live marks. The gap is also earnable only on
    channels 3-5 (`SIGMA_FLOOR = 0.02` saturates the flat baseline member on
    thickness-like channels living at ~0.002-0.008), and channels 4-5 are
    deterministic functions of 0-3 (register item 3), so the achievable margin is
    capped regardless of how structured the sweeps are. The block is EVIDENCE
    ONLY: no EFE term, VFE term, preference, precision belief, policy prior, or
    policy posterior reads it, and it carries `declaredAs` saying so.
    `bootstrap_composition_train_steps` is a GRADIENT BUDGET, not an objective
    term: it appears in no objective, and at fixed parameters it cannot change
    which policy wins. Two measured properties of the probe set: (a) the declared
    acceptance margin is bounded by the TIGHTER of two null models,
    `max(gap(blank), gap(cell-shuffled))`, and measured at 2700 gradient steps
    the binding one is BLANK (-0.05) rather than the marginal-preserving shuffle
    (-74), because a well-trained code scores the shuffle as strongly
    out-of-distribution -- a shuffle-only criterion would report a margin earned
    by overconfidence; (b) the raw probe range is reported but is NOT a criterion
    for the same reason (measured 296 vs 147 across the two bootstrap generators,
    driven almost entirely by how confidently negative each model is about the
    iid-scatter probe). Cost, measured: 330-450 ms per composition gradient step
    at `batch_size = 32`, dominated by `relational_observations`' per-sample numpy
    round-trip, which is why the budget defaults to 0.
36. AMORTIZED POLICY PROPOSAL. `PolicyProposalNetwork` is a factorized
    recognition density over mark and passage latents conditioned on flattened
    canvas and relational posterior means. Its training target is the
    normalized base-EFE posterior over the finite candidates already generated,
    so it inherits any misspecification and truncation in that posterior. It has
    no density for immediate stop or compound passage plans, and continuous
    factors are conditionally independent given the shared features. The
    hand-written and learned branches share deterministic decoders, but that
    cancellation does not remove finite-candidate bias or the self-training loop
    created when future training support is partly supplied by the learned
    proposal itself. `learned_proposal_mix=0.0` keeps emitted candidates on the
    hand-written baseline by default. `tests/test_proposal.py` now checks the
    normalized/support-bounded densities, hand-sampler agreement, mixture
    attribution, posterior-only objective, fallback refusal, checkpoint
    continuation, and no EFE change under the default gate.
    `tests/test_proposal_convergence.py` adds an exact equal-EFE multiplicity
    control plus a fixed 360-cell spatial grid. Stop posterior mass and
    deep-horizon winning geometry did not converge under the tested budgets.
    The accepted approximation is a posterior conditional on the sampled set,
    not a proposal-invariant posterior over continuous painting policies.

## 14. Required Next Decisions

This specification records the following current decision state:

- `AI-102`: ANSWERED by `docs/VARIABLE_SENSOR_ACCESS_LEDGER.md` and its
  machine-readable inventory. Replacing the privileged paths remains M2 work;
- `AI-103`: ANSWERED by `docs/OBSERVATION_FACTOR_AUDIT.md` and the focused
  factorization/unit tests;
- `AI-104`: ANSWERED by `tests/test_reference_oracle.py` and
  `docs/REFERENCE_MODEL_ACCEPTANCE_2026-08-04.md`. Scalar and multivariate
  conjugate fixtures verify posterior moments, every VFE component, analytic
  minimization, four-order precision behavior, and the local outside-patch
  identity prior. The spatial estimator's complexity, NLL, and total match
  independent fine-grid integration to 1e-6 (measured 4.6e-9). The summary
  estimator's complexity matches the analytic Gaussian KL to 1e-4 (measured
  1e-16), and its declared 4096-sample total is accepted inside +-0.05 nats
  (measured maximum absolute error 0.02317; register item 22);
- `AI-105`: ANSWERED by the same acceptance record and harness.
  Terminal coverage risk equals the reference full-KL risk to 1e-3 for coverage
  means in `[0.70, 0.90]` (measured worst case 5.8e-5; see register item 21 for
  the band's limits). The transition decomposition satisfies
  `R_transition + A_transition = -I_theta` exactly, and the project's
  `epistemic_value` is confirmed NOT to be subtracted a second time — unlike the
  reference's Dirichlet parameter novelty, which is a different quantity and is.
  The policy posterior equals `policy_posterior_full` to 1e-6 (measured 2.2e-8)
  for both painters. The harness also found and forced the fix of a real defect:
  the motor modality double-counted ambiguity at all six EFE total sites. An
  enumerated matrix now isolates deterministic, ambiguity-only, purely
  epistemic, and preference-dominated controls and verifies posterior
  normalization with and without policy priors. Low-coverage terminal behavior
  is assigned to AI-106; the unnormalized self-trained composition preference
  remains explicitly deferred to AI-110 rather than being accepted here;

- `AI-107`: ANSWERED FOR THE CURRENT LOCAL LIKELIHOODS, WITH A NEGATIVE M2
  CALIBRATION RESULT. Validation-only scalar variance scaling left both CNN
  and cVAE nominal 90% intervals at about 99.4% empirical coverage; an action-
  footprint-only diagnostic confirmed the failure. A fixed-condition CNN's
  ensemble disagreement increased only 1.087x on held-out dynamic-roll
  conditions versus the declared 1.50x gate. Learned likelihood, cVAE latent,
  ensemble, target-posterior, fixed camera likelihood, fixed identity
  likelihood, and EFE precision terms are reported separately. All offline
  precision-ledger entries were unobserved declared priors. AI-109 subsequently
  tested the data/capacity and likelihood-family hypotheses;
- `AI-109`: LOCAL BRANCH MEASURED; HIERARCHY BRANCH ACTIVE. Twenty-seven
  three-seed local runs show modest, non-monotonic data sensitivity and no
  generic CNN capacity benefit. Five ensemble members improve density over
  one. The material-posterior cVAE does not materially improve the CNN and
  compounds recursive error; this does not test the proposed visual cVAE. A
  normalized identity/consequence mixture improves mean exact test
  NLL by 1.392 nats and retains 0.0738 eight-mark RMSE, but validation-scaled
  50%/90% intervals cover 90.60%/95.75% and every seed fails calibration. It
  remains shadow-only. Hierarchy learning curves are structurally unavailable
  because all accepted corpus endpoints are fixed-horizon truncations, not
  policy-selected terminal paintings;
- `AI-110`: whether the composition preference can be retained;
- `AI-111`: ANSWERED FOR THE CURRENT M1 BASELINE, WITH A NEGATIVE CONVERGENCE
  RESULT. Hand-written and learned proposals are explicitly separate from
  policy priors; mixture weights are computational choices; and the current
  posterior is accepted only as `Q(pi | sampled candidate set S)`. The
  learned-proposal mix remains zero. M3 must implement and validate a complete
  base-measure/prior/proposal correction before making proposal-invariant
  posterior claims;
- `AI-112`: what learned and episodic state may persist between runs.

No claim of principled sensor-based active inference should pass the M1 gate
until those decisions have evidence.
