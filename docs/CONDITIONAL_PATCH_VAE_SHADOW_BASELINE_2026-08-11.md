# Conditional Patch-Transition VAE Shadow Baseline

Date: 2026-08-11

Status: implemented offline/shadow experiment; not policy-active

Canonical model identifier: `conditional-local-material-transition-cvae-v0`

## 2026-08-12 attribution correction

This experiment is **not** the first implementation of the owner's original
VAE proposal. That proposal was an action-conditioned visual mark-consequence
model operating on fresh pre/post image patches. This experiment instead
predicts coarse material-posterior channels. Its measurements remain valid for
that declared target, but they do not test, implement, or reject the proposed
visual VAE. The canonical distinction is
`VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.

## Decision

The project contains a conditional variational autoencoder (cVAE) ensemble as
an experimental learned local material-transition likelihood. The experiment
is intentionally narrower than the eventual painting hierarchy:

- it models possible local material consequences of a selected mark;
- it does not decide which part of the painting is disordered;
- it does not define compositional order;
- its reconstruction error is not an aesthetic reward or prior preference;
- it does not yet replace `LocalSpatialDynamicsEnsemble` or the physics-based
  counterfactual path used by the live planner.

This separation preserves the governing active-inference boundary. The cVAE
is a candidate likelihood in the generative model, not an independently named
controller or reward.

## Factorization

For one local action-supported patch, each bootstrap member implements

```math
p_{\theta_e}(x^P_{t+1}
  \mid x^P_t,\log\sigma^{2,P}_t,a^P_t,b_t,m_t,z_t),
\qquad z_t \sim \mathcal{N}(0,I),
```

with recognition density

```math
q_{\phi_e}(z_t
  \mid x^P_t,\log\sigma^{2,P}_t,
  x^P_{t+1},\log\sigma^{2,P}_{t+1},a^P_t,b_t,m_t).
```

Here:

- `x` is the camera-likelihood-derived material posterior mean, not exact
  process material;
- the current and next posterior log variances are retained as recognition and
  conditioning inputs rather than discarded;
- `a` is the rasterized selected mark (footprint, start/end, width, amount,
  tone, and curvature);
- `m` is the conditional motor realization represented by the existing motor
  channels in the action raster;
- `b` is a five-field compact pre-stroke brush condition: inferred load mean,
  load standard deviation, inferred black-fraction mean, black-fraction
  standard deviation, and an availability bit;
- `z` represents outcome variation not fixed by those declared conditions.

The decoder predicts a diagonal Gaussian over the next material-posterior mean
and then applies the existing material-support projection. Only independent
material channels enter the reconstruction likelihood; derived contrast and
coverage remain deterministic transforms.

## VFE / negative ELBO

Training uses the standard beta-one objective. For each patch:

```math
F =
  -\mathbb{E}_{q_\phi(z)}
    [\log p_\theta(x^P_{t+1}\mid x^P_t,a^P_t,b_t,m_t,z)]
  + D_{KL}[q_\phi(z)\|\mathcal{N}(0,I)].
```

The implementation records separately:

- reconstruction negative log likelihood in nats;
- latent KL in nats;
- their sum, the negative ELBO / variational free energy;
- VFE divided by the number of valid independent patch elements for batch
  optimization and cross-size reporting.

Padded cells are excluded by an explicit validity mask. The Gaussian
normalization constant is retained, so reported density units are not the
constant-omitting convention used by the older local CNN report.

## Corpus and sensor-access boundary

`trajectory-posterior-corpus-v2` extends the whole-trajectory corpus with the
compact inferred pre-stroke brush posterior. The driver freezes the belief
that existed after any selected reload/preserve preparation and before stroke
depletion. The record contains no exact held-paint amount or bristle
microstructure.

Legacy `trajectory-posterior-corpus-v1` shards remain readable. Their four
brush fields are zero and their fifth availability field is zero. This denotes
missing context; it must never be interpreted as observed zero brush load.

The cVAE corpus adapter names every accepted field and rejects any shard that
does not explicitly declare
`process_truth_used_as_training_input = false`. It only extracts local patches
after the manifest has split complete trajectories into train, validation,
and test sets.

Current accepted inputs are:

1. current camera-derived material posterior mean and log variance;
2. selected painting action;
3. conditional motor realization;
4. optional compact inferred pre-stroke brush posterior;
5. causally later camera-derived material posterior mean and log variance.

Forbidden inputs include exact simulator canvas/contact state, exact brush
reservoir state, bristle microstructure, and future policy outcomes.

## Uncertainty decomposition

The prior-predictive ensemble reports three different quantities:

1. `likelihood_variance`: decoder Gaussian variance conditional on one member
   and one latent sample;
2. `latent_variance`: variation between multiple `z` samples inside a fixed
   member;
3. `epistemic_variance`: disagreement between bootstrap member means after
   marginalizing their finite latent sample sets.

The first two sum to `outcome_variance`; all three sum to `total_variance`.
This is a finite-Monte-Carlo approximation. It is more informative than
calling every stochastic effect "aleatoric," but it is not proof that the
latent has discovered physically meaningful brush modes.

## Shadow capability report

`active_painter.conditional_vae_train` writes a separate checkpoint and JSON
report. It evaluates validation and test trajectories without using either for
gradient updates. The report contains:

- mean member negative ELBO per valid element;
- reconstruction NLL per valid element and latent KL per patch;
- importance-weighted held-out NLL estimate;
- paired correct-versus-shuffled action checks;
- paired correct-versus-motor-channel-ablated checks;
- a shuffled-brush check when at least two valid brush contexts exist;
- mean likelihood, latent, and epistemic variances;
- approximate 50% and 90% coverage under a moment-matched Gaussian.

Condition ablations reuse paired random samples. A positive
`ablation_minus_correct` gap is evidence that the learned held-out density uses
that condition. A zero or negative gap fails that capability check; the code
does not reinterpret failure as success.

Run:

```powershell
python -m active_painter.conditional_vae_train `
  --manifest runs/corpus/split_manifest.json `
  --output-checkpoint runs/checkpoints/conditional-patch-cvae.pt `
  --report-path runs/checkpoints/conditional-patch-cvae.report.json `
  --device cuda `
  --ensemble-size 3 `
  --latent-dim 16 `
  --gradient-steps 2000
```

The default is 16 latent dimensions, three bootstrap members, 32 hidden
channels, two residual blocks, and variable masked patch sizes. These are
engineering starting points, not identified model dimensions.

## Admission gates before policy use

The model must remain shadow-only until a manifested, live-scale corpus shows:

1. improved validation/test importance-weighted NLL versus the existing local
   Gaussian CNN and an identity/local-deposition baseline;
2. positive held-out action, amount/tone/curvature, brush-load, and motor
   conditioning gaps in strata where those variables actually vary;
3. uncertainty calibration reported by condition and not only in aggregate;
4. epistemic disagreement that falls with in-support data and rises on
   deliberately out-of-support actions;
5. physically plausible samples after material projection without posterior
   collapse or a decoder that ignores `z`;
6. stable sequential multi-mark rollout error, not only one-step patch fit;
7. an explicit EFE integration design showing which cVAE outputs are
   likelihood, ambiguity, and epistemic-value terms.

Passing these gates would justify a separate integration decision for this
material-posterior model. It would not, by itself, establish the later
mesoscopic order/compressibility model. That later layer must explain whether
a patch is economically described
as one or several coherent brush events and whether it is compatible with
slower painting structure.

## Implemented files

- `src/active_painter/conditional_patch_vae.py`: corpus adapter, masked cVAE,
  beta-one VFE, importance-weighted density estimate, and ensemble uncertainty;
- `src/active_painter/conditional_vae_train.py`: offline training, paired
  capability ablations, calibration report, and isolated checkpoint;
- `src/active_painter/trajectory_corpus.py`: v2 optional brush-posterior fields
  and v1 compatibility;
- `src/active_painter/arm_agent_driver.py`: pre-stroke brush-belief propagation
  to observation-only corpus callbacks;
- `tests/test_conditional_patch_vae.py`: probabilistic, conditioning,
  uncertainty, access-boundary, callback, and checkpoint tests.

Focused verification on 2026-08-11: 10 tests passed in 3.99 seconds across
`tests/test_conditional_patch_vae.py` and `tests/test_trajectory_corpus.py`.
The full driver suite exceeded a two-minute command budget without producing a
failure traceback; this is not counted as a full-suite pass. The directly
affected driver callback is covered by a focused test.

## Known limitations and measured update

- AI-107 and AI-109 subsequently trained substantial instances. Across the
  three-seed AI-109 comparison, the material cVAE did not improve beyond seed
  variation and accumulated unstable recursive error. It remains shadow-only;
  see `AI109_PREDICTIVE_LEARNING_CURVES_TECHNICAL_2026-08-12.md`.
- Current reconstruction target is the next posterior mean. Posterior log
  variance conditions recognition and prediction but is not yet integrated as
  uncertainty in the reconstruction observation measure.
- Raw registered camera images are not persisted, so this is not end-to-end
  camera perception learning.
- Old v1 corpus samples cannot teach brush-load sensitivity.
- A finite diagonal-Gaussian decoder and moment-matched calibration summary can
  miss multimodal tail behavior.
- Bootstrap-member disagreement is an approximate parameter posterior, not a
  formal Bayesian neural-network posterior.
- This one-step model does not yet infer multi-event brushmark explanations,
  patch compressibility, local order, or compatibility with global structure.
- Most importantly, its target is a coarse material posterior rather than the
  registered post-mark image. It is not the action-conditioned visual model
  selected in `VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.
