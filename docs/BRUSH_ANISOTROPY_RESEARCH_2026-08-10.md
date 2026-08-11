# Anisotropic Round-Brush Contact: Research And Process Baseline

Status: implemented provisional generative-process baseline
Date: 2026-08-10
Canonical arm: current upper-arm-roll design; the angled wrist-roll exploration
remains non-canonical and is not required by this work.

## Why this exists

The earlier canvas process treated the brush as a round swept disc. That made
upper-arm roll weakly identifiable from painted consequences and made pushing
the bristles ferrule-first nearly equivalent to pulling them handle-first.
Inclined bristles interact asymmetrically with canvas tooth, while the selected
round tuft changes contact extent with pressure and handle incidence.

An initial implementation incorrectly represented that geometric anisotropy as
a fixed-aspect flat/chisel brush tied to a material width axis. Owner review on
2026-08-10 caught the mismatch. The corrected canonical process is
axisymmetric: normal incidence is circular, handle tilt stretches the contact
patch along the handle projection, and radial roll about an unchanged handle
axis does not rotate a fictional chisel edge.

This implementation does **not** add a reward for handle-leading motion.  It
changes the generative process.  Painting policies can consequently predict
different camera-visible material transitions, bristle-tip path errors, and
loads for different directions and roll realizations through the existing
likelihood and EFE path.

## Prior work used

This is a reduced implementation of established brush/friction ideas rather
than a new phenomenological chatter generator.

1. William Baxter and Ming Lin, *A Versatile Interactive 3D Brush Model*,
   Pacific Graphics 2004, DOI
   [10.1109/PCCGA.2004.1348363](https://doi.org/10.1109/PCCGA.2004.1348363).
   Baxter explicitly models bristle tips catching in surface tooth more when
   pushed than pulled.  Its anisotropic Coulomb term lowers friction smoothly
   in the preferred pull direction, while normal force scales friction.  It
   also separates static and kinetic contact and projects a deforming 3D brush
   into a 2D paint footprint.
2. C. Canudas de Wit, H. Olsson, K. J. Astrom, and P. Lischinsky, *A New Model
   for Control of Systems with Friction*, IEEE TAC 1995, DOI
   [10.1109/9.376053](https://doi.org/10.1109/9.376053).  The LuGre model shows
   how an aggregate bristle-deflection state can represent pre-sliding
   displacement, breakaway, hysteresis, and stick-slip without resolving every
   microscopic asperity.
3. Paolo Gidoni and Antonio DeSimone, *On the Genesis of Directional Friction
   Through Bristle-like Mediating Elements*, ESAIM COCV 2017, DOI
   [10.1051/cocv/2017030](https://doi.org/10.1051/cocv/2017030).  This supports
   treating directional friction as a joint consequence of bristle geometry,
   surface fluctuations, and normal load.
4. Nelson Chu et al., *Detail-Preserving Paint Modeling for 3D Brushes*, NPAR
   2010, [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2010/06/PaintModel_NPAR_2010.pdf).
   It supports the practical footprint-impression architecture: project brush
   geometry to a 2D contact map, modulate it with canvas tooth, and use that map
   for bidirectional paint transfer.
5. Pavel V. Sviatov et al., *Physically Motivated Model of a Painting Brush for
   Robotic Painting and Calligraphy*, Robotics 2024, DOI
   [10.3390/robotics13060094](https://doi.org/10.3390/robotics13060094).
   This work specifically models a round artistic brush, separates the robot
   handle/TCP path from the lagging center of the contact-force patch, and
   calibrates footprint width and lag as functions of canvas approach.
6. Mai Otsuki et al., *MAI Painting Brush: An Interactive Device that Realizes
   the Feeling of Real Painting*, UIST 2010,
   [paper](https://www.rm2c.ise.ritsumei.ac.jp/mclab/pdf/UIST10_otsuki.pdf).
   Its reduced round-brush footprint changes size with pressure/bending,
   stretches along brush tilt while retaining cross-width, and follows the
   projected tip direction. That is the closest precedent for the inexpensive
   angle-dependent footprint used here.

The full Baxter solver uses multiple skeletal bristle spines and constrained
energy minimization.  That is too expensive for the current counterfactual
budget. The present approximation keeps one coherent tuft state and one
angle-dependent round-tuft footprint.

## Implemented process equations

Let `d` be canvas-plane handle/contact displacement for one simulation step,
`z` the aggregate tangential bristle deflection, `N` normal force, and `p` the
unit direction from bristle tip toward the handle.  For nonzero travel,

```text
a = clamp(dot(unit(d), p), -1, 1)
eta = C_pull * max(0, a)^k
s_tooth = 1 + C_tooth * (2 * tooth - 1)
mu_s(a,tooth) = mu_s0 * (1 - eta) * s_tooth
mu_k(a,tooth) = mu_k0 * (1 - eta) * s_tooth
z_candidate = z + d
```

The tuft remains stuck while

```text
K * norm(z_candidate) <= mu_s(a,tooth) * N.
```

At breakaway it releases to the kinetic limit:

```text
z = unit(z_candidate) * mu_k(a,tooth) * N / K.
```

The aggregate painting point is the rigid brush contact minus `z`.  During a
sticking interval the painting point remains nearly fixed while the handle
moves; release produces a jump.  Repeated elastic loading and release yields a
stop/play stick-slip trace.  Frozen canvas tooth changes the local breakaway
threshold, so irregularity is tied to a persistent surface cause rather than
fresh random noise.

The normal force remains separate from the reported tangential force.  The
native abstract plant receives their combined magnitude as a provisional load
on the following step.  MuJoCo continues to report its own physical contact
force/current; the reduced brush model does not overwrite a simulated sensor.

## Angle-dependent round-brush footprint

`ArmKinematics.forearm_rotation` defines local `+Y` from elbow/handle toward
the brush tip. Let `h` be that unit handle-to-tip axis and let `beta` be the
acute angle **between that end-effector axis and the canvas plane**. Thus
`beta=90 degrees` is perpendicular to the canvas and `beta` approaches zero at
grazing incidence. Let `h_parallel` be the axis's canvas-plane projection. The
selected 12.7 mm / 0.5 in bundle diameter supplies the tuft envelope. The
contacting fraction grows as `0.12 + 0.88 * sqrt(pressure)` from tip contact
toward that envelope, after which pressure can add up to 20% bounded radial
splay. The corrected footprint uses

```text
aspect = clamp(1 + k_angle * cot(beta),
               1,
               aspect_max)
r_major = r * aspect
r_minor = r
```

The major direction is `h_parallel`; at normal incidence its norm is zero and
the patch is circular. This tangent-law form is an explicitly provisional
continuous interpolation, not a calibrated constitutive equation. It captures
the established qualitative facts that tilt stretches a round-brush footprint
along the tip/handle direction while cross-width remains set primarily by
pressure and splay. The cap avoids a singular footprint at grazing incidence.
Current parameters are `brush_round_canvas_angle_elongation_gain = 0.35` and
`brush_round_max_aspect_ratio = 1.65`. These deliberately replace the initial
oversized 1.15 gain / 2.10 cap. The nominal diameter now comes from
`brush_round_bundle_diameter_world = 0.50`, aligned with the MJCF, instead of
the unrelated historical `0.10 + 0.42 * pressure` radius law.

Upper-arm roll can still matter because, through the accepted shoulder/elbow
kinematics, it can change the whole handle axis relative to the canvas. But
rotation about an unchanged round-brush axis has no independent footprint
effect. Bristle furrows organize across actual brush travel rather than a
fixed material edge. Direct `paint_at` calls without a supplied handle axis
retain the circular footprint for compatibility.

The stroke-exit trajectory unloads actual Cartesian depth and intended
pressure with the same final taper that narrows footprint width. The lift phase
starts at that unloaded endpoint with taper flow held at zero. Previously the
path's smootherstep decelerated to zero under a fixed 0.2-world-unit
penetration, and lift reset flow to one; repeated zero-motion contact frames
therefore collapsed into a circular stamp after every tapered tail. The
correction changes conventional trajectory realization below the selected
painting policy, adds no paint gate, and retains deposition under any positive
physical contact.

## Active-inference boundary

This file describes the **generative process**, not the desired final agent
model.  The eventual counterfactual model should infer a compact stochastic
brush state and transition likelihood from camera and proprioceptive history.
It should predict distributions over footprint, bristle lag/release, and load;
it should not call the hidden process implementation or receive exact process
state.

Current execution counterfactuals construct independent plant/material/brush
state and independent future noise, but still reuse much of the same physics
implementation as the process.  That remains an uncalibrated simulation-only
integration baseline.  Independence from the live state is necessary but is
not the same as having a learned, misspecified, uncertain generative model.

The new telemetry fields are intended to make the later split measurable:

- `brush_tangential_force_n`;
- `brush_pull_alignment`;
- `brush_sticking` and `brush_stick_slip_transition`;
- `brush_slip_fraction`;
- rigid `brush_*` position versus aggregate `painting_point_*` position.

No direct preference, reward, or EFE term is attached to pull alignment,
sticking, or slip.  Decision relevance comes only through predicted sensory
and material consequences and their declared likelihood/preference terms.

## Approximations and calibration needs

- The selected physical brush geometry, bristle stiffness, static/kinetic
  coefficients, and footprint-versus-pressure/incidence law have not been
  measured. The current tangent law and aspect cap are simulation priors.
- One aggregate tuft cannot reproduce individual-hair modes, ferrule contact,
  twisting, or multiple simultaneous stick/slip regions.
- Canvas tooth is a fixed procedural prior, not a scanned physical canvas.
- The native plant maps tangential force magnitude to its existing scalar
  contact-load approximation; it does not apply a full Cartesian wrench.
- MuJoCo does not yet feed this reduced tangential state into a calibrated
  compliant brush/contact body.
- A later camera/proprioceptive likelihood must infer the compact brush state;
  exact process fields are evaluation and training labels only.

Useful bench calibration is a matrix of constant-speed straight pulls and
pushes over the same canvas strip at several normal loads and handle incidence
angles. Measure handle pose, motor current/force, and the resulting mark. Fit
static breakaway, kinetic drag, stiffness, footprint width/length/lag, and
uncertainty jointly; do not tune them to make a preferred controller win.

## Tests

`tests/test_arm_sim.py` checks that:

- normal incidence yields a circular round-brush footprint, while matched
  handle tilts elongate it along the projected handle direction;
- pushing has higher tangential load and more irregular bristle-tip increments
  than pulling under matched conditions;
- normal force, direction, and frozen tooth each condition friction;
- zero normal force produces zero tangential force and released deflection;
- legacy direct calls remain round and existing material-conservation tests
  continue to pass.
