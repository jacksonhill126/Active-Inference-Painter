# AGENTS.md — Active-Inference Painter

## Governing constraint

Painting cognition must be active inference from top to bottom. Do not add vague aesthetic heuristics, scalar rewards, or weighted score soups.

At painting level, every decision-relevant quantity must be explicitly grounded as one of:

- a likelihood in the generative model;
- a transition prior;
- a prior preference over outcomes or latent trajectories;
- a precision belief;
- a variational free-energy term;
- an expected-free-energy term;
- a policy prior or posterior.

## Allowed conventional engineering boundary

Conventional forward kinematics, inverse kinematics, trajectory interpolation, robot dynamics, motor control, collision checking, and hard safety constraints are allowed below the selected painting policy.

IK may realize or predict a Cartesian painting policy. It must not choose the painting policy.

## Terminal coverage rule

Coverage is a material state derived from paint thickness, not visible tone. White paint on white ground still increases coverage.

The strong 80–90% coverage preference is terminal and conditional on `stop`. Do not apply it at every intermediate time step. Every candidate policy must terminate in `stop`, and the immediate `stop` policy must always be available.

## Contact rule

Do not introduce a globally preferred contact-pressure scalar. Pressure/contact predictions must be conditional on intended mark consequences, stroke phase, speed, curvature, brush state, local canvas state, and model uncertainty.

## Higher-level priors

Separate posterior beliefs, transition priors, and preferences. Higher-level priors must have slower dynamics and higher temporal depth; do not let them copy lower-level observations through fast moving averages.

## Safety

Hard joint, current, force, workspace, watchdog, and non-finite-state limits remain external to the active-inference painting model.

## Required development practice

- Add tests for each probabilistic claim.
- Log VFE and EFE decompositions separately.
- Mark approximations as approximations.
- Never rename an ordinary reward or controller as active inference.
