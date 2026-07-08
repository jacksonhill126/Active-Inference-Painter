# Codex continuation brief

## Non-negotiable architecture

Painting cognition must be formulated as active inference. Do not add heuristic rewards such as `+closure`, `-ugliness`, or weighted aesthetic score soups.

Every painting-level term must be identifiable as one of:

- likelihood in the generative model;
- state-transition prior;
- prior preference over outcomes or latent trajectories;
- precision belief;
- variational free energy term;
- expected free energy term;
- policy prior/posterior.

Conventional IK, robot dynamics, servo control, and hard safety are allowed only beneath the selected painting policy.

## Immediate task

Connect the existing terminal-coverage active-inference planner to a MuJoCo arm while preserving these interfaces:

1. `StrokeAction` remains a Cartesian/contact policy representation.
2. The painting planner infers strokes and `stop`; IK only realizes the selected Cartesian path.
3. Canvas state must include independent spatial fields for thickness, wetness, visible pigment, and material coverage.
4. White-on-white paint must increase thickness and coverage despite little visual change.
5. Observation uncertainty should increase conditionally with wetness and thickness because smearing and mixing become less predictable.
6. The terminal coverage preference applies only to policies that terminate in `stop`, centered near 0.87 with an 80–90% high-probability region.
7. Include the immediate `stop` policy and multi-stroke-then-stop policies in every planning cycle.
8. Keep hard maximum paint mass, current, joint, workspace, and watchdog limits external to active inference.

## Required tests

- `stop` has low expected free energy around 0.85–0.90 coverage.
- continuation is preferred at low coverage when predicted strokes approach the terminal band.
- `stop` is preferred over a stroke expected to overshoot the terminal band.
- white paint on white ground increases material coverage.
- increased wetness/thickness increases predicted observation entropy.
- learned model uncertainty gates epistemic policy selection.
