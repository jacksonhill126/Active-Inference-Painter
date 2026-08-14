# What We Corrected About The Painter's Internal Model

Date: 2026-08-12

## The short version

The simulator should know detailed paint physics. The painter does not need a
canvas-wide internal spreadsheet of physical paint variables.

The painter first needs to see and predict the image: tone, edges, masses, and
how a proposed brush movement is likely to change them.

## The key distinction

The **generative process** is the simulated or physical world. It can contain
wetness, thickness, pigment mass, bristle mechanics, sticking, pickup, and
blending because those variables determine what actually happens.

The **generative model** is the agent's economical internal prediction. It
should not mirror the process variable-for-variable. At the present stage,
explicit coarse wetness consumes capacity without helping the agent represent
an angled boundary or coherent tone mass.

Instead, the painter should look freshly at the place where it is considering
a mark and predict a family of possible visible results. An internal temporary
latent may implicitly summarize whether that patch behaves as wet, dry, thick,
sticky, or broken, but it does not need a human-readable name and need not be
remembered across the whole canvas.

## What the model should remember

Persistently:

- the registered visual image or its predictive multiscale representation;
- tone masses and their uncertain continuous boundaries;
- important relations between regions;
- slower structure across marks, passages, and the painting.

Temporarily, when evaluating a proposed mark:

- a high-resolution image crop around its path;
- the current brush belief;
- an uncertain interaction latent answering, in effect, “what kinds of marks
  will this movement probably make here?”

After the mark, the painter looks again. Unpredicted bristle and paint behavior
becomes new visual evidence rather than a demand that the old prediction name
every physical cause correctly.

## The VAE correction

Your earlier VAE proposal referred to this stochastic visual mark predictor.
The model we actually built instead predicted coarse material-posterior
channels. That experiment was legitimate, and it produced a useful negative
result for its own target, but it was not a test of your visual VAE proposal.

The documentation now separates those two ideas so future agents cannot cite
the material cVAE result as evidence against the visual model you meant.

## What changes next

The current corpus preserved 16x16 material posteriors but not the registered
camera frames needed to learn visual fidelity. Future collection must retain
pre- and post-mark rectified images, camera provenance, action-aligned crops,
brush belief, action, and termination labels.

The next model should be judged by whether it predicts unseen visual outcomes,
preserves tone and oriented edges, remains calibrated, and supports stable
multistep visual rollouts. It should not be judged by whether it reconstructs
unobserved coarse wetness.

The full canonical technical boundary is
`docs/VISUAL_GENERATIVE_MODEL_BOUNDARY.md`.
