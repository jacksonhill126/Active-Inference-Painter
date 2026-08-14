# Where The Visual Hierarchy And Attention Work Stand

Date: 2026-08-14

## The short version

The project is not changing into a project about moving a crop around a video
frame.

The central goal is still the hierarchical active-inference painter you
described: a model that can understand a painting in terms of tone, edges,
continuous masses, relationships among separated regions, recurring motifs,
and slower structure across marks and passages.

Attention remains important because the order in which the painter inspects
parts of the canvas may help create and test those relationships. But an
attention trajectory does not have to mean a rectangular window gliding
around the image.

## What an attention trajectory means here

An attention trajectory is the history of where the model allocated detailed
visual information and confidence over time.

It might look like:

```text
first:  broad attention to the whole painting
then:   detailed evidence from one ambiguous junction
then:   detailed evidence from a distant related mass
then:   return to the first region after making a mark
```

Those detailed observations could be implemented as:

- one high-resolution crop;
- several sparse image tiles;
- a broad low-resolution view plus one high-resolution region;
- eventually, another budgeted readout from the camera.

The current crop request is useful because it gives us a clean way to prevent
the agent from secretly seeing every high-resolution pixel. It is an
implementation tool, not a claim that cognition consists of steering a box.

## What is actually built

The simulation and camera path are substantial. The detailed paint process,
dual-camera geometry, registered global observations, requested native-derived
crops, body inference, and repeated sensor-mediated painting loop exist.

The earlier material transition experiments also produced useful negative
evidence: adding a VAE to the coarse 16x16 material fields did not solve the
important visual problem.

The corrected visual mark VAE path is now implemented. It uses a fresh image
of the proposed target, the mark, brush belief, camera, and motor realization
to predict a distribution over post-mark images. Its small smoke run worked
end to end. A larger collection reached 18 complete trajectories before a
Windows Update restart and can be resumed.

What is not yet built is the central hierarchy itself. There is not yet an
accepted learned state for continuous masses, spatial relationships, or visual
motifs, and the visual VAE does not yet influence painting policy.

## What we should work toward

The hierarchy should have four understandable levels:

1. **Tone and edges** - what light/dark structure and boundary directions are
   actually visible.
2. **Masses** - which boundaries and tones appear to belong to the same
   continuous light or dark region.
3. **Relationships** - how masses and marks relate through angle, spacing,
   scale, alignment, overlap, repetition, and transformation.
4. **Motifs and passages** - configurations that recur or evolve across
   multiple marks and longer periods of painting.

The larger levels should help predict the smaller ones. If two distant marks
belong to one larger structure, observing one should change what the painter
expects to see at the other.

This gives a concrete meaning to one part of the canvas referring to another.
It does not require a hand-written rule saying that repetition, continuity, or
harmony is beautiful.

## The narrowed order of work

To prevent the project from spreading outward, the order should be:

1. Finish measuring the local visual mark predictor without endlessly scaling
   it.
2. Build a visual representation that preserves broad tone and boundary angle.
3. Test whether a continuous mass state improves prediction.
4. Test whether separated regions can predict one another.
5. Add slower relationship and motif states only when the simpler hierarchy
   earns predictive evidence.
6. Connect the accepted hierarchy to mark selection.
7. Then compare active attention with random and uniform observation.

Attention therefore stays in the research program, but it does not get to
become an elaborate independent subsystem before the model can represent the
painting structure that attention is supposed to investigate.

## What we are deliberately not building now

We are not currently committing to:

- simulated biological eyes;
- smooth crop-steering trajectories;
- cortical columns or spiking neurons;
- semantic object recognition;
- hand-authored contour, motif, harmony, or beauty scores;
- a large general-purpose vision model;
- complicated long-horizon joint attention and painting policies.

The useful lesson from visual cortex is narrower: represent local evidence at
several scales, let larger spatial hypotheses explain it, and let those larger
hypotheses feed predictions back down.

## Bottom line

The cortex-inspired hierarchy is core work. Attention trajectories are an
important later mechanism and experiment. The moving crop is optional.

The next meaningful success is not an animated attention box. It is evidence
that the model can represent a mass or relationship, use one region to predict
another, and allow that shared belief to change what mark it expects or
selects next.
