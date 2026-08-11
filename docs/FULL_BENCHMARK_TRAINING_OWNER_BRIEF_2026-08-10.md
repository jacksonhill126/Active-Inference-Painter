# What tonight's benchmark and training run established

## The short version

We now have a real trained local brush/canvas dynamics checkpoint, not just a
pipeline smoke test. It learned from 1,216 training transitions and improved
substantially on 192 held-out transitions that it never trained on.

What we do **not** yet have is a trained composition model. The simulated
painters never chose to stop within their 64-mark caps, so none of the saved
canvases can honestly be called a finished painting. The code correctly
refused to train composition on those arbitrary endpoints.

Everything here is still an uncalibrated simulation-only baseline.

## What the speed test means

Running multiple independent painters in parallel works. One worker produced
about 0.056 transitions per second. Eight workers produced about 0.474 per
second—8.4 times the throughput in this particular benchmark. That is enough
to make simulation training practical on this laptop rather than requiring
literal real-time painting.

There is an important catch: keeping the same worker alive across several
paintings became progressively slower. The first 64-mark paintings took about
19 minutes, while a later wave was still unfinished after roughly 42 minutes.
The likely solution is simple in concept: let each process paint one trajectory,
save it, exit, and start a fresh process for the next one. That keeps useful
learned parameters centralized while preventing each simulator's online state
from growing indefinitely.

## What was actually learned

The trained component predicts the local material consequence of a selected
mark, conditioned on the camera-derived canvas belief and action/motor
realization. This is the part that needs to learn things such as:

- how an angled round-brush footprint changes deposited material;
- how existing thickness, wetness, tone, and uncertainty affect the next
  canvas belief;
- how black versus white paint and curved versus straight actions differ;
- how conditional upper-arm-roll realization changes the observed outcome.

The held-out error improved strongly:

- validation NLL went from -1.177 to -4.765;
- test NLL went from -1.541 to -4.891.

Lower is better. These values can be negative because the reported Gaussian
negative log density intentionally leaves out an additive constant; the
important information is the large improvement on data excluded from
training, not whether the absolute number is above or below zero.

The checkpoint is
`runs/checkpoints/shared-pretraining-20260810-1408.pt`.

## Why composition did not train

The hierarchy has capacity for slower composition-level beliefs, but it needs
examples whose terminal status has a defensible meaning. A canvas saved only
because we reached a 64-action compute cap is not the same as a canvas on which
the policy inferred that stopping was appropriate.

All 22 saved trajectories were cap-truncated. There were zero genuine stop
decisions. Training composition on their final frames would teach the model
that an arbitrary timeout means “finished,” which would directly undermine the
active-inference design. Skipping composition was therefore a successful
guardrail, not a failed training command.

## Problems found during the run

One of eight workers finished its first archive but stalled while finalizing
the filename. The archive itself was complete and was recovered and verified.
The other seven continued normally. The code now ignores temporary archives
when discovering shards, so a future interrupted write cannot accidentally
enter a dataset split.

The trainer also initially rejected the corpus because each worker had a
different canvas-grain and brush-microstructure seed. That diversity is
intentional: it teaches shared parameters across different surface/brush
realizations. The guard now allows only those two seeds to vary; it still
rejects changes to the actual model, plant, likelihood, or learning settings.

## What I would do next

The next priority is not simply “run longer.” First, make each collection
process short-lived and add phase timings. Then run a fresh corpus and inspect
why the stop policy is never selected. We need genuine terminal canvases before
composition learning can begin.

After that, load tonight's checkpoint into a fresh simulation runtime and do a
controlled trained-versus-untrained comparison. That will tell us whether the
better held-out transition likelihood changes policy selection, mark variety,
roll use, and stopping behavior in the active-inference loop—not merely whether
the neural predictor fits stored patches.
