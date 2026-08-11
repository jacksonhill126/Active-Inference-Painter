# Compact Dual-IMX296 Camera Rig Decision — 2026-08-07

## Decision

The initial hardware-oriented baseline now uses exactly two Raspberry Pi
Global Shutter Cameras (Sony IMX296), selected but not yet purchased, on a
single rigid crossbar. Both use a provisional 4 mm CS-mount lens. This replaces
the full-size four-view proposal; there is no separate inspection, overhead,
or profile camera in the current baseline.

The change reduces the spatial envelope, removes the fold-away inspection
mechanism, and provides two matched global-shutter observation channels. It
does not prove that two cameras satisfy every future embodiment requirement.

## Nominal geometry

Canvas center is `(0.075, 0.4826, 0.350) m`.

| Quantity | Right | Left |
| --- | ---: | ---: |
| Optical center (m) | `(0.375, -0.1674, 0.570)` | `(-0.225, -0.1674, 0.570)` |
| Lateral offset | +300 mm | -300 mm |
| Normal standoff | 650 mm | 650 mm |
| Height above canvas center | 220 mm | 220 mm |
| Distance to canvas center | 748.9 mm | 748.9 mm |
| Canvas incidence | 29.78 degrees | 29.78 degrees |

Camera-center span is 600 mm. Placeholder housings imply roughly 640 mm total
width and 670 mm forward extent from the canvas plane before crossbar ends,
connectors, cables, or lights. This is an envelope, not mounting CAD.

## Optical assumptions

The declaration uses the 1456 x 1088 IMX296 array at 3.45 micrometre pitch.
The provisional 4 mm pinhole model gives 50.27 degrees vertical field of view.
Both views declare global shutter, CSI-2, provisional 60 Hz sampling, 10-bit
quantization, 512 x 512 registered global observations, and 256 x 256
requested foveae. Lens SKU, distortion, focus, exposure, attainable paired
rate, latency, synchronization, noise, and dropout remain unmeasured.

## Reproducible geometric evidence

```powershell
python -m active_painter.camera_pose_sweep `
  --output-dir docs\figures\camera_observability `
  --grid-size 9 --ray-grid-size 9
```

The 243-pose sweep covers a 9 x 9 canvas grid at roll -32, 0, and +32 degrees.
Maximum IK residual was `2.20e-9 m`. Each camera independently kept the
sampled bristle tip in frame and unoccluded in 100% of poses. Minimum canvas
visibility was 81.5% right and 75.3% left; means were 93.8% and 94.2%.

The check excludes camera/mount visual geoms from rays and uses MuJoCo
segmentation. It does not model real paint photometry, lens distortion, glare,
blur, exposure, or learned edge/tip detection.

## Alignment and next gates

The authoritative record is `models/active_inference_painter.xml` plus
`models/README.md`; `AGENTS.md` repeats the baseline for later agents. The web
smoke renderer uses 640 x 480 per view as a throughput approximation.

Next: purchase matched modules/lenses; confirm two-camera CSI throughput and
timing on the chosen controller; design the crossbar, cables, lighting, and
collision envelope; calibrate both cameras; then collect real arm/brush images
to validate the likelihood and edge/tip observation model.
