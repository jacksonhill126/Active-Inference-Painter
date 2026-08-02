# Camera Observability And Grayscale Input Brief

**Rig:** `provisional-multiview-v4`

**Simulation model:** `mujoco-robstride-electromechanical-v4`

**Sweep:** 243 contact poses, 9 × 9 canvas locations, upper-arm roll at
−32°, 0°, and +32°

## Finding

The opposing oblique cameras provide complete combined brush-tip visibility
over the sampled contact workspace. Individually, the right camera sees the
bristle/contact point in 99.59% of poses and the left camera in 98.35%; their
continuous union is 100%. The proposed overhead edge-profile camera sees the
brush tip in 100% of sampled poses and provides a direct image consequence of
brush-to-canvas normal separation.

The head-on camera still satisfies its intended role: complete canvas
inspection while the arm is in `camera_clear_park`. Its contact-pose frames
below are diagnostic only because the camera must be stowed during painting.

![Representative grayscale frames](figures/camera_observability/camera_pose_sweep_plate.png)

*Figure 1. Representative model-input frames. The orange reticle is the
projected brush tip, not an observation supplied to the agent. The park-only
inspection camera is explicitly marked when shown in hypothetical contact
poses.*

![Brush-tip visibility maps](figures/camera_observability/camera_visibility_maps.png)

*Figure 2. Bristle visibility at each sampled canvas location. Red cells are
out-of-frame or occluded for that particular camera. The two-camera continuous
union remains complete.*

## Geometry Decision

The owned lenses are now explicit in the XML: 25 mm on the OM-1 and 35 mm on
the A7R II, with the latter provisionally using its Super 35 capture mode. The
nominal full-frame equivalents are therefore 50 and 52.5 mm. Based on 17.3 mm
and 23.5 mm active 16:9 widths, their provisional vertical fields of view are
22.03° and 21.39°. These are nominal pinhole values, not measured intrinsics.

Both optical centers are 7% farther from canvas center than the v3 poses while
retaining the same incidence angles. In a nominal 3840 × 2160 frame, the
canvas spans about 1587–1628 pixels horizontally and 1874–1924 pixels
vertically, leaving at least 94 pixels at every vertical edge. The 12.7 mm
brush spans approximately 40–48 native pixels depending on view and canvas
direction. This leaves useful calibration and mount tolerance without giving
up foveal detail.

The fourth camera is `brush_standoff_overhead`, located above the canvas at
`(0.075, 0.250, 1.250) m` and looking downward along a direction tangent to the
canvas plane. Its image is an `canvas_edge_profile`, not a canvas image:

- it is never passed through the canvas homography;
- its visible canvas edge establishes the plane reference;
- the bristle silhouette supplies a likelihood for normal separation,
  approach, contact, and compliance;
- at 640 × 480, one millimetre of normal motion spans approximately 0.50–0.90
  pixels across the canvas height;
- the present 12 mm compliance range therefore spans approximately 6–11
  pixels before subpixel edge fitting.

This camera should be mounted above the canvas rather than at its left or right
edge. The overhead location avoids consuming lateral robot workspace and
preserves useful normal-distance scale across the full canvas width.

## Grayscale Observation Contract

The model-facing input is now explicitly single-channel grayscale:
`linear_grayscale_float32_normalized_0_1`.

| View | Provisional hardware and lens | Capture mode | Acquisition | Global input | Fovea | Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Right oblique | Owned OM System OM-1, 25 mm | MFT full-width 16:9 | 3840 × 2160 | 512 × 512 | 256 × 256 | 30 Hz |
| Left oblique | Owned Sony A7R II, 35 mm | Super 35 full-width 16:9 | 3840 × 2160 | 512 × 512 | 256 × 256 | 30 Hz |
| Head-on inspection | Additional camera TBD | TBD | 3840 × 2160 target | 512 × 512 | 256 × 256 | 5 Hz / on demand |
| Overhead profile | Low-cost global shutter TBD | Native 4:3 TBD | 1456 × 1088 target | 640 × 480 | None | 60 Hz |

The owned OM-1/25 mm and A7R II/35 mm combinations are the provisional
opposing continuous pair. This assignment may swap left/right after mount,
focus, latency, and occlusion measurements. Neither owned body is
simultaneously claimed as the separate fixed head-on camera. Sony documents
clean 4K HDMI output from the A7R II at
24/25/30p; OM System documents 4K HDMI output for the OM-1:

- [Sony A7R II help guide](https://helpguide.sony.net/ilc/1520/v1/en/print.pdf)
- [Sony A7R II APS-C/Super 35 mode](https://helpguide.sony.net/ilc/1520/v1/en/contents/TP0001077177.html)
- [OM System OM-1 help guide](https://download.omsystem.com/pages/inst/om1/manual_om1_v1.7_ENU.pdf)
- [OM System 25 mm F1.8 nominal lens specifications](https://explore.omsystem.com/us/en/m-zuiko-25mm-f1-8-black)

The XML now also declares a provisional acquisition model. The two oblique
views use read-noise standard deviation 0.003, signal-dependent noise
coefficient 0.006, 8-bit model-input quantization, and one 30 Hz frame period
of latency. The inspection view uses 0.002/0.004 and 200 ms; the overhead
profile uses 0.004/0.008 and one 60 Hz frame period. Dropout is presently zero.
These are explicit simulation assumptions, not measured camera parameters.

For a 508 mm square canvas, the canonical 512 × 512 canvas image represents
approximately 0.99 mm per pixel and a 12.7 mm brush spans 12.8 canonical
pixels. That is a useful first baseline for global prediction without making
the observation tensor unnecessarily large. Native frames must be retained
for calibration, diagnostics, and foveal crops. A fovea is sampled directly
from the undistorted 4K acquisition in canvas UV coordinates; it is never
cropped from or upsampled out of the 512 × 512 global tensor.

The recommended physical acquisition path is:

1. acquire the two clean 3840 × 2160 HDMI streams with host timestamps;
2. lock and calibrate exposure, focus, aperture, white balance, capture-card
   conversion, rolling-shutter timing, and relative camera latency;
3. undistort using the measured lens model;
4. derive the 512 × 512 global view and selected 256 × 256 foveae separately
   from the native acquisition;
5. preserve calibration-derived validity, timing, and uncertainty metadata; exact
   simulator occlusion masks remain evaluation-only;
6. convert calibrated intensity to normalized floating-point grayscale for
   the likelihood/encoder.

`active_painter.camera_calibration` now implements the intrinsic portion of
this procedure. It generates the metric A3 checkerboard in
`calibration/cameras/checkerboard_11x8_28mm.svg`, detects the target in native
HDMI frames, and writes a versioned Brown-Conrady calibration with per-view
residuals, field coverage, tilt diversity, and explicit acceptance gates.
Capture and operator instructions are in `calibration/cameras/README.md`.
There is not yet an accepted physical calibration for either owned camera.
Camera-to-canvas and cross-camera extrinsics remain a separate post-mount
stage.

This does not make white paint on a white ground directly observable as
material coverage. That ambiguity is accepted for the present milestone.
Coverage remains hidden material state rather than an agent input.

## Implemented Observation Process

`CameraObservationProcess` now produces versioned, timestamped, multi-rate
grayscale products from the MJCF rig. Each exposure is rendered at the
XML-declared acquisition resolution by default. Provisional sensor noise and
quantization are applied once to that native frame. A 512 x 512 global canvas
view is then rectified independently from it, and each requested 256 x 256
fovea is sampled directly from the same native frame rather than cropped from
the global product. Native, global, and foveal products from one exposure
share camera name, capture sequence, and timestamps. Explicit low-resolution
native overrides exist only for bounded tests and resource-limited
diagnostics.

MuJoCo supplies arm/brush geometry, lighting, and occlusion. The Python
process supplies only the current superficial grayscale canvas appearance.
Internal MuJoCo segmentation is used to composite that appearance onto pixels
where the canvas surface rendered; the segmentation and exact visibility
labels are discarded before any `CameraFrame` is constructed. Rendered arm
and brush pixels survive rectification and foveation, so occlusion is an image
consequence rather than a mask given to the model. The overhead view retains
its native frame and also emits an edge-profile product.

Foveation has no simulator-selected default. A `FoveaRequest` addresses a
center and span in shared canvas UV coordinates and records its expiry,
selection basis, and basis revision. Permitted bases are a sensor-derived
posterior, a policy prediction, or an explicitly diagnostic operator request;
exact simulator pose, contact, visibility, segmentation, and material state
are not valid selectors. No request means no foveal product. Configured
capture rates and latency are enforced by `CameraObservationBundle`; the
head-on view is emitted only when its park-availability condition is supplied.

For operator inspection, the web runtime publishes a canvas-registered trace
of the foveal products that were actually delivered. It does not expose queued
requests as observations. The latest delivery is highlighted and older
deliveries fade over 10 seconds by default. This duration is explicitly a
visualization fallback, not a claim about perceptual memory; the runtime adopts
an agent-declared `foveation_memory_horizon_s` when one exists. Pointer-selected
requests are labeled `operator_diagnostic` and therefore remain distinct from
future sensor-posterior or policy-prediction gaze selection.

The provisional specular term is deliberately small and narrow in scope. It
mixes a bounded positive residual from MuJoCo's rendered canvas lighting into
the superficial grayscale appearance. It is not a wet-paint BRDF, does not
read the hidden wetness field, and should be replaced after lighting/camera
measurements.

## Owned Hardware Baseline and Upgrade Path

The OM-1/A7R II pair is the starting hardware, not a temporary visualizer
stand-in. Both are rolling-shutter Bayer cameras, so their view-specific
likelihoods must retain separate photometric, timing, and precision
parameters. They need not be color-matched into a fictitious common camera.
The controller receives calibrated grayscale observations and uncertainty.

The industrial-camera class below is an upgrade path only if measurements
show that rolling shutter, synchronization, capture latency, or consumer
camera processing materially limits held-out prediction or calibration.

A typical primary sensor for this application is a monochrome industrial
machine-vision area camera with:

- global shutter;
- C/CS-mount fixed-focus lens;
- USB3 Vision or GigE Vision transport;
- hardware trigger for synchronizing the two continuous views;
- manual exposure, gain, aperture, and focus;
- at least 10-bit acquisition even though the first model input may be
  normalized to a lower-precision tensor;
- native resolution around 1–3 MP.

For scale, Basler documents monochrome ace cameras with 1920 × 1200 global-
shutter sensors, USB3 or GigE transport, and hardware triggering. Teledyne FLIR
documents Blackfly S global-shutter monochrome cameras with 10- or 12-bit ADCs,
hardware I/O, and ROI modes. These are representative camera classes, not yet
vendor selections:

- [Basler acA1920-155um documentation](https://docs.baslerweb.com/aca1920-155um)
- [Teledyne FLIR Blackfly S BFS-PGE-14Y3M specification](https://softwareservices.flir.com/BFS-PGE-14Y3/latest/Model/spec.html)

For the low-cost overhead profile camera, a monochrome OV9281-class global-
shutter module is appropriate. Arducam documents a 1280 × 800 monochrome
global-shutter OV9281 module; cropping or downsampling it to 640 × 480 leaves
ample margin for this profile task:

- [Arducam OV9281 module datasheet](https://cdn.arducam.com/downloads/modules/OV9281/OV9281_MIPI_Camera_Module_Standalone_DS.pdf)

The Raspberry Pi Global Shutter Camera is another inexpensive prototype
option, with a 1456 × 1088 Sony IMX296 sensor, RAW10 output, C/CS mount, and
external synchronization support. It is a colour sensor, however, so a true
monochrome module is preferable when grayscale sensitivity and the absence of
demosaicing artefacts matter:

- [Raspberry Pi Global Shutter Camera](https://www.raspberrypi.com/products/raspberry-pi-global-shutter-camera/)

## Interpretation Limits

This is a geometric observability result plus a provisional owned-hardware
contract, not a completed calibrated sensor model.

- Bristle visibility uses MuJoCo segmentation at the declared model-input
  resolution; it is not a learned detection probability.
- Camera housing geoms are excluded because final mounts and the inspection
  stow mechanism do not exist in CAD yet.
- Lens distortion, calibration residuals, defocus, auto-exposure, rolling
  shutter, and measured dropout are not modeled.
- Read/signal noise, latency, quantization, and glare are provisional declared
  approximations, not calibrated physical-camera models.
- The canvas input is superficial grayscale only. White-on-white coverage and
  other hidden material variables remain unobservable.
- The overhead view measures a visual profile. It must not be converted into
  exact simulator standoff before inference.
- `CameraSpatialLikelihood` now consumes registered global/foveal products.
  It predicts superficial grayscale from latent thickness and surface tone,
  infers a mean-field occlusion/outlier responsibility from image residuals,
  and reports state-complexity, occlusion-complexity, and expected negative
  log likelihood separately. It is a provisional analytic likelihood, not a
  learned camera encoder or a calibrated physical-camera model.
- The simulator now renders at declared native resolution by default and
  produces native-derived global and requested foveal products. This is not
  live HDMI acquisition, and physical crop, exposure, lens, timing, and
  photometric behavior remain provisional.
- Product derivation latency is not yet separated from the declared camera
  delivery latency.

The complete numeric results are stored beside the figures in
`camera_pose_sweep.json`, `contact_poses.csv`, and
`camera_pose_metrics.csv`. Regenerate them with:

```powershell
python -m active_painter.camera_pose_sweep `
  --output-dir docs\figures\camera_observability `
  --grid-size 9 `
  --ray-grid-size 9
```
