# Camera Observability And Grayscale Input Brief

**Rig:** `provisional-multiview-v2`

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

The two oblique views retain their existing positions but now use a 24°
vertical field of view instead of 45°. At the simulated 1024 × 1024 acquisition
resolution, the canvas occupies approximately 737–740 pixels horizontally and
867–871 pixels vertically. The 12.7 mm brush spans about 17.7 source pixels at
canvas center. The narrower view therefore uses substantially more of the
sensor without clipping the canvas.

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

| View | Role | Availability | Model input | Rate | Registration |
| --- | --- | --- | ---: | ---: | --- |
| Right oblique | Contact and surface observation | Continuous | 512 × 512 | 30 Hz | Canvas UV homography |
| Left oblique | Contact and surface observation | Continuous | 512 × 512 | 30 Hz | Canvas UV homography |
| Head-on inspection | Whole-painting inspection | Park only | 512 × 512 | 5 Hz / on demand | Canvas UV homography |
| Overhead profile | Brush standoff/compliance | Continuous | 640 × 480 | 60 Hz | Calibrated edge profile |

For a 508 mm square canvas, the canonical 512 × 512 canvas image represents
approximately 0.99 mm per pixel and a 12.7 mm brush spans 12.8 canonical
pixels. That is a useful first baseline for global prediction without making
the observation tensor unnecessarily large. Native frames should still be
retained for calibration, diagnostics, and later foveal crops.

The recommended physical acquisition path is:

1. acquire synchronized global-shutter monochrome frames at the camera's
   native 1–3 MP resolution;
2. subtract dark response and apply flat-field/exposure calibration;
3. undistort using the measured lens model;
4. rectify the three canvas views into the shared 512 × 512 canvas frame;
5. preserve per-pixel validity, occlusion, and precision masks;
6. convert calibrated intensity to normalized floating-point grayscale for
   the likelihood/encoder.

This does not make white paint on a white ground directly observable as
material coverage. Coverage remains hidden material state. Controlled
directional illumination, surface texture, and limited specular cues provide
visual evidence, with ambiguity represented as posterior uncertainty.

## Camera Class

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

This is a geometric observability result, not a completed sensor model.

- Bristle visibility uses MuJoCo segmentation at the declared model-input
  resolution; it is not a learned detection probability.
- Camera housing geoms are excluded because final mounts and the inspection
  stow mechanism do not exist in CAD yet.
- The simulation does not yet include lens distortion, calibration residuals,
  defocus, exposure dynamics, glare, shot/read noise, latency, or dropout.
- MuJoCo renders blank canvas shading only. The Python material process has not
  yet been inserted into these camera frames.
- The overhead view measures a visual profile. It must not be converted into
  exact simulator standoff before inference.

The complete numeric results are stored beside the figures in
`camera_pose_sweep.json`, `contact_poses.csv`, and
`camera_pose_metrics.csv`. Regenerate them with:

```powershell
python -m active_painter.camera_pose_sweep `
  --output-dir docs\figures\camera_observability `
  --grid-size 9 `
  --ray-grid-size 9
```
