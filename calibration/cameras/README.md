# Fixed-Camera Intrinsic Calibration

This is the first physical calibration stage for the continuous oblique
cameras. It measures each delivered HDMI image independently. Do not calibrate
from still photographs: movie crop, scaling, stabilization, and lens correction
can differ from the 3840 x 2160 feed used by the controller.

## Target

`checkerboard_11x8_28mm.svg` has 11 x 8 inner corners and 28 mm squares. Its
physical page is 376 x 292 mm, which fits on A3. Print at **100% / actual size**
with all fit-to-page scaling disabled. Verify that ten adjacent square widths
measure exactly 280 mm, then bond the print to a flat rigid panel.

Regenerate it with:

```powershell
python -m active_painter.camera_calibration generate-target `
  --output calibration\cameras\checkerboard_11x8_28mm.svg
```

## Locked Capture Modes

Use the exact modes intended for operation:

- `canvas_right_oblique`: OM System OM-1, 25 mm, UHD 3840 x 2160 at 30 Hz,
  full-width 16:9;
- `canvas_left_oblique`: Sony A7R II, 35 mm, UHD 3840 x 2160 at 30 Hz,
  APS-C/Super 35 enabled.

For both cameras, use the intended HDMI capture device and keep the following
fixed for the entire dataset: resolution, frame rate, crop mode, manual focus,
aperture, shutter, ISO, white balance, lens-correction settings, and capture-
card pixel conversion. Disable IBIS, electronic stabilization, digital zoom,
and autofocus. Start around f/5.6 on the OM-1 and f/8 on the A7R II, with enough
light for a short exposure and sharp corners across the board.

## Images

Capture 25-30 sharp frames per camera. The board should appear:

- near every image edge and corner as well as the center;
- at several moderate horizontal and vertical tilts;
- at several rotations in the image plane;
- large enough that individual squares remain crisp;
- entirely visible, with no motion blur or glare obscuring corners.

Use frames extracted from the HDMI stream without resizing. Suggested folders:

```text
calibration/cameras/om1_25mm/raw/
calibration/cameras/a7rii_35mm_super35/raw/
```

## Solve

```powershell
python -m active_painter.camera_calibration calibrate `
  --camera-name canvas_right_oblique `
  --images calibration\cameras\om1_25mm\raw `
  --output calibration\cameras\om1_25mm\intrinsics.json

python -m active_painter.camera_calibration calibrate `
  --camera-name canvas_left_oblique `
  --images calibration\cameras\a7rii_35mm_super35\raw `
  --output calibration\cameras\a7rii_35mm_super35\intrinsics.json
```

The command reports Brown-Conrady intrinsics, distortion, measured field of
view, per-view residuals, frame coverage, tilt range, and explicit quality
gates. A result is not copied into the authoritative MJCF until it is
`accepted` and its residual plots and held-out canvas registration have been
reviewed. Camera-to-canvas extrinsics are a separate second stage performed
after the final mounts exist.
