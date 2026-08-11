# Compact Dual-Camera Intrinsic Calibration

This is the first physical calibration stage for the two selected Raspberry
Pi Global Shutter Cameras (Sony IMX296). The cameras and provisional 4 mm
CS-mount lenses have not yet been purchased, so no calibrated result exists.

## Target

`checkerboard_11x8_28mm.svg` has 11 x 8 inner corners and 28 mm squares. Its
376 x 292 mm page fits on A3. Print at **100% / actual size**, disable
fit-to-page scaling, verify that ten adjacent square widths measure exactly
280 mm, and bond the print to a flat rigid panel.

Regenerate it with:

```powershell
python -m active_painter.camera_calibration generate-target `
  --output calibration\cameras\checkerboard_11x8_28mm.svg
```

## Locked Capture Mode

Use the exact operational pipeline for both `canvas_right_oblique` and
`canvas_left_oblique`: full IMX296 1456 x 1088 array, global shutter, the final
frame rate, and the intended CSI receiver. Keep resolution, crop, manual
focus, aperture, exposure, gain, white balance, correction settings, and pixel
conversion fixed throughout. Disable automatic focus/exposure and digital
scaling. If the purchased lens or mode differs from the provisional 4 mm / 60
Hz contract, update the MJCF metadata before calibration.

## Images

Capture 25-30 sharp frames per camera. Put the board near every image edge and
corner as well as the center, with several horizontal/vertical tilts and
in-plane rotations. Keep the board fully visible, crisp, and glare-free.
Extract native frames without resizing.

Suggested folders:

```text
calibration/cameras/imx296_right_4mm/raw/
calibration/cameras/imx296_left_4mm/raw/
```

## Solve

```powershell
python -m active_painter.camera_calibration calibrate `
  --camera-name canvas_right_oblique `
  --images calibration\cameras\imx296_right_4mm\raw `
  --output calibration\cameras\imx296_right_4mm\intrinsics.json

python -m active_painter.camera_calibration calibrate `
  --camera-name canvas_left_oblique `
  --images calibration\cameras\imx296_left_4mm\raw `
  --output calibration\cameras\imx296_left_4mm\intrinsics.json
```

The solver reports Brown-Conrady intrinsics/distortion, measured field of
view, per-view residuals, frame coverage, tilt range, and quality gates. Do not
copy a result into the authoritative MJCF until it is `accepted` and residual
plots plus held-out canvas registration have been reviewed. Camera-to-canvas
extrinsics and cross-camera timing are separate stages performed after the
final rigid crossbar exists.
