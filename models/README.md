# Models

The accepted native simulator reference is documented in
[`docs/NATIVE_PLANT_REFERENCE.md`](../docs/NATIVE_PLANT_REFERENCE.md) as
`native-abstract-v0`.

No MuJoCo XML model is currently accepted in this directory. The first S1
artifact must be an abstract clone of the native joint order, axes, ranges,
home pose, link lengths, canvas frame, and brush-tip site before physical
geometry or calibrated actuator parameters are introduced.

Expected future model lineages:

- `mujoco-abstract-v0`: native kinematic clone;
- `mujoco-calibrated-v0`: parameterized from physical measurements;
- `mujoco-safety-v0`: conservative hardware operating-envelope model.
