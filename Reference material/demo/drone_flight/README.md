# Drone flight demo

Hybrid-lite aerial navigation: discrete **EFE** waypoint lattice (Chapter 9), smooth executed path with preference growth (Chapter 13 navigation stub), and **LGS** fusion of noisy position reads (Chapter 3).

## Run

```bash
python demo/drone_flight/visualize_drone_flight.py --save
```

## Library APIs

- `simulate_pomdp_agent`, `make_gridworld`
- `simulate_robot_navigation`
- `LinearGaussianSystem.posterior_batch`

## Outputs

- Figure: `output/figures/demo/drone_flight/visualize_drone_flight.png`
- Data: `output/data/demo/drone_flight/visualize_drone_flight.{npz,json}`

## Limitations (v1)

The kinematic path is not closed-loop controlled by the POMDP planner; LGS uses i.i.d. reads as a teaching fusion pattern.
