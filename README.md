# PINNs for the 1D Heat Equation

This repository contains a set of Python experiments demonstrating how to solve the 1D heat equation with Physics-Informed Neural Networks (PINNs) using TensorFlow.

The project trains a neural network to satisfy:

- the governing PDE residual,
- the initial condition,
- and the boundary conditions,

without requiring a large labeled dataset.

## What is included

The repository contains several scripts exploring different initial conditions for the heat equation on the unit interval:

- `PINN_Heat1D_sin_human.py` — uses a sinusoidal initial condition
- `PINN_Heat1D_Gauss_human.py` — uses a Gaussian initial condition
- `PINN_Heat1D_Heaviside.py` — uses a Heaviside-style step initial condition
- `saving_an_animation.py` — utilities for generating animated visualizations
- `test2.py`, `test3.py` — additional exploratory scripts

## PDE being solved

The implemented model solves the 1D heat equation:

u_t = alpha * u_xx

with boundary conditions and initial data chosen per experiment. The network learns a solution u(x, t) over the domain x in [0, 1], t in [0, 1].

## Method

Each script follows the standard PINN approach:

1. Define a neural network that takes `(x, t)` as input and outputs `u(x, t)`.
2. Compute the PDE residual using automatic differentiation.
3. Penalize the residual, the initial condition mismatch, and the boundary condition mismatch.
4. Train with Adam optimizer.
5. Visualize the predicted solution with contour plots, 3D surface plots, and frame-based animations.

## Dependencies

Install the required Python packages:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Run an example

From the repository root:

```bash
python PINN_Heat1D_sin_human.py
```

or:

```bash
python PINN_Heat1D_Gauss_human.py
```

or:

```bash
python PINN_Heat1D_Heaviside.py
```

These scripts generate plots and save animation frames/GIF outputs in the current working directory.

## Project structure

```text
PINNs/
├── PINN_Heat1D_sin_human.py
├── PINN_Heat1D_Gauss_human.py
├── PINN_Heat1D_Heaviside.py
├── saving_an_animation.py
├── test2.py
├── test3.py
├── medias/
├── README.md
└── frames/   # generated during execution
```

## Notes

- This project is designed as a compact educational/demo implementation.
- Training can take time depending on the chosen number of collocation points and epochs.
- The outputs are intended to visualize how PINNs can approximate PDE solutions with differentiable loss functions.

## License

This project does not currently declare a license in the repository. If you plan to reuse or distribute the code, add an appropriate open-source license before publication.

## Acknowledgements

This work is based on the idea of Physically Informed Neural Networks for solving partial differential equations and is intended for experimentation and learning.
