# Solidification Process

This repository contains a minimal CUDA implementation of a 2D phase–field model for dendritic solidification. The solver is written in Python using PyCUDA and mirrors the anisotropic dendrite example from FiPy.

## Requirements
- Python 3
- [PyCUDA](https://documen.tician.de/pycuda/)

Install the requirements with pip:

```bash
pip install pycuda
```

## Running the solver
Execute the script directly after installing the dependencies and ensuring a CUDA-capable GPU is available:

```bash
python anisotropic_dendrite_solver.py
```

The solver initializes a circular solid seed with a small random perturbation and automatically chooses a stable time step based on the grid spacing and material parameters. When finished it prints the minimum, maximum and mean values of the phase field along with a small center slice.
