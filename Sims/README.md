# Plasma Cosmology v0.1 - Galaxy Rotation

**Author:** Bucurenciu Mihai Alexandru  
**Version:** 0.1  
**Date:** May 2026  
**DOI:** (will be assigned by Zenodo)

## Abstract

We present a 3D GPU-accelerated ideal magnetohydrodynamic (MHD) simulation framework designed to test whether magnetic tension and J×B forces from a strong toroidal magnetic field (Z-pinch configuration) can provide significant radial support in galactic disks, potentially contributing to the observed flat rotation curves without invoking dark matter.

The code implements:
- Staggered constrained transport (CT) magnetic field evolution
- Conservative HLLD Riemann solver
- FFT-based self-gravity (Poisson solver)
- Strong toroidal magnetic field initialization
- Realistic galactic rotation seed
- Rotation curve diagnostics and basic J×B analysis

Preliminary results show that the inclusion of a strong toroidal magnetic field improves the match between simulated and observed flat rotation curves compared to gravity-only models. This work represents an early numerical exploration within the Plasma Cosmology framework, investigating electromagnetic effects as a possible contributor to galactic dynamics.

## Features

- CUDA/GPU accelerated using CuPy
- 3D Cartesian grid with periodic boundaries
- Self-gravity via FFT Poisson solver
- Strong toroidal (azimuthal) magnetic field (Z-pinch like)
- Rotation curve visualization
- Density and pressure floors for stability

## How to Run

```bash
pip install cupy-cuda12x matplotlib numpy
python main.py
