# Tensor_DMD

This repository implements **Tensor Dynamic Mode Decomposition (TDMD)** based on the **T-product** for third-order dynamical systems.  
The code applies DMD directly to tensor-valued data without flattening the spatial structure into vectors, which allows the method to better preserve multidimensional correlations in applications such as video and climate data.

In particular, this repository includes two numerical experiments corresponding to the examples in the paper:

1. **Synthetic video experiment**: foreground/background separation for a moving object in a noisy video.
2. **Climate data experiment**: reconstruction of sea surface temperature (SST) data.

---

## Overview

This repository implements **Tensor Dynamic Mode Decomposition (TDMD)** based on the **T-product** for third-order dynamical systems.

Given a sequence of tensor snapshots  
X₀, X₁, …, X_T,  
TDMD models the dynamics as

X_{t+1} = A ★ X_t,

where ★ denotes the **T-product**.

Unlike classical DMD, which requires flattening multidimensional data into vectors, TDMD operates directly on **3D tensors** and preserves the intrinsic spatial structure of the data.

The implementation in this repository focuses on the **T-product formulation** and demonstrates its use for:

- low-rank tensor dynamical modeling  
- reconstruction of tensor snapshots  
- separation of persistent and transient components  
- comparison with standard matrix-based DMD

---

## File description

### `TDMD_fgbg.m`
This script corresponds to the **first numerical experiment** in the paper.  
It applies TDMD to a synthetic video sequence and performs **foreground/background separation**.

More specifically, the script:
- loads or generates a 3D video tensor,
- applies TDMD based on the T-product,
- identifies dynamic modes,
- separates the approximately static **background** from the moving **foreground** object.

This is the script for the **synthetic moving-square experiment**.

---

### `Climate_all.m`
This script corresponds to the **second numerical experiment** in the paper.  
It applies TDMD to the **sea surface temperature (SST)** dataset.

More specifically, the script:
- loads the SST tensor data,
- applies TDMD to the climate tensor,
- reconstructs the monthly SST fields,
- evaluates reconstruction performance.

This is the script for the **climate data experiment**.

---

### `sst_60x60xT_sub.mat`
This file contains the **climate dataset** used in the second experiment.

It is a tensor of sea surface temperature data on a \(60 \times 60 \times T\) grid, where:
- the first two modes correspond to the spatial grid,
- the third mode corresponds to time.

This dataset is used by `Climate_all.m`.

---

### `bcirc.m`
This file provides an auxiliary function for constructing the **block circulant matrix** associated with a third-order tensor.

The block circulant operator is a fundamental building block in the T-product framework.  
It converts a tensor into a structured block matrix so that tensor multiplication under the T-product can be implemented through matrix multiplication.

This function is a core helper routine used in TDMD-related computations.

---

### `tt_matrix.m`
This file provides another auxiliary function used in the tensor computations required by TDMD.

It supports the implementation of tensor-based matrix operations that arise in the T-product framework and is used together with `bcirc.m` as a helper routine in the experiments.

---

### `README.md`
This file provides an overview of the repository, describes the purpose of the code, and explains the role of each file.

---

## Numerical experiments

### Experiment 1: Foreground/Background Separation
The script `TDMD_fgbg.m` implements the first experiment.  
A synthetic video sequence with a moving foreground object and a noisy background is analyzed using TDMD.  
The dominant tensor modes are used to separate:
- the **background** (persistent component),
- the **foreground** (transient/moving component).

---

### Experiment 2: Climate Data Reconstruction
The script `Climate_all.m` implements the second experiment.  
It applies TDMD to the SST tensor stored in `sst_60x60xT_sub.mat` and studies the reconstruction of the climate data over time.

---

## Notes

- `TDMD_fgbg.m` is the script for the **first experiment** (video foreground/background separation).
- `Climate_all.m` is the script for the **second experiment** (climate data reconstruction).
- `bcirc.m` and `tt_matrix.m` are **auxiliary functions** needed by the tensor-based implementation.
- `sst_60x60xT_sub.mat` is the **SST dataset** used in the climate example.

