# SpecTra-Net: Unified Spatiotemporal Learning for High-Activity Regimes

**SpecTra-Net** is a deep learning framework designed for complex spatiotemporal forecasting tasks. It unifies Grid and Node-based representations and incorporates advanced mechanisms to handle high-activity and complex data regimes.

## Key Components

The architecture consists of four core modules:
1.  **SFE (Spectral Feature Extraction):** Captures frequency-domain patterns using FFT-based processing.
2.  **T-Gating (Temporal Gating):** Dynamically filters temporal information based on input complexity.
3.  **GOT (Geometric Optimal Transport):** Handles spatial dependencies using optimal transport theory with prototype learning.
4.  **MGAT (Multi-Graph Attention):** Enhances spatial representation through graph attention mechanisms considering geodesic distances.

## Features

* **Unified Input Support:** Seamlessly handles both Grid (`T, H, W, C`) and Node (`T, N, C`) data formats.
* **Safe Crop Mechanism:** Automatically applies cropping only to grid data while preserving node data integrity.
* **Robust Spatial Modeling:** Combines geometric embeddings and learnable prototypes.
* **Comprehensive Metrics:** Evaluates using RMSE, MAE, SMAPE, and R2 (Global & Per-channel).

## Requirements

* Python 3.8+
* PyTorch >= 1.10
* NumPy

## Installation

```bash
pip install -r requirements.txt
