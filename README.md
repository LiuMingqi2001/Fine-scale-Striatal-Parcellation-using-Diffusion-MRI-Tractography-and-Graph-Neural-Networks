# Fine-scale Striatal Parcellation using Diffusion MRI Tractography and Graph Neural Networks

This repository contains the source code for the paper **"Fine-scale striatal parcellation using diffusion MRI tractography and graph neural networks."** The code leverages graph neural networks (GNNs) to perform fine-scale parcellation of the striatum based on diffusion MRI tractography data. All code is located in the `src/` folder.

> **Note:** This repository currently contains only the `src` part of the project. You may later add additional directories (e.g., for data, logs, or pre-trained parameters) as your project evolves.

---

## Overview

The project pipeline consists of two main stages:

1. **Pretraining:**  
   A graph convolutional network (GCN) is pre-trained to learn latent representations from tractography data. The network uses TransformerConv layers with edge features derived from both Euclidean distances and a sigmoid-transformed normalization of voxel distances.

2. **Finetuning & Clustering:**  
   The pre-trained model is further fine-tuned with a joint loss that incorporates reconstruction, clustering, and diversity losses. K-Means clustering is integrated into the training loop to drive the parcellation process.

---

## Code Structure

The code is organized into the following modules within the `src/` directory:

- **`data_loader.py`**  
  Contains functions for loading and preprocessing diffusion MRI tractography data. This module builds graph representations (using PyTorch Geometric's `Data` objects) from subject-specific files.

- **`model.py`**  
  Implements the graph convolutional network (`GCNNet`). The network is constructed using TransformerConv layers along with batch normalization and ReLU activations.

- **`trainer.py`**  
  Provides the pretraining routine. This module defines a training loop that optimizes the reconstruction loss, saves the best model, and logs training progress using TensorBoard.

- **`finetuner.py`**  
  Implements the finetuning and clustering procedure. This module integrates K-Means clustering with the model’s latent representations and defines a joint loss (reconstruction, clustering, and diversity losses) for fine-scale parcellation.

- **`utils.py`** (Optional)  
  You can include helper functions (e.g., logging, metrics, or additional utilities) in this module.

---

## Requirements

Make sure you have the following installed:

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- NumPy
- SciPy
- scikit-learn
- TensorBoard
- Matplotlib (optional, for visualization)

Install the Python dependencies using:

```bash
pip install torch torch_geometric numpy scipy scikit-learn tensorboard matplotlib
