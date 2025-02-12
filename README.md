# Fine-scale Striatal Parcellation using Diffusion MRI Tractography and Graph Neural Networks

This repository contains the source code for the paper **"Fine-scale striatal parcellation using diffusion MRI tractography and graph neural networks."**

## Citation
Gao, J., Liu, M., Qian, M., Tang, H., Wang, J., Ma, L., Li, Y., Dai, X., Wang, Z., Lu, F., and Zhang, F., 2025. Fine-scale striatal parcellation using diffusion MRI tractography and graph neural networks. *Medical Image Analysis*, p.103482. The code leverages graph neural networks (GNNs) to perform fine-scale parcellation of the striatum based on diffusion MRI tractography data.

> **Note:** This repository now includes scripts for running pretraining and finetuning, located in the `scripts/` folder.

---

## Paper Information

This repository is associated with the following paper:

**Gao, J., Liu, M., Qian, M., Tang, H., Wang, J., Ma, L., Li, Y., Dai, X., Wang, Z., Lu, F., and Zhang, F. (2025).** Fine-scale striatal parcellation using diffusion MRI tractography and graph neural networks. *Medical Image Analysis*, p.103482.

## Overview

The project pipeline consists of two main stages:

1. **Pretraining:**  
   A graph convolutional network (GCN) is pre-trained to learn latent representations from tractography data. The network uses TransformerConv layers with edge features derived from both Euclidean distances and a sigmoid-transformed normalization of voxel distances.

2. **Finetuning & Clustering:**  
   The pre-trained model is further fine-tuned with a joint loss that incorporates reconstruction, clustering, and diversity losses. K-Means clustering is integrated into the training loop to drive the parcellation process.

---

Additionally, the fiber tract tracing preprocessing part of the HCP dataset is included in this repository.

## Code Structure

The code is organized into the following modules:

### **Main Scripts (Newly Added in `scripts/` Directory)**

- **`scripts/run_pretrain.py`**  
  - Loads diffusion MRI graph data from the HCP dataset.
  - Initializes a GCN model and trains it using reconstruction loss.
  - Saves the pre-trained model in `params/pretrained_model.pth`.
  
  **Usage:**
  ```bash
  python scripts/run_pretrain.py
  ```

- **`scripts/run_finetune.py`**  
  - Loads the pre-trained model.
  - Uses a joint loss function incorporating clustering and diversity constraints.
  - Saves the fine-tuned model in `params/finetuned_model.pth`.

  **Usage:**
  ```bash
  python scripts/run_finetune.py
  ```

### **Core Modules (`src/` Directory)**

- **`src/data_loader.py`**  
  Loads and preprocesses diffusion MRI tractography data, converting them into graph representations.

  - **Note on `omatrix_folder`:**  
    The parameter `omatrix_folder` specifies the folder name where the connectivity matrices and coordinate files are stored for each subject. For example, `"probtrackx_R_omatrix2"` refers to data for the right brain hemisphere, whereas `"probtrackx_L_omatrix2"` refers to data for the left brain hemisphere.

- **`src/model.py`**  
  Implements the graph convolutional network (`GCNNet`). The network is constructed using TransformerConv layers along with batch normalization and ReLU activations.

- **`src/trainer.py`**  
  Provides the pretraining routine, optimizing the reconstruction loss and logging training progress using TensorBoard.

- **`src/finetuner.py`**  
  Implements the finetuning and clustering procedure by integrating K-Means clustering with the model’s latent representations.

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

Install the dependencies using:

```bash
pip install torch torch_geometric numpy scipy scikit-learn tensorboard matplotlib
```

---

## HCP Dataset Preprocessing
The fiber tract tracing preprocessing part of the HCP dataset is available at: [https://github.com/LiuMingqi2001/Fiber-Tract-Tracing-Pipeline](https://github.com/LiuMingqi2001/Fiber-Tract-Tracing-Pipeline)

## Notes and Recommendations

1. **Ensure Correct Paths:** Update dataset paths (`/path/to/HCP/data`, `/path/to/sub_ids`, etc.) before running the scripts.
2. **Check `params/` Directory:** If saving models in `params/`, make sure the directory exists:
   ```python
   import os
   os.makedirs("params", exist_ok=True)
   ```
3. **GPU/CPU Compatibility:** If running on a system without CUDA, modify the device assignment:
   ```python
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   ```
4. **KMeans Initialization:** If training is slow, consider reducing `n_init=100` in `KMeans(n_clusters=..., n_init=100)`.

---

This code was uploaded after refactoring and optimization, and some minor bugs will be fixed later.

