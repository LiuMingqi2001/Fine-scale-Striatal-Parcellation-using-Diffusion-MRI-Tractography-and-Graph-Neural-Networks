# Fine-scale Striatal Parcellation using Diffusion MRI Tractography and Graph Neural Networks

This repository provides source code for the paper:

> **Gao, J., Liu, M., Qian, M., Tang, H., Wang, J., Ma, L., Li, Y., Dai, X., Wang, Z., Lu, F., & Zhang, F. (2025).**  
> *Fine-scale striatal parcellation using diffusion MRI tractography and graph neural networks.*  
> Medical Image Analysis, p.103482.

---

## Overview

This project uses **graph neural networks (GNNs)** to perform **fine-scale parcellation of the striatum** based on diffusion MRI tractography data. It consists of two main stages:

1. **Pretraining**  
   A GCN model (using TransformerConv layers) is trained on tractography data to learn latent representations. Euclidean distances and sigmoid-transformed voxel distances serve as edge features.

2. **Finetuning & Clustering**  
   The pre-trained model is further optimized with a joint loss that combines reconstruction, clustering, and diversity terms. **K-Means** clustering is integrated into the training loop to drive parcellation.

> **New**: The `scripts/` directory now includes complete pretraining and finetuning scripts.

---

## Citation

If you find this code useful, please cite:

```
@article{gao2025fine,
  title   = {Fine-scale striatal parcellation using diffusion MRI tractography and graph neural networks},
  author  = {Gao, J. and Liu, M. and Qian, M. and Tang, H. and Wang, J. and Ma, L. and Li, Y. and Dai, X. and Wang, Z. and Lu, F. and Zhang, F.},
  journal = {Medical Image Analysis},
  volume  = {},
  pages   = {103482},
  year    = {2025}
}
```

---

## Repository Structure

```
.
├── scripts/
│   ├── run_pretrain.py  <-- Pretraining script
│   └── run_finetune.py  <-- Finetuning script
├── src/
│   ├── data_loader.py   <-- Loads & preprocesses diffusion MRI data
│   ├── model.py         <-- Defines GCN architecture (TransformerConv layers)
│   ├── trainer.py       <-- Pretraining routine (reconstruction loss)
│   └── finetuner.py     <-- Finetuning & clustering (K-Means integration)
├── params/              <-- (Optional) Directory for saving trained models
└── README.md
```

### Scripts in `scripts/` Directory

- **`run_pretrain.py`**  
  1. Loads diffusion MRI graph data (HCP dataset).  
  2. Initializes a GCN model.  
  3. Trains using a reconstruction loss function.  
  4. Saves the pretrained model to `params/pretrained_model.pth`.

  **Usage**:
  ```bash
  python scripts/run_pretrain.py
  ```

- **`run_finetune.py`**  
  1. Loads the pretrained model.  
  2. Applies a joint loss with clustering and diversity constraints.  
  3. Saves the finetuned model to `params/finetuned_model.pth`.

  **Usage**:
  ```bash
  python scripts/run_finetune.py
  ```

### Core Modules in `src/` Directory

- **`data_loader.py`**  
  Handles data loading and preprocessing for diffusion MRI tractography, converting data into graph representations.

  - **Omatrix Folder Note**:  
    The `omatrix_folder` parameter (e.g., `"probtrackx_R_omatrix2"` or `"probtrackx_L_omatrix2"`) indicates where connectivity matrices and coordinate files for each subject are stored.

- **`model.py`**  
  Defines the `GCNNet` class. Uses **TransformerConv** layers with batch normalization and ReLU activation.

- **`trainer.py`**  
  Implements the pretraining loop, optimizing reconstruction loss and logging via TensorBoard.

- **`finetuner.py`**  
  Combines the pretrained model with K-Means for clustering-driven finetuning, incorporating reconstruction, clustering, and diversity losses.

---

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- NumPy
- SciPy
- scikit-learn
- TensorBoard
- Matplotlib (optional for visualization)

Install dependencies:
```bash
pip install torch torch_geometric numpy scipy scikit-learn tensorboard matplotlib
```

---

## HCP Dataset Preprocessing

For fiber tract tracing of the HCP dataset, please refer to:
- [Fiber-Tract-Tracing-Pipeline (LiuMingqi2001)](https://github.com/LiuMingqi2001/Fiber-Tract-Tracing-Pipeline)
- [BAI-Net-Brain-Atlas-Individualization-Network (trichtu)](https://github.com/trichtu/BAI-Net-Brain-Atlas-Individualization-Network)

---

## Notes & Recommendations

1. **Path Configuration**  
   Update dataset paths (e.g., `/path/to/HCP/data`, `/path/to/sub_ids`) in your scripts to match your environment.

2. **Model Saving**  
   If using `params/` for model checkpoints, ensure it exists:
   ```python
   import os
   os.makedirs("params", exist_ok=True)
   ```

3. **GPU/CPU Compatibility**  
   Modify the device assignment if no CUDA GPU is available:
   ```python
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   ```

4. **K-Means Initialization**  
   If training is slow or memory-intensive, consider lowering `n_init` (default = `100`) in `KMeans`.

---

**This code was refactored for clarity and optimization; minor bugs may remain.**  
Feel free to open issues or pull requests to report any problems or suggest improvements.
