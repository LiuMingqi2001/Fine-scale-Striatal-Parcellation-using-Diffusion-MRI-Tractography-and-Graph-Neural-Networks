import os
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

def load_hcp_graphs(data_directory, subject_names_file, omatrix_folder="probtrackx_R_omatrix2", distance_threshold=5):
    """Load graphs for pretraining.
    
    Parameters:
        data_directory (str): Base directory for the HCP data.
        subject_names_file (str): File listing subject directories.
        omatrix_folder (str): Folder name (e.g., "probtrackx_R_omatrix2" for pretraining,
                               "probtrackx_L_omatrix2" for finetuning).
        distance_threshold (int): Threshold used for edge calculation.
    
    Returns:
        list: List of torch_geometric.data.Data objects.
    """
    graphs = []
    with open(subject_names_file, 'r') as file:
        subject_names = [line.strip() for line in file.readlines()]
    
    for subject_dir in subject_names:
        data_path = os.path.join(data_directory, subject_dir, omatrix_folder, "finger_print_fiber.npz")
        data_array = sp.load_npz(str(data_path))
        subject_data = data_array.toarray()
        
        # Load voxel coordinates
        coor_path = os.path.join(data_directory, subject_dir, omatrix_folder, "coords_for_fdt_matrix2")
        coordinates = np.loadtxt(coor_path)
        subject_coords = torch.tensor(coordinates[:, :3].astype(int), dtype=torch.float)
        
        # Compute distances between voxels (for edges)
        diff = subject_coords.unsqueeze(1) - subject_coords.unsqueeze(0)
        distances = torch.norm(diff, dim=-1, p=2)
        
        # Create edges (here you can change the condition to include a threshold if needed)
        edges = torch.nonzero((0 < distances), as_tuple=False).t()
        
        # Compute edge features (using a sigmoid on a normalized Euclidean distance)
        x = torch.tensor(subject_data, dtype=torch.float)
        euclidean_distances = torch.sqrt(torch.sum((x.unsqueeze(1) - x.unsqueeze(0))**2, dim=2))
        sorted_distances, _ = torch.sort(euclidean_distances, dim=1)
        k_smallest_distances = sorted_distances[:, 26]
        normalized_distances = euclidean_distances - k_smallest_distances
        sigmoid_distances = 1 / (1 + torch.exp(normalized_distances))
        edge_features = sigmoid_distances[edges[0], edges[1]].unsqueeze(1)
        
        # Build graph data object
        graph = Data(x=x, pos=subject_coords, edge_index=edges, edge_attr=edge_features)
        graphs.append(graph)
        
    return graphs
