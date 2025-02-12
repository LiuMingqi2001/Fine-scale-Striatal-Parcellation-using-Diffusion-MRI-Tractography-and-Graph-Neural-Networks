import os
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

def load_hcp_graphs(data_directory: str, subject_names_file: str, omatrix_folder: str, distance_threshold: float = 5) -> list:
    """
    Load diffusion MRI tractography data and convert them into graph representations.

    Each subject's data is expected to be located in its own folder under `data_directory`. Within each subject's folder,
    the connectivity matrix and corresponding voxel coordinates are stored in a subfolder specified by `omatrix_folder`.
    
    **Note on `omatrix_folder`:**
      - Use `"probtrackx_R_omatrix2"` for data corresponding to the right hemisphere.
      - Use `"probtrackx_L_omatrix2"` for data corresponding to the left hemisphere.

    Parameters:
        data_directory (str): Base directory containing subject folders.
        subject_names_file (str): Path to a text file listing the subject folder names.
        omatrix_folder (str): Subfolder name containing the connectivity matrix and coordinate files.
        distance_threshold (float): Maximum distance for considering an edge between voxels (optional filtering).

    Returns:
        graphs (list): A list of `torch_geometric.data.Data` objects, one per subject.
    """
    graphs = []
    # Read subject folder names from file
    with open(subject_names_file, 'r') as file:
        subject_names = [line.strip() for line in file.readlines()]

    for subject_dir in subject_names:
        # Construct path for the connectivity matrix file (.npz format)
        data_path = os.path.join(data_directory, subject_dir, omatrix_folder, "finger_print_fiber.npz")
        data_array = sp.load_npz(str(data_path))
        subject_data = data_array.toarray()

        # Load voxel coordinates from the coordinate file
        coor_path = os.path.join(data_directory, subject_dir, omatrix_folder, "coords_for_fdt_matrix2")
        coordinates = np.loadtxt(coor_path)
        # Only use the first three columns (x, y, z)
        subject_coords = torch.tensor(coordinates[:, :3].astype(int), dtype=torch.float)

        # Compute pairwise Euclidean distances between voxels
        diff = subject_coords.unsqueeze(1) - subject_coords.unsqueeze(0)
        distances = torch.norm(diff, dim=-1, p=2)

        # Create edges: select all pairs with distance > 0.
        # You can modify this logic to apply filtering using `distance_threshold` if desired.
        edges = torch.nonzero((0 < distances), as_tuple=False).t()

        # Calculate edge features using a normalized version of the Euclidean distances
        x = torch.tensor(subject_data, dtype=torch.float)
        euclidean_distances = torch.sqrt(torch.sum((x.unsqueeze(1) - x.unsqueeze(0)) ** 2, dim=2))
        sorted_distances, _ = torch.sort(euclidean_distances, dim=1)
        # Use the 27th smallest distance (index 26) for normalization; adjust as needed
        k_smallest_distances = sorted_distances[:, 26].unsqueeze(1)
        normalized_distances = euclidean_distances - k_smallest_distances
        sigmoid_distances = 1 / (1 + torch.exp(normalized_distances))
        # Assign edge features based on the computed sigmoid values
        edge_features = sigmoid_distances[edges[0], edges[1]].unsqueeze(1)

        # Create the PyTorch Geometric Data object for this subject's graph
        graph = Data(x=x, pos=subject_coords, edge_index=edges, edge_attr=edge_features)
        graphs.append(graph)

    return graphs
