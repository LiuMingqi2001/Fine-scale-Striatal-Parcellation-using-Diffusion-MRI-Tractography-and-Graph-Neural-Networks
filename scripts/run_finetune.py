import os
from torch_geometric.loader import DataLoader
from src.data_loader import load_hcp_graphs
from src.finetuner import GCN_KMeans

def main():
    # Update these paths as necessary for your environment:
    data_directory = "/path/to/HCP/data"         # Base directory containing subject folders
    subject_names_file = "/path/to/sub_ids"       # File listing the subject folder names

    # For finetuning, select the appropriate hemisphere folder.
    # "probtrackx_L_omatrix2" is used here for the left hemisphere.
    omatrix_folder = "probtrackx_L_omatrix2"
    distance_threshold = 5

    # Load graph data for each subject
    graphs = load_hcp_graphs(data_directory, subject_names_file, omatrix_folder, distance_threshold)
    
    # Create a DataLoader for batching (adjust batch_size as needed)
    loader = DataLoader(graphs, batch_size=5, shuffle=False)

    # Initialize the finetuning and clustering module
    # Hyperparameters: adjust n_clusters, alpha, merge_threshold, hidden_channels, output_channels as needed.
    finetuner = GCN_KMeans(n_clusters=5, alpha=1.0, merge_threshold=6.0, hidden_channels=96, output_channels=128)

    # Start the finetuning process.
    # epochs: number of training epochs.
    # stopthreshold: a threshold for label convergence (optional).
    # lambda_distance and lambda_DEC: weighting factors for the joint loss.
    # device: 'cuda:0' if using GPU (ensure CUDA is available) or 'cpu'.
    # log_dir: directory for TensorBoard logs.
    finetuner.train_model(
        graphs,
        loader,
        epochs=1000,
        stopthreshold=0.005,
        lambda_distance=1.5,
        lambda_DEC=30,
        device='cuda:0',
        log_dir="logs/finetune"
    )

if __name__ == '__main__':
    main()
