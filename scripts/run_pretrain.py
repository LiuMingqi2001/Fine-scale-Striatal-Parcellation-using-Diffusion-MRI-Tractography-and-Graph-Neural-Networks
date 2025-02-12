import torch
from torch_geometric.loader import DataLoader
from src.data_loader import load_hcp_graphs
from src.model import GCNNet
from src.trainer import pretrain

def main():
    data_dir = "/path/to/HCP"
    subject_file = "/path/to/sub_ids"
    graphs = load_hcp_graphs(data_dir, subject_file, omatrix_folder="probtrackx_R_omatrix2", distance_threshold=5)
    loader = DataLoader(graphs, batch_size=10, shuffle=True)
    
    model = GCNNet(in_channels=72, hidden_channels=96, out_channels=128)
    pretrain(model, loader, epochs=2000, log_interval=10, 
             save_path="params/pretrained_model.pth", device='cuda:0')

if __name__ == '__main__':
    main()
