import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
import numpy as np
from src.model import GCNNet

class GCN_KMeans:
    def __init__(self, n_clusters=3, alpha=1.0, merge_threshold=0.5, hidden_channels=96, output_channels=128):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.merge_threshold = merge_threshold
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t().clone().detach().requires_grad_(True)

    @staticmethod
    def lossf(q, p):
        return 0.5 * (F.kl_div(torch.log(q), (p + q) / 2, reduction='batchmean') +
                      F.kl_div(torch.log(p), (p + q) / 2, reduction='batchmean'))

    def get_t_distribution(self, x, cluster_centers, device='cuda:0'):
        # Compute the Student t-distribution for clustering
        xe = torch.unsqueeze(x, 1).to(device) - cluster_centers.to(device)
        q = 1.0 / (1.0 + (torch.sum(xe ** 2, dim=2) / self.alpha))
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def joint_loss(self, output_de, output_ae, data, mbk, cluster_centers,
                   lambda_distance=2, lambda_DEC=10, device='cuda:0'):
        recon_loss = F.mse_loss(output_de, data.x.float())
        assigned_clusters = mbk.predict(output_ae.cpu().detach().numpy())
        cluster_centers_tensor = torch.tensor(cluster_centers[assigned_clusters], device=device)
        distances = torch.norm(output_ae - cluster_centers_tensor, dim=-1, p=2)
        
        # Diversity loss (using a margin)
        cluster_centers_tensor_full = torch.tensor(cluster_centers, device=device)
        diff = cluster_centers_tensor_full.unsqueeze(0) - cluster_centers_tensor_full.unsqueeze(1)
        centro_distance = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        mask = torch.eye(centro_distance.size(0), device=device).bool()
        centro_distance = centro_distance.masked_select(~mask)
        margin = 3.5
        diversity_loss = torch.relu(margin - torch.min(centro_distance))
        
        q = self.get_t_distribution(output_ae, torch.tensor(cluster_centers, device=device), device=device)
        p = self.target_distribution(q)
        DEC_loss = self.lossf(q, p)
        
        joint_loss = (recon_loss +
                      lambda_distance * distances.mean() +
                      lambda_DEC * DEC_loss +
                      diversity_loss)
        return joint_loss, recon_loss, lambda_distance * distances.mean(), lambda_DEC * DEC_loss, diversity_loss

    def train_model(self, graphs, loader, epochs=10, stopthreshold=0.001,
                    lambda_distance=1.5, lambda_DEC=10, device='cuda:0', log_dir="logs/finetune"):
        # Initialize model and load pre-trained parameters
        model = GCNNet(in_channels=72, hidden_channels=self.hidden_channels, out_channels=self.output_channels).to(device)
        # Load pre-trained parameters (adjust the path as needed)
        pretrained_path = f'params/<your_pretrained_model>.pth'
        model.load_state_dict(torch.load(pretrained_path))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = StepLR(optimizer, step_size=1500, gamma=0.1)
        
        # Initialize KMeans with features from a subset of graphs
        mbk = KMeans(n_clusters=self.n_clusters, n_init=100)
        features = []
        model.eval()
        with torch.no_grad():
            for i in range(min(80, len(graphs))):
                data = graphs[i].to(device)
                result_ae, _ = model(data)
                features.append(result_ae.cpu().detach().numpy())
        features_np = np.concatenate(features)
        mbk.fit(features_np)
        cluster_centers = mbk.cluster_centers_
        
        writer = SummaryWriter(log_dir)
        best_loss = float('inf')
        best_model_params = None
        prev_assignments = None

        for epoch in range(1, epochs + 1):
            for data in loader:
                data = data.to(device)
                model.train()
                optimizer.zero_grad()
                output_ae, output_de = model(data)
                loss_tuple = self.joint_loss(output_de, output_ae, data, mbk, cluster_centers, lambda_distance, lambda_DEC, device=device)
                loss = loss_tuple[0]
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Evaluation step: compute losses and update cluster centers periodically
            epoch_loss = 0.0
            current_assignments = []
            clusters_features = []
            model.eval()
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    output_ae, output_de = model(data)
                    loss_tuple = self.joint_loss(output_de, output_ae, data, mbk, cluster_centers, lambda_distance, lambda_DEC, device=device)
                    epoch_loss += loss_tuple[0]
                    current_clusters = mbk.predict(output_ae.cpu().numpy())
                    clusters_features.append(output_ae.cpu().detach().numpy())
                    current_assignments.extend(current_clusters)
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar("Joint Loss", epoch_loss, epoch)
            print(f'Epoch [{epoch}/{epochs}], Joint Loss: {epoch_loss:.4f}')
            
            # Optionally: update cluster centers every few epochs
            if epoch % 50 == 0:
                clusters_features = np.concatenate(clusters_features)
                mbk = KMeans(n_clusters=self.n_clusters, n_init=100)
                mbk.fit(clusters_features)
                cluster_centers = mbk.cluster_centers_
                
            # (Optional) Check for convergence based on assignment changes
            if prev_assignments is not None:
                num_non_converged = np.sum(np.array(current_assignments) != np.array(prev_assignments))
                if num_non_converged < stopthreshold * len(current_assignments):
                    best_model_params = model.state_dict()
            prev_assignments = current_assignments

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_params = model.state_dict()
        
        # Save best model and cluster centers
        torch.save(best_model_params, f'params/finetuned_model_{epochs}.pth')
        # Optionally save cluster centers (e.g., using joblib)
        writer.close()
