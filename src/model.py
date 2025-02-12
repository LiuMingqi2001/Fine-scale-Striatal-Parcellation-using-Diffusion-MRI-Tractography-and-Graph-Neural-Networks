import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
import torch.nn.init as init

class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = out_channels
        
        # Define TransformerConv layers (uncomment others if you want to experiment)
        self.conv1 = torch_geometric.nn.TransformerConv(in_channels, hidden_channels // 2, heads=2, bias=True, edge_dim=1)
        self.conv2 = torch_geometric.nn.TransformerConv(hidden_channels, out_channels // 2, heads=2, bias=True, edge_dim=1)
        self.conv3 = torch_geometric.nn.TransformerConv(out_channels, hidden_channels // 2, heads=2, bias=True, edge_dim=1)
        self.conv4 = torch_geometric.nn.TransformerConv(hidden_channels, in_channels // 2, heads=2, bias=True, edge_dim=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(in_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            if hasattr(layer, 'weight') and layer.weight is not None:
                init.xavier_uniform_(layer.weight.data, gain=init.calculate_gain('relu'))
            if hasattr(layer, 'bias') and layer.bias is not None:
                init.constant_(layer.bias.data, 0.0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x_ae = F.relu(x)

        x = self.conv3(x_ae, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x_de = F.relu(x)

        return x_ae, x_de
