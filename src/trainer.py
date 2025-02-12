# src/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from src.model import GCNNet

def pretrain(model, loader, epochs, log_interval, save_path, device='cuda:0'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=700, gamma=0.1)
    mseloss = nn.MSELoss().to(device)
    best_loss = float('inf')
    best_model_params = None
    train_loss = []
    writer = SummaryWriter("logs/pretrain")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output_ae, output_de = model(data)
            loss = mseloss(output_de, data.x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Evaluate on training data
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                _, output_de = model(data)
                epoch_loss += mseloss(output_de, data.x)
            epoch_loss = epoch_loss / len(loader)
            train_loss.append(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_params = model.state_dict()
        writer.add_scalar("Reconstruction Loss", epoch_loss, epoch)
        if epoch % log_interval == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    
    torch.save(best_model_params, save_path)
    writer.close()
