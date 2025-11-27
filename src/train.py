import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import numpy as np

from Dataloader import WeatherDiseaseDataset
from model import ContrastiveHybridModel
from utils import (
    save_model,
    save_predictions,
    save_metrics,
    evaluate_per_timestep_variable,
    save_per_timestep_variable_metrics,
)
from attention_visualization import register_attention_hooks, global_attn_weights, visualize_all, clear_visualization_buffers

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    task_losses = None

    with torch.no_grad():
        for inputx, inputy in dataloader:
            inputx = inputx.to(device)
            inputy = inputy.to(device)
            pred, _, _ = model(inputx)
            loss = 0.0
            if task_losses is None:
                task_losses = torch.zeros(pred.shape[-2], device=device)
            # compute raw task losses for metrics
            for task in range(pred.shape[-2]):
                task_loss = criterion(pred[..., task, 0], inputy[..., task])
                loss += task_loss
                task_losses[task] += task_loss.detach()
            total_loss += loss.item()
            all_preds.append(pred.squeeze(-1).cpu())
            all_targets.append(inputy.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    task_losses = task_losses.cpu().numpy() / len(dataloader)
    return total_loss / len(dataloader), all_preds, all_targets, task_losses

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = WeatherDiseaseDataset(config['data']['inputx_path'], config['data']['inputy_path'], split='train')
    val_ds = WeatherDiseaseDataset(config['data']['inputx_path'], config['data']['inputy_path'], split='val')
    test_ds = WeatherDiseaseDataset(config['data']['inputx_path'], config['data']['inputy_path'], split='test')
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
  
    model = ContrastiveHybridModel(config)
    num_tasks = model.num_tasks
    log_vars = nn.Parameter(torch.zeros(num_tasks, device=device))  # learnable log variances

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    log_vars = log_vars.to(device)

    clear_visualization_buffers()
    register_attention_hooks(model, enable=False)

    optimizer = optim.Adam(list(model.parameters()) + [log_vars], lr=config['learning_rate'])
    criterion = nn.MSELoss()

    result_dir = f"results/{config['experiment_name']}"
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, 'training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss')
        for k in range(num_tasks): f.write(f',val_task{k+1}')
        f.write(',log_var1')
        f.write("\n")

    best_val_loss = float('inf')
    patience, patience_ctr = config.get('patience',10), 0

    for epoch in range(1, config['epochs']+1):
        model.train()
        train_loss = 0.0
        for inputx, inputy in train_loader:
            inputx, inputy = inputx.to(device), inputy.to(device)
            pred, _, _ = model(inputx)
            # Uncertainty weighting: loss = sum_k (task_loss * exp(-log_var_k) + log_var_k)
            loss = 0.0
            for k in range(num_tasks):
                #print(pred[...,k,0].shape,inputy[...,k].shape)
                
                task_loss = criterion(pred[...,k,0], inputy[...,k])
                loss += task_loss #* torch.exp(-log_vars[k]) + log_vars[k] * 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss/len(train_loader)

        model.eval()
        val_loss, val_preds, val_targets, val_task_losses = evaluate(model, val_loader, criterion, device)

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{avg_train:.4f},{val_loss:.4f}")
            for tl in val_task_losses: f.write(f",{tl:.4f}")
            for v in log_vars.detach().cpu().numpy(): f.write(f",{v:.4f}")
            f.write("\n")

        print(f"Epoch {epoch}/{config['epochs']} TL:{avg_train:.4f} VL:{val_loss:.4f} TaskLosses:{val_task_losses} LogVars:{log_vars.data.cpu().numpy()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(result_dir,'best_model.pt'))
            torch.save(log_vars, os.path.join(result_dir,'best_log_vars.pt'))
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr>=patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(os.path.join(result_dir,'best_model.pt')))
    log_vars = torch.load(os.path.join(result_dir,'best_log_vars.pt')).to(device)

    register_attention_hooks(model, enable=True)
    test_loss, test_preds, test_targets, test_task_losses = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} Task losses: {test_task_losses}")


    num_samples, N, T, K = test_preds.shape
    preds = test_preds.permute(0,2,1,3).reshape(-1, K).numpy()
    targets = test_targets.permute(0,2,1,3).reshape(-1, K).numpy()
    import pandas as pd
    df = pd.DataFrame(preds, columns=[f'pred_task{i+1}' for i in range(K)])
    for i in range(K):
        df[f'true_task{i+1}'] = targets[:, i]
    df.to_csv(f'{result_dir}/pred_true.csv', index=False)

    save_metrics(test_preds, test_targets, f'{result_dir}/all_metric.csv')
    tm = evaluate_per_timestep_variable(test_preds.numpy(), test_targets.numpy())
    save_per_timestep_variable_metrics('test', tm, f'{result_dir}/disease_metric.csv')

    attention_dir = os.path.join(result_dir, 'attention_plots')
    os.makedirs(attention_dir, exist_ok=True)
    visualize_all(global_attn_weights, selected_node=0, save_dir=attention_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
