import torch
import pandas as pd
import os
#from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    return mean_squared_error(
        y_true, y_pred,
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=False, 
    )


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def save_predictions(pred, target, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    df = pd.DataFrame({'prediction': pred_np.flatten(), 'true': target_np.flatten()})
    df.to_csv(path, index=False)

def save_metrics(pred, target, path='results/all_metrics.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)
    mse = ((pred_np - target_np) ** 2).mean()
    mae = abs(pred_np - target_np).mean()
    mae = mean_absolute_error(target_np, pred_np)
    rmse = root_mean_squared_error(target_np, pred_np)
    #F_norm = la.norm(target_np-pred_np,'fro')/la.norm(target_np,'fro')
    r2 = r2_score(target_np, pred_np)
    var = 1-(np.var(target_np-pred_np))/np.var(target_np)
    df = pd.DataFrame({'mse': [mse], 'mae': [mae], 'r2': [r2], 'var': [var]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def evaluate_per_timestep_variable(pred, target):
    B, T, N, D = pred.shape
    metrics = []


    for d in range(D):
        pred_td = pred[:, :, :, d].reshape(-1)
        target_td = target[:, :, :, d].reshape(-1)

        mae = mean_absolute_error(target_td, pred_td)
        rmse = root_mean_squared_error(target_td, pred_td)
        #F_norm = la.norm(target_td-pred_td,'fro')/la.norm(target_td,'fro')
        r2 = r2_score(target_td, pred_td)
        var = 1-(np.var(target_td-pred_td))/np.var(target_td)

        metrics.append({
            'variable': d,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'var': var
        })

    return metrics

def save_per_timestep_variable_metrics(epoch, metrics, path='results/disease_metrics.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(metrics)
    df['epoch'] = epoch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
