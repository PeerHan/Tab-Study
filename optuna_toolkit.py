#!/usr/bin/env python3

import torch
from torch import nn
from torch_toolkit import *
import joblib
#from tab_transformer_pytorch import FTTransformer
from FTTransformer import FTTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import xgboost
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from numpy import mean, std
from torchmetrics import regression
from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
import chemprop
from lightning import pytorch as pl 
from NDF import Forest
from copy import deepcopy

def chemprop_objective(
        trial : "Trial",
        train : "DataFrame",
        target_columns : str | list = "activity",
        n_tasks : int = 1,
        log_path : str = "",
        device="cuda",
    
) -> float:
    """
    Function to build a MPNN network (chemprop) based on selected parameter which optuna will find.
    Furthermore the network will be trained with different parameter to find the best HP for the model. A simple logging is provided to track the amount of finished trials.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data which should be split in train and val
        - val_df : Dataframe for hp validation
        - target_columns : name of the target(s) - can be multiple
        - n_tasks : specifiy for multitask
        - log_path : Folder for log files
    
    The following hyper parameter will be optimized:
        - depth for MP layer
        - hidden dim of MP layer
        - activation function for MP layer
        - dropout prob. for MP layer
        - aggregation method
        - amount of layers for FFN
        - droput prob for FFN
        - activation function for FFN
        - Use batchnorm after MP 
        - warmup epochs
        - max epochs
        - initial lr
        - batchsize
    
    
    """

    
    # Start Optuna
    depth = trial.suggest_int("depth", 1, 5)
    hidden_dim_bond = trial.suggest_int("hidden_bond", 100, 1000, 100)
    activation_bond = "elu"
    dropout = trial.suggest_float("dropout_bond", 0., 0.5, step=0.05)
    
    mp = chemprop.nn.BondMessagePassing(d_h=hidden_dim_bond, activation=activation_bond, dropout=dropout, depth=depth)
    aggregation = trial.suggest_categorical("aggregation", choices=["sum", "mean", "norm"])
    agg_map = {"sum" : chemprop.nn.SumAggregation(), "mean" : chemprop.nn.MeanAggregation(), "norm" : chemprop.nn.NormAggregation()}
    agg = agg_map[aggregation]
    
    n_layers = trial.suggest_int("layer", 1, 5)
    activation_ffn = "elu"
    ffn_hidden_dim = trial.suggest_int("hidden_ffn", 100, 2000, 100)
    ffn_dropout = trial.suggest_float("dropout_ffn", 0, 0.5, step=0.05)
    ffn = chemprop.nn.RegressionFFN(output_transform=None, input_dim=hidden_dim_bond, hidden_dim=ffn_hidden_dim, n_layers=n_layers, activation=activation_ffn, n_tasks=n_tasks, dropout=ffn_dropout)

    
    bn = trial.suggest_categorical("BN", choices=["True", "False"])
    batch_norm = bn == "True"
    metric_list = [chemprop.nn.metrics.MAEMetric(), chemprop.nn.metrics.RMSEMetric()]
    warmup_epochs = trial.suggest_int("warmup_epochs", 2, 10)
    init_lr = trial.suggest_float("init_lr", 1e-4, 1e-3, log=True)
    final_lr = trial.suggest_float("init_lr", 1e-6, 1e-4, log=True)

    mean_r2 = []
    mean_mae = []
    
    kfold = KFold(shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(train):
        
        train_df = train.loc[train_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)
        
        train_smiles = train_df.loc[:, "smiles"].values
        train_targets = train_df.loc[:, target_columns].values
        if n_tasks == 1:
            train_targets = train_targets.reshape(-1 , 1)
        train_data = [chemprop.data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_smiles, train_targets)]
        featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = chemprop.data.MoleculeDataset(train_data, featurizer)
        #scaler = train_dset.normalize_targets()

        val_smiles = val_df.loc[:, "smiles"].values
        val_targets = val_df.loc[:, target_columns].values
        if n_tasks == 1:
            val_targets = val_targets.reshape(-1, 1)
        val_data = [chemprop.data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(val_smiles, val_targets)]
        val_dset = chemprop.data.MoleculeDataset(val_data, featurizer)
        #val_dset = val_dset.normalize_targets(scaler)
        batch_size = trial.suggest_categorical("batch_size", choices=[16, 32, 64, 128])

        train_loader = chemprop.data.build_dataloader(train_dset, num_workers=5, shuffle=True, batch_size=batch_size)
        val_loader = chemprop.data.build_dataloader(val_dset, num_workers=5, shuffle=False, batch_size=batch_size)
        
        model = chemprop.models.MPNN(mp, agg, ffn, batch_norm, metric_list, warmup_epochs=warmup_epochs, init_lr=init_lr, final_lr=final_lr)
    
        early_stopping = EarlyStopping('val_loss', patience=10, mode="min")

        trainer = pl.Trainer(
                logger=False,
                accelerator=device,
                devices=[6],
                max_epochs=500,
                callbacks=[early_stopping])

        trainer.fit(model, train_loader, val_loader)

        out = trainer.predict(model, val_loader)
        preds = pd.DataFrame(torch.concat(out).numpy())
        targets = val_df[target_columns]
        res = trainer.test(model, val_loader)
        mae = res[0]["batch_averaged_test/mae"]
        r2 = 0

        if n_tasks == 1:
            r2 = pd.concat((targets, preds), axis=1).corr().iloc[0, 1] ** 2
        # Multitask
        else:
            mask = ~targets.isna()
            valid_preds = preds[mask]
            r2s = []
            for i, col in enumerate(target_columns):
                mask = ~targets[col].isna()
                r2 = np.corrcoef(preds[i][mask].dropna(), targets[col].dropna()) ** 2
                r2s.append(r2)
            r2 = mean(r2s)
            
        mean_r2.append(r2)
        mean_mae.append(mae)
    
    mean_r2 = np.mean(mean_r2)
    mean_mae = np.mean(mean_mae)

    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(mean_r2) + " - MAE : " + str(mean_mae) + "\n")
        
    return r2, mae


def create_optuna_chemprop(
        best_model : "Series",
        n_tasks : int = 1,
    ) -> "Model":
    """
    Function to create a pytorch chemprop model based on the best HP-Search with optuna.
    The returned model only consists of the following HP:
        - depth for MP layer
        - hidden dim of MP layer
        - activation function for MP layer
        - dropout prob. for MP layer
        - aggregation method
        - amount of layers for FFN
        - droput prob for FFN
        - activation function for FFN
        - Use batchnorm after MP 
        - warmup epochs
        - max epochs
        - initial lr
    
    Parameters:
        - best_model : Pandas series object of the best optuna trial
        - n_tasks : Single or Multitask
    """
    
    depth = best_model["depth"]
    hidden_dim_bond = best_model["hidden_bond"]
    activation_bond = "elu"
    dropout = best_model["dropout_bond"]
    
    mp = chemprop.nn.BondMessagePassing(d_h=hidden_dim_bond, activation=activation_bond, dropout=dropout, depth=depth)
    aggregation = best_model["aggregation"]
    agg_map = {"sum" : chemprop.nn.SumAggregation(), "mean" : chemprop.nn.MeanAggregation(), "norm" : chemprop.nn.NormAggregation()}
    agg = agg_map[aggregation]
    
    n_layers = best_model["layer"]
    activation_ffn = "elu"

    ffn_hidden_dim = best_model["hidden_ffn"]
    ffn_dropout = best_model["dropout_ffn"]
    ffn = chemprop.nn.RegressionFFN(output_transform=None, input_dim=hidden_dim_bond, hidden_dim=ffn_hidden_dim, n_layers=n_layers, activation=activation_ffn, n_tasks=n_tasks, dropout=ffn_dropout)

    
    bn = best_model["BN"]
    batch_norm = bn == "True"
    metric_list = [chemprop.nn.metrics.MAEMetric(), chemprop.nn.metrics.RMSEMetric()]
    warmup_epochs = best_model["warmup_epochs"]
    max_epoch = 500
    init_lr = best_model["init_lr"]
    
    model = chemprop.models.MPNN(mp, agg, ffn, batch_norm, metric_list, warmup_epochs=warmup_epochs, init_lr=init_lr)
    

    return model

def define_regressor(
        trial : "Trial", 
        input_feats : int, 
        output : int = 1,
        consider_layer_norm : bool = False
    ) -> "Model":
    """
    Function to build a regression model based on optuna.
    
    Parameters:
        - trial : Optuna trial object
        - input_feats : Number of features as input features for the network
        - output : Number of regresion tasks
    
    Hyparameter:
        - Amount of layers (1-10)
        - Activation function (ReLU, Mish, SiLU, leaky ReLU)
        - Hidden Units (10 - 500)
        - Dropout prob (0 - 0.4)
    """
    # HP 1 : How many Layers?
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    if consider_layer_norm:
        layer_norm = bool(trial.suggest_int("layer_norm", 0, 1))
    else:
        layer_norm = False
    

    hidden_dim = trial.suggest_int("neurons", 100, 2000, step=100)
    dropout = trial.suggest_float("dropout", 0., 0.5, step=0.05)
    for i in range(n_layers):
        # HP 3 : Amount of Hidden units
        if layer_norm and i == 0:
            layers.append(nn.LayerNorm(input_feats))
            layers.append(nn.Mish())
        output_feats = hidden_dim
        layers.append(nn.Linear(input_feats, output_feats))
        # HP 4 : Dropout Probability
        #if batch_norm:
         #   layers.append(nn.BatchNorm1d(output_feats))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Mish())
        input_feats = output_feats
    
    layers.append(nn.Linear(input_feats, output))
    
    return nn.Sequential(*layers)

def mlp_objective(
        trial : "Trial", 
        train : "DataFrame", 
        target : str = "activity",
        device : str = "cuda", 
        input_feats : int = 1340,
        log_path : str = "mlp_logs.txt",
    ) -> float:
    """
    Function to build a network based on selected parameter which optuna will find.
    Furthermore the network will be trained with different parameter to find the best HP for the model. A simple logging is provided to track the amount of finished trials.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data which should be split in train and val
        - val_df : Dataframe for hp validation
        - target : name of the target
        - device : Should be always cuda if a gpu is available
        - input_feats : Input feats for the network
        - log_path : Folder for log files
    
    The following hyper parameter will be optimized:
        - Layer
        - Activation Function
        - Hidden units per layer
        - Dropout probability
        - Batchsize
        - LR Scheduler patience
        - Model initialization
        - Learning rate
        - Weight decay
    """
    
    batch_size = trial.suggest_categorical("batch_size", choices=[16, 32, 64, 128])
    patience = 10
    delta = 0.
    lr_pat = 10
    max_epochs = 500
    
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)    
    loss_f = nn.MSELoss()
    
    mean_r2 = []
    mean_mae = []
    kfold = KFold(shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(train):
        
        train_df = train.loc[train_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)

        r2 = None
        mae = None
        last_epoch = None
        # Model Section
        model = define_regressor(trial, input_feats=1340, output=1)
        train_loader, val_loader = create_dataloader(
            {"Train" : train_df,
             "Test" : val_df
            },
            batch_size=batch_size,
            target=target
        ).values()

        model = model.to(device)
        xavier_init(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        min_val_loss = 1000

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=lr_pat)
        early_stopper = EarlyStopper(patience=patience, min_delta=delta)
        for epoch in range(max_epochs):
            t_loss = training(train_loader, model, loss_f, optimizer, device=device)
            v_loss = validation(val_loader, model, loss_f, optimizer, device=device)

            scheduler.step(v_loss)
            if v_loss < min_val_loss:
                min_val_loss = v_loss
                torch.save(model.state_dict(), "Weights/MLP.pt")

            if early_stopper.early_stop(v_loss.item()):
                break
        last_epoch = epoch

        model.load_state_dict(torch.load("Weights/MLP.pt", weights_only=True))

        _, test_pred, test_target = testing(val_loader, model, loss_f, device=device)

        r2 = np.corrcoef(test_target, test_pred)[0, 1] ** 2
        mae = mean_absolute_error(test_target, test_pred)
        
        mean_r2.append(r2)
        mean_mae.append(mae)

    
    mean_r2 = np.mean(mean_r2)
    mean_mae = np.mean(mean_mae)
    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(mean_r2) + " - MAE : " + str(mean_mae) + " - Epochs " + str(last_epoch) + "\n")
        
    return mean_r2, mean_mae

def deep_lasso_mlp_objective(
        trial : "Trial", 
        train : "DataFrame", 
        target : str = "activity",
        device : str = "cuda", 
        input_feats : int = 1340,
        log_path : str = "mlp_logs.txt",
    ) -> float:
    """
    Function to build a network based on selected parameter which optuna will find.
    Furthermore the network will be trained with different parameter to find the best HP for the model. A simple logging is provided to track the amount of finished trials.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data which should be split in train and val
        - val_df : Dataframe for hp validation
        - target : name of the target
        - device : Should be always cuda if a gpu is available
        - input_feats : Input feats for the network
        - log_path : Folder for log files
    
    The following hyper parameter will be optimized:
        - Layer
        - Activation Function
        - Hidden units per layer
        - Dropout probability
        - Batchsize
        - LR Scheduler patience
        - Model initialization
        - Learning rate
        - Weight decay
    """
    
    batch_size = trial.suggest_categorical("batch_size", choices=[16, 32, 64, 128])
    lasso_weight = trial.suggest_float("lasso_weight", 0, 0.4)
    patience = 10
    delta = 0.
    lr_pat = 10
    max_epochs = 500
    
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)    
    loss_f = nn.MSELoss()
    
    mean_r2 = []
    mean_mae = []
    kfold = KFold(shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(train):
        
        train_df = train.loc[train_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)

        r2 = None
        mae = None
        last_epoch = None
        # Model Section
        model = define_regressor(trial, input_feats=1340, output=1)
        train_loader, val_loader = create_dataloader(
            {"Train" : train_df,
             "Test" : val_df
            },
            batch_size=batch_size,
            target=target
        ).values()

        model = model.to(device)
        xavier_init(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        min_val_loss = 1000

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=lr_pat)
        early_stopper = EarlyStopper(patience=patience, min_delta=delta)
        for epoch in range(max_epochs):
            t_loss = deep_lasso_training(train_loader, model, loss_f, optimizer, lasso_weight, device)
            v_loss = validation(val_loader, model, loss_f, optimizer, device=device)

            scheduler.step(v_loss)
            if v_loss < min_val_loss:
                min_val_loss = v_loss
                torch.save(model.state_dict(), "Weights/MLP.pt")

            if early_stopper.early_stop(v_loss.item()):
                break
        last_epoch = epoch

        model.load_state_dict(torch.load("Weights/MLP.pt", weights_only=True))

        _, test_pred, test_target = testing(val_loader, model, loss_f, device=device)

        r2 = np.corrcoef(test_target, test_pred)[0, 1] ** 2
        mae = mean_absolute_error(test_target, test_pred)
        
        mean_r2.append(r2)
        mean_mae.append(mae)

    
    mean_r2 = np.mean(mean_r2)
    mean_mae = np.mean(mean_mae)
    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(mean_r2) + " - MAE : " + str(mean_mae) + " - Epochs " + str(last_epoch) + "\n")
        
    return mean_r2, mean_mae

def xgb_objective(
        trial : "Trial", 
        train : "DataFrame", 
        target : str = "activity",
        log_path : str = "xgb_log.txt",
    ) -> (float, float, float):
    """
    Function to find the best hyperparameter for an XGBoost model based on the validation data.
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data
        - val_df : Dataframe with val data
        - target : name of the target
        - log_path : Folder for log files 
    
    The following hyper parameter will be optimized:
        - max_depth
        - learning_rate
        - n_estimators
        - min_child_weight
        - gamma
        - subsample
        - colsample_bytree
        - reg_alpha
        - reg_lambda
    """
    
    param = {
        'max_depth': trial.suggest_int('max_depth', 1, 75),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 0., 3.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 3.0),
        'n_jobs' : 30
    }
    
    
    mean_r2 = []
    mean_mae = []
    kfold = KFold(shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(train):
        
        train_df = train.loc[train_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)
        mae = None
        r2 = None

        X_train, y_train = train_df.drop(target, axis=1), train_df[target]
        X_val, y_val = val_df.drop(target, axis=1), val_df[target]

        xgb = xgboost.XGBRegressor(**param)
        xgb.fit(X_train, y_train)

        test_pred = xgb.predict(X_val)

        r2 = np.corrcoef(y_val, test_pred)[0, 1] ** 2
        mae = mean_absolute_error(y_val, test_pred)
        
        mean_r2.append(r2)
        mean_mae.append(mae)
        
    mean_r2 = np.mean(mean_r2)
    mean_mae = np.mean(mean_mae)

    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(mean_r2) + " - MAE : " + str(mean_mae) + "\n")
        
    return mean_r2, mean_mae


def create_optuna_mlp(
        best_model : "Series",
        outputs : int = 1,
        input_feats : int = 1340,
    ) -> "Model":
    """
    Function to create a pytorch model based on the best HP-Search with optuna.
    The returned model only consists of the following HP:
        - Layers
        - Neurons per layer
        - Dropout probability per layer
        - Activation function
    
    Parameters:
        - best_model : Pandas series object of the best optuna trial
        - input_feats : Amount of input features for the first layer
    """
    
    layers = []
    N = best_model.params_n_layers
    input_feats = input_feats
    
    activation_func = nn.Mish()
    
    for i in range(N):
        output_feats = best_model["params_neurons"]
        dropout_p = best_model["params_dropout"]
        input_feats = int(input_feats)
        output_feats = int(output_feats)
        layers.append(nn.Linear(input_feats, output_feats))
        layers.append(activation_func)
        layers.append(nn.Dropout(dropout_p))
        input_feats = output_feats
    
    # delete last dropout
    del layers[-1]
    layers.append(nn.Linear(input_feats, outputs))
    
    model = nn.Sequential(*layers)

    return model
        
        
def best_trial_mlp(
        study_df : "DataFrame",
    ) -> dict:
    """
    Function to get all HPs of the best optuna trial (MLP) and the corresponding pytorch model.
    All objects are stored in a dictionary with the following key-value pairs:
        - batch_size : Batch size for training
        - lr : Initial learning rate
        - lr_patience : Patience for the learning rate scheduler
        - weight_decay : Weight decay regularization value
        - epoch : Amount of epochs
        - trial : Pandas Series of the best model
        
    Parameters:
        - study_df : Dataframe from an optuna study
    """
    important_params = ["params_batch_size", "params_lr",
                        "params_dropout", "params_n_layers", "params_neurons"]
    
    study_df = study_df.rename({"values_0" : "R2", "values_1" : "MAE"}, axis=1)
    
    for q in [0.01, 0.025, 0.05, 0.075, 0.1]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break
                    
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
    
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["trial"] = pd.Series(best_model)
    
    return trial_data

def best_trial_xgb(
        study_df : "DataFrame",
        n_jobs=30
    ) -> dict:
    """
    Function to get all Hps for the best optuna trial (XGB) and the corresponding xgb-model.
    All parameters are stored in a dictionary with the following key-parameter pairs:
        - max_depth 
        - learning_rate
        - n_estimators
        - min_child_weight
        - gamma
        - subsample
        - colsample_bytree
        - reg_alpha
        - reg_lambda
        - n_jobs
    """
    
    important_params = ["params_max_depth", "params_learning_rate", "params_n_estimators",
                        "params_min_child_weight", "params_gamma", "params_subsample",
                        "params_colsample_bytree", "params_reg_alpha", "params_reg_lambda"]
    
    study_df = study_df.rename({"values_0" : "R2", "values_1" : "MAE"}, axis=1)

    
    for q in [0.01, 0.025, 0.05, 0.075, 0.1]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break
                    
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
        
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["n_jobs"] = n_jobs
    
    return trial_data

def best_trial_chemprop(
        study_df : "DataFrame",
    ) -> dict:
    """
    Function to get all HPs of the best optuna trial (chemprop GNN) and the corresponding pytorch model.
    All objects are stored in a dictionary with key value pairs to rebuild the best model.
        
    Parameters:
        - study_df : Dataframe from an optuna study
    """
    important_params = ['params_BN', 'params_aggregation', 'params_batch_size',
       'params_depth', 'params_dropout_bond', 'params_dropout_ffn',
       'params_hidden_bond', 'params_hidden_ffn', 'params_init_lr',
       'params_layer', 'params_warmup_epochs']
    
    study_df = study_df.rename({"values_0" : "R2", "values_1" : "MAE"}, axis=1)

        
    for q in [0.01, 0.025, 0.05, 0.075, 0.1]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break
                    
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
    
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["trial"] = pd.Series(best_model)
    
    return trial_data

def best_trial_tabtransformer(
        study_df : "DataFrame",
    ) -> dict:
    """
    Function to get all HPs of the best optuna trial (TabTransformer) and the corresponding pytorch model.
    All objects are stored in a dictionary with key value pairs to rebuild the best model.
        
    Parameters:
        - study_df : Dataframe from an optuna study
    """
    important_params = ["params_amount_heads", "params_attn_dropout", "params_batch_size",
                        "params_dropout", "params_encoder_output_dim", "params_ff_dropout",
                        "params_head_dim", "params_layer_norm", "params_lr",
                        "params_n_layers", "params_neurons", "params_transformer_depth"]
    
    study_df = study_df.rename({"values_0" : "R2", "values_1" : "MAE"}, axis=1)

        
    for q in [0.01, 0.025, 0.05, 0.075, 0.1]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break
                    
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
    
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["trial"] = pd.Series(best_model)
    
    return trial_data

def ft_transformer_objective(
        trial : "Trial",
        train : "DataFrame",
        target : str = "activity",
        log_path : str = "",
        device : str = "cuda:9"
) -> float:
    """
    FT Transformer
    """
    # HP 5 : Batchsize
    batch_size = trial.suggest_categorical("batch_size", choices=[16, 32, 64, 128])
    patience = 10
    delta = 0.
    # HP 6 : LR Scheduler patience
    # Cut
    lr_pat = 10
    epochs = 500
    
    # HP 7: Learning Rate
    # Check decay
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    mean_r2 = []
    mean_mae = []
    kfold = KFold(shuffle=True, random_state=1)
    loss_f = nn.MSELoss()
    
    dim = trial.suggest_int("encoder_output_dim", 128, 300)
    dim_head = trial.suggest_int("head_dim", 32, 128)
    depth = trial.suggest_int("transformer_depth", 2, 5)
    heads = trial.suggest_int("amount_heads", 2, 5)
    attn_dropout = trial.suggest_float("attn_dropout", 0., 0.5, step=0.05)
    ff_dropout = trial.suggest_float("ff_dropout", 0., 0.5, step=0.05)
    r2_torch = regression.r2.R2Score()
    mae_torch = regression.mae.MeanAbsoluteError()
    for i, (train_idx, val_idx) in enumerate(kfold.split(train)):
        
        train_df = train.loc[train_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)
        
        train_dset = NumericalTabSet(train_df.drop(target, axis=1), train_df[target])
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        train_scaler = train_dset.scaler
        val_dset = NumericalTabSet(val_df.drop(target, axis=1), val_df[target], scaler=train_scaler)
        val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)

        r2 = None
        mae = None
        last_epoch = None
        # Model Section
        model = FTTransformer(
            num_continuous=train_dset.conts.shape[1],
            projected_dim=512,
            dim_head=dim_head,
            dim=dim,
            dim_out=1,                                  
            depth=depth,                                       
            heads=heads,                           
            attn_dropout=attn_dropout,       
            ff_dropout=ff_dropout,
        )
        predictor  = define_regressor(trial, input_feats=dim, output=1, consider_layer_norm=True)
        xavier_init(predictor)
        model.to_logits = predictor
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=lr_pat)
        early_stopper = EarlyStopper(patience=patience, min_delta=delta)

        min_val_loss = 1000
        for epoch in range(epochs):
            t_loss = 0
            for conts, y in train_loader:
                optimizer.zero_grad()
                conts = conts.float().to(device)
                res = model(conts)
                conts = conts.to("cpu")
                res = res.to("cpu")
                l = loss_f(res.flatten(), y.float())
                l.backward()
                optimizer.step()
                t_loss += l.item()
            
            model.eval()
            res_vec = []
            y_vec = []
            v_loss = 0
            for conts, y in val_loader:
                conts = conts.float().to(device)
                res = model(conts)
                conts = conts.to("cpu")
                res_vec.append(res.to("cpu"))
                y_vec.append(y)
                l = loss_f(res.flatten().to("cpu"), y.float())
                v_loss += l.item()
            res = torch.concat(res_vec).flatten()
            y = torch.concat(y_vec)
            #print(r2_torch(target=y, preds=res), mae_torch(target=y, preds=res), v_loss)
            if v_loss < min_val_loss:
                min_val_loss = v_loss
                torch.save(model.state_dict(), "Weights/Tab.pt")
            scheduler.step(v_loss)
                
            if early_stopper.early_stop(v_loss):
                break
                

        last_epoch = epoch

        model.load_state_dict(torch.load("Weights/Tab.pt", weights_only=True))

        res_vec = []
        y_vec = []
        with torch.no_grad():
            model.eval()
            for conts, y in val_loader:
                res = model(conts.to(device))
                res_vec.append(res.to("cpu"))
                y_vec.append(y)

        res = torch.concat(res_vec)
        y = torch.concat(y_vec)
        res = res.squeeze(1)
        r2 = r2_torch(target=y, preds=res)
        mae = mae_torch(target=y, preds=res)

        mean_r2.append(r2)
        mean_mae.append(mae)
        
        model.to("cpu")
        del model
        del optimizer
        torch.cuda.empty_cache()

    
    mean_r2 = np.mean(mean_r2)
    mean_mae = np.mean(mean_mae)
    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(mean_r2) + " - MAE : " + str(mean_mae) + " - Epochs " + str(last_epoch) + "\n")
        
    return mean_r2, mean_mae


def tabpfn_objective(
        trial : "Trial", 
        train : "DataFrame", 
        target : str = "activity",
        log_path : str = "tabpfn_log.txt",
    ) -> (float, float, float):
    """
    Function to find the best hyperparameter for a TabPFN model based on the validation data.
    https://www.nature.com/articles/s41586-024-08328-6
    
    Parameter :
        - trial : Optuna trial object
        - train_df : Dataframe with train data
        - val_df : Dataframe with val data
        - target : name of the target
        - log_path : Folder for log files 
    
    The following hyper parameter will be optimized:
        - n_estimators
        - softmax_temperature
        - average_before_softmax
    """
    
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 100),
        'softmax_temperature': trial.suggest_float('softmax_temperature', 0.05, 1.0),
        'average_before_softmax': trial.suggest_categorical('average_before_softmax', [True, False]),
        'ignore_pretraining_limits' : True,
        'device' : 'cuda:3',
        'n_jobs' : 30
    }
    
    
    mean_r2 = []
    mean_mae = []
    kfold = KFold(shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(train):
        
        train_df = train.loc[train_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)
        mae = None
        r2 = None

        X_train, y_train = train_df.drop(target, axis=1), train_df[target]
        X_val, y_val = val_df.drop(target, axis=1), val_df[target]

        model = TabPFNRegressor(**param)
        model.fit(X_train, y_train)

        test_pred = model.predict(X_val)

        r2 = np.corrcoef(y_val, test_pred)[0, 1] ** 2
        mae = mean_absolute_error(y_val, test_pred)
        
        mean_r2.append(r2)
        mean_mae.append(mae)
        
    mean_r2 = np.mean(mean_r2)
    mean_mae = np.mean(mean_mae)

    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(mean_r2) + " - MAE : " + str(mean_mae) + "\n")
        
    return mean_r2, mean_mae


def best_trial_tabpfn(
        study_df : "DataFrame",
        n_jobs=30
    ) -> dict:
    """
    Function to get all Hps for the best optuna trial (TabPFN) and the corresponding TabPFN-model.
    All parameters are stored in a dictionary with the following key-parameter pairs:
        - n_estimators
        - softmax_temperature
        - average_before_softmax
        - n_jobs
    """
    
    important_params = ["params_softmax_temperature", "params_average_before_softmax", "params_n_estimators"]
    
    study_df = study_df.rename({"values_0" : "R2", "values_1" : "MAE"}, axis=1)

    
    for q in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break
    
    if len(subset) == 0:
        subset = study_df
                    
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
        
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["n_jobs"] = n_jobs
    trial_data['ignore_pretraining_limits'] = True
    
    return trial_data
    
   
def forest_objective(
        trial : "Trial", 
        train : "DataFrame", 
        target : str = "activity",
        device : str = "cuda", 
        input_feats : int = 1340,
        log_path : str = "forest_logs.txt",
    ) -> float:
    batch_size = trial.suggest_categorical("batch_size", choices=[128, 256])
    patience = 3
    delta = 0.
    lr_pat = 3
    max_epochs = 100
    
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)    
    loss_f = nn.MSELoss()
    
    mean_r2 = []
    mean_mae = []
    kfold = KFold(shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(train):
        
        train_df = train.loc[train_idx, :].reset_index(drop=True)
        val_df = train.loc[val_idx, :].reset_index(drop=True)

        r2 = None
        mae = None
        last_epoch = None
        # Model Section
        model = Forest(
            n_tree=trial.suggest_int("n_tree", 5, 100),
            tree_depth=trial.suggest_int("tree_depth", 1, 5),
            n_in_feature=1340,
            tree_feature_rate=trial.suggest_float("feature_rate", 0.1, 0.95, step=0.05),
            n_class=1,
            jointly_training=False
        )
        train_loader, val_loader = create_dataloader(
            {"Train" : train_df,
             "Test" : val_df
            },
            batch_size=batch_size,
            target=target
        ).values()

        model = model.to(device)
        xavier_init(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        min_val_loss = 1000

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=lr_pat)
        early_stopper = EarlyStopper(patience=patience, min_delta=delta)
        for epoch in range(max_epochs):
            t_loss = training(train_loader, model, loss_f, optimizer, device=device)
            v_loss = validation(val_loader, model, loss_f, optimizer, device=device)

            scheduler.step(v_loss)
            if v_loss < min_val_loss:
                min_val_loss = v_loss
                #torch.save(model.state_dict(), "Weights/Forest.pt")

            if early_stopper.early_stop(v_loss.item()):
                break
        last_epoch = epoch

        #model.load_state_dict(torch.load("Weights/Forest.pt", weights_only=True))

        _, test_pred, test_target = testing(val_loader, model, loss_f, device=device)

        r2 = np.corrcoef(test_target, test_pred)[0, 1] ** 2
        mae = mean_absolute_error(test_target, test_pred)
        
        mean_r2.append(r2)
        mean_mae.append(mae)

    
    mean_r2 = np.mean(mean_r2)
    mean_mae = np.mean(mean_mae)
    with open(log_path, "a") as log_file:
        log_file.write("Current Trial : " + str(trial.number) + " - R2 : " + str(mean_r2) + " - MAE : " + str(mean_mae) + " - Epochs " + str(last_epoch) + "\n")
        
    return mean_r2, mean_mae


def best_trial_ndf(
        study_df : "DataFrame",
    ) -> dict:
    """
    Function to get all HPs of the best optuna trial (MLP) and the corresponding pytorch model.
    All objects are stored in a dictionary with the following key-value pairs:
        - batch_size : Batch size for training
        - lr : Initial learning rate
        - lr_patience : Patience for the learning rate scheduler
        - weight_decay : Weight decay regularization value
        - epoch : Amount of epochs
        - trial : Pandas Series of the best model
        
    Parameters:
        - study_df : Dataframe from an optuna study
    """
    important_params = ["params_batch_size", "params_lr", "params_feature_rate",
                        "params_n_tree", "params_tree_depth"]
    
    study_df = study_df.rename({"values_0" : "R2", "values_1" : "MAE"}, axis=1)
    
    for q in [0.01, 0.025, 0.05, 0.075, 0.1]:
        subset = study_df[(study_df.R2 >= study_df.R2.quantile(1-q)) & (study_df.MAE <= study_df.MAE.quantile(q))]
        if len(subset) != 0:
            break
                    
    best_model = subset.sort_values("R2", ascending=False).reset_index(drop=True).iloc[0, :]
    
    trial_data = {param.replace("params_", "") : best_model[param] for param in important_params}
    
    trial_data["trial"] = pd.Series(best_model)
    
    return trial_data