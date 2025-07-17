#!/usr/bin/env python3

import optuna
from optuna.samplers import TPESampler
from optuna_toolkit import *
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
args = parser.parse_args()
model = args.model.upper()
assert model in ["XGB", "MLP", "MPNN", "TAB", "LASSO_MLP", "TABPFN", "FOREST"]
model_map = {"XGB" : xgb_objective, 
             "MLP" : mlp_objective,
             "MPNN" : chemprop_objective,
             "TAB" : ft_transformer_objective,
             "LASSO_MLP" : deep_lasso_mlp_objective,
             "TABPFN" : tabpfn_objective,
             "FOREST" : forest_objective}

objective_func = model_map[model]

for endpoint in ["rPPB", "RLM", "HLM"]:
    print(f"Current endpoint: {endpoint}")

    train_df = pd.read_csv(f"Data/{endpoint}/ADME_{endpoint}_train_feat.csv").drop("ID", axis=1)
    if model == "MPNN":
        train_df = pd.read_csv(f"Data/{endpoint}/ADME_{endpoint}_train.csv")        

    study = optuna.create_study(directions=["maximize", "minimize"], study_name=f"{model}", sampler=TPESampler())

    study.optimize(lambda trial: objective_func(trial, train_df, log_path=f"Logs/{endpoint}/{model}.txt"),
                   n_trials=50,
                   n_jobs=1)

    study_df = study.trials_dataframe()

    study_df.to_csv(f"Experiment_Data/{endpoint}/{model}_results.csv", index=False)