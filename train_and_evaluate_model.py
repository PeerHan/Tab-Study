from pathlib import Path
import pandas as pd
from torch_toolkit import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from optuna_toolkit import *
import xgboost
from math import sqrt
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
import pickle
from tabpfn import TabPFNRegressor
from Evaluation import get_metrics
from FTTransformer import FTTransformer
import argparse
from NDF import *
from lightning import pytorch as pl

repetitions = 30
device = "cuda"
data_path = "./"

# Evaluate all training strategies
for endpoint in ["hPPB", "Sol", "HLM", "rPPB", "RLM", "MDR1_ER"]: 

    print("Current Endpoint : " + endpoint, flush=True)

    for model in ["FOREST"]:

        suffix = "_feat.csv" if model != "MPNN" else ".csv"
        train_data = f"{data_path}/Data/{endpoint}/ADME_{endpoint}_train{suffix}"
        test_data = f"{data_path}/Data/{endpoint}/ADME_{endpoint}_test{suffix}"

        train_df = pd.read_csv(train_data)
        test_df = pd.read_csv(test_data)
        if model != "MPNN":
            train_df = train_df.drop("ID", axis=1)
            test_df = test_df.drop("ID", axis=1)
        
        print("Current Model : " + model, flush=True)
        trial_data = f"{data_path}/Experiment_Data/{endpoint}/{model}_results.csv"
        trial_df = pd.read_csv(trial_data)

        # Finding best training run
        best_predictions = None
        min_mae = 100
        # Neural Network schedule
        if model in ["MLP", "LASSO_MLP"]:
            train_set = MixedDescriptorSet(train_df)

            test_loader = DataLoader(MixedDescriptorSet(test_df), batch_size=len(test_df), shuffle=False)
            mlp_results = {}
            trial_data = best_trial_mlp(trial_df)
            best_trial = trial_data["trial"]
            loss = nn.MSELoss()
            bs = trial_data["batch_size"]
            train_loader = DataLoader(train_set, shuffle=True, batch_size=int(bs))

            # Start the training
            for loop in range(1, repetitions+1):
                print("Training and Testing Modell " + str(loop), flush=True)
                mlp_model = create_optuna_mlp(best_trial)
                xavier_init(mlp_model)
                mlp_model.to(device)
                optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=trial_data["lr"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                # Train on Train/Val Set
                for i in range(150):
                    _ = training(train_loader, mlp_model, loss, optimizer, device=device)

                    scheduler.step()


                preds = []
                # Amount of MC Models
                for i in range(50):
                    # Test on Intern and Extern
                    _, test_pred, test_target = testing(test_loader, mlp_model, loss, device, mc_dropouts=True)

                    preds.append(test_pred)



                preds = np.array(preds)
                pred = np.mean(preds, axis=0)       


                metrics = get_metrics(pred, test_target)
                
                if metrics["MAE"] < min_mae:
                    min_mae = metrics["MAE"]
                    best_preds = pred

                    
                
                mlp_results[f"{model}_" + str(loop)] = metrics


            mlp_df = pd.DataFrame(mlp_results).T
            mlp_df.to_csv(f"{data_path}/Evaluation_Data/{endpoint}/{model}_results.csv", index=False)

        # XGB schedule
        elif model == "XGB":

            xgb_results = {}
            trial_data = best_trial_xgb(trial_df)

            for loop in range(1, repetitions+1):
                print("Training and Testing Modell " + str(loop), flush=True)
                xgb_model = xgboost.XGBRegressor(**trial_data)
                # Prevent Deterministic Behaviour
                xgb_model.random_state = loop 
                xgb_model.seed = loop
                X, y = train_df.drop("activity", axis=1), train_df.activity
                xgb_model.fit(X, y)
                X_test, y_test = test_df.drop("activity", axis=1), test_df.activity
                pred = xgb_model.predict(X_test)
                metrics = get_metrics(pred, y_test)
                
                if metrics["MAE"] < min_mae:
                    min_mae = metrics["MAE"]
                    best_preds = pred


                xgb_results["XGB_" + str(loop)] = metrics
                
            xgb_df = pd.DataFrame(xgb_results).T
            xgb_df.to_csv(f"{data_path}/Evaluation_Data/{endpoint}/XGB_results.csv", index=False)


        # Chemprop schedule
        elif model == "MPNN":
            chemprop_results = {}
            trial_data = best_trial_chemprop(trial_df)
            bs = int(trial_data["batch_size"])

            test_loader = chemprop_loader_from_df(test_df, shuffle=False, bs=len(test_df))

            train_loader = chemprop_loader_from_df(train_df, shuffle=True, bs=bs)

            for loop in range(1, repetitions+1):
                print("Training and Testing Modell " + str(loop))
                chemprop = create_optuna_chemprop(trial_data, 
                                                  n_tasks=1)

                trainer = pl.Trainer(
                        logger=False,
                        accelerator="gpu",
                        devices=1,
                        max_epochs=500)

                trainer.fit(chemprop, train_loader)

                preds = torch.concat(trainer.predict(chemprop, test_loader)).numpy().reshape(-1, )
                targets = test_df.activity
                metrics = get_metrics(preds, targets)
                     
                if metrics["MAE"] < min_mae:
                    min_mae = metrics["MAE"]
                    best_preds = preds
                
                chemprop_results["Chemprop_" + str(loop)] = metrics
            
            chemprop_df = pd.DataFrame(chemprop_results).T
            chemprop_df.to_csv(f"{data_path}/Evaluation_Data/{endpoint}/MPNN_results.csv", index=False)
        
        elif model == "TAB":
            tab_results = {}
            trial_data = best_trial_tabtransformer(trial_df)
            train_set = NumericalTabSet(train_df.drop("activity", axis=1), train_df.activity)
            train_scaler = train_set.scaler
            test_loader = DataLoader(NumericalTabSet(test_df.drop("activity", axis=1), test_df.activity, scaler=train_scaler), batch_size=len(test_df), shuffle=False)
            best_trial = trial_data["trial"]
            loss = nn.MSELoss()
            bs = trial_data["batch_size"]
            train_loader = DataLoader(train_set, shuffle=True, batch_size=int(bs))

            # Start the training
            for loop in range(1, repetitions+1):
                print("Training and Testing Modell " + str(loop), flush=True)

                tab_model = FTTransformer(
                    num_continuous=1340,
                    projected_dim=512,
                    dim=trial_data["encoder_output_dim"],
                    dim_head=trial_data["head_dim"],
                    dim_out=1,                                  
                    depth=trial_data["transformer_depth"],                                
                    heads=trial_data["amount_heads"],                           
                    attn_dropout=trial_data["attn_dropout"],       
                    ff_dropout=trial_data["ff_dropout"],
                )
                mlp_model = create_optuna_mlp(best_trial, input_feats=trial_data["encoder_output_dim"])
                tab_model.to_logits = mlp_model
                xavier_init(tab_model)
                tab_model.to(device)
                optimizer = torch.optim.Adam(tab_model.parameters(), lr=trial_data["lr"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                # Train on Train/Val Set
                for i in range(150):
                    for conts, y in train_loader:
                        optimizer.zero_grad()
                        conts = conts.float().to(device)
                        res = tab_model(conts)
                        conts = conts.to("cpu")
                        res = res.to("cpu")
                        l = loss(res.flatten(), y.float())
                        l.backward()
                        optimizer.step()

                    scheduler.step()



                res_vec = []
                for conts, y in test_loader:
                    conts = conts.float().to(device)
                    res = tab_model(conts)
                    conts = conts.to("cpu")
                    res_vec.append(res.to("cpu"))
                res = torch.concat(res_vec).flatten().detach().numpy()                    

                metrics = get_metrics(res, test_df.activity)
                
                if metrics["MAE"] < min_mae:
                    min_mae = metrics["MAE"]
                    best_preds = res

                    
                
                tab_results[f"{model}_" + str(loop)] = metrics


            tab_df = pd.DataFrame(tab_results).T
            tab_df.to_csv(f"{data_path}/Evaluation_Data/{endpoint}/{model}_results.csv", index=False)
        
        elif model == "TABPFN":
            tabpfn_results = {}
            trial_data = best_trial_tabpfn(trial_df)
            for loop in range(1, repetitions+1):
                print("Training and Testing Modell " + str(loop), flush=True)
                tabpfn_model = TabPFNRegressor(**trial_data)
                # Prevent Deterministic Behaviour
                tabpfn_model.random_state = loop 
                X, y = train_df.drop("activity", axis=1), train_df.activity
                tabpfn_model.fit(X, y)
                X_test, y_test = test_df.drop("activity", axis=1), test_df.activity
                pred = tabpfn_model.predict(X_test)
                metrics = get_metrics(pred, y_test)
                
                if metrics["MAE"] < min_mae:
                    min_mae = metrics["MAE"]
                    best_preds = pred


                tabpfn_results["TabPFN_" + str(loop)] = metrics
                
            tabpfn_df = pd.DataFrame(tabpfn_results).T
            tabpfn_df.to_csv(f"{data_path}/Evaluation_Data/{endpoint}/TabPFN_results.csv", index=False)
        
        elif model == "FOREST":
            device = "cuda:5"
            ndf_results = {}
            trial_data = best_trial_ndf(trial_df)
            train_set = MixedDescriptorSet(train_df)

            test_loader = DataLoader(MixedDescriptorSet(test_df), batch_size=len(test_df), shuffle=False)
            best_trial = trial_data["trial"]
            loss = nn.MSELoss()
            bs = trial_data["batch_size"]
            train_loader = DataLoader(train_set, shuffle=True, batch_size=int(bs))

            # Start the training
            for loop in range(1, repetitions+1):
                print("Training and Testing Modell " + str(loop), flush=True)
                # Init + create a model based on the best trial
                ndf_model = Forest(
                    n_tree=trial_data["n_tree"],
                    tree_depth=trial_data["tree_depth"],
                    n_in_feature=1340,
                    tree_feature_rate=trial_data["feature_rate"],
                    n_class=1,
                    jointly_training=False,
                    device=device
                )
                xavier_init(ndf_model)
                ndf_model.to(device)
                optimizer = torch.optim.AdamW(ndf_model.parameters(), lr=trial_data["lr"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                # Train on Train/Val Set
                for i in range(150):
                    _ = training(train_loader, ndf_model, loss, optimizer, device=device)

                    scheduler.step()


                _, pred, test_target = testing(test_loader, ndf_model, loss, device, mc_dropouts=False)


                metrics = get_metrics(pred, test_target)
                
                if metrics["MAE"] < min_mae:
                    min_mae = metrics["MAE"]
                    best_preds = pred

                    
                
                ndf_results[f"{model}_" + str(loop)] = metrics


            ndf_df = pd.DataFrame(ndf_results).T
            ndf_df.to_csv(f"{data_path}/Evaluation_Data/{endpoint}/{model}_results.csv", index=False)

        
        
        test_df = pd.read_csv(f"{data_path}/Data/{endpoint}/ADME_{endpoint}_test.csv")
        test_df["Preds"] = best_preds
        
        test_df[["smiles", "activity", "Preds"]].to_csv(f"{data_path}/Evaluation_Data/{endpoint}/{model}_preds.csv", index=False)

            