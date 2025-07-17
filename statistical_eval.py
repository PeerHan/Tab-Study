#!/usr/bin/env python3

from Evaluation import *
import pandas as pd

print("Start Eval...")
extern_reference_R = {"hPPB" : {"model" : "LightGBM", "R" : 0.8}, 
                      "rPPB" : {"model" : "MPNN2", "R" : 0.74}, 
                      "Sol" : {"model" : "MPNN2", "R" : 0.66},
                      "RLM" : {"model" : "MPNN2", "R" : 0.75},
                      "HLM" : {"model" : "MPNN2", "R" : 0.73},
                      "MDR1_ER" : {"model" : "MPNN2", "R" : 0.78}}

extern_df = pd.DataFrame(extern_reference_R).T.reset_index(names="Endpoint")
extern_df["Model"] = "Biogen"
extern_df["R2"] = extern_df.R ** 2

dfs = []
for endpoint in extern_reference_R.keys():
    df = model_df_on_set(endpoint, "Evaluation_Data/")
    dfs.append(df)

dfs = pd.concat(dfs, ignore_index=True)
df = pd.concat((dfs, extern_df), ignore_index=True, axis=0)

df.Model = df.Model.str.replace("TABPFN", "TabPFN")
df.Model = df.Model.str.replace("TAB", "FTT")
df.Model = df.Model.str.replace("FOREST", "NDF")

models = ["MLP", "XGB", "FTT", "TabPFN", "NDF"]
df = df[df.Model.isin(models + ["Biogen"])].reset_index(drop=True)

# Significance Plots
tukey_subplot(df, "Imgs/TukeyPlots_R2.png", models=models)
tukey_subplot(df, "Imgs/TukeyPlots_MAE.png", models=models, metric="MAE")
# Effect Size Plots
for endpoint in extern_reference_R.keys():
    cohen_plot(df, models, f"Imgs/Cohenplot_{endpoint}_R2.png", endpoint)    
    cohen_plot(df, models, f"Imgs/Cohenplot_{endpoint}_MAE.png", endpoint, metric="MAE")   

# Distribution of R2 Values
plot_distribution(df, "R2", kind="violin", save_path="Imgs/Summaryplot_R2_violin.png")
plot_distribution(df, "R2", kind="strip", save_path="Imgs/Summaryplot_R2_strip.png")
plot_distribution(df[df.Model != "Biogen"], "MAE", kind="violin", save_path="Imgs/Summaryplot_MAE_violin.png",
                  hue_order=models)
plot_distribution(df[df.Model != "Biogen"], "MAE", kind="strip", save_path="Imgs/Summaryplot_MAE_strip.png",
                  hue_order=models)
                                
# Applicability Domain
applicability_domain_plots("Imgs/", list(extern_reference_R.keys()), models)
 