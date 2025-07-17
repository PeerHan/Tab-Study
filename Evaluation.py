import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from scipy import stats
import numpy as np
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from math import sqrt
from rdkit import Chem
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import pandas as pd
from statsmodels.stats.anova import AnovaRM 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import tukey_hsd
import scipy.stats as stats

sns.set_style("darkgrid")
sns.set(font_scale=1.25)


def get_metrics(
        preds : np.ndarray, 
        targets : np.ndarray,
    ) -> dict:
    """
    Function to calculate metrics based on the model prediction and the ground truth.
    The following metrics (sklearn) are calculated:
        - Mean absolute error
        - Mean squared error
        - Root mean squared error
        - Pearson R
        - R^2 
        - Spearman R
    
    Parameters:
        - preds : Prediction array
        - targets : Target array
        - dataset : Label for the test-set
    """
    
    results = {}
    results["MAE"] = mean_absolute_error(targets, preds)
    results["MSE"] = mean_squared_error(targets, preds)
    results["RMSE"] = sqrt(results["MSE"])
    results["R"] = np.corrcoef(targets, preds)[0, 1]
    results["R2"] = results["R"] ** 2
    results["SPR"] = spearmanr(targets, preds).statistic
    
    return results

def model_df_on_set(
        endpoint : str, 
        data_path : str, 
    ) -> "DataFrame":
    """
    Function to create a combined dataframe per training set (Trained on intern or mixed (inex) data) on a specific endpoint.
    Parameters:
        - endpoint : Desired Endpoint (hPPB, rPPB, Sol, HLM, RLM)
        - data_path : Path to the set folder
        - training_set : Get the results from models which are trained either on Intern oder InEx (mixed) data
        - mlp_metric : Choose if the best MAE or R2 Trial should be used for the comparison
    """
    models = ["MLP", "XGB", "MPNN", "LASSO_MLP", "TAB", "TabPFN", "FOREST"]
    model_dfs = [pd.read_csv(f"{data_path}/{endpoint}/{model}_results.csv") for model in models ]
    for model_df, model in zip(model_dfs, models):
        model_df["Model"] = model.replace("_", " ")
    result_df = pd.concat(model_dfs, ignore_index=True)
    result_df["Endpoint"] = endpoint
    return result_df

def tukey_hsd_same_endpoint(df, endpoint, metric="R2"):
    
    sub_df = df[(df.Endpoint == endpoint) & (df.Model != "Biogen")][[metric, "Model", "Endpoint"]]
    sub_df["Subject"] = [i for i in range(1, 31)] * (df.Model.unique().shape[0] - 1)
    anova = AnovaRM(sub_df, depvar=metric, subject="Subject", within=["Model"]).fit()
    
    mlp = sub_df[(sub_df.Model == "MLP")][metric].tolist()
    xgb = sub_df[(sub_df.Model == "XGB")][metric].tolist()
    #mpnn = sub_df[(sub_df.Model == "MPNN")].R2.tolist()
    #lasso = sub_df[(sub_df.Model == "LASSO MLP")].R2.tolist()
    forest = sub_df[(sub_df.Model == "NDF")][metric].tolist()
    tabtrans = sub_df[(sub_df.Model == "FTT")][metric].tolist()
    tabpfn = sub_df[(sub_df.Model == "TabPFN")][metric].tolist()
    res = tukey_hsd(mlp, xgb, tabtrans, forest, tabpfn) # Add lasso, mpnn if desired

    return res.pvalue, anova.anova_table.iloc[0, -1]

def plot_distribution(
        df : "DataFrame", 
        metric : str = "R2", 
        save_path : str = None,
        hue_order : list = ["MLP", "XGB", "FTT", "TabPFN", "NDF", "Biogen"],
        kind : str = "violin"
    ) -> None:
    """
    Function to plot the distribution of a specified metric.
    The hue visualizes the used model, while the x-axis shows the used training set.
    The figure consists of 2 parts : One Plot for the metric distribution on the intern test set and one for the extern test set.
    Parameters:
        - endpoint_df : Dataframe which was created from result_df_per_endpoint
        - endpoint : Current endpoint (title)
        - extern_reference_dict : Dictionary to reference a threshold. The dictionary should have the keys value (y metric) und model (name of the used model). The value should be measured as Pearson R.
        - metric : Specify the distribution of a specific metric (R, R2, SPR (Spearman R), MAE, MSE, RSME)
        - save_path : If specified the fig will be saved in 'path/file_name.svg'
        - kind : kind for seaborn catplot
    """

    g = sns.catplot(df,
                    kind=kind, 
                    x="Endpoint", 
                    y=metric, 
                    hue="Model", 
                    palette="tab10", 
                    hue_order=hue_order,
                    height=6,
                    aspect=2.5)

    
    g.fig.suptitle(f"Distribution of the best Model for each Endpoint", y=1.05, size=30)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=550)
        plt.close()
    else:
        plt.show()
    return

def create_fp(
        df : "DataFrame", 
        nbits : int = 1024, 
        radius : int = 2,
    ) -> [list, list]:
    """
    Transforms smiles to MorganFingerprint.
    Returns fp and the smiles as lists.
    Parameters:
        - nbits : Amount of bits for the fp (default = 1024)
        - radius : Radius for the fp (default = 2)
    """
    smiles = df.smiles.tolist()
    ms = [Chem.MolFromSmiles(x) for x in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, nBits=nbits, radius=radius) for mol in ms]
    return fps, smiles

def calc_dice_sim(
        test_fps : list, 
        train_fps : list, 
        test_smiles : list, 
        train_smiles : list
    ) -> "DataFrame":
    """
    Creates a similarity dataframe based on the Sorensen Dice Coefficient.
    The dataframe has the following properties:
        - sm1 : smiles from the test df
        - sm2 : smiles from the train df
        - sim : similarity
    Parameters:
        - test_fps : test fingerprints
        - train_fps : train fingerprints
        - test_smiles : test smiles
        - train_smiles : train smiles
    """
    qu, ta, sim = [], [], []
    for n in range(len(test_fps)):
        s = DataStructs.BulkDiceSimilarity(test_fps[n], train_fps[:]) 
        # collect the SMILES and values
        for m in range(len(s)):
            qu.append(test_smiles[n])
            ta.append(train_smiles[:][m])
            sim.append(s[m])
    sim_df = pd.DataFrame({"sm1" : qu, "sm2" : ta, "sim" : sim})
    return sim_df

def get_sim_df(
        test_df : "DataFrame", 
        train_df : "DataFrame", 
        nbits : int = 1024, 
        radius : int = 2
    ) -> "DataFrame":
    """
    Wrapper function to create a smilarity dataframe based on the sorensen dice coefficient 
    based on a train and test dataframe (smile_col = smiles).
    Parameters:
        - test_df : Dataframe with test smiles
        - train_df : Dataframe with train smiles
        - nbits and radius : Parameters for the morgan fingerprint
    """
    train_fps, train_smiles = create_fp(train_df, nbits, radius)
    test_fps, test_smiles = create_fp(test_df, nbits, radius)
    sim_df = calc_dice_sim(test_fps, train_fps, test_smiles, train_smiles)
    sim_df = sim_df.sort_values("sim", ascending=False)
    return sim_df

def aggregate_sim_df(
        sim_df : "DataFrame", 
        top_k : int = 5
    ) -> "DataFrame":
    """
    Takes in a similarity dataframe and calculates the top_k neighbours based on the similarity, returning
    an aggregated dataframe.
    The top_k neighbours are the most similar k train smiles related to a test smile.
    Parameters:
        - sim_df : similarity dataframe
        - top_k : amount of k neighbours which should be considered for the aggregation on the sim score
    """
    sim_df = sim_df.sort_values(["sm1", "sim"], ascending=False)
    group_df = sim_df.groupby("sm1").head(top_k).sort_values("sm1").groupby("sm1").sim.agg(["min", "max", "mean", "std"]).sort_values("min")
    group_df = group_df.reset_index()
    group_df = group_df.rename({"sm1" : "smiles"}, axis=1)
    return group_df

def get_merged_df(
        group_df : "DataFrame", 
        pred_df : "DataFrame"
    ) -> "DataFrame":
    """
    Merging the grouped similarity DF with a prediction dataframe.
    Furthermore, the MAE is calculated and the mean similarity values are binned.
    Parameters:
        - group_df : Aggregated similarity df
        - pred_df : Predictions of a model containing smiles, activity and prediction
    """
    merged = group_df.merge(pred_df, on="smiles")
    merged["MAE"] = abs(merged.activity - merged.Preds)
    merged["Mean of top 5 similarity (Binned)"] = pd.cut(merged["mean"], bins=np.arange(0., 1.1, 0.1))
    merged = merged.sort_values("Mean of top 5 similarity (Binned)")
    return merged

def cohen_d(vector1, vector2):
    mu1 = vector1.mean()
    mu2 = vector2.mean()
    var1 = vector1.var()
    var2 = vector2.var()
    cohen_d = (mu1 - mu2) / np.sqrt((var1 + var2) / 2)
    effect_size = "Huge" if cohen_d > 2 else "Very Large" if cohen_d > 1.2 else "large" if cohen_d > 0.8 else "medium" if cohen_d > 0.5 else "small" if cohen_d > 0.2 else "Very small"
    return cohen_d, effect_size

def calc_cohen_matrix(df, models, endpoint, metric="R2"):
    matrix = np.empty((len(models), len(models)))

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            d, effect = cohen_d(df[(df.Model == model1) & (df.Endpoint == endpoint)][metric],
                                df[(df.Model == model2) & (df.Endpoint == endpoint)][metric])
            matrix[i, j] = d
            
    return matrix

def cohen_plot(df, models, save_path, endpoint, metric="R2"):
    hue_order = ["MLP", "XGB", "FTT", "TabPFN", "NDF"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    fig.tight_layout()
    sub_df = df[df.Endpoint == endpoint]
    mat = calc_cohen_matrix(sub_df, models, endpoint, metric=metric)
    sns.heatmap(mat,
                xticklabels=models,
                yticklabels=models,
                cmap="flare_r",
                annot=True,
                ax=ax2)
    sns.kdeplot(sub_df[(sub_df.Model != "Biogen")], 
                x=metric, 
                hue="Model",
                bw_adjust=2,
                palette="tab10",
                hue_order=hue_order,
                ax=ax1)
    ax1.title.set_text(f"KDE Plot of {metric} Values")
    ax2.title.set_text("Pairwise Cohen d Values")
    fig.suptitle(f"Cohen D Statistic and {metric} Distribtion for Endpoint {endpoint}", y=1.05)
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(save_path, bbox_inches="tight", dpi=250)
    plt.close()
    
def tukey_subplot(df, save_path, models, metric="R2"):
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(20, 10), dpi=250)
    fig.tight_layout()
    
    j = k = 0

    for i, endpoint in enumerate(["hPPB", "rPPB", "Sol", "RLM", "HLM", "MDR1_ER"]):
        tukey_hm, anova_p = tukey_hsd_same_endpoint(df, endpoint, metric=metric)
        hm = sns.heatmap(tukey_hm,
                         xticklabels=models,
                         yticklabels=models,
                         ax=axs[j, k],
                         vmin=0,
                         vmax=0.05,
                         cmap="flare",
                         annot=True,
                         cbar=0)
        axs[j, k].title.set_text(f"Test {metric} for {endpoint}(ANOVA P-val: {anova_p:.2})")

        k += 1
        if k > 2:
            k = 0
            j += 1

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position der Colorbar anpassen
    fig.colorbar(hm.collections[0], cax=cbar_ax, label="P-values of Tukey Posthoc Test")
    fig.subplots_adjust(right=0.9)
    fig.suptitle("Significant Model Differences per Endpoint", y=1.05, size=30)
    fig.savefig(save_path, bbox_inches="tight", dpi=250)
    plt.close()
    
def applicability_domain_plots(save_folder, endpoints, models):

    for endpoint in endpoints:
        train_df = pd.read_csv(f"Data/{endpoint}/ADME_{endpoint}_train.csv")
        test_df = pd.read_csv(f"Data/{endpoint}/ADME_{endpoint}_test.csv")

        sim_df = get_sim_df(test_df, train_df)
        group_df = aggregate_sim_df(sim_df)

        fig, axs = plt.subplots(2, 3, figsize=(20, 10), dpi=250, sharex=True, sharey=True)

        j = k = 0
        model_dict = {"FTT" : "TAB",
                      "TabPFN" : "TABPFN",
                      "LASSO MLP" : "LASSO_MLP",
                      "NDF" : "FOREST"}
        for i, model in enumerate(models):
            model_name = model_dict[model] if model in model_dict else model
            preds = pd.read_csv(f"Evaluation_Data/{endpoint}/{model_name}_preds.csv")
            merged = get_merged_df(group_df, preds)
            cat_codes = merged["Mean of top 5 similarity (Binned)"].cat.codes
            merged["Mean of top 5 similarity (right Bin)"] = merged["Mean of top 5 similarity (Binned)"].apply(lambda x : x.right)
            sns.barplot(merged, x="Mean of top 5 similarity (right Bin)", y="MAE", ax=axs[j, k])
            axs[j, k].set_xlim(cat_codes.min() - 0.5, cat_codes.max() + 0.5)
            axs[j, k].title.set_text(model)
            k += 1
            if k == 3:
                k = 0
                j += 1
            merged.to_csv(f"Data/Applic/{model}_{endpoint}.csv", index=False)
        axs[1, 2].set_visible(False)
        #axs[1, 0].set_position([0.24, 0.125, 0.228, 0.343])
        #axs[1, 1].set_position([0.55, 0.125, 0.228, 0.343])
        fig.tight_layout()
        fig.suptitle(f"Applicability Domain per Model for Endpoint {endpoint}", y=1.05, size=30)
        fig.savefig(f"{save_folder}/{endpoint}_applicability_domain.png",
                    bbox_inches="tight",
                    dpi=250)
        plt.close()