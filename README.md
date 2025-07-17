# The Experiment
Valid prediction of ADME (Absorption, Distribution, Metabolism, and Excretion) properties is crucial in drug discovery. Recent advances in tabular deep learning models including Feature Tokenizer Transformers (FTT), Neural Decision Forests (NDF), and Tabular Prior-Data Fitted Network (TabPFN) claim to outperform traditional methods like eXtreme Gradient Boosting (XGBoost) and Multilayer Perceptrons (MLPs) on tabular data. This study systematically evaluates these modern architectures against established machine learning techniques in the domain of chemoinformatics using tabular data for ADME prediction. Using publicly available Biogen datasets, various models are benchmarked on six key ADME endpoints. The results show that while modern deep learning models perform comparably, they do not consistently surpass traditional methods. Overall [FTT](https://arxiv.org/abs/2106.11959v2), [NDF](https://openaccess.thecvf.com/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf), [TabPFN](https://arxiv.org/abs/2207.01848), [XGB](https://arxiv.org/abs/1603.02754), MLP, [MPNN](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237), and [Deep-Lasso MLPs]( https://arxiv.org/pdf/2311.05877) are compared against each other.

# About the Repository

## Data
Only the public ADME data from [Fang et al.](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c00160) is used, therefore the ADME prediction involves the endpoints hPPB, rPPB, solubility, RLM, HLM, and MDR1-MDCK ER.

## Scripts
- **FTTransformer:** Using a [implementation](https://github.com/lucidrains/tab-transformer-pytorch) of the FTTransformer and adapting it slighlty.
- **NDF:** Using a [implementation](https://github.com/jingxil/Neural-Decision-Forests) of NDF and adapting it to regression problems.
- **optuna_toolkit:** Containing functions for bayesian hyperparameter tuning using optuna and a k-fold cv.
- **hyperparameter_tuning:** Running the hp tuning.
- **train_and_evaluate_model:** Use the best model configuration and re-train instances with different random seeds and evaluate on the test split.
- **Evaluation**: Containing visualization functions and statistical methods to compare the results.
- **statistical_eval:** Running a comparison by using the evaluation functions.
- **torch_toolkit:** Plain PyTorch functions to make life easier.

## Results
- **Imgs:** Contains the results from the evaluation using Cohen-D plots, applicability domain plots, Tukey heatmaps, and distribution plots for the performance
- **Experiment_Data:** Contains the hyperparameter tuning results for every model on every endpoint. A model configuration is assigned to a validation metric.
- **Evaluation_Data:** Contains the test metric for every model on every endpoint for every random seed.

# Conclusion
The results demonstrate that while modern deep learning architectures perform competitively, they do not consistently outperform traditional ML models for tabular based ADME prediction. XGBoost and MLPs remain a highly effective choice achieving superior results across multiple endpoints. These findings suggest that, despite the growing interest in Transformer based models, traditional machine learning techniques like MLPs and XGB remain the most practical and reliable choice for structured ADME prediction tasks. However, TabPFN proved itself as a promising contender since it achieved the second-best test results on every endpoint with the lowest standard deviation although the model assumption regarding the shape of the dataset are violated.