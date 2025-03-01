## Predicting critical properties with fully connected neural networks (NNs)

This folder contains following files:
1. *grouped_table.csv* is the table with structures and critical parameters - **Tc**, **Pc**, **omega**, created during datasets curation;
2. *descriptors calculation.ipynb* describes the process of getting descriptors of all molecules in dataset;
3. *table_with_descriptors.csv* - contains all descriptors of all molecules with structures and critical values ;
4. *features_to_drop.pickle* contains a list of descriptors. After calculation of all [RDKit](https://www.rdkit.org/) descriptors, it was found that some of them are highly correlated. Therefore, we decided to drop desciptors that have  high correlation coefficient. This file contains such a features.
5. *calculator_py.py* is a code, that calculates critical parameters using the best models.   To make calculator work correctly, one should have RDKit, [Scikit-learn](https://scikit-learn.org/stable/) and [PyTorch](https://pytorch.org/). It would be better, if CUDA is installed. It can be launched with
```
python caclulator_py.py

```
 And after that you will be prompted to type SMILES of your molecule
8. *calculator_notebook.ipynb* is Jupyter Notebook version of the calculator

Folders *Critical pressure*, *Critical temperature*, *Omega* contains files, related to models. The process of their development is also described in corresponding READMEs. We followed the way **Omega** (as the worst to predict) -> **Pc** -> **Tc**. The consequence is important as result of search for the best **omega** prediction is then used for the work with **Pc** and **Tc**


The best models have folowing parameters:

* **Tc**: RMSE for test dataset is 29 K, $R^{2}$ is 0.89
* **Pc**: RMSE for test dataset is 2.19 bar, $R^{2}$ is 0.95
* **omega**: RMSE for test dataset is 0.078, $R^{2}$ is 0.79  
However, there is a huge problem that you need to keep a lot of files: column transofrmers (for each separate model), imputer, list of descriptors to drop, etc., so it's  recommended to try Graph Neural Networks)
---