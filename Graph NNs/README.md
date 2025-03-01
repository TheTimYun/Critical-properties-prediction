## Predicting critical properties with Graph Neural Networks

This set of files is dedicated to the prediction of critical properties using Graph Neural Network. As opposed to fully connected NNs requiring scaled set of certain features, GNNs use only moleculular structure as input.

This folder contains following files:
1. *best_config_Temperature.pickle*, *best_config_Pressure.pickle*, *best_config_omega.pickle* - contain dictionaries with optimal parameters for model training - number of neurons on layers, number of epochs and learning rate for each critical value - **Tc**, **Pc**, **omega**.
2. *omega_result.pickle*, *pressure_result.pickle*, *temperature_result.pickle* - contain instances of Ray Tune *Result* files for the search for optimal model hyperparameters.
3. *omega_model.pth*, *presure_model.pth*, *temperature_model.pth* - are trained instances of GraphNet class (desctibed below).
4. *Model_parametrization_Omega.ipynb*, *Model_parametrization_pressure.ipynb*, *Model_parametrization_temperature.ipynb* - contain the workflow of the search for optimal  hyperparameters for each critical value.
5. *Model_architecture_search.ipynb* - contains the workflow of the search for the best model architecture.
6. *Calculator_notebook.ipynb* and *calculator_py.py* are created to calculate Critical properties on the base of trained models. *.py* file can be launched in terminal 

```
python calculator_py.py

```
After then, you will be prompted to type molecule SMILES and get critical values. 

To use models, several libraries must be installed:
* [PyTorch](https://pytorch.org/) 2.6, it would be better that CUDA is present.
* [RDKit](https://www.rdkit.org/docs/Install.html) 2024.09.5
* [ChemLib](https://chemlib.readthedocs.io/en/latest/)  2.2.4
* [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) 2.7.0 is obligatory for GNNs  

To launch Notebook, you should also have [SciKit-learn](https://scikit-learn.org/stable/) 1.6.1 to perform ```train_test_split``` and [Ray tune](https://docs.ray.io/en/latest/tune/index.html) to perform the search for hyperparameters

The base class for critical properties preiction, is ```GraphNet```, inheriting ```nn.Module```, which consists of several layers:  
NNConv(17 -> n1) -> ReLU -> EdgeConv(n1, n2) -> ReLU -> EdgeConv(n2, n3) -> ReLU -> GraphConv(n3, n4) -> ReLU -> Global_mean_pool -> dropout(p = 0.3) -> Linear(n4, 1000) -> CELU  - > Linear(1000, 1000) -> CELU -> Linear(1000,1)

For **Tc** prediction, best parameters are: n1 = 500, n2 = 1000, n3 = 800, n4 = 900, 
For **Pc** prediction, best parameters are: n1 = 300, n2 = 200, n3 = 400, n4 = 300, n_epochs = 45, lr =  0.00037787236069767406 
For **omega** prediction, best parameters are: n1 = 400, n2 = 700, n3 = 500, n4 = 800  

---
The best models of Graph Neural Networks have following performance:

* **Tc**: RMSE for test dataset is 28 K, $R^{2}$ is 0.91
* **Pc**: RMSE for test dataset is 2.02 bar, $R^{2}$ is 0.96
* **omega**: RMSE for test dataset is 0.10, $R^{2}$ is 0.67

  ---
