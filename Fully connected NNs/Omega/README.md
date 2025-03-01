## Predicting omega values with fully connected neural networks (NNs)

This folder contains following files:
1. *table_with_descriptors.csv* - contains all descriptors of all molecules with structures and critical values. It was obtained on the earlier step and the process is described in *descriptors_calculation.ipynb*.
2. *Run_result.ipynb* - contains the run result of [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) during the search of optimal hyperparameters for **omega** prediction. 
3. *omega_model.pth* is the model (not state_dict!) for the best omega prediction. It's architecture can be found below in this file.
5. *column_transformer_omega.pickle* is the instance of SciKit-learn column transformer to scale numeric columns of descriptors dataset and input data. 
6. *Model_selection_omega.ipynb* - describes the process of the search for optimal model architecture with [Wandb](https://wandb.ai/site/)
7. *Model_parameterization_omega.ipynb* - describes the process of the search for optimal hyperparameters with Ray Tune and Hyperopt and training the model

---

The best model for **omega** prediction is:
Linear(2204->2700) -> CELU(alpha = 0.01) -> dropout(p = 0.3) -> Linear(2700 -> 900) ->CELU(alpha = 0.01) -> dropout(p = 0.3) -> Linear(900 -> 1)  
Learning rate is 0.00010525231047880925  
It's performance for **Pc** prediction is: $R^{2}$ 0.79, RMSE is 0.078


---