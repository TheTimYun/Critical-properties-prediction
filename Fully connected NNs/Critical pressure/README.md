## Predicting critical pressure with fully connected neural networks (NNs)

This folder contains following files:
1. *table_with_descriptors.csv* - contains all descriptors of all molecules with structures and critical values. It was obtained on the earlier step and the process is described in *descriptors_calculation.ipynb*.
2. *Run_result_omega.ipynb* - in order to reduce the number of final model architectures, we check the hypothesis, that optimal architecture and learning rate, obtained for **omega** prediction, can also work satisfactorily for other parameters, such as **Pc** and **Tc**. Therefore, we save result of [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) run for **omega** here. A little spoiler: its performance is not much lower, than of the optimal **Pc** model so that is used in final calculator.
3. *Run_result_pressure.ipynb* - is the result of Ray Tune run for **Pc**
4. *model_1_dict_state.pth* and *model_2_dict_state.pth* are state dicts for the best **omega** model and for the best **Pc** model, respectively. What is the best **omega** model and why it is used here, is described slightly above.
5. *column_transformer_Pc.pickle* is the instance of SciKit-learn column transformer instance to scale columns of descriptors dataset. 
6. *Model_selection_pc.ipynb* - describes the process of the search for optimal model architecture with [Wandb](https://wandb.ai/site/)
7. *Model_parameterization_pc.ipynb* - describes the process of the search for optimal hyperparameters with Ray Tune and Hyperopt and training the model

---

The best model for **omega** prediction is:
Linear(2204->2700) -> CELU(alpha = 0.01) -> dropout(p = 0.3) -> Linear(2700 -> 900) ->CELU(alpha = 0.01) -> dropout(p = 0.3) -> Linear(900 -> 1)  
Learning rate is 0.00010525231047880925  
It's performance for **Pc** prediction is: $R^{2}$ 0.95, RMSE 2.19 bar

The best model for **Pc** prediction is:
Linear(2204->800) -> GELU(approximate='none') -> dropout(p = 0.3) -> Linear(800 -> 1100) -> GELU(approximate='none') -> dropout(p = 0.3) -> Linear(1100 -> 1)  
Learning rate is 0.0006656808561573917  
It's performance for **Pc** prediction is: $R^{2}$ 0.96, RMSE 1.96 bar

---