## Predicting critical properties with Neural Networks

This project is dedicated to the prediction of critical properties of small **organic** molecules from their structure, using Neural Networks. Three critical parameters - **Tc**, **Pc**, **omega** can be calculated with workflow, presented in this project    

Project has several folders. Every folder contains README which guides thourough the content of the folder:
* *Datasets* - here datasets from [*Chemicals*](https://github.com/CalebBell/chemicals) library and datasets from [paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c00546) are curated and digested;
* *Fully connected NNs* - here descriptors are caclulated and fully-connected NNs are generated. Here the final calculator to get critical values can also be found
* *Graph NNs* - here Graph Neural Networks, that uses only molecular structure, are created. Again, final calculator to get critical values can be found.

To use models, several libraris must be installed:
* [PyTorch](https://pytorch.org/) 2.6, it would be better that CUDA is present.
* [RDKit](https://www.rdkit.org/docs/Install.html) 2024.09.5
* [ChemLib](https://chemlib.readthedocs.io/en/latest/)  2.2.4
* [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) 2.7.0 is obligatory for GNNs
* [SciKit-learn](https://scikit-learn.org/stable/) 1.6.1

Standard libraries - Pandas, Numpy, MatPlotLib, Pickle should also be present :) 

During model preparation some other librarires are used and they may be found in the beginning of each Jupyter Notebook in the import section



----
The best models of Fully-connected Neural Networks have folowing performance:


* **Tc**: RMSE for test dataset is 29 K, $R^{2}$ is 0.89  
* **Pc**: RMSE for test dataset is 2.19 bar, $R^{2}$ is 0.95  
* **omega**: RMSE for test dataset is 0.078, $R^{2}$ is 0.79    
However, there is a small problem that you need to keep a lot of files: column transofrmers (for each separate model), imputer, list of descriptors to drop, etc., so it's  recommended to try Graph Neural Networks, which require only molecular structure ^:-))
---
The best models of Graph  Neural Networks have folowing performance:

* **Tc**: RMSE for test dataset is 28 K, $R^{2}$ is 0.91
* **Pc**: RMSE for test dataset is 2.02 bar, $R^{2}$ is 0.96
* **omega**: RMSE for test dataset is 0.10, $R^{2}$ is 0.67

  ---
  If you have any problems or just want to say some feedback words, you can write me to timyun96@gmail.com
