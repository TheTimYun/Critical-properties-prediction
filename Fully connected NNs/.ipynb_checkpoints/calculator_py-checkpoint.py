import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd
import numpy as np
from rdkit.Chem import DataStructs
from sklearn.impute import KNNImputer
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")



with open('features_to_drop.pickle', 'rb') as inp:
    features_to_drop = pickle.load(inp)


with open('imputer.pickle', 'rb') as inp:
    imputer = pickle.load(inp)

def mols_to_descriptors(list_of_mols):
    descriptor_dict = {desc[0]:[] for desc in Descriptors.descList }
    for mol in list_of_mols:
        for descriptor, func in Descriptors.descList:
            try:
                descriptor_dict[descriptor].append(func(mol))
            except:
                descriptor_dict[descriptor].append(np.nan)
    return pd.DataFrame(descriptor_dict)


def mols_to_fingerprints(list_of_mols):
    fp_list = []
    for mol in list_of_mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 3)
        dest_array = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fp, dest_array)
        fp_list.append(dest_array)
    return pd.DataFrame(np.stack(fp_list), columns = ['fp{}'.format(i) for i in range(2048)])
    
    
def mol_to_final_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    desc_df = mols_to_descriptors([mol])
    desc_df = desc_df.drop(columns = features_to_drop)
    desc_df_columns = desc_df.columns
    desc_df = pd.DataFrame(imputer.transform(desc_df), columns = desc_df_columns)
    fp_df = mols_to_fingerprints([mol])
    concatenated_table = pd.concat([desc_df, fp_df], axis = 1)
    return concatenated_table


smiles = input('Please, enter the SMILES of your molecule')

desc_df = mol_to_final_descriptors(smiles)
with open('./Omega predictions/column_transformer_omega.pickle', 'rb') as inp:    
    ct_omega = pickle.load(inp)

omega_dataset = torch.Tensor(ct_omega.transform(desc_df))

class MyModel(nn.Module):
    def __init__(self, l1 = 2700, l2 = 900):
        super().__init__()
        self.linear_1 = nn.Linear(2204, l1)
        self.a1 = nn.CELU(alpha = 0.01)
        self.dropout_1 = nn.Dropout(p = 0.3)
        self.linear_2 = nn.Linear(l1, l2)
        self.a2 = nn.CELU(alpha = 0.01)
        self.dropout_2 = nn.Dropout(p = 0.3)
        self.linear_3 = nn.Linear(l2, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.a1(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.a2(x)
        x = self.dropout_2(x)
        x = self.linear_3(x)
        return x

with open('./Omega predictions/omega_model.pth', 'rb') as inp:
    omega_model = torch.load(inp)
omega_model.to('cpu')

omega = omega_model(omega_dataset)
print('Omega is equal to ', omega.item())


with open('./Critical pressure/column_transformer_Pc.pickle', 'rb') as inp:    
    ct_Pc = pickle.load(inp)
pressure_dataset = torch.Tensor(ct_Pc.transform(desc_df))
    
with open('./Critical pressure/model_1_dict_state.pth', 'rb') as inp:
    pressure_model = MyModel()
    pressure_model.load_state_dict(torch.load(inp, weights_only = True))

pressure_model.to('cpu')
pressure = pressure_model(pressure_dataset)
print('Pc is equal to ', pressure.item())


with open('./Critical temperature/column_transformer_Tc.pickle', 'rb') as inp:    
    ct_Tc = pickle.load(inp)
temperature_dataset = torch.Tensor(ct_Tc.transform(desc_df))

with open('./Critical temperature/model_1_dict_state.pth', 'rb') as inp:
    temperature_model = MyModel()
    temperature_model.load_state_dict(torch.load(inp, weights_only = True))

temperature_model.to('cpu')
temperature = temperature_model(temperature_dataset)
print('Tc is equal to ', temperature.item())
