import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
import torch.nn as nn
import torch
from chemlib import Element
from torch_geometric.nn import global_mean_pool, GraphConv, NNConv, EdgeConv
import pickle
import warnings
warnings.filterwarnings("ignore")

def make_edge_features(mol):
    edge_features = np.zeros([mol.GetNumBonds(), 4])
    for i, bond in enumerate(mol.GetBonds()):
        if bond.GetBondTypeAsDouble() == 1.0:
            edge_features[i,0] = 1
        elif bond.GetBondTypeAsDouble() == 2.0:
            edge_features[i,1] = 1
        elif bond.GetBondTypeAsDouble() == 3.0:
            edge_features[i, 2] = 1
        else:
            edge_features[i, 3] = 1
    return torch.tensor(edge_features, dtype = torch.float32)


def make_x(mol):
    atom_properties = []
    
    for atom in mol.GetAtoms():
        _el = Element(atom.GetSymbol())
        atom_properties.append([atom.GetAtomicNum(), atom.GetMass(), int(atom.GetIsAromatic()), atom.GetExplicitValence(), 
                                atom.GetImplicitValence(), atom.GetTotalValence(), 
                                atom.GetNumExplicitHs(), atom.GetNumImplicitHs(), atom.GetTotalNumHs(),
                                atom.GetDegree(), atom.GetTotalDegree(), atom.GetFormalCharge(),
                               int(atom.IsInRing()), _el.Electronegativity, _el.FirstIonization, _el.AtomicRadius, _el.SpecificHeat
                               ])
    x = torch.tensor(atom_properties, dtype = torch.float32)
    return x

def make_edge_indices(mol):
    start_atoms = []
    end_atoms = []
    for bond in mol.GetBonds():
        start_atoms.append(bond.GetBeginAtomIdx())
        end_atoms.append(bond.GetEndAtomIdx())
    edge_indice = torch.tensor([start_atoms, end_atoms], dtype = torch.int64)
    return edge_indice


class GraphNet(nn.Module):
    def __init__(self, num_node_features = 17,
                n1 = 200,
                n2 = 500,
                n3 = 500,
                n4 = 500):
        super().__init__()
        conv_1_nn = nn.Sequential(nn.Linear(4, 200), nn.ReLU(), nn.Linear(200, 17*n1))
        self.conv_1 = NNConv(17, n1, nn = conv_1_nn)
        self.a1_con = nn.ReLU()
        
        conv_2_nn = nn.Sequential(nn.Linear(2*n1, n2), nn.ReLU(), nn.Linear(n2, n2))
        self.conv_2 = EdgeConv(nn = conv_2_nn)
        self.a2_con = nn.ReLU()
        
        conv_3_nn = nn.Sequential(nn.Linear(2*n2, n3), nn.ReLU(), nn.Linear(n3, n3))
        self.conv_3 = EdgeConv(nn = conv_3_nn)
        self.a3_con = nn.ReLU()
        
        self.conv_4 = GraphConv(in_channels = n3, out_channels = n4)
        self.a4_con = nn.ReLU()
        self.dropout_1 = nn.Dropout(p = 0.3)
        self.a1 = nn.CELU()
        self.linear_1 = nn.Linear(in_features = n4, out_features = 1000)
        self.a2 = nn.CELU()
        self.linear_2 = nn.Linear(in_features = 1000, out_features = 1000)
        self.a3 = nn.CELU()
        self.linear_3 = nn.Linear(in_features = 1000, out_features = 1)
    def forward(self, data):
        x, edge_index, edge_attrs =  (data.x, data.edge_index, data.edge_features)
        x = self.a1_con(self.conv_1(x = x, edge_index = edge_index, edge_attr = edge_attrs))
        x = self.a2_con(self.conv_2(x = x, edge_index = edge_index))
        x = self.a3_con(self.conv_3(x = x, edge_index = edge_index))
        x = self.a4_con(self.conv_4(x = x, edge_index = edge_index))
        if hasattr(data, 'batch'):
            x = global_mean_pool(x, batch = data.batch)
        else:
            x = torch.mean(x, dim = 0, keepdim = True)
        #x = self.a1(x)
        x = self.dropout_1(x)
        x = self.linear_1(x)
        x = self.a2(x)
        x = self.linear_2(x)
        x = self.a3(x)
        x = self.linear_3(x)
        return x


smi = input('Enter SMILES')
mol = Chem.MolFromSmiles(smi)
data = Data(x = make_x(mol), edge_index = make_edge_indices(mol), edge_features = make_edge_features(mol))
with open('best_config_temperature.pickle', 'rb') as inp:
    best_config_Tc = pickle.load(inp)

temperature_model = GraphNet(n1 = best_config_Tc['n1'],
                             n2 = best_config_Tc['n2'], 
                             n3 = best_config_Tc['n3'],
                             n4 = best_config_Tc ['n4'])

with open('temperature_model.pth', 'rb') as inp:
    temperature_model = torch.load(inp)
temperature_model.to('cpu')

print('Tc of your molecule is : ', temperature_model(data).item())

with open('best_config_pressure.pickle', 'rb') as inp:
    best_config_Pc = pickle.load(inp)

pressure_model = GraphNet(n1 = best_config_Pc['n1'],
                             n2 = best_config_Pc['n2'], 
                             n3 = best_config_Pc['n3'],
                             n4 = best_config_Pc ['n4'])

with open('pressure_model.pth', 'rb') as inp:
    pressure_model = torch.load(inp)
pressure_model.to('cpu')
print('Pc of your molecule is : ', pressure_model(data).item())

with open('best_config_omega.pickle', 'rb') as inp:
    best_config_omega = pickle.load(inp)

omega_model = GraphNet(n1 = best_config_omega['n1'],
                             n2 = best_config_omega['n2'], 
                             n3 = best_config_omega['n3'],
                             n4 = best_config_omega ['n4'])

with open('omega_model.pth', 'rb') as inp:
    omega_model = torch.load(inp)
omega_model.to('cpu')
print('Omega of your molecule is : ', omega_model(data).item())