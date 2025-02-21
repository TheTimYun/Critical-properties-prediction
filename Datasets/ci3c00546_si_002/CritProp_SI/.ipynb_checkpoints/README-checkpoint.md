## Details on the properties

The data set includes the following 8 properties:
- Tc: critical temperature, in K
- Pc: critical pressure, in bar
- rhoc: critical density, in mol/L
- omega: acentric factor, unitless
- Tb: boiling point, in K
- Tm: melting point, in K
- dHvap: enthalpy of vaporization at boiling point, in kJ/mol
- dHfus: enthalpy of fusion at melting point, in kJ/mol

## Details on the files

- `all_data`: includes the data sets used in this work. All data points are listed for each chemical compound as well as
its corresponding data source. The details of the data sources are listed in the next section. The distribution of the data set is included in each folder.
  - `estimated_data_for_pretraining`: contains the estimated data from Yaws' handbook that are used to pre-train our machine learning
    (ML) model.
  - `experimental_data`: contains the experimental data (references 1 - 15) used to fine-tune our final ML model.
- `additional_features`: includes the additional features tested for the ML model. The Abraham features are generated for all data (references 1 - 15)
while the acsf, qm, and rdkit features are only generated for the data from references 1 - 9.
  - abraham: Abraham solute parameters (E, S, A, B, L). Molecular features.
  - acsf: ACSF (atom-centered symmetry functions). Atomic features that are converted from the 3D coordinates of the compound.
  - qm_atom: QM (quantum chemical) atomic feature. 
  - qm_mol: QM molecular feature.
  - rdkit: Selected RDKit 2D molecular features.
- `data_splits_and_model_predictions`: contains the training and test sets used to evaluate the model. It also
contains the predicted values from our final ML model for each test set.
  - random and scaffold splits: training and test sets that include the data from references 1 - 9.
  - external test set: a test set that includes the data from only references 10 - 15.

## Data sources

The experimental data are collected from the following sources:

1. Green, D. W.; Perry, R. H. <i>Perry’s Chemical Engineers’ Handbook (Section 2 - Physical
and Chemical Data)</i>, 8th ed.; McGraw-Hill’s AccessEngineering; McGraw-Hill: New
York, 2000.
2. Kleiber, M.; Joh, R.; Span, R. D3 Properties of Pure Fluid Substances:
Datasheet from VDI-Buch Volume: “VDI Heat Atlas” in SpringerMaterials
(https://doi.org/10.1007/978-3-540-77877-6). https://materials.springer.com/lb/docs/sm_nlb_978-3-540-77877-6_18.
3. Yaws, C. L. Chapter 1: Critical properties and acentric factor — Organic compounds. In <i>Thermophysical Properties of Chemicals and Hydrocarbons</i>; William Andrew, 2009.
4. Yaws, C. L. Chapter 2: Critical properties and acentric factor — Inorganic compounds. In <i>Thermophysical Properties of Chemicals and Hydrocarbons</i>; William Andrew, 2009.
5. Linstrom, P. J.; Mallard, W. G.; Eds., NIST Chemistry WebBook, NIST Standard
Reference Database Number 69. https://doi.org/10.18434/T4D303.
6. Yaws, C. L. Chapter 11: Enthalpy of fusion at freezing point — Organic compounds \& Chapter 12: Enthalpy of fusion at freezing point — Inorganic compounds. In <i>Thermophysical Properties of Chemicals and Hydrocarbons</i>; William Andrew, 2009.
7. Yaws, C. L. Chapter 9: Enthalpy of vaporization at boiling point — Organic compounds \& Chapter 10: Enthalpy of vaporization at boiling point — Inorganic compounds. In <i>Thermophysical Properties of Chemicals and Hydrocarbons</i>; William Andrew, 2009.
8. Rumble, J. R.; ed., <i>CRC Handbook of Chemistry and Physics, 102nd Edition (Internet
Version 2021)</i>; CRC Press/Taylor & Francis: Boca Raton, FL.
9. Joback, K. G. A unified approach to physical property estimation using multivariate
statistical techniques. Ph.D. thesis, Massachusetts Institute of Technology, 1984.
10. Kudchadker, A. P.; Ambrose, D.; Tsonopoulos, C. Vapor-liquid critical properties of elements and compounds. 7. Oxygen compounds other than alkanols and cycloalkanols. 
<i>J. Chem. Eng. Data</i> <b>2001</b>, 46, 457–479.
11. Tsonopoulos, C.; Ambrose, D. Vapor-liquid critical properties of elements and compounds. 8. Organic sulfur, silicon, and tin compounds (C+ H+ S, Si, and Sn). 
<i>J. Chem. Eng. Data</i> <b>2001</b>, 46, 480–485.
12. Marsh, K. N.; Young, C. L.; Morton, D. W.; Ambrose, D.; Tsonopoulos, C. Vapor-liquid critical properties of elements and compounds. 9. Organic compounds containing nitrogen. 
<i>J. Chem. Eng. Data</i> <b>2006</b>, 51, 305–314
13. Marsh, K. N.; Abramson, A.; Ambrose, D.; Morton, D. W.; Nikitin, E.; Tsonopoulos, C.; Young, C. L. Vapor-liquid critical properties of elements and compounds. 10. Organic compounds containing halogens. 
<i>J. Chem. Eng. Data</i> <b>2007</b>, 52, 1509–1538.
14. Ambrose, D.; Tsonopoulos, C.; Nikitin, E. D. Vapor-Liquid Critical Properties of Elements and Compounds. 11. Organic Compounds Containing B+ O; Halogens+ N,+ O,+ O+ S,+ S,+ Si; N+ O; and O+ S,+ Si. 
<i>J. Chem. Eng. Data</i> <b>2009</b>, 54, 669–689.
15. Ambrose, D.; Tsonopoulos, C.; Nikitin, E. D.; Morton, D. W.; Marsh, K. N. Vapor–liquid critical properties of elements and compounds. 12. Review of recent data for hydrocarbons and non-hydrocarbons. 
<i>J. Chem. Eng. Data</i> <b>2015</b>, 60, 3444–3482

For the data from Yaws's handbook (ref 3, 4, 6, 7), `-expt` indicates the experimental data, and `-est` indicates the estimated data.

## Authors

Sayandeep Biswas, Yunsie Chung, Josephine Ramirez, Haoyang Wu, William H. Green

Green Group, Department of Chemical Engineering
Massachusetts Institute of Technology

Email: whgreen@mit.edu


## License

The materials are open access and distributed under the terms and conditions of the Creative Commons Attribution (CC BY 4.0) license (https://creativecommons.org/licenses/by/4.0/).

