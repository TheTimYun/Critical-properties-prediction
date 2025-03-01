## Working with Datasets

This folder contains following files:
1. *Appendix to PSRK Revision 4.tsv*, *CRCCriticalOrganics.tsv*, *DIPPRPinaMartines.tsv*, *IUPACOrganicCriticalProps.tsv*, *PassutDanner1973.tsv*, *Yaws Collection.tsv* are datasets from [*Chemicals*](https://chemicals.readthedocs.io/index.html) library. 
2. *From_chemical_datasets.csv* contains conctenated raw data from all previously described *Chemicals* datasets. There is only data about **CAS number**, **Chemical name**, **Tc**, **Pc**, **omega** and **SMILES**. No duplicate removal was made! It was used as temporary file during curation. 
3. Folder *ci3c00546_si_002* is supplementary information from [paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c00546). File *'CritProp_SI/all_data/experimental_data/critprop_data_only_smiles_mean_value_expt.csv'* contains experimental data on critical properties, that was used for the project - **Tc**, **Pc**, **omega**. 
4. *Datasets curation.ipynb* makes all the work. We upload datasets from *Chemicals*, merge them, retrieve SMILES from PubChem, upload dataset from paper, exclude *NaN*s, inorganic molecules and salts, average values from different sources.
5. *grouped_table.csv* is the final result of Dataset curation

---