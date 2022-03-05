import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from typing import List
import pyrfume
import requests

CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"

class RDKit_2D:
    def __init__(self, smiles: pd.Series):
        self.smiles: pd.Series = smiles
        self.molecules: List[Chem.rdchem.Mol] = [Chem.MolFromSmiles(smile)
                                                 for smile in self.smiles]    

    def compute_descriptors(self) -> pd.DataFrame:
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
          [x[0] for x in Descriptors._descList])
        names = calculator.GetDescriptorNames()
        values = [calculator.CalcDescriptors(molecule) for molecule in self.molecules]
        df = pd.DataFrame(values, columns=names)
        df.insert(loc=0, column="SMILES", value=self.smiles.values)
        return df, names

data = pyrfume.load_data("keller_2016/molecules.csv", remote=True)
smiles = data.IsomericSMILES
type(smiles)
kit = RDKit_2D(smiles)
df, names = kit.compute_descriptors()
headers = list(names)
headers.insert(0, "SMILES")


file = open(r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\Molecule_Descriptors.csv',
            'w+', newline='')
output_path = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\Molecule_Descriptors.csv'

df.to_csv(output_path, columns = headers)



