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


def smiles_to_iupac(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text

name = smiles_to_iupac(smiles[0])
print(name)
"""
iupac_names = []
for smile in smiles:
    name = smiles_to_iupac(smile)
    iupac_names.append(name)
headers.insert(0, "Names")
df["Names"] = iupac_names
""" 


file = open('/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/Molecule_Descriptors1.csv',
            'w+', newline='')
output_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/Molecule_Descriptors.csv'

df.to_csv(output_path, columns = headers)



