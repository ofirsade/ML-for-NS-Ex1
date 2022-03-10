import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from typing import List
import pyrfume
import pubchempy as pcp


input_path_data = r"C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\12868_2016_287_MOESM1_ESM.xlsx"

df2 = pd.read_excel(input_path_data, header=2)
df2 = df2[["CID", "Odor","Odor dilution","EDIBLE ","BAKERY ","SWEET ","FRUIT ", "FISH", "GARLIC ","SPICES ", "COLD", "SOUR","BURNT ", "ACID ", "WARM ", "MUSKY ", "SWEATY ","AMMONIA/URINOUS","DECAYED","WOOD ","GRASS ","FLOWER ", "CHEMICAL "]]
df2.columns = df2.columns.str.replace(' ', '')
df2.rename(columns={'Odordilution': 'Odor_dilution'}, inplace=True)
df2.fillna(0,inplace=True)
cid = df2['CID']
odor = df2['Odor']
for i in range(len(cid)):
    if "-" in str(cid[i]):
        s = pcp.get_compounds(odor[i], 'name')
        cid[i] = s[0].cid
df2["CID"] = cid
df2 = df2.groupby(['CID', 'Odor_dilution'], sort=False).mean()
c = ['1st Max','2nd Max','3rd Max']
df2 = df2.apply(lambda x: pd.Series(x.nlargest(3).index, index=c), axis=1).reset_index()

y = []
smiles=[]
cid = df2['CID']
for i in range(len(df2)):
    row = df2.iloc[i].values
    y.append(int("SOUR" in row))
    s = pcp.Compound.from_cid(str(cid[i]))
    smile = s.isomeric_smiles
    smiles.append(smile)

df2["SMILES"] = smiles
df2 = df2[["CID", "Odor", "Odor dilution", "SMILES"]]
df2 = pd.get_dummies(df2, columns=['Odor_dilution'])


class RDKit_2D:
    def __init__(self, smiles: pd.Series):
        self.smiles: pd.Series = smiles
        self.molecules: List[Chem.rdchem.Mol] = [Chem.MolFromSmiles(smile)
                                                 for smile in self.smiles]    

    def compute_descriptors(self) -> pd.DataFrame:
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
          [x[0] for x in Descriptors._descList]
        )
        names = calculator.GetDescriptorNames()
        values = [calculator.CalcDescriptors(molecule)              
                for molecule in self.molecules]
        df = pd.DataFrame(values, columns=names)
        df.insert(loc=0, column="SMILES", value=self.smiles.values)
        return df

data = pyrfume.load_data("keller_2016/molecules.csv", remote=True)
smiles = data.IsomericSMILES
kit = RDKit_2D(smiles)
df = kit.compute_descriptors()

X = pd.merge(df2, df, how = "left", on = "SMILES")

file_y = open(r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx',
            'w+', newline='')
output_path_y = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx'

y.to_excel(output_path_y)

file_X = open(r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx',
            'w+', newline='')
output_path_X = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx'

X.to_excel(output_path_X)
