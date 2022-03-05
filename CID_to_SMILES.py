import pandas as pd
import numpy as np
import pubchempy as pcp


input_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/12868_2016_287_MOESM1_ESM.xlsx'

df = pd.read_excel(input_path)
lst = df['CID'].to_numpy()

smiles = []
i = 1
short_lst = lst[:1000]
print(len(short_lst))
for c in short_lst:
    s = pcp.Compound.from_cid(str(c))
    smile = s.isomeric_smiles
    print(i, " ", str(c), " ", smile)
    i += 1
    smiles.append(smile)
#print(smiles)
smiles1 = smiles * 55
#print(smiles1)
df['Smiles'] = smiles1


file = open('/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/SMILES.xlsx',
            'w+', newline='')
output_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/SMILES.xlsx'

df.to_excel(output_path)

