import pandas as pd
import numpy as np
import pubchempy as pcp


#input_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/Molecule_Descriptors.csv'
input_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/Corrected_CID.csv'

#input_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/12868_2016_287_MOESM1_ESM.csv'


df = pd.read_csv(input_path)#, usecols = col_names)
lst = df['CID'].to_numpy()

smiles = []
i = 1
for c in lst:
    s = pcp.Compound.from_cid(str(c))
    smile = s.isomeric_smiles
    print(i, smile)
    i += 1
    smiles.append(smile)

df['Smiles'] = smiles


file = open('/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/SMILES.csv',
            'w+', newline='')
output_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/SMILES.csv'

df.to_csv(output_path)#, columns = headers)

