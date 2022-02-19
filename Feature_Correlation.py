import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)

input_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/All_Descriptors.csv'

df = pd.read_csv(input_path)

plt.figure(figsize=(12,10))

corr = df.corr(method = 'pearson')

sns.heatmap(corr)#, annot = True)

plt.show()




"""
col_names = ['Odor', 'Odor dilution', 'Gender',
             'Race ("unknown" indicates that the subject did not wish to specify)', 'Ethnicity',
             'Age', 'Detection (yes/no)', 'Familiarity (yes/no)', 'Intensity (0-100)',
             'Pleasantness (0-100)', 'Familiarity (1-100)']

df = pd.read_csv(input_path, usecols = col_names)

col_names2 = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',
              'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt',
              'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons',
              'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
              'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2',
              'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
              'BCUT2D_CHGHI', 'BCUT2D_CHGLO']
"""

