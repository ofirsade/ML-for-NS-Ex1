import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)

input_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/SMILES.csv'

input_path2 = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/Molecule_Descriptors.csv'


df = pd.read_csv(input_path)
df['Detection (yes/no)'] = (df['Detection (yes/no)'] == 'I smell something').astype(int)
df['Familiarity (yes/no)'] = (df['Familiarity (yes/no)'] == 'I know what the odor is').astype(int)
df['Gender'] = (df['Gender'] == 'M').astype(int)
df1 = df[['SMILES', 'Odor', 'Odor dilution', 'Subject # (DREAM challenge) ', 'Gender', 'Age', 'Detection (yes/no)', 'Familiarity (yes/no)',
          'Intensity (0-100)', 'Pleasantness (0-100)', 'Familiarity (0-100)']]
#print(df)

df2 = pd.read_csv(input_path2)

#print("DF1 Columns: ", list(df.columns),
#      "\nDF2 Columns: ", list(df2.columns))


result_df = pd.merge(df1, df2, how = "left", on = "SMILES")

file = open('/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/All_Descriptors.csv',
            'w+', newline='')
output_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/All_Descriptors.csv'

result_df.to_csv(output_path)

#plt.figure(figsize=(12,10))

#corr = result_df.corr(method = 'pearson')

#sns.heatmap(corr, annot = True)

#plt.show()




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

