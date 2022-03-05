import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)

input_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/12868_2016_287_MOESM1_ESM.xlsx'

input_path2 = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/Molecule_Descriptors.xlsx'


df = pd.read_excel(input_path, header = 2)
df["CAN OR CAN'T SMELL"] = (df["CAN OR CAN'T SMELL"] == "I smell something").astype(int)
df["KNOW OR DON'T KNOW THE SMELL"] = (df["KNOW OR DON'T KNOW THE SMELL"] == 'I know what the odor is').astype(int)
df['Gender'] = (df['Gender'] == 'M').astype(int)

df2 = pd.read_excel(input_path2)

result_df = pd.merge(df, df2, how = "left", on = "SMILES")

result_df = pd.get_dummies(result_df, columns=['Odor dilution'])

file = open('/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/All_Descriptors.xlsx',
            'w+', newline='')
output_path = '/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/All_Descriptors.xlsx'

result_df.to_excel(output_path)


