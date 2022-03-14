import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)


### Load the dataset
input_path = input("Input path to file including file name: ")
df = pd.read_excel(input_path)
df = df[['MinEStateIndex', 'BCUT2D_MRLOW', 'BalabanJ', 'Chi4n', 'MolMR']]
df = df.loc[:, df.any()]
numeric_features = df.select_dtypes(include=np.number)
df.loc[:, numeric_features.columns] = df.loc[:, numeric_features.columns].apply(lambda l: l.fillna(l.mean()),axis=0)
scaler = StandardScaler()

# Get scaling parameters with the train sample exclusively, using the Scaler.fit() function
scaler.fit(df.loc[:, numeric_features.columns])


# Scale data using Scaler.transform()
df.loc[:, numeric_features.columns] = scaler.transform(df.loc[:, numeric_features.columns])


### Create the correlation matrix
plt.figure(figsize=(12,10))
corr = df.corr(method = 'pearson')
sns.heatmap(corr, annot = True)
plt.show()
