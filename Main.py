import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None


input_path_X = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx'
input_path_y = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx'
X = pd.read_excel(input_path_X, index_col=0)
y = pd.read_excel(input_path_y, index_col=0)
y = list(y[0])

X = X.loc[:, X.any()]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

scaler = StandardScaler()
numeric_features = X_train.select_dtypes(include=np.number)
# Get scaling parameters with the train sample exclusively, using the Scaler.fit() function
scaler.fit(X_train.loc[:, numeric_features.columns])

# Scale data using Scaler.transform()
X_train.loc[:, numeric_features.columns] = scaler.transform(X_train.loc[:, numeric_features.columns])
X_test.loc[:, numeric_features.columns] = scaler.transform(X_test.loc[:, numeric_features.columns])

