from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

input_path_X = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx'
input_path_y = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx'
X = pd.read_excel(input_path_X, index_col=0)
y = pd.read_excel(input_path_y, index_col=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# numeric_features = df.select_dtypes(np.float)
# scaler = StandardScaler()
# df.loc[:, numeric_features.columns] = scaler.fit_transform(df.loc[:, numeric_features.columns])