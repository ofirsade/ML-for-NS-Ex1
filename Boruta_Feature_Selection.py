import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#Loading the dataset
input_path = "/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/All_Descriptors.xlsx"
df = pd.read_excel(input_path)
print("File has been loaded successfully!")

df.drop(labels = "SMILES", axis = 1, inplace = True)
df.drop(labels = "Odor", axis = 1, inplace = True)
df.drop(labels = "Subject # (DREAM challenge) ", axis = 1, inplace = True)
df1 = df.replace(np.nan, -1)
X = df1.drop(labels = "SOUR", axis = 1)
y = df1['SOUR']

###initialize Boruta
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 5
)
boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 100 # number of trials to perform
)

print("Boruta has been initialised successfully!")

### fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X), np.array(y))

print("Boruta has been fitted successfully!")

### print results
green_area = X.columns[boruta.support_].to_list()
blue_area = X.columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)
