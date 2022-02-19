from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)

#Loading the dataset
input_path = "/Users/ofirsade/Desktop/UNI/Masters/Courses/SEMESTER I/למידה חישובית למדעי המוח/HW/Final Project/All_Descriptors.csv"
df = pd.read_csv(input_path)

df.drop(labels = "SMILES", axis = 1, inplace = True)
df.drop(labels = "Odor", axis = 1, inplace = True)
df.drop(labels = "Odor dilution", axis = 1, inplace = True)
df.drop(labels = "Subject # (DREAM challenge) ", axis = 1, inplace = True)
df.drop(labels = "Gender", axis = 1, inplace = True)
df1 = df.replace(np.nan, -1)

X = df1.drop(labels = "Familiarity (yes/no)", axis = 1)   #Feature Matrix
y = df1["Familiarity (yes/no)"]          #Target Variable
df.head()

#Adding constant column of ones, mandatory for sm.OLS model
##X_1 = sm.add_constant(X)

#Fitting sm.OLS model
##model = sm.OLS(y,X_1).fit()
##p = model.pvalues

#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p = []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.009):
        cols.remove(feature_with_p_max)
        print("Removed ", feature_with_p_max, "with pval ", pmax)
    else:
        break
selected_features_BE = cols
print(len(selected_features_BE),
      "\n", selected_features_BE)

