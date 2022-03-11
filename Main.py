import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from collections import Counter
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
pd.options.mode.chained_assignment = None


input_path_X = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx'
input_path_y = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx'
X = pd.read_excel(input_path_X, index_col=0)
y = pd.read_excel(input_path_y, index_col=0)
y = list(y[0])

X = X.loc[:, X.any()]
numeric_features = X.select_dtypes(include=np.number)
X.loc[:, numeric_features.columns] = X.loc[:, numeric_features.columns].apply(lambda l: l.fillna(l.mean()),axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

scaler = StandardScaler()

# Get scaling parameters with the train sample exclusively, using the Scaler.fit() function
scaler.fit(X_train.loc[:, numeric_features.columns])

# Scale data using Scaler.transform()
X_train.loc[:, numeric_features.columns] = scaler.transform(X_train.loc[:, numeric_features.columns])
X_test.loc[:, numeric_features.columns] = scaler.transform(X_test.loc[:, numeric_features.columns])

###initialize Boruta
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 5
)
boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 500,
   random_state=1
)

### fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X_train.loc[:, numeric_features.columns]), np.array(y_train))
### print results
green_area = X_train.loc[:, numeric_features.columns].columns[boruta.support_].to_list()
blue_area = X_train.loc[:, numeric_features.columns].columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)

# ### Instanciate a PCA object for the sake of easy visualisation
# pca = PCA(n_components = 2)

# ### Fit and transform x to visualise inside a 2D feature space
# X_vis = pca.fit_transform(X_train)

# ### Apply the random over-sampling
# ada = ADASYN()
# X_resampled, y_resampled = ada.fit_resample(X_train, y_train) ### This is the oversampled data that we want to use.
# print(sorted(Counter(y_resampled).items()))
# X_res_vis = pca.transform(X_resampled)

# ### Two subplots
# f, (ax1, ax2) = plt.subplots(1, 2)

# c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="NON-SOUR",
#                  alpha=0.5)
# c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="SOUR",
#                  alpha=0.5)
# ax1.set_title('Original set')

# ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
#             label="NON-SOUR", alpha=.5)
# ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
#             label="SOUR", alpha=.5)
# ax2.set_title('ADASYN')

# ### Plotting
# for ax in (ax1, ax2):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     ax.spines['left'].set_position(('outward', 10))
#     ax.spines['bottom'].set_position(('outward', 10))
#     ax.set_xlim([-6, 8])
#     ax.set_ylim([-6, 6])

# plt.figlegend((c0, c1), ('NON-SOUR', 'SOUR'), loc='lower center',
#               ncol=2, labelspacing=0.)
# plt.tight_layout(pad=3)
# plt.show()
