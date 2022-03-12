import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from collections import Counter
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import *
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
pd.options.mode.chained_assignment = None

# input path for X and y
input_path_X = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\X.xlsx'
input_path_y = r'C:\Users\dell\Documents\ML final project\ML-for-NS-Ex1\y.xlsx'

# loading X and y
X = pd.read_excel(input_path_X, index_col=0)
y = pd.read_excel(input_path_y, index_col=0)

# casting y to a list
y = list(y[0])

# removing empty columns 
X = X.loc[:, X.any()]

# droping non-numeric columns
numeric_features = X.select_dtypes(include=np.number)
X = X.loc[:, numeric_features.columns]

# filling na's with columns means
X = X.apply(lambda l: l.fillna(l.mean()), axis=0)

# stratify splitting the data- test size of 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

scaler = StandardScaler()

# get scaling parameters with the train sample exclusively, using the Scaler.fit() function
scaler.fit(X_train)

# scale data using Scaler.transform()
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def find_optimal_model(X_train, X_test, y_train, y_test):
    """
    Running a grid search cv on the the data to find the optimal model.
    Models used: Random Forest, SVM, Logistic Regression and XGboost.
    Then testing the optimal model on the test data, and plotting confusion matrix.
    """

    # Initialze the estimators for random forest, SVM, logistic regression and XGboost
    clf1 = RandomForestClassifier(random_state=42)
    clf2 = SVC(probability=True, random_state=42, kernel="rbf")
    clf3 = LogisticRegression(random_state=42, max_iter=10000)
    clf4 = XGBClassifier(objective='binary:logistic', seed=42, use_label_encoder=False, eval_metric='logloss')
    
    # Initiaze the hyperparameters for each dictionary

    # parameters for random forest
    param1 = {}
    param1['classifier__n_estimators'] = [50, 100]
    param1['classifier__max_depth'] = [2, 5]
    param1['classifier__min_samples_split'] = [2, 70]
    param1['classifier__min_samples_leaf'] = [1, 50]
    param1['classifier'] = [clf1]

    # parameters for SVM
    param2 = {}
    param2['classifier__C'] = [2e-3, 2e7]
    param2['classifier__gamma'] = [2e-7, 2e3]
    param2['classifier'] = [clf2]

    # parameters for logistic regression
    param3 = {}
    param3['classifier__C'] = [100, 10, 1.0, 0.1, 0.01]
    param3['classifier__penalty'] = ['l1', 'l2']
    param3['classifier__solver'] = ['saga', 'liblinear']
    param3['classifier'] = [clf3]

    # parameters for XGboost
    param4 = {}
    param4['classifier__max_depth'] = [2,5]
    param4['classifier__min_child_weight'] = [1,6]
    param4['classifier__gamma'] = [0.1,10]
    param4['classifier__reg_alpha'] = [0.1,20]
    param4['classifier__reg_lambda'] = [0.001,100]
    param4['classifier__learning_rate'] = [0.01,1]
    param4['classifier__n_estimators'] = [10, 200]
    param4['classifier'] = [clf4]

    pipeline = Pipeline([('classifier', clf1)])
    params = [param1, param2, param3, param4]

    # Train the grid search model
    print("Searching for the best model... \n")
    gs = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='f1', verbose=1).fit(X_train, y_train)
    
    # printing best performing model and its corresponding hyperparameters
    print("The best model is:\n ", gs.best_params_)

    # printing the f1 score for the best model on training data
    print("The f1 score of the model on training data:\n ", gs.best_score_)

    # predicting using the best model with X_test
    y_pred = gs.predict(X_test)

    # printing the f1 score for the best model on test data
    print("The f1 score of the model on test data:\n ",f1_score(y_test, y_pred))

    # printing classification report
    report = classification_report(y_test, y_pred)
    print("classification report:\n ", report)

    # plotting confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap=plt.cm.Greens)
    plt.title(f"Confusion Matrix")
    plt.xlabel("Predicted label", fontsize = 15)
    plt.ylabel("True label", fontsize = 15)
    plt.show()

    # calculatin feature permutation importance 
    r = permutation_importance(gs, X_test, y_test, n_repeats=30, random_state=1)

    print("Feature permutation importance: \n")
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X.columns[i]:<8}  "
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

    

# running the function on the dataset
find_optimal_model(X_train, X_test, y_train, y_test)

#initialize Boruta
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 5
)

print("\n Running Boruta feature selection...\n")

boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 500,
   random_state=1
)

# fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X_train), np.array(y_train))

# print results
green_area = X.columns[boruta.support_].to_list()
blue_area = X.columns[boruta.support_weak_].to_list()
print('features in the green area:\n', green_area)
print('features in the blue area:\n', blue_area)

# call transform() on X_train to filter it down to selected features
X_train_boruta = boruta.transform(X_train)

# call transform() on X_test to filter it down to selected features
X_test_boruta = boruta.transform(X_test)

# running the function on the dataset after Boruta feature selection
find_optimal_model(X_train_boruta, X_test_boruta, y_train, y_test)

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

