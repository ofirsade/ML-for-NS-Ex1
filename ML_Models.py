import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import plot_tree
from sklearn.tree import export_graphviz
import graphviz
import os
import warnings
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
#import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report


warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("No Warning Shown\n")

""" calculates F1 """
def calc_f1(precision, recall):
    f1=2*((precision*recall)/(precision+recall))
    return f1

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

matrix = {}
#main_list = ['Number of localizations','Polygon surronding flat cluster median size (nm)']
#main_list= ['Number of localizations','Polygon surronding flat cluster median size (nm)',
#            'Polygon median density','Polygon surronding flat cluster mean size (nm)','Number of clusters']


Splits = ["Split 1", "Split 2", "Split 3"]

path1 = r"C:\Users\User\Documents\שנה ד\סמסטר ב\פרויקט גמר\Final Evaluation 2\HDBSCAN"

path1 = input("Input path and file name: ")

for split in Splits:
    print(split)
    train_file= os.path.join(path1, split, "Train.csv")
    test_file= os.path.join(path1, split, "Validation.csv")
    lis = {}
    for c in list(itertools.combinations('01234',5)):# choose the number of features
        results = {}
        L2 =[]
        for i in range(5):## Change it to be = number of features
            L2.append(main_list[int(c[i])])
        print("features: ", L2)
        ### Read train features
        df_train = pd.read_csv(train_file, usecols = L2)

        ### Read test features
        df_test = pd.read_csv(test_file,usecols = L2)
        

        ### Read labels
        df_label_train=pd.read_csv(train_file, usecols = ['Label'])
        df_label_test=pd.read_csv(test_file, usecols = ['Label'])


        ################################### XGBoost tree #################################
        print("############################ XGBoost tree ###############################")
        ### Initialise the model
        model = xgb.XGBClassifier()

        ### Build the tree for Train set.

        ### Orint the hyperparametes of the model
        model.fit(df_train, df_label_train)

        #plt =
        plot_tree(model)
        ### The model's tree
        #plt.show()

        predict_test = model.predict(df_test)
        print("XG_Boost prediction on test set : \n", predict_test)

        ### Test set error Analysis
        #cm = confusion_matrix(df_label_test, predict_test)
        #plt.figure()
        #sns.heatmap(cm, annot=True)
        #print("XG_Boost confusion matrix of test set : \n")
        #plt.show()
        #error anly
        print("XGBOOST report: ", classification_report(df_label_test, predict_test))





        
        precision = precision_score(df_label_test, predict_test)
        recall = recall_score(df_label_test, predict_test)
        results['Precision of XG_Boost on test set'] = format(precision_score(df_label_test, predict_test))
        results['Recall of XG_Boost on test set'] = format(recall_score(df_label_test, predict_test))
        results['Accuracy of XG_Boost on test set'] = format(accuracy_score(df_label_test, predict_test))
        results['F1 scoreof XG_Boost on test set'] = calc_f1(precision, recall)




        ### Predict on Train set to test overfitting.
        predict_train = model.predict(df_train)
        print("XG_Boost prediction on Train set : \n",predict_train)

        ### Train set error Analysis
        #predict_train = model.predict(df_train)
        cm = confusion_matrix(df_label_train, predict_train)
        #plt.figure()
        #sns.heatmap(cm, annot=True)
        #print("XG_Boost confusion matrix of train set : \n")
        #plt.show()


        precision = precision_score(df_label_train, predict_train)
        recall = recall_score(df_label_train, predict_train)
    ##    results['Precision of XG_Boost on train set'] = format(precision_score(df_label_train, predict_train))
    ##    results['Recall of XG_Boost on train set']= format(recall_score(df_label_train, predict_train))
    ##    results['Accuracy of XG_Boost on train set'] = format(accuracy_score(df_label_train, predict_train))
    ##    results['F1 scoreof XG_Boost on train set'] = calc_f1(precision, recall)



        ##################################### SVM ########################################
        print("############################ SVM MODEL ###############################")
        model = svm.SVC(kernel='linear')
        clf = svm.SVC(probability=True)

        ### Fit: Train set
        " SVM " ,clf.fit(df_train, df_label_train)

        ### Predict: Test set.
        predict_test = clf.predict(df_test)
        print("SVM prediction on test set : \n", predict_test)

        ### Test set error Analysis
        cm = confusion_matrix(df_label_test, predict_test)
        #plt.figure()
        sns.heatmap(cm, annot=True)
        #print("SVM confusion matrix of test set : \n")
        #plt.show()

        print("SVM report: ", classification_report(df_label_test, predict_test))
        precision = precision_score(df_label_test, predict_test)
        recall = recall_score(df_label_test, predict_test)
        results['Precision of SVM on test set'] = format(precision_score(df_label_test, predict_test))
        results['Recall of SVM on test set']= format(recall_score(df_label_test, predict_test))
        results['Accuracy of SVM on test set'] = format(accuracy_score(df_label_test, predict_test))
        results['F1 scoreof SVM on test set'] = calc_f1(precision, recall)


        ### Predict: Train set to test overfitting.
        predict_train= clf.predict(df_train)
        print("SVM prediction on Train set : \n",predict_train)

        ### Train set error Analysis
        predict_train = clf.predict(df_train)
        cm = confusion_matrix(df_label_train, predict_train)
        #plt.figure()
        sns.heatmap(cm, annot=True)
        #print("SVM confusion matrix of train set : \n")
        #plt.show()

        print("SVM report on train set: ", classification_report(df_label_train, predict_train))
        precision = precision_score(df_label_train, predict_train)
        recall = recall_score(df_label_train, predict_train)
        precision = precision_score(df_label_train, predict_train)
        recall = recall_score(df_label_train, predict_train)
        results['Precision of SVM on train set'] = format(precision_score(df_label_train, predict_train))
        results['Recall of SVM on train set']= format(recall_score(df_label_train, predict_train))
        results['Accuracy of SVM on train set'] = format(accuracy_score(df_label_train, predict_train))
        results['F1 score of SVM on train set'] = calc_f1(precision, recall)

        lis[c] = results
    print("lis = ", lis.keys())## this is important to know whar are the features at each line(their indexis)
    matrix[split] = list(lis.values())
    
table = {"split 1": matrix["Split 1"],"Split 2": matrix["Split 2"],"split 3": matrix["Split 3"]}
p1 = input("Please input output path and file name: ")
p2= os.path.join(p1, split + ".xlsx")
DF2 = pd.DataFrame(table)
DF2.to_excel(r"C:\Users\User\Documents\שנה ד\סמסטר ב\פרויקט גמר\CODE\matrix.xlsx")


        
