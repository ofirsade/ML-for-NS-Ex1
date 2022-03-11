###Final Evaluation

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import warnings
from sklearn import svm
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("No Warning Shown\n")

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


arr = {}
main_list = ['Number of localizations','Polygon surronding flat cluster median size (nm)','Polygon median density',
             'Polygon surronding flat cluster mean size (nm)','Number of clusters']
#path1 = r"C:\Users\User\Documents\שנה ד\סמסטר ב\פרויקט גמר\Final Evaluation 2\HDBSCAN"
path1=r"C:\Users\User\Documents\שנה ד\סמסטר ב\פרויקט גמר\Final Evaluation 2\HDBSCAN"
Predict = {}
subjects_Label = [1,1,0,0,1,1,0,0,1,1,0,0]
##arr = {"HW2":df_test[0:31],"PL4":df_test[32:51],"C004":df_test[52:76],"C002":df_test[77:104],
##       "NB8":df_test[0:30],"DA1":df_test[31:61],"C011":df_test[62:89],"C008":df_test[90:111],
##       "PD7":df_test[0:26],"PD6":df_test[37:55],"C012":df_test[56:78],"C009":df_test[79:107]
##       }
############################################Split 1
#[HW2:0-31,pl4:32-51,C004:52-76,c002:77-104]

train_file= os.path.join(path1, "Split 1", "HDBSCAN_Train.csv")
test_file= os.path.join(path1, "Split 1", "HDBSCAN_Validation.csv")
df_train = pd.read_csv(train_file, usecols = main_list)

#read test features
df_test = pd.read_csv(test_file,usecols = main_list)
arr = {"HW2":df_test[0:31],"PL4":df_test[32:51],"C004":df_test[52:76],"C002":df_test[77:104]}

#read labels
df_label_train=pd.read_csv(train_file, usecols = ['Label'])
df_label_test=pd.read_csv(test_file, usecols = ['Label'])

model = svm.SVC(kernel='linear')
clf = svm.SVC(probability=True)

# fit on Train set
" SVM " ,clf.fit(df_train, df_label_train)

#predict on Test set.
predict_test = clf.predict(df_test)
HW2 = predict_test[0:31]
#print(HW2)
predict_HW2 = predict_test[0:31]
#print(predict_HW2)
#print(predict_HW2)
if (list(predict_HW2).count(0)/len(predict_HW2 )> 0.4):
    Predict["HW2"] = 0
elif(list(predict_HW2).count(1)/len(predict_HW2 )> 0.4):
    Predict["HW2"] = 1
else:
    Predict["HW2"] = 0#wrong pred

    
predict_PL4 = predict_test[32:51]
#print(predict_PL4)
if (list(predict_PL4).count(0)/len(predict_PL4 )> 0.4):
    Predict["PL4"] = 0
elif(list(predict_PL4).count(1)/len(predict_PL4 )> 0.4):
    Predict["PL4"] = 1
else:
    Predict["PL4"] = 0#wrong pred


    
predict_C004 = predict_test[52:76]
#print(predict_C004)
if (list(predict_C004 ).count(0)/len(predict_C004  )> 0.4):
    Predict["C004"] = 0
elif(list(predict_C004 ).count(1)/len(predict_C004  )> 0.4):
    Predict["C004"] = 1
else:
    Predict["C004"] = 1#wrong pred


    
predict_C002 =  predict_test[77:104]
#print(predict_C002)
if (list(predict_C002).count(0)/len(predict_C002 )> 0.4):
    Predict["C002"] = 0
elif(list(predict_C002).count(1)/len(predict_C002 )> 0.4):
    Predict["C002"] = 1
else:
    Predict["C002"] = 1#wrong pred

#####################################################################
############################################Split 2
#[Nb8:0-30,DA1:31-61,C011:62-89,c008:90-111]

train_file= os.path.join(path1, "Split 2", "HDBSCAN_Train.csv")
test_file= os.path.join(path1, "Split 2", "HDBSCAN_Validation.csv")
df_train = pd.read_csv(train_file, usecols = main_list)

#read test features
df_test = pd.read_csv(test_file,usecols = main_list)
arr = {"NB8":df_test[0:30],"DA1":df_test[31:61],"C011":df_test[62:89],"C008":df_test[90:111]}
#read labels
df_label_train=pd.read_csv(train_file, usecols = ['Label'])
df_label_test=pd.read_csv(test_file, usecols = ['Label'])

model = svm.SVC(kernel='linear')
clf = svm.SVC(probability=True)

# fit on Train set
" SVM " ,clf.fit(df_train, df_label_train)

#predict on Test set.
predict_test = clf.predict(df_test)
predict_NB8 = predict_test[0:30]
#print(predict_NB8)
if (list(predict_NB8).count(0)/len(predict_NB8 )> 0.4):
    Predict["NB8"] = 0
elif(list(predict_NB8).count(1)/len(predict_NB8 )> 0.4):
    Predict["NB8"] = 1
else:
    Predict["NB8"] = 0#wrong pred

    
predict_DA1 = predict_test[31:61]
#print(predict_DA1)
if (list(predict_DA1).count(0)/len(predict_DA1 )> 0.4):
    Predict["DA1"] = 0
elif(list(predict_DA1).count(1)/len(predict_DA1 )> 0.4):
    Predict["DA1"] = 1
else:
    Predict["DA1"] = 0#wrong pred

    
predict_C011 = predict_test[62:89]
#print(predict_C011)
if (list(predict_C011).count(0)/len(predict_C011 )> 0.4):
    Predict["C011"] = 0
elif(list(predict_C011).count(1)/len(predict_C011 )> 0.4):
    Predict["C011"] = 1
else:
    Predict["C011"] = 1#wrong pred

    
predict_C008 = predict_test[90:111]
#print(predict_C008)
if (list(predict_C008).count(0)/len(predict_C008 )> 0.4):
    Predict["C008"] = 0
elif(list(predict_C008).count(1)/len(predict_C008 )> 0.4):
    Predict["C008"] = 1
else:
    Predict["C008"] = 1#wrong pred
    
#####################################################################
############################################Split 3
#[PD7:0-26,PD6:37-55,C012:56-78,c009:79-107]

train_file= os.path.join(path1, "Split 3", "HDBSCAN_Train.csv")
test_file= os.path.join(path1, "Split 3", "HDBSCAN_Validation.csv")
df_train = pd.read_csv(train_file, usecols = main_list)

#read test features
df_test = pd.read_csv(test_file,usecols = main_list)
arr =  {"PD7":df_test[0:26],"PD6":df_test[37:55],"C012":df_test[56:78],"C009":df_test[79:107]}


#read labels
df_label_train=pd.read_csv(train_file, usecols = ['Label'])
df_label_test=pd.read_csv(test_file, usecols = ['Label'])

model = svm.SVC(kernel='linear')
clf = svm.SVC(probability=True)

# fit on Train set
" SVM " ,clf.fit(df_train, df_label_train)

#predict on Test set.
predict_test = clf.predict(df_test)
predict_PD7 = predict_test[0:26]
#print(predict_PD7)
if (list(predict_PD7).count(0)/len(predict_PD7 )> 0.4):
    Predict["PD7"] = 0
elif(list(predict_PD7).count(1)/len(predict_PD7 )> 0.4):
    Predict["PD7"] = 1
else:
    Predict["PD7"] = 0#wrong pred
    
predict_PD6 = predict_test[37:55]
#print(predict_PD6)
if (list(predict_PD6).count(0)/len(predict_PD6 )> 0.4):
    Predict["PD6"] = 0
elif(list(predict_PD6).count(1)/len(predict_PD6 )> 0.4):
    Predict["PD6"] = 1
else:
    Predict["PD6"] = 0#wrong pred
    
    
predict_C012 = predict_test[56:78]
#print(predict_C012)
if (list(predict_C012).count(0)/len(predict_C012 )> 0.4):
    Predict["C012"] = 0
elif(list(predict_C012).count(1)/len(predict_C012 )> 0.4):
    Predict["C012"] = 1
else:
    Predict["C012"] = 1#wrong pred
    
    
predict_C009 = predict_test[79:107]
#print(predict_C009)
if (list(predict_C009).count(0)/len(predict_C009 )> 0.4):
    Predict["C009"] = 0
elif(list(predict_C009).count(1)/len(predict_C009 )> 0.4):
    Predict["C009"] = 1
else:
    Predict["C009"] = 1#wrong pred

#####################################################################
#print(Predict)
print("Subjects labels are: ", subjects_Label)
print("The predicted values of the model are: ",Predict.values())
arr = list(Predict.values())
#arr_org = list(subjects_Label.values())
print("Model report : ", classification_report(subjects_Label, arr))













        
