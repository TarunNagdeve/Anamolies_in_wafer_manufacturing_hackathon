import pandas as pd
import numpy as np
train=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Anamolies_in_Wafer_manufacturing\Participants_Data_WH18\Train.csv')
print(train.corr()['Class'])
train['new_1']=train['feature_2']*train['feature_1']
#train['new_2']=train['feature_3']+train['feature_4']
Labels=train['Class']
train=train.drop('Class',axis=1)
Features=train[:]

from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc=StandardScaler()
x_sc=sc.fit_transform(Features)
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
pca=PCA(616)
x_pca = pca.fit_transform(x_sc)



xtrain, xtest, ytrain, ytest = train_test_split(x_pca, Labels, test_size=0.3, random_state=101)
lmodel = LinearSVC(C=0.0000001,random_state=67, max_iter=150000,intercept_scaling=0.01)
lmodel.fit(x_pca,Labels)


test=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Anamolies_in_Wafer_manufacturing\Participants_Data_WH18\Test.csv')
test['new_1']=test['feature_2']*test['feature_1']

test_features=test[:]

test_sc=sc.transform(test_features)
test_pca=pca.transform(test_sc)
ans=lmodel.predict(test_pca)
df=pd.DataFrame(ans)
df.to_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Anamolies_in_Wafer_manufacturing\Participants_Data_WH18\Finally.csv')

















