#to access files and folders
import os
#data analysis and manipulation library
import pandas as pd
#math operations for multi-dimensional arrays and matrices
import numpy as np
#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #use a specific color theme

import warnings
warnings.simplefilter("ignore")#ignore warnings during executiona

import sys
sys.path.append('/kaggle/working/cogload/')
from processing_data import Preprocessing
from selection_feature import Feature_Selection

#Using model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score,log_loss
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

#read the data
data_folder_path = '/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/'
#read the data
label_df = pd.read_excel(data_folder_path+'labels.xlsx',index_col=0)
temp_df= pd.read_excel(data_folder_path+'temp.xlsx',index_col=0)
hr_df= pd.read_excel(data_folder_path+'hr.xlsx',index_col=0)
gsr_df = pd.read_excel(data_folder_path+'gsr.xlsx',index_col=0)
rr_df= pd.read_excel(data_folder_path+'rr.xlsx',index_col=0)
print('Done reading data')

#check 30-second segments
print("Data shapes:")
print('Labels',label_df.shape)
print('Temperature',temp_df.shape)
print('Heartrate',hr_df.shape)
print('GSR',gsr_df.shape)
print('RR',rr_df.shape)

# Khởi tạo đối tượng Preprocessing
processing_data = Preprocessing(window_size=1, 
                                temp_df=temp_df, 
                                hr_df=hr_df, 
                                gsr_df=gsr_df, 
                                rr_df=rr_df, 
                                normalize="Standard")

# Lấy dữ liệu
data = processing_data.get_data()

test_ids = ['3caqi','6frz4','bd47a','f1gjp','iz3x1']

train_ids = ['1mpau', '2nxs5', '5gpsc', '7swyk', '8a1ep', 'b7mrd',
       'c24ur', 'dkhty', 'e4gay', 'ef5rq', 'f3j25', 'hpbxa',
       'ibvx8', 'iz2ps', 'rc1in', 'tn4vl', 'wjxci', 'yljm5']

X_train = []
y_train = []
X_test = []
y_test = []
user_train = []
user_test = []


for user in label_df.user_id.unique():
    if user in train_ids:
        user_features = data[label_df.user_id == user]
        X_train.append(user_features)
        y = label_df.loc[label_df.user_id == user, 'level'].values
        
        # Convert labels (rest,0,1,2) to binary (rest vs task)
        y[y == 'rest'] = -1
        y = y.astype(int) + 1
        y[y > 0] = 1
        y_train.extend(y)
        
        temp = label_df.loc[label_df.user_id==user,'user_id'].values #labels
        user_train.extend(temp)
    elif user in test_ids:
        user_features = data[label_df.user_id == user]
        X_test.append(user_features)
        y = label_df.loc[label_df.user_id == user, 'level'].values
        
        # Convert labels (rest,0,1,2) to binary (rest vs task)
        y[y == 'rest'] = -1
        y = y.astype(int) + 1
        y[y > 0] = 1
        y_test.extend(y)
        
        temp = label_df.loc[label_df.user_id==user,'user_id'].values #labels
        user_test.extend(temp)

# Concatenate and convert to DataFrame/NumPy array
X_train = pd.concat(X_train)
y_train = np.array(y_train)
X_test = pd.concat(X_test)
y_test = np.array(y_test)

print('Train data:', X_train.shape, y_train.shape)
print('Test data:', X_test.shape, y_test.shape)

X_train, X_test = Feature_Selection.selected_RFECV(X_train, X_test, y_train, user_train)
print(X_train)

