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
                                label_df = label_df,
                                normalize="Standard")
X_train, y_train, X_test, y_test, user_train, user_test = processing_data.get_data()
print(X_train)

X_train, X_test = Feature_Selection.selected_SFS(X_train, X_test, y_train, model = SVC(kernel='linear'), k_features = 11, forward = False, floating = True)
print(X_train)
