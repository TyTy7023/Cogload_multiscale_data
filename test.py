#to access files and folders
import os
from datetime import datetime
#data analysis and manipulation library
import pandas as pd
from argparse import ArgumentParser

import warnings
warnings.simplefilter("ignore")#ignore warnings during executiona

import sys
sys.path.append('/kaggle/working/cogload/')
from processing_data import Preprocessing
from selection_feature import Feature_Selection
from best_model import train_model

from sklearn.svm import SVC

#argument parser
parser = ArgumentParser()
parser.add_argument("--data_folder_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")
parser.add_argument("--window_size", default = 1, type = int, help = "Window size for feature extraction SMA")
parser.add_argument("--normalize", default = "Standard", type = str, help = "Normalization method, Standard or MinMax")
parser.add_argument("--model_selected_feature", default = "None", type = str, help = "None, RFECV, SFS")
parser.add_argument("--k_features", default = 11, type = int, help = "k of feature selected of SFS")
parser.add_argument("--forward", default = False, type = bool, help = "True to use backward, False to use forward")
parser.add_argument("--floating", default = True, type = bool, help = "True to use sfs with floating, False with no floating")
parser.add_argument("--features_to_remove", nargs='+', default = [], type = str, help="List of features to remove")
parser.add_argument("--debug", default = 0, type = int, help="debug mode 0: no debug, 1: debug")

args = parser.parse_args()

args_dict = vars(args)
log_args = pd.DataFrame([args_dict])

#read the data
label_df = pd.read_excel(args.data_folder_path+'labels.xlsx',index_col=0)
temp_df= pd.read_excel(args.data_folder_path+'temp.xlsx',index_col=0)
hr_df= pd.read_excel(args.data_folder_path+'hr.xlsx',index_col=0)
gsr_df = pd.read_excel(args.data_folder_path+'gsr.xlsx',index_col=0)
rr_df= pd.read_excel(args.data_folder_path+'rr.xlsx',index_col=0)
print("Data shapes:")
print('Labels',label_df.shape)
print('Temperature',temp_df.shape)
print('Heart Rate',hr_df.shape)
print('GSR',gsr_df.shape)
print('RR',rr_df.shape)

# Khởi tạo đối tượng Preprocessing
processing_data = Preprocessing(window_size = args.window_size, 
                                temp_df = temp_df, 
                                hr_df = hr_df, 
                                gsr_df = gsr_df, 
                                rr_df = rr_df,
                                label_df = label_df,
                                normalize=args.normalize)
X_train, y_train, X_test, y_test, user_train, user_test = processing_data.get_data(features_to_remove = args.features_to_remove)

if(args.model_selected_feature == "RFECV"):
    X_train, X_test = Feature_Selection.selected_RFECV(X_train = X_train,
                                                        X_test = X_test, 
                                                        y_train = y_train,
                                                        user_train = user_train
                                                        )
elif(args.model_selected_feature == "SFS"):
    X_train, X_test = Feature_Selection.selected_SFS(X_train = X_train,
                                                     X_test = X_test, 
                                                     y_train = y_train,
                                                     model = SVC(kernel='linear'),
                                                     k_features = args.k_features, 
                                                     forward = args.forward,
                                                     floating = args.floating)
print(X_train.shape,end="\n\n")

# Save log
features = ','.join(X_train.columns)
log_feature = pd.DataFrame({"feature": [features]})
log = pd.concat([log_args, log_feature], axis=1)

# is valid directory
directory_name = '/kaggle/working/log/'
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# create folder with timestamp
timestamp = datetime.now().strftime('%H_%M_%S')
sub_directory = directory_name + f"_{timestamp}"
os.makedirs(sub_directory)

# Save log_args and log_feature
file_name = f'args_and_feature_selected.csv'  # Tên file tự động
log.to_csv(os.path.join(sub_directory, file_name), index=False)

train_model(X_train, y_train, X_test, y_test, user_train, n_splits=args.GroupKFold, path = sub_directory, debug = args.debug)


