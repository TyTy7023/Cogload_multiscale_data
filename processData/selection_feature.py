import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier
from sklearn.svm import SVC

sys.path.append('/kaggle/working/cogload/')
from model.best_model import train_model

class Feature_Selection:
    @staticmethod
    def selected_feature(selected_features, X_train, X_test):
        fs_train_orig = pd.DataFrame()
        fs_test_orig = pd.DataFrame()

        # Thay đổi cách nối cột và tên cột
        for i in selected_features:
            # Thêm cột vào fs_train và fs_test với tên cột tương ứng
            fs_train_orig[i] = X_train[i]
            fs_test_orig[i] = X_test[i]
        return fs_train_orig, fs_test_orig

    @staticmethod
    def selected_RFECV(X_train, X_test, y_train, user_train, estimator = XGBClassifier(n_jobs=-1)):
        gk = GroupKFold(n_splits=len(np.unique(user_train)))
        splits = gk.get_n_splits(X_train, y_train, user_train) #generate folds to evaluate the models using leave-one-subject-out
        fs_clf = RFECV(estimator=estimator, #which estimator to use
                    step = 1, #how many features to be removed at each iteration
                    cv = splits,#use pre-defined splits for evaluation (LOSO)
                    scoring='accuracy',
                    min_features_to_select=1,
                    n_jobs=-1)
        fs_clf.fit(X_train, y_train)#perform feature selection. Depending on the size of the data and the estimator, this may last for a while
        selected_features = X_train.columns[fs_clf.ranking_==1]
        print(f"Selected feature : {selected_features}")
        return Feature_Selection.selected_feature(selected_features, X_train, X_test)

    @staticmethod
    def selected_SFS(X_train, X_test, y_train, model = SVC(kernel='linear'), k_features = 11, forward = False, floating = True):
        original_columns = list(X_train.columns)
        sfs = SFS(model, 
                k_features=k_features, 
                forward = forward, 
                floating = floating, 
                scoring = 'accuracy',
                cv = 4,
                n_jobs = -1)
        sfs = sfs.fit(X_train, y_train)
        selected_feature_indices = sfs.k_feature_idx_

        # Dùng chỉ số để lấy tên cột đã được chọn từ danh sách
        selected_features = [original_columns[i] for i in selected_feature_indices]
        print(f"Selected feature : {selected_features}")
        return Feature_Selection.selected_feature(selected_features, X_train, X_test)

    @staticmethod
    def mine_SFS(X_train, X_test, y_train, y_test, user_train):
        directory_name = '/kaggle/working/log/remove'
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        test_accuracies = []
        features = X_train.columns.tolist() 
        i = 0
        name_max_number = []
        
        while(i<39):
            df = pd.DataFrame({
                'Features_removing': [],
                'Accuracy': [],
            })
            directory_name = f'/kaggle/working/log/remove/remove_{i}_feature.csv'
            i += 1
            df.to_csv(directory_name, index=False)
            
            for feature in features:
                X_train_cp = X_train.drop(columns=[f'{feature}'])
                X_test_cp = X_test.drop(columns=[f'{feature}'])
                
                train_model(X_train_cp, 
                            y_train, 
                            X_test_cp, 
                            y_test, 
                            user_train,
                            feature_remove=feature, 
                            n_splits=3, 
                            path = directory_name, 
                            debug = 0,
                            models = ['LDA', 'SVM', 'RF'])
                    
            df = pd.read_csv(directory_name)
            max_number = df['Accuracy'].max()
            name_max_number = df.loc[df['Accuracy'].idxmax(), 'Features_removing']
            
            X_train = X_train.drop(columns=[name_max_number])
            X_test = X_test.drop(columns=[name_max_number])
            print(f"REMAIN: {X_train.columns} - ACC: {max_number}")   
            test_accuracies.append((X_train.columns, max_number)) 
            
            features = X_train.columns.tolist() 

        feature_counts = [len(features) for features, _ in test_accuracies]
        accuracies = [accuracy for _, accuracy in test_accuracies]
        
        plt.figure(figsize=(8, 5))
        plt.plot(feature_counts, accuracies, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs. Number of Features (Backward Selection)')
        plt.grid(True)
        plt.show()
        
        best_column, max_accuracy = max(test_accuracies, key=lambda x: x[1])
        return best_column