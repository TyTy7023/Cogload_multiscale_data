import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier
from sklearn.svm import SVC

import sys
sys.path.append('/kaggle/working/cogload/model/')
from best_model import train_model 

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
    def selected_SBS(X_train, X_test, y_train, y_test, user_train, models):
        directory_name = '/kaggle/working/log/remove'
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        raw_train = X_train.copy(deep=True)
        raw_test = X_test.copy(deep=True)

        X_train_cp = X_train.copy(deep=True)
        X_test_cp = X_test.copy(deep=True)
        features = X_train.columns.tolist() 
        best_columns = []
        accs = []
        y_probs = []

        for model in models:
            result = []
            test_accuracies = []
            X_train_cp = raw_train.copy(deep=True)
            X_test_cp = raw_test.copy(deep=True)
            X_train = X_train_cp.copy(deep=True)
            X_test = X_test_cp.copy(deep=True)
            features = X_train.columns.tolist()

            REMAIN = []
            ACC = []
            Y_PROBS = []
            print(f"MODEL: {model} - SHAPE: {X_train.shape}")

            i = 0
            directory_name = f'/kaggle/working/log/remove/{model}/'
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            while(i<39):
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
                                    models = [model],
                                    index_name = i)
                        
                df = pd.read_csv(directory_name + f'{i}_results_model.csv')
                max_number = df['accuracy'].max()
                name_max_number = df.loc[df['accuracy'].idxmax(), ['features_remove', 'y_probs']]
            
                X_train = X_train.drop(columns=[name_max_number['features_remove']])
                X_test = X_test.drop(columns=[name_max_number['features_remove']])
                

                REMAIN.append(X_train.columns)
                ACC.append(max_number) 
                Y_PROBS.append(name_max_number['y_probs'])

                test_accuracies.append((X_train.columns, max_number, name_max_number['y_probs'])) 
                
                features = X_train.columns.tolist() 
                i += 1
            df = pd.DataFrame({'features': REMAIN, 'accuracy': ACC, 'y_probs': Y_PROBS})
            df.to_csv(f'/kaggle/working/log/remove/result/{model}.csv', index=False)
            
            feature_counts = [len(features) for features, _, _ in test_accuracies]
            accuracies = [accuracy for _, accuracy, _ in test_accuracies]
            
            plt.figure(figsize=(8, 5))
            plt.plot(feature_counts, accuracies, marker='o')
            plt.xlabel('Number of Features')
            plt.ylabel(f'Test Accuracy {model}')
            plt.title('Test Accuracy vs. Number of Features (Backward Selection)')
            plt.grid(True)
            plt.savefig(f'/kaggle/working/log/remove/result/{model}_acc.png')
            plt.show()
            
            best_column, max_accuracy, y_probs = max(test_accuracies, key=lambda x: x[1])
            best_columns.append(best_column)
            accs.append(max_accuracy)

        result.append({
            'Model': models,
            'Best Column': best_columns,
            'Shape': len(best_column),
            'Accuracy': accs,
            'Y Probs': y_probs
        })
        result = pd.DataFrame(result)
        result.to_csv('/kaggle/working/log/remove/result/result.csv', index=False)