import numpy as np
import pandas as pd
import os

from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/kaggle/working/cogload/')
from EDA import EDA

sys.path.append('/kaggle/working/cogload/model/')

def train_model(X_train, y_train, X_test, y_test, user_train, path, feature_remove = ['None'], n_splits=3 , debug = 0, models = ['LDA', 'SVM', 'RF'], index_name = 1):
        # K-Fold Cross-Validation với 6 folds
    kf = GroupKFold(n_splits=n_splits)

    for model in models:
        log_results = []
        best_model = None
        best_score = 0
        y_vals = []
        y_pred_vals = []
        
        # Lặp qua từng fold
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train, groups = user_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            y_vals.append(y_val_fold)

            # Train model
            if model == 'LDA':
                estimator = LDA(shrinkage = 0.5, solver = 'lsqr')     
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

            elif model == 'SVM':
                estimator = SVC(kernel='rbf', C = 100, degree = 2, gamma = 0.001, probability=True, random_state=42)
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

                estimator.fit(X_train_fold, y_train_fold)
            elif model == 'RF':
                estimator = RF(n_estimators=300, max_depth=10, random_state=42, min_samples_leaf=2, min_samples_split=5)
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

            elif model == 'XGB':
                estimator = XGBClassifier(colsample_bytree= 1.0, gamma= 0, learning_rate= 0.2, max_depth= 5, min_child_weight= 4, n_estimators= 100, subsample= 0.8, n_jobs=-1)
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]
            
            elif model == 'MLP_Sklearn':
                from MLP_model import MLP
                estimator = MLP.MLP_Sklearn()
                estimator.fit(X_train_fold, y_train_fold, user_train)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:, 1]

            elif model == 'MLP_Keras':
                from MLP_model import MLP
                estimator = MLP.MLP_Keras()
                estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, path)
                y_pred_prob = estimator.predict_proba(X_val_fold)

            elif model == 'TabNet':
                from Tabnet_model import TabNet
                estimator = TabNet()
                estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:, 1]

            if model == []:
                return
            
            y_val_pred = estimator.predict(X_val_fold)
            y_pred_vals.append(y_pred_prob)

            accuracy = accuracy_score(y_val_fold, y_val_pred)

            if accuracy > best_score:
                best_score = accuracy
                best_model = estimator

        y_pred = best_model.predict(X_test)

        # Đánh giá mô hình trên tập kiểm tra
        acc = accuracy_score(y_test, y_pred)

        if not os.path.isfile(f'{path}{index_name}_results_model.csv'):
            print('Create new file')
        # Tạo một DataFrame trống (nếu file cần chứa dữ liệu dạng bảng)
            df = pd.DataFrame({
                "model": model,
                "accuracy": f"{acc}",
                "features_remove": [feature_remove]
            })
            df.to_csv(f'{path}{index_name}_results_model.csv', index=False)

        df_existing = pd.read_csv(f'{path}{index_name}_results_model.csv')
        print(df_existing)
        if df_existing.empty: 
            df_to_append = pd.DataFrame({
                "model": model,
                "accuracy": f"{acc}",
                "features_remove": [feature_remove]
            })
            df_to_append.to_csv(f'{path}{index_name}_results_model.csv', index=False)
        else:
            df_to_append = pd.DataFrame({
            "model": model,
            "accuracy": f"{acc}",
            "features_remove": [feature_remove]
            }, columns=df_existing.columns)
            print(df_to_append)
        # Ghi thêm vào file CSV
            df_to_append.to_csv(f'{path}{index_name}_results_model.csv', mode='a', header=False, index=False)



