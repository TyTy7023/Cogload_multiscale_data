import numpy as np
import pandas as pd
import os

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/kaggle/working/cogload/')
from EDA import EDA

def train_model(X_train, y_train, X_test, y_test, user_train, path, feature_remove = ['None'], n_splits=3 , debug = 0, models = ['LDA', 'SVM', 'RF']):
        # K-Fold Cross-Validation với 6 folds
    kf = GroupKFold(n_splits=n_splits)

    for model in models:
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
            if model == 'SVM':
                estimator = SVC(kernel='rbf', C = 100, degree = 2, gamma = 0.001, probability=True, random_state=42)
            if model == 'RF':
                estimator = RF(n_estimators=300, max_depth=10, random_state=42, min_samples_leaf=2, min_samples_split=5)
            if model == []:
                return
            estimator.fit(X_train_fold, y_train_fold)
            
            y_val_pred = estimator.predict(X_val_fold)
            y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]
            y_pred_vals.append(y_pred_prob)

            accuracy = accuracy_score(y_val_fold, y_val_pred)

            if accuracy > best_score:
                best_score = accuracy
                best_model = estimator

        y_pred = best_model.predict(X_test)

        # Đánh giá mô hình trên tập kiểm tra
        acc = accuracy_score(y_test, y_pred)
        
        # Đọc file CSV gốc để lấy danh sách cột
        df_existing = pd.read_csv(path)
        if df_existing.empty:   
            df_to_append = pd.DataFrame({
                'Model': [model],
                'Features_removing': [feature_remove],
                'Accuracy': [acc]
            })
            df_to_append.to_csv(path, index=False)
        else:
            df_to_append = pd.DataFrame({
            'Model': [model],
            'Features_removing': [feature_remove],
            'Accuracy': [acc], 

            }, columns=df_existing.columns)
        # Ghi thêm vào file CSV
            df_to_append.to_csv(path, mode='a', header=False, index=False)



