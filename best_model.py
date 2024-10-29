import os
import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/kaggle/working/cogload/')
from EDA import EDA

def train_model(X_train, y_train, X_test, y_test, user_train, path, n_splits=3 , debug = 0):
        # K-Fold Cross-Validation với 6 folds
    kf = GroupKFold(n_splits=n_splits)

    best_model = None
    best_score = 0
    accuracy_all = []
    logloss_all = []
    y_vals = []
    y_pred_vals = []
    log_results = []


    # Lặp qua từng fold
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train, groups = user_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        y_vals.append(y_val_fold)
        
        id_user = np.array(user_train)
        # Kiểm tra nhóm trong fold
        train_groups = id_user[train_index]
        val_groups = id_user[val_index]
        
        print(f'User of train_fold({fold}) : {np.unique(train_groups)}')
        print(f'User of val_fold({fold}) :{np.unique(val_groups)}')    

        estimator = LDA(shrinkage = 0.5, solver = 'lsqr')     
        estimator.fit(X_train_fold, y_train_fold)
        
        y_val_pred = estimator.predict(X_val_fold)
        y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

        y_pred_vals.append(y_pred_prob)
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        accuracy_all.append(accuracy)

        logloss = log_loss(y_val_fold, y_pred_prob)
        logloss_all.append(logloss)

        if accuracy > best_score:
            best_score = accuracy
            best_model = estimator

    # ROC tâp validation K-Fold
    EDA.draw_ROC(path, y_vals, y_pred_vals, 'LDA')

    print(f"Best parameters found: {best_model.get_params()}\n")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    # Đánh giá mô hình trên tập kiểm tra
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred, average=None)
    logloss = log_loss(y_test, y_pred_proba)

    print("Report:" + class_report)
    print(f"ACCURACY: {acc}")
    print(f"LOGLOSS: {logloss}")


    # Xác định các lớp để hiển thị trong ma trận nhầm lẫn
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=unique_labels.tolist(), 
                yticklabels=unique_labels.tolist())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    accuracy_all = np.array(accuracy_all)
    logloss_all = np.array(logloss_all)
    print(f"Accucracy all fold: {accuracy_all}\nMean: {accuracy_all.mean()} ---- Std: {accuracy_all.std()}")
    print(f"LogLoss all fold: {logloss_all}\nMean: {logloss_all.mean()} ---- Std: {logloss_all.std()}")

    f1Score = ','.join(map(str, f1Score))
    log_results.append({
        "model": "LDA",
        "accuracy": f"{acc} +- {accuracy_all.std()}",
        "logloss": f"{logloss} +- {logloss_all.std()}",
        "best_model": best_model.get_params(),
        "f1_score": f1Score,
        "confusion_matrix": conf_matrix
    })

