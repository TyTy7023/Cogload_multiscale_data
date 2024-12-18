import os
from datetime import datetime
import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/kaggle/working/cogload/')
from EDA import EDA
from model_method_I import EnsembleModel_7GB

def train_model(X_train, y_train, X_test, y_test, user_train, path, n_splits=3 , debug = 0, models = ['MLB','Tabnet','E7GB','ESVM']):
    np.random.seed(42)
    path = os.path.dirname(path)
    path_EDA = path + '/EDA/'

    if debug == 1:
        models = models[:2]
    log_results = []
    test_accuracy_models = []
    accuracies_all = []
    f1_score_models = []
    y_pred_tests = []

    for model in models:
        print(f"\n\t\tMODEL: {model}")
        # K-Fold Cross-Validation với 6 folds
        kf = GroupKFold(n_splits=n_splits)

        best_model = None
        best_score = 0
        accuracy_all = []
        logloss_all = []
        y_vals = []
        y_pred_vals = []

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

            if model == 'E7GB':
                estimator = useModel(model)
            else:
                estimator, param_grid = useModel(model) 
            
            if model != 'E7GB' and model != 'ESVM':
                grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=GroupKFold(n_splits=3), scoring='accuracy', verbose=1)
                grid_search.fit(X_train_fold, y_train_fold, groups = train_groups)
                
                y_val_pred = grid_search.predict(X_val_fold)
                y_pred_prob = grid_search.predict_proba(X_val_fold)[:,1]

            else:
                estimator.fit(X_train_fold, y_train_fold)
                y_val_pred = estimator.predict(X_val_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)
                if model == 'ESVM':
                    y_pred_prob = y_pred_prob[:,1]

            y_pred_vals.append(y_pred_prob)
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            accuracy_all.append(accuracy)

            logloss = log_loss(y_val_fold, y_pred_prob)
            logloss_all.append(logloss)

            if accuracy > best_score:
                best_score = accuracy
                if model != 'E7GB' and model != 'ESVM':
                    best_model = grid_search
                else:
                    best_model = estimator

        # ROC tâp validation K-Fold
        EDA.draw_ROC(path_EDA + "/models/", y_vals, y_pred_vals, model)

        # Dự đoán trên tập kiểm tra
        if model == 'ESVM':
            print(f"Best parameters found: {useModel(model)[1]}\n")
        else:
            print(f"Best parameters found: {best_model.best_params_}\n")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        if model == 'E7GB':
            y_pred_tests.append(y_pred_proba)
        else: 
            y_pred_tests.append(y_pred_proba[:, 1])

        # Đánh giá mô hình trên tập kiểm tra
        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        f1Score = f1_score(y_test, y_pred, average=None)

        test_accuracy_models.append(acc)
        accuracies_all.extend(accuracy_all)
        f1_score_models.append(f1Score.mean())

        print("Report:" + class_report)
        print(f"ACCURACY: {acc}")

        accuracy_all = np.array(accuracy_all)
        logloss_all = np.array(logloss_all)
        print(f"Accucracy all fold: {accuracy_all}\nMean: {accuracy_all.mean()} ---- Std: {accuracy_all.std()}")
        print(f"LogLoss all fold: {logloss_all}\nMean: {logloss_all.mean()} ---- Std: {logloss_all.std()}")

        f1Score = ','.join(map(str, f1Score))
        log_results.append({
            "model": model,
            "accuracy": f"{acc} +- {accuracy_all.std()}",
            "best_model": best_model.best_params_ if model != "ESVM" else  useModel(model)[1],
            "f1_score": f1Score,
            "confusion_matrix": conf_matrix
        })
        print("\n===================================================================================================================================\n")
    log_results = pd.DataFrame(log_results)
    file_name = f'results_model.csv'  # Tên file tự động
    log_results.to_csv(os.path.join(path, file_name), index=False)

    EDA.draw_Bar(path_EDA, models, test_accuracy_models, 'Accuracy Test')
    EDA.draw_BoxPlot(path_EDA, list(itertools.chain.from_iterable([[i]*3 for i in models])), accuracies_all, 'Accuracy train')
    EDA.draw_Bar(path_EDA, models, f1_score_models, 'F1 Score')
    EDA.draw_ROC(path_EDA, y_test, y_pred_tests, models)

def useModel(model):
# Train model
    
    if model == 'E7GB':
        estimator = EnsembleModel_7GB()
    elif model == 'ESVM':
        base_estimator = SVC(probability=True, random_state=42)
        estimator = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)
        param_grid = estimator.get_params()
    elif model == 'MLP':
        estimator = MLPClassifier(random_state=42)
        param_grid = {
            'hidden_layer_sizes': [(100,), (50, 50), (100, 100)],  # Số lượng nơ-ron ẩn trong mỗi layer
            'activation': ['relu', 'tanh', 'logistic'],              # Hàm kích hoạt
            'solver': ['adam', 'sgd'],                                # Thuật toán tối ưu
            'alpha': [0.0001, 0.001, 0.01],                           # L2 penalty (regularization term) parameter
            'learning_rate': ['constant', 'invscaling', 'adaptive']   # Phương pháp cập nhật learning rate
        }

    if model == 'E7GB':
        return estimator
    else:
        return estimator, param_grid
        
