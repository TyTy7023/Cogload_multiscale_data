import os
from datetime import datetime
import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
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

def train_model(X_train, y_train, X_test, y_test, user_train, path, n_splits=3 , debug = False):
    np.random.seed(42)
    models = ['ESVM','E7GB', 'MLP', 'LR', 'LDA', 'KNN', 'RF', 'AB', 'GB', 'SVC', 'XGB']
    if debug:
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

            estimator, param_grid = useModel(model) 
            
            if model != 'E7GB':
                grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=GroupKFold(n_splits=3), scoring='accuracy', verbose=1)
                grid_search.fit(X_train_fold, y_train_fold, groups = train_groups)
                
                y_val_pred = grid_search.predict(X_val_fold)
                y_pred_prob = grid_search.predict_proba(X_val_fold)[:,1]

            else:
                estimator.fit(X_train_fold, y_train_fold)
                y_val_pred = estimator.predict(X_val_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)

            y_pred_vals.append(y_pred_prob)
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            accuracy_all.append(accuracy)

            logloss = log_loss(y_val_fold, y_pred_prob)
            logloss_all.append(logloss)

            if accuracy > best_score:
                best_score = accuracy
                if model != 'E7GB':
                    best_model = grid_search
                else:
                    best_model = estimator

        # ROC tâp validation K-Fold
        EDA.draw_ROC(path, y_vals, y_pred_vals, model)

        # Dự đoán trên tập kiểm tra
        print(f"Best parameters found: {best_model.best_params_}\n" )
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
        logloss = log_loss(y_test, y_pred_proba)

        test_accuracy_models.append(acc)
        accuracies_all.extend(accuracy_all)
        f1_score_models.append(f1Score.mean())

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
            "model": model,
            "accuracy": f"{acc} +- {accuracy_all.std()}",
            "logloss": f"{logloss} +- {logloss_all.std()}",
            "best_model": best_model.best_params_,
            "f1_score": f1Score,
            "confusion_matrix": conf_matrix
        })
        print("\n===================================================================================================================================\n")
    log_results = pd.DataFrame(log_results)
    file_name = f'results_model.csv'  # Tên file tự động
    log_results.to_csv(os.path.join(path, file_name), index=False)

    EDA.draw_Bar(path, models, test_accuracy_models, 'Accuracy Test')
    EDA.draw_BoxPlot(path, list(itertools.chain.from_iterable([[i]*3 for i in models])), accuracies_all, 'Accuracy train')
    EDA.draw_Bar(path, models, f1_score_models, 'F1 Score')
    EDA.draw_ROC(path, y_test, y_pred_tests, models)

def useModel(model):
# Train model
    if model == 'LR':
        estimator = LR(random_state=42)
        # Find best parmeter 
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],        
            'solver': ['liblinear']         
        }
    elif model == 'LDA':
        estimator = LDA()
        param_grid = {
            'solver': ['svd', 'lsqr', 'eigen'],  
            'shrinkage': [None, 'auto', 0.1, 0.5, 0.9] 
        }
    elif model == 'KNN':
        estimator = KNN()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],        # Số lượng láng giềng k
            'weights': ['uniform', 'distance'],     # Trọng số: uniform (các điểm đều quan trọng), distance (trọng số theo khoảng cách)
            'metric': ['euclidean', 'manhattan', 'minkowski']  # Loại khoảng cách: Euclidean, Manhattan hoặc Minkowski
        }                
    elif model == 'AB':
        estimator = AB(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],  # Số lượng bộ phân loại cơ sở (number of weak learners)
            'learning_rate': [0.01, 0.1, 1.0],  # Tốc độ học (learning rate)
        }    
    elif model == 'RF':
        estimator = RF(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],  # Number of trees in the forest
            'max_depth': [10, 20, 30],        # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required at each leaf node
        }     
    elif model == 'GB':
        estimator = GB(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],  # Number of boosting stages to be run
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
            'max_depth': [3, 5, 7],          # Maximum depth of the tree
            'min_samples_split': [2, 5, 10], # Minimum number of samples required to split a node
            'min_samples_leaf': [1, 2, 4]    # Minimum number of samples required at each leaf node
        }   
    elif model == 'SVC':
        estimator = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 100],                # Điều chỉnh độ phạt sai số
            'kernel': ['linear', 'rbf', 'poly'],    # Các loại kernel
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Tham số gamma cho RBF, poly kernels
            'degree': [2, 3, 4]                    # Bậc của polynomial kernel (nếu dùng 'poly')
        }
    elif model == 'XGB':
        estimator = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],  # Number of boosting stages to be run
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
            'max_depth': [3, 5, 7],          # Maximum depth of the tree
            'min_child_weight': [1, 2, 4],   # Minimum sum of instance weight (hessian) needed in a child
            'subsample': [0.8, 1.0],         # Subsample ratio of the training instance
            'colsample_bytree': [0.8, 1.0],  # Subsample ratio of columns when constructing each tree
            'gamma': [0, 0.1, 0.2]           # Minimum loss reduction required to make a further partition on a leaf node of the tree
        }
    elif model == 'E7GB':
        estimator = EnsembleModel_7GB()
    elif model == 'MLP':
        estimator = MLPClassifier(random_state=42)
        param_grid = {
            'hidden_layer_sizes': [(100,), (50, 50), (100, 100)],  # Số lượng nơ-ron ẩn trong mỗi layer
            'activation': ['relu', 'tanh', 'logistic'],              # Hàm kích hoạt
            'solver': ['adam', 'sgd'],                                # Thuật toán tối ưu
            'alpha': [0.0001, 0.001, 0.01],                           # L2 penalty (regularization term) parameter
            'learning_rate': ['constant', 'invscaling', 'adaptive']   # Phương pháp cập nhật learning rate
        }
    elif model == 'ESVM':
        base_estimator = SVC(probability=True, random_state=42)
        param_grid = {
            'base_estimator__C': [0.1, 1, 5, 10],
            'base_estimator__kernel': ['linear','rbf'],
            'base_estimator__gamma': ['scale', 0.1],
            'learning_rate': [0.1, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }

        # Tạo mô hình AdaBoost với SVM
        estimator = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

    return estimator, param_grid
