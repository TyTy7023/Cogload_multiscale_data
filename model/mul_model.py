import os
import numpy as np
import pandas as pd
import itertools
import random

from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import sys
sys.path.append('/kaggle/working/cogload/')
from EDA import EDA
from MLP_model import MLP
from Tabnet_model import TabNet

def train_model(X_train, y_train, X_test, y_test, user_train, path, feature_remove = ['None'], n_splits=3 , debug = 0, models = ['MLP_Sklearn', 'MLP_Keras','TabNet'], index_name = 1):
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

            # Train model
            if model == 'MLP_Sklearn':
                estimator = MLP.MLP_Sklearn()
                estimator.fit(X_train_fold, y_train_fold, train_groups)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:, 1]

            elif model == 'MLP_Keras':
                estimator = MLP.MLP_Keras()
                estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, path)
                y_pred_prob = estimator.predict_proba(X_val_fold)

            elif model == 'TabNet':
                estimator = TabNet()
                estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:, 1]
            
            else:
                raise ValueError(f"Model {model} is not supported")
            y_pred_vals.append(y_pred_prob)

            y_val_pred = estimator.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            accuracy_all.append(accuracy)

            if accuracy > best_score:
                best_score = accuracy
                best_model = estimator

        # ROC tâp validation K-Fold
        EDA.draw_ROC(path_EDA + "/models/", y_vals, y_pred_vals, model)

        # Dự đoán trên tập kiểm tra
        print(f"Best parameters found: {best_model.best_params}\n")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        if model == 'MLP_Keras':
            y_pred_tests.append(y_pred_proba)
            y_pred_proba = [item[0] for item in y_pred_proba]
        else: 
            y_pred_tests.append(y_pred_proba[:, 1])
            y_pred_proba = y_pred_proba[:, 1]

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
        print(f"Accuracy all fold: {accuracy_all}\nMean: {accuracy_all.mean()} ---- Std: {accuracy_all.std()}")
        
        file_name = 'results_model.csv'
        os.makedirs(path, exist_ok=True)  # Đảm bảo thư mục tồn tại
        
        df_existing = pd.read_csv(os.path.join(path, file_name))
        df_to_append = pd.DataFrame({
            "model": [model],
            "accuracy": [f"{acc}"],
            "best_model": [str(best_model.best_params)],
            "feature_remove": [feature_remove],
            "Y Probs": [y_pred_proba]
        }, columns=df_existing.columns)
        
        df_to_append.to_csv(os.path.join(path, file_name), mode='a', header=False, index=False)
        print("\n===================================================================================================================================\n")

    EDA.draw_Bar(path_EDA, models, test_accuracy_models, 'Accuracy Test')
    EDA.draw_BoxPlot(path_EDA, list(itertools.chain.from_iterable([[i]*3 for i in models])), accuracies_all, 'Accuracy train')
    EDA.draw_Bar(path_EDA, models, f1_score_models, 'F1 Score')
    EDA.draw_ROC(path_EDA, y_test, y_pred_tests, models)

