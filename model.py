
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X_train, y_train, X_test, y_test, user_train, n_splits=6, log_results = []):
    np.random.seed(42)
    models = ['LogisticRegression', 'LinearDiscriminantAnalysis', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'SVC']
    for model in models:
        print(f"\n\t\tMODEL: {model}")
        # K-Fold Cross-Validation với 6 folds
        kf = GroupKFold(n_splits=n_splits)

        best_model = None
        best_score = 0
        fold_results = []
        accuracy_all = []
        logloss_all = []

        # Lặp qua từng fold
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train, groups = user_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            id_user = np.array(user_train)
            # Kiểm tra nhóm trong fold
            train_groups = id_user[train_index]
            val_groups = id_user[val_index]
            
            print(f'User of train_fold({fold}) : {np.unique(train_groups)}')
            print(f'User of val_fold({fold}) :{np.unique(val_groups)}')    

            # Train model
            if model == 'LogisticRegression':
                estimator = LogisticRegression(random_state = 42)
                # Find best parmeter 
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', None],        
                    'solver': ['saga']         
                }
            elif model == 'LinearDiscriminantAnalysis':
                estimator = LinearDiscriminantAnalysis()
                param_grid = {
                    'solver': ['svd', 'lsqr', 'eigen'],  
                    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9] 
                }
            elif model == 'KNeighborsClassifier':
                estimator = KNeighborsClassifier()
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9, 11],        # Số lượng láng giềng k
                    'weights': ['uniform', 'distance'],     # Trọng số: uniform (các điểm đều quan trọng), distance (trọng số theo khoảng cách)
                    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Loại khoảng cách: Euclidean, Manhattan hoặc Minkowski
                    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], # Thuật toán: auto, ball_tree, kd_tree, brute
                    'leaf_size': list(range(10, 101, 10))
                }                
            elif model == 'DecisionTreeClassifier':
                estimator = DecisionTreeClassifier(random_state=42)
                param_grid = {
                    'splitter' : ["best", "random"], # Phương pháp chia: best (chọn cách chia tốt nhất), random (chọn cách chia ngẫu nhiên)
                    'criterion': ['gini', 'entropy'],   # Chỉ số phân chia: Gini hoặc Entropy
                    'max_depth': [None, 10, 20, 30, 40, 50],  # Chiều sâu tối đa của cây
                    'min_samples_split': [2, 10, 20],  # Số mẫu tối thiểu để chia nút
                    'min_samples_leaf': [1, 5, 10],    # Số mẫu tối thiểu tại mỗi nút lá
                    'max_features': [None, 'auto', 'sqrt', 'log2']  # Số lượng đặc trưng được xem xét tại mỗi nút phân chia
                }  
            elif model == 'GaussianNB':
                estimator = GaussianNB()
                param_grid = {
                            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Điều chỉnh biến nhỏ để tăng độ ổn định tính toán
                            }  
            elif model == 'AdaBoostClassifier':
                estimator = AdaBoostClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],  # Số lượng bộ phân loại cơ sở (number of weak learners)
                    'learning_rate': [0.01, 0.1, 1.0],  # Tốc độ học (learning rate)
                    'estimator': [ DecisionTreeClassifier(max_depth=d) for d in range(1, 21) ] + [SVC(probability=True)],  # Cây quyết định hoặc máy vector hỗ trợ
                    'algorithm': ['SAMME', 'SAMME.R']  # Thuật toán tối ưu hóa
                }    
            elif model == 'RandomForestClassifier':
                estimator = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [100, 200, 300],  # Số lượng bộ phân loại cơ sở (cây quyết định) trong rừng
                    'max_depth': [10, 20, 30],         # Độ sâu tối đa của cây quyết định
                    'min_samples_split': [2, 5, 10],   # Số mẫu tối thiểu để chia một nút của cây
                    'min_samples_leaf': [1, 2, 4],     # Số mẫu tối thiểu ở mỗi lá của cây quyết định
                    'max_features': ['auto', 'sqrt', 'log2'],  # Số lượng đặc trưng được chọn để chia tại mỗi lần phân tách
                    'bootstrap': [True, False],        # Phương pháp chọn mẫu: True (chọn mẫu với phép lặp), False (không phép lặp)
                    'criterion': ['gini', 'entropy', "log_loss"],  # Hàm dùng để đo chất lượng của lần chia (Gini, Entropy hoặc Log Loss)
                    'class_weight': ['balanced', 'balanced_subsample', None],  # Trọng số lớp (được tính tự động nếu chọn 'balanced')
                    'ccp_alpha': [0.0, 0.1, 0.2]       # Tham số cắt tỉa cây để giảm độ phức tạp của mô hình (Minimal Cost-Complexity Pruning)
                }   
            elif model == 'GradientBoostingClassifier':
                estimator = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [100, 200, 300],  # Số lượng giai đoạn boosting cần chạy
                    'learning_rate': [0.01, 0.1, 0.2],  # Tốc độ học (shrinkage bước đi)
                    'max_depth': [3, 5, 7],             # Độ sâu tối đa của cây quyết định
                    'min_samples_split': [2, 5, 10],    # Số mẫu tối thiểu cần thiết để chia một nút của cây
                    'min_samples_leaf': [1, 2, 4],      # Số mẫu tối thiểu cần thiết tại mỗi lá của cây quyết định
                    'max_features': ['auto', 'sqrt', 'log2'],  # Số lượng đặc trưng được xem xét khi tìm kiếm lần chia tốt nhất
                    'subsample': [0.8, 0.9, 1.0],       # Tỷ lệ mẫu được sử dụng để huấn luyện các bộ phân loại cơ sở
                    'loss': ['deviance', 'exponential']  # Hàm mất mát (loss function) cần tối ưu
                }   
            elif model == 'SVC':
                estimator = SVC(probability=True, random_state=42)
                param_grid = {
                    'C': [0.1, 1, 10, 100],                # Điều chỉnh độ phạt sai số
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],    # Các loại kernel
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Tham số gamma cho RBF, poly kernels
                    'degree': [2, 3, 4],                    # Bậc của polynomial kernel (nếu dùng 'poly')
                    'coef0': [0.0, 0.1, 0.5, 1.0],          # Tham số độc lập trong các kernel sigmoid và polynomial
                    'shrinking': [True, False],              # Sử dụng hay không sử dụng thuật toán thu hẹp biên
                    'class_weight': ['balanced', None],      # Trọng số lớp (được tính tự động nếu chọn 'balanced')
                    'decision_function_shape': ['ovo', 'ovr']  # Chọn giữa 'ovo' (one-vs-one) và 'ovr' (one-vs-rest)
                }

            grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=GroupKFold(n_splits=3), scoring='accuracy', verbose=1)
            grid_search.fit(X_train_fold, y_train_fold, groups = train_groups)
            
            y_val_pred = grid_search.predict(X_val_fold)
            y_pred_prob = grid_search.predict_proba(X_val_fold)

            accuracy = accuracy_score(y_val_fold, y_val_pred)
            conf_matrix = confusion_matrix(y_val_fold, y_val_pred)
            class_report = classification_report(y_val_fold, y_val_pred)
            logloss = log_loss(y_val_fold, y_pred_prob)

            fold_results.append({
                "fold": fold,
                "model": estimator,
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix,
                "classification_report": class_report
                })
            accuracy_all.append(accuracy)
            logloss_all.append(logloss)

            if accuracy > best_score:
                best_score = accuracy
                best_model = grid_search

        # Dự đoán trên tập kiểm tra
        print(f"Best parameters found: {best_model.best_params_}\n" )
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        # Đánh giá mô hình trên tập kiểm tra
        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
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

        log_results.append({
            "model": model,
            "accuracy": acc,
            "std_accuracy": accuracy_all.std(),
            "logloss": logloss,
            "std_logloss": logloss_all.std(),
            "best_model": best_model.best_params_,
            "report": class_report,
            "confusion_matrix": conf_matrix
        })
        print("\n===================================================================================================================================\n")
