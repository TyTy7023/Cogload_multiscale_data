import scikeras
import warnings
from tensorflow import get_logger
import numpy as np
import random
import itertools

get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Setting the random state for TF")

from keras.models import Sequential
from keras.layers import Dense
from keras_tuner import RandomSearch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

class MLP:
    class MLP_Keras:
        def __init__(self):
            self.best_model = None
            self.best_params = None

        # Hàm tạo mô hình với tham số units và số lượng lớp ẩn
        def build(self, hp, input_shape):
            model = Sequential()
            num_hidden_layers = hp.Int('num_hidden_layers', min_value=2, max_value=5, step=1)
            model.add(Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), activation="relu", input_shape=(input_shape,)))
            for i in range(num_hidden_layers - 1):  # Vì lớp input đã được thêm rồi
                model.add(Dense(units=hp.Int(f'units_{i+1}', min_value=32, max_value=128, step=32), activation="relu"))
            model.add(Dense(1, activation="sigmoid"))
            
            model.compile(loss="binary_crossentropy", optimizer=hp.Choice('optimizer', values=['adam', 'sgd']), metrics=["accuracy"])
            return model
        
        def tuner(self, directory):
            return RandomSearch(
                hypermodel=self.build,
                objective='val_accuracy',
                max_trials=10, # Số lượng thử nghiệm
                tune_new_entries=True, # Cho phép thêm tham số mới
                allow_new_entries=True, # Cho phép thêm tham số mới
                max_retries_per_trial = 3, # Số lần thử lại tối đa cho mỗi thử nghiệm không thành công
                max_consecutive_failed_trials = 3, # Số lần thử nghiệm không thành công tối đa liên tiếp
                directory = directory,
                project_name='MLP_Keras_output',
                seed=42
            )

        def fit(self, X_train, y_train, X_test, y_test, directory):
            tuner = self.tuner(directory, len(X_train.columns))
            tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
            self.best_model = tuner.get_best_models(num_models=1)[0]
            best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
            self.best_params = best_trial.hyperparameters.values
            self.best_model.summary()

        def predict_proba(self, X_test):
            if self.best_model is not None:
                return self.best_model.predict_proba(X_test)
            else:
                raise ValueError("Model is not trained yet. Call fit() first.")
        
        def predict(self, X_test):
            if self.best_model is not None:
                return np.round(self.best_model.predict(X_test))
            else:
                raise ValueError("Model is not trained yet. Call fit() first.")
            
    class MLP_Sklearn:
        def __init__(self):
            self.best_model = None
            self.best_params = None

        def set_Params(self):
            layer_sizes = [32, 64, 128]  # Các số nơ-ron có thể có trong mỗi lớp
            hidden_layer_sizes = []

            random.seed(12)
            # Thử nghiệm với nhiều chiều và số lớp từ 1 đến 5 lớp
            for num_layers in range(1, 4):  # Số lớp từ 1 đến 5
                for combo in itertools.product(layer_sizes, repeat=num_layers):  # Tạo tất cả các kết hợp với số lớp và nơ-ron khác nhau
                    hidden_layer_sizes.append(combo)

            estimator = MLPClassifier(random_state=42)
            param = {
                'hidden_layer_sizes': hidden_layer_sizes,  # Số lượng nơ-ron ẩn trong mỗi layer
                'activation': ['relu', 'tanh', 'logistic'],              # Hàm kích hoạt
                'solver': ['adam', 'sgd'],                                # Thuật toán tối ưu
                'alpha': [0.0001, 0.001, 0.01],                           # L2 penalty (regularization term) parameter
                'learning_rate': ['constant', 'invscaling', 'adaptive']   # Phương pháp cập nhật learning rate
            }   
            return estimator, param

        def fit(self, X_train, y_train, train_groups):
            estimator, param_distributions = self.set_Params()
            tuner = RandomizedSearchCV(estimator, param_distributions, n_iter=10, random_state=42, cv=GroupKFold(n_splits=3))  # 5-fold cross-validation
            tuner.fit(X_train, y_train, groups = train_groups)
            self.best_model = tuner.best_estimator_
            self.best_params = tuner.best_params_

        def predict_proba(self, X_test):
            if self.best_model is not None:
                return self.best_model.predict_proba(X_test)
            else:
                raise ValueError("Model is not trained yet. Call fit() first.")
        
        def predict(self, X_test):
            if self.best_model is not None:
                return self.best_model.predict(X_test)
            else:
                raise ValueError("Model is not trained yet. Call fit() first.")

    