
import warnings
from tensorflow import get_logger
import numpy as np
import subprocess
import sys
import random
import itertools


get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Setting the random state for TF")

from keras.models import Sequential
from keras.layers import Dense
from keras_tuner import RandomSearch
from sklearn.neural_network import MLPClassifier
from keras.backend import clear_session
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

class MLP:
    class MLP_Keras:
        def __init__(self):            
            def install_and_import(package):
                try:
                    __import__(package)
                except ImportError:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

            # Cài đặt và import thư viện
            install_and_import("scikeras")
            install_and_import("keras-tuner")

            import scikeras
            self.best_model = None
            self.best_params = None

        def build(self, hp=None):
            # Tạo mô hình MLP với các tham số cố định
            model = Sequential()

            # Giả sử self.shape là số lượng đặc trưng trong dữ liệu
            print(self.shape)
            clear_session()

            input_shape = (self.shape,)
            num_hidden_layers = 3  # Cố định số lượng lớp ẩn là 3
            units = 128  # Cố định số lượng units cho lớp đầu tiên
            units_1 = 64  # Cố định số lượng units cho lớp ẩn thứ 1
            units_2 = 128  # Cố định số lượng units cho lớp ẩn thứ 2
            units_3 = 128  # Cố định số lượng units cho lớp ẩn thứ 3
            units_4 = 32  # Cố định số lượng units cho lớp cuối cùng

            model.add(Dense(units=units, activation="relu", input_shape=input_shape)) # input_shape là số lượng đặc trưng trong dữ liệu
            model.add(Dense(units=units_1, activation="relu"))
            model.add(Dense(units=units_2, activation="relu"))
            model.add(Dense(units=units_3, activation="relu"))
            model.add(Dense(units=units_4, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))
            
            # Cố định optimizer là 'adam'
            model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
            return model

        def tuner(self, directory):
            return RandomSearch(
                hypermodel=self.build,
                objective='val_accuracy',
                max_trials=1,  # Giảm số lần thử nghiệm vì bạn không cần tìm kiếm tham số nữa
                seed=42
            )

        def fit(self, X_train, y_train, X_test, y_test, directory):
            self.shape = X_train.shape[1]  # Lấy số lượng đặc trưng từ X_train
            
            # Khởi tạo lại mô hình cho mỗi lần huấn luyện
            tuner = self.tuner(directory)
            tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

            # Lấy mô hình tốt nhất
            self.best_model = tuner.get_best_models(num_models=1)[0]
            best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
            self.best_params = best_trial.hyperparameters.values
            self.best_model.summary()

        def predict_proba(self, X_test):
            if self.best_model is not None:
                return self.best_model.predict(X_test)
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
            # Fixed parameters
            fixed_params = {
                'hidden_layer_sizes': [(32,)],  # Fixed layer sizes
                'activation': ['logistic'],      # Fixed activation function
                'solver': ['adam'],              # Fixed solver
                'alpha': [0.001],                # Fixed regularization term
                'learning_rate': ['constant']   # Fixed learning rate
            }   
            estimator = MLPClassifier(random_state=42)
            return estimator, fixed_params

        def fit(self, X_train, y_train, train_groups):
            estimator, param_distributions = self.set_Params()
            tuner = RandomizedSearchCV(estimator, param_distributions, n_iter=1, random_state=42, cv=GroupKFold(n_splits=3))  # 3-fold cross-validation
            tuner.fit(X_train, y_train, groups=train_groups)
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
