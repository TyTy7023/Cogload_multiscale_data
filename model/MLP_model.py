
import warnings
from tensorflow import get_logger
import numpy as np
import subprocess
import sys
import random
import itertools
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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
            clear_session()

            self.fixed_params = {
                "units": 128,
                "units_1": 64,
                "units_2": 128,
                "units_3": 128,
                "units_4": 32,
                "activation": "relu",
                "optimizer": "adam",
                "loss": "binary_crossentropy",
            }
        
            model.add(Dense(units=self.fixed_params["units"], activation=self.fixed_params["activation"], input_shape=self.input_shape))
            model.add(Dense(units=self.fixed_params["units_1"], activation=self.fixed_params["activation"]))
            model.add(Dense(units=self.fixed_params["units_2"], activation=self.fixed_params["activation"]))
            model.add(Dense(units=self.fixed_params["units_3"], activation=self.fixed_params["activation"]))
            model.add(Dense(units=self.fixed_params["units_4"], activation=self.fixed_params["activation"]))
            model.add(Dense(1, activation="sigmoid"))
            
            # Sử dụng optimizer cố định
            model.compile(
                loss=self.fixed_params["loss"],
                optimizer=self.fixed_params["optimizer"],
                metrics=["accuracy"]
            )
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
            self.input_shape = X_train.shape[1]
            estimator, param_distributions = self.set_Params()
            tuner = RandomizedSearchCV(estimator, param_distributions, n_iter=1, random_state=42, cv=GroupKFold(n_splits=3))  # 3-fold cross-validation
            tuner.fit(X_train, y_train, groups=train_groups)
            self.best_model = tuner.best_estimator_
            self.best_params = self.fixed_params

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
