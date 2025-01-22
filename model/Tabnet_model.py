import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Cài đặt và import thư viện
install_and_import("pytorch_tabnet")

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

class TabNet:
    def __init__(self):
        self.best_model = None
        self.best_params = None

    def build(self, config):
        tabnet_params = {
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=config['lr']),
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            "scheduler_params": {"step_size": config['step_size'], "gamma": config['gamma']},
            "mask_type": config['mask_type'],
            "n_d": config['n_d'],
            "n_a": config['n_a'],
            "n_independent": config['n_independent'],
            "n_shared": config['n_shared'],
            "n_steps": config['n_steps'],
            "gamma": 1.3
        }
        model = TabNetClassifier(**tabnet_params)
        return model

    def train_tabnet(self, X_train, y_train, X_valid, y_valid):
        # Tham số cố định
        config = {
            'lr': 0.001,
            'step_size': 10,
            'gamma': 0.5,
            'mask_type': 'entmax',
            'n_d': 32,
            'n_a': 32,
            'n_independent': 2,
            'n_shared': 2,
            'n_steps': 5,
            'batch_size': 512,
            'virtual_batch_size': 128,
        }
        self.best_params = config
        model = self.build(config)
        self.best_params = config
        model = self.build(config)

        # Kiểm tra và chuẩn hóa dữ liệu
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        X_valid = X_valid.values if isinstance(X_valid, pd.DataFrame) else X_valid
        y_valid = y_valid.values if isinstance(y_valid, pd.DataFrame) else y_valid

        # Tạo thư mục checkpoint nếu chưa tồn tại
        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        class CheckpointCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 10 == 0:
                    model.save_model(checkpoint_dir)

        checkpoint_callback = CheckpointCallback()

        # Huấn luyện mô hình
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            max_epochs=100,
            patience=10,
            batch_size=config['batch_size'],
            virtual_batch_size=config['virtual_batch_size'],
            num_workers=1,
            drop_last=False,
            callbacks=[checkpoint_callback]
        )

        # Dự đoán và tính accuracy
        y_pred = model.predict(X_valid)
        val_accuracy = accuracy_score(y_valid, y_pred)
        return val_accuracy

    def fit(self, X_train, y_train, X_test, y_test):
        # Tham số cố định
        config = {
            'lr': 0.001,
            'step_size': 10,
            'gamma': 0.5,
            'mask_type': 'entmax',
            'n_d': 32,
            'n_a': 32,
            'n_independent': 2,
            'n_shared': 2,
            'n_steps': 5,
            'batch_size': 512,
            'virtual_batch_size': 128,
        }

        self.best_model = self.build(config)

        # Chuẩn hóa dữ liệu
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test = y_test.values if isinstance(y_test, pd.DataFrame) else y_test

        # Huấn luyện mô hình với dữ liệu train và test
        self.best_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=100,
            patience=10,
            batch_size=config['batch_size'],
            virtual_batch_size=config['virtual_batch_size']
        )

    def predict_proba(self, X_test):
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        if self.best_model is not None:
            probas = self.best_model.predict_proba(X_test)
            return probas
        else:
            raise ValueError("Model is not trained yet. Call fit() first.")

    def predict(self, X_test):
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        if self.best_model is not None:
            return np.round(self.best_model.predict(X_test))
        else:
            raise ValueError("Model is not trained yet. Call fit() first.")
