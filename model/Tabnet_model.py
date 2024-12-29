import os
import pandas as pd
import numpy as np
import optuna
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
            "n_steps": 3,
            "gamma": 1.3
        }
        model = TabNetClassifier(**tabnet_params)
        return model

    def train_tabnet(self, trial, X_train, y_train, X_valid, y_valid):
        config = {
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
            'step_size': trial.suggest_categorical('step_size', [10, 20, 50]),
            'gamma': trial.suggest_uniform('gamma', 0.1, 0.9),
            'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
            'n_d': trial.suggest_categorical('n_d', [8, 16, 32, 64]),
            'n_a': trial.suggest_categorical('n_a', [8, 16, 32, 64]),
            'n_independent': trial.suggest_categorical('n_independent', [2, 3, 4]),
            'n_shared': trial.suggest_categorical('n_shared', [2, 3, 4]),
        }

        model = self.build(config)

        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        X_valid = X_valid.values if isinstance(X_valid, pd.DataFrame) else X_valid
        y_valid = y_valid.values if isinstance(y_valid, pd.DataFrame) else y_valid

        checkpoint_dir = "checkpoint"
        if os.path.exists(checkpoint_dir):
            model.load_model(checkpoint_dir)

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
            batch_size=256,
            virtual_batch_size=128,
            num_workers=1,
            drop_last=False,
            callbacks=[checkpoint_callback]
        )

        # Dự đoán và tính accuracy
        y_pred = model.predict(X_valid)
        val_accuracy = accuracy_score(y_valid, y_pred)
        return val_accuracy

    def optimize_tabnet(self, X_train, y_train, X_valid, y_valid):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.train_tabnet(trial, X_train, y_train, X_valid, y_valid), n_trials=10)
        best_trial = study.best_trial
        print(f"Best trial found at {best_trial.number}")
        print(f"Best hyperparameters: {best_trial.params}")
        return best_trial

    def fit(self, X_train, y_train, X_test, y_test):
        best_trial = self.optimize_tabnet(X_train, y_train, X_test, y_test)
        self.best_params = best_trial.params
        self.best_model = self.build(self.best_params)

        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test = y_test.values if isinstance(y_test, pd.DataFrame) else y_test

        self.best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=100, patience=10)

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
