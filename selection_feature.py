import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

class Feature_Selection:
    @staticmethod
    def selected_feature(selected_features):
        fs_train_orig = pd.DataFrame()
        fs_test_orig = pd.DataFrame()
        columns = X_train.columns

        # Thay đổi cách nối cột và tên cột
        for i in selected_features:
            # Thêm cột vào fs_train và fs_test với tên cột tương ứng
            fs_train_orig[columns[i]] = X_train[columns[i]]
            fs_test_orig[columns[i]] = X_test[columns[i]]
        return fs_train_orig, fs_test_orig

    @staticmethod
    def selected_RFECV(X_train, y_train, user_train, estimator = XGBClassifier(n_jobs=-1)):
        gk = GroupKFold(n_splits=len(np.unique(user_train)))
        splits = gk.get_n_splits(X_train, y_train, user_train) #generate folds to evaluate the models using leave-one-subject-out
        print(splits)
        fs_clf = RFECV(estimator=estimator, #which estimator to use
                    step=1, #how many features to be removed at each iteration
                    cv=splits,#use pre-defined splits for evaluation (LOSO)
                    scoring='accuracy',
                    min_features_to_select=1,
                    n_jobs=-1)
        fs_clf.fit(X_train, y_train)#perform feature selection. Depending on the size of the data and the estimator, this may last for a while
        selected_features = X_train.columns[fs_clf.ranking_==1]
        print(f"Selected feature : {selected_features}")
        return selected_feature(selected_features)
