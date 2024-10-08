import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessing :
    def __init__(self, temp_df, hr_df, gsr_df, rr_df, label_df, window_size = 1, normalize = "Standard"):
        if(window_size > len(temp_df.columns)):
            raise ValueError("Window size is greater than the number of samples. Please choose a smaller window size.")
        self.window_size = window_size
        if normalize != "Standard" and normalize != "MinMax":
            raise ValueError("Invalid normalization method. Please choose between 'Standard' and 'MinMax'")
        self.normalize = normalize
        self.temp_df = temp_df
        self.hr_df = hr_df
        self.gsr_df = gsr_df
        self.rr_df = rr_df
        self.label_df = label_df

    def SMA(self):
        self.temp_df = self.temp_df.rolling(self.window_size,axis=1).mean()
        self.hr_df = self.hr_df.rolling(self.window_size,axis=1).mean()
        self.gsr_df = self.gsr_df.rolling(self.window_size,axis=1).mean()
        self.rr_df = self.rr_df.rolling(self.window_size,axis=1).mean()

    def extract_features(self):
        temp_features = self.extract_stat_features(self.temp_df,'temp')
        hr_features = self.extract_stat_features(self.hr_df,'hr')
        gsr_features = self.extract_stat_features(self.gsr_df,'gsr')
        rr_features = self.extract_stat_features(self.rr_df,'rr')
        return pd.concat([temp_features,hr_features,gsr_features,rr_features],axis=1)

    @staticmethod
    def extract_stat_features(df,data_type=''):
        stat_features_names = ['mean','std','skew','kurtosis','diff','diff2','q25','q75','qdev','max-min']
        final_names =  [data_type + '_' + x for x in stat_features_names]
        features = pd.DataFrame(columns = stat_features_names) #create empty dataframe
        values = [df.mean(axis=1).values, #mean
                    df.std(axis=1).values,  #standard deviation
                    df.skew(axis=1).values, #skewness
                    df.kurtosis(axis=1).values, #kurtosis
                    df.diff(axis=1).mean(axis=1).values, #mean value of first derivative
                    df.diff(axis=1).diff(axis=1).mean(axis=1).values, #mean value of second derivative
                    df.quantile(0.25,axis=1).values, #25th quantile
                    df.quantile(0.75,axis=1).values,#75th quantile
                    df.quantile(0.75,axis=1).values-df.quantile(0.25,axis=1).values, #quartile deviation
                    df.max(axis=1).values-df.min(axis=1).values] #range
        values  = np.column_stack(values)
        return pd.DataFrame(values,columns = final_names) 
    
    def extract_features(self):
        temp_features = Preprocessing.extract_stat_features(self.temp_df,'temp')
        hr_features = Preprocessing.extract_stat_features(self.hr_df,'hr')
        gsr_features = Preprocessing.extract_stat_features(self.gsr_df,'gsr')
        rr_features = Preprocessing.extract_stat_features(self.rr_df,'rr')
        self.stat_feat_all = pd.concat([temp_features,hr_features,gsr_features,rr_features],axis=1)

    def splits_train_test(self):
        test_ids = ['3caqi','6frz4','bd47a','f1gjp','iz3x1']
        train_ids = ['1mpau', '2nxs5', '5gpsc', '7swyk', '8a1ep', 'b7mrd',
               'c24ur', 'dkhty', 'e4gay', 'ef5rq', 'f3j25', 'hpbxa',
               'ibvx8', 'iz2ps', 'rc1in', 'tn4vl', 'wjxci', 'yljm5']

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        self.user_train = []
        self.user_test = []

        for user in self.label_df.user_id.unique():
            if user in train_ids:
                user_features = self.stat_feat_all[self.label_df.user_id == user]
                X_train.append(user_features)
                y = self.label_df.loc[self.label_df.user_id == user, 'level'].values

                # Convert labels (rest,0,1,2) to binary (rest vs task)
                y[y == 'rest'] = -1
                y = y.astype(int) + 1
                y[y > 0] = 1
                y_train.extend(y)

                temp = self.label_df.loc[self.label_df.user_id==user,'user_id'].values #labels
                self.user_train.extend(temp)
                
            elif user in test_ids:
                user_features = self.stat_feat_all[self.label_df.user_id == user]
                X_test.append(user_features)
                y = self.label_df.loc[self.label_df.user_id == user, 'level'].values

                # Convert labels (rest,0,1,2) to binary (rest vs task)
                y[y == 'rest'] = -1
                y = y.astype(int) + 1
                y[y > 0] = 1
                y_test.extend(y)

                temp = self.label_df.loc[self.label_df.user_id==user,'user_id'].values #labels
                self.user_test.extend(temp)

        # Concatenate and convert to DataFrame/NumPy array
        self.X_train = pd.concat(X_train)
        self.y_train = np.array(y_train)
        self.X_test = pd.concat(X_test)
        self.y_test = np.array(y_test)
        
    def normalize_data(self, data):
        standard = StandardScaler()
        minmax = MinMaxScaler()

        if self.normalize == "Standard":
            return standard.fit_transform(data)
        elif self.normalize == "MinMax":
            return minmax.fit_transform(data)

    def get_data(self):
        if(self.window_size > 1):
            self.SMA()
        self.extract_features()
        self.splits_train_test()
        return self.normalize_data(self.X_train), self.y_train, self.normalize_data(self.X_test), self.y_test, self.user_train, self.user_test

