import pandas as pd
import numpy as np

class Preprocessing :
    def __init__(self, window_size, temp_df, hr_df, gsr_df, rr_df, normalize = "Standard"):
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
        temp_features = self.extract_stat_features(self.temp_df,'temp')
        hr_features = self.extract_stat_features(self.hr_df,'hr')
        gsr_features = self.extract_stat_features(self.gsr_df,'gsr')
        rr_features = self.extract_stat_features(self.rr_df,'rr')
        self.stat_feat_all = pd.concat([temp_features,hr_features,gsr_features,rr_features],axis=1)

    def normalize_data(self):
        if self.normalize == "Standard":
            self.stat_feat_all = (self.stat_feat_all - self.stat_feat_all.mean())/self.stat_feat_all.std()
        elif self.normalize == "MinMax":
            self.stat_feat_all = (self.stat_feat_all - self.stat_feat_all.min())/(self.stat_feat_all.max() - self.stat_feat_all.min())
        return self.stat_feat_all

    def get_data(self):
        if(self.window_size > 1):
            self.SMA()
        self.extract_features()
        return self.normalize_data()

