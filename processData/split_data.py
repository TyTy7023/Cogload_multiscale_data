import pandas as pd

import sys
sys.path.append('/kaggle/working/cogload/model/')
from processing_data import Preprocessing

class split_data () :
    def __init__(self, window_size, temp_df, hr_df, gsr_df, rr_df, label_df, normalize):
        self.window_size = window_size
        self.temp_df = temp_df
        self.hr_df = hr_df
        self.gsr_df = gsr_df
        self.rr_df = rr_df
        self.label_df = label_df
        self.normalize = normalize
        
        self.temp = []
        self.temp.append(self.temp_df)
        self.hr = []
        self.hr.append(self.hr_df)
        self.gsr = []
        self.gsr.append(self.gsr_df)
        self.rr = []
        self.rr.append(self.rr_df)

        if self.window_size > 1:
            self.SMA()

    def split_data(self, split = 2):
        for i in range(split):
            self.temp.append(self.temp_df.iloc[:,i::split])
            self.hr.append(self.hr_df.iloc[:,i::split])
            self.gsr.append(self.gsr_df.iloc[:,i::split])
            self.rr.append(self.rr_df.iloc[:,i::split])

    def SMA(self):
        self.temp = self.temp_df.rolling(self.window_size,axis=1).mean()
        self.hr = self.hr_df.rolling(self.window_size,axis=1).mean()
        self.gsr = self.gsr_df.rolling(self.window_size,axis=1).mean()
        self.rr = self.rr_df.rolling(self.window_size,axis=1).mean()

    def get_data(self):
        self.all_data_train = []
        self.all_data_test = []
        for i in range(len(self.temp)):
            processing_data = Preprocessing( 
                                temp_df = self.temp[i], 
                                hr_df = self.hr[i], 
                                gsr_df = self.gsr[i], 
                                rr_df = self.rr[i],
                                label_df = self.label_df,
                                normalize=self.normalize,
                                data_type= f"_{i}_")
            X_train, self.y_train, X_test, self.y_test, self.user_train, self.user_test = processing_data.get_data(features_to_remove = "None")

            self.all_data_train.append(X_train)
            self.all_data_test.append( X_test)
        return sum(self.all_data_train), self.y_train, sum(self.all_data_test), self.y_test, self.user_train, self.user_test
            

    

    