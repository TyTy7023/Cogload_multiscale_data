import pandas as pd

import sys
sys.path.append('/kaggle/working/cogload/model/')
from processing_data import Preprocessing

class split_data () :
    def __init__(self, window_size, temp_df, hr_df, gsr_df, rr_df, label_df, normalize, split = 2):
        self.window_size = window_size
        self.temp_df = temp_df
        self.hr_df = hr_df
        self.gsr_df = gsr_df
        self.rr_df = rr_df
        self.label_df = label_df
        self.normalize = normalize
        self.split = split
        
        self.temp = []
        self.hr = []
        self.gsr = []
        self.rr = []

        if self.window_size > 1:
            self.SMA()
    
    def SMA(self):
        self.temp_df = self.temp_df.rolling(self.window_size,axis=1).mean()
        self.hr_df = self.hr_df.rolling(self.window_size,axis=1).mean()
        self.gsr_df = self.gsr_df.rolling(self.window_size,axis=1).mean()
        self.rr_df = self.rr_df.rolling(self.window_size,axis=1).mean()

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
                                normalize=self.normalize)
            processing_data.splits_train_test()
            processing_data.X_train, processing_data.X_test = processing_data.normalize_data(processing_data.X_train, processing_data.X_test)
            
            self.all_data_train.append(processing_data.X_train)
            self.all_data_test.append(processing_data.X_test)
        
        return sum(self.all_data_train), processing_data.y_train, sum(self.all_data_test), processing_data.y_test, processing_data.user_train, processing_data.user_test
    
    def split_data(self, split = 2):
        temp_split = []
        hr_split = []
        gsr_split = []
        rr_split = []
        num_cols = len(self.temp_df.columns)

        # Tính toán số cột mỗi nhóm (chia đều trước)
        base_step = num_cols // split
        extra = num_cols % split  # Cột dư để phân bổ vào các nhóm đầu
        start = 0

        for i in range(split):  
            step = base_step + (1 if i < extra else 0)
            end = start + step
            processing_data = Preprocessing( 
                                    temp_df = self.temp_df.iloc[:,start :end], 
                                    hr_df = self.hr_df.iloc[:,start :end], 
                                    gsr_df = self.gsr_df.iloc[:,start :end], 
                                    rr_df = self.rr_df.iloc[:,start :end],
                                    label_df = self.label_df,
                                    normalize=self.normalize)
            start = end
            
            processing_data.extract_features()
            temp_split.append(processing_data.temp_stat_features)
            hr_split.append(processing_data.hr_stat_features)
            gsr_split.append(processing_data.gsr_stat_features)
            rr_split.append(processing_data.rr_stat_features)
        
        self.temp.append(sum(temp_split)/len(temp_split))
        self.hr.append(sum(hr_split)/len(hr_split))
        self.gsr.append(sum(gsr_split)/len(gsr_split))
        self.rr.append(sum(rr_split)/len(rr_split))

            