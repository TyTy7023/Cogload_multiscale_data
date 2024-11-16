# cogLoad_experiment

- baselines: chứa 4 thư mục (none, rfecv, sbfs, sffs) là những baseline thực hiện trên kaggle với các feature selection khác nhau và kết quả của các thực nghiệm được lưu tại https://docs.google.com/spreadsheets/d/1zgkSr6DWsrN9pbjXBCUEjU8tTATNmxz-gy427TRlKXw/edit?gid=0#gid=0
- data: gồm 3 thư mục (23_objects, allFeatures, last_30s_segments)
	+ 23_objects: dữ liệu thô của 23 đối tượng được ghi nhận trong quá trình thu thập dữ liệu.
	+ allFeatures: gồm statFeatures.csv và features.csv
		*statFeatures.csv: 10 thống kê đặc trưng cơ bản (tham khảo từ bài báo gốc:https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing) bao gồm mean, standard deviation, skewness, kurtosis, diff, diff2, 25th quantile, 75th, quantile, qdev, max-min.
		*allFeatures.csv: bao gồm thống kê đặc trưng cơ bản và và các đặc trưng chuyên gia (tham khảo từ bài báo gốc https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing)
	+ last_30s_segments: gồm các tín hiệu sinh lý (gsr, hr, rr, temp) được trích xuất 30s cuối của mỗi tín hiệu và thông tin nhãn (labels)
-model.py:
   + Hàm chạy lần lượt 9 models = ['LogisticRegression', 'LinearDiscriminantAnalysis', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'SVC'] với random_state = 42
-processing_data.py:
   + File chứa class xử lý preprocessing với statistical, splits train test, normalize
-selection_feature.py:
  + File chứa class xử lý selected feature. Có 2 mô hình selected: (RFECV(estimator = XGBClassifier(n_jobs=-1)), SFS(estimator = SVM(kernel='linear')
  + test.py: file chạy experiment
  
# CÁCH THỰC NGHIỆM MỘT BASELINE: 
- Kết nối Kaggle với GitHub với tên thư mục là cogload
- %run /kaggle/working/cogload/test.py
- Các thông số truyền vào có thể thay đổi
	+ --data_folder_path, default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/" 
	+ --GroupKFold, default = 3, type = int, help = "Slip data into k group for training model"
	+ --window_size, default = 1, help ="Window size for feature extraction SMA"
	+ --normalize, default = "Standard", help = "Normalization method, Standard or MinMax")
	+ --k_features, default = 11, help = "k of feature selected of SFS")
	+ --forward, default = False, type = bool, help = "True to use backward, False to use forward"
	+ --floating, default = True, type = int, help = "True to use sfs with floating, False with no floating"
	+ --model_selected_feature", default = "None", help = "None, RFECV, SFS"
	+ --split, default=[], type = int, help = "split the segment to extract"
	+ --estimator_RFECV, default='SVM', type=str, help="model for RFECV"
	+ --debug, default = 0, type = int, help="debug mode 0: no debug, 1: debug"
- Kết quả sẽ được lưu vào kaggle/ouput/log/
