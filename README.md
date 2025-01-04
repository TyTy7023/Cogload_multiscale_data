# Author
- Hướng dẫn:
	+ TS. Đỗ Như Tài
	+ TS. Nguyễn Quốc Huy
- Thực hiện:
	+ Đỗ Minh Quân
	+ Lê Thị Mỹ Hương
	+  Trần Bùi Ty Ty
# cogLoad_experiment

- baselines: chứa 4 thư mục (none, rfecv, sbfs, sffs) là những baseline thực hiện trên kaggle với các feature selection khác nhau và kết quả của các thực nghiệm được lưu tại https://docs.google.com/spreadsheets/d/1zgkSr6DWsrN9pbjXBCUEjU8tTATNmxz-gy427TRlKXw/edit?gid=0#gid=0
- data: gồm 3 thư mục (23_objects, allFeatures, last_30s_segments)
	+ 23_objects: dữ liệu thô của 23 đối tượng được ghi nhận trong quá trình thu thập dữ liệu.
	+ allFeatures: gồm statFeatures.csv và features.csv
		*statFeatures.csv: 10 thống kê đặc trưng cơ bản (tham khảo từ bài báo gốc:https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing) bao gồm mean, standard deviation, skewness, kurtosis, diff, diff2, 25th quantile, 75th, quantile, qdev, max-min.
		*allFeatures.csv: bao gồm thống kê đặc trưng cơ bản và và các đặc trưng chuyên gia (tham khảo từ bài báo gốc https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing)
	+ last_30s_segments: gồm các tín hiệu sinh lý (gsr, hr, rr, temp) được trích xuất 30s cuối của mỗi tín hiệu và thông tin nhãn (labels)
- model:
   + Bao gồm các file model trong đó có 3 file chính:
		+ mul_model.py: chứa các model MLP và Tabnet
		+ single_model.py: chứa các model đơn giản của thư viện sklearn,...
		+ model.py: là file chạy model ban đầu, chứa gần như tất cả model đã được thử nghiệm
- processData:
   + Gồm các file xử lý data, có 3 file:
   		+ processing_data: chứa class xử lý preprocessing với statistical, splits train test, normalize
		+ split_data: xử lý multiscale data
		+ selection_feature.py: File chứa class xử lý selected feature. Có 2 mô hình selected: (RFECV(estimator = XGBClassifier(n_jobs=-1)), selected_SFS(estimator = SVM(kernel='linear')),  selected_SBS())
  + test: chứa các file test
  + main.py: file chạy chính của model
  
# CÁCH THỰC NGHIỆM MỘT BASELINE: 
- Kết nối Kaggle với GitHub với tên thư mục là cogload
- %run /kaggle/working/cogload/main.py
- Các thông số truyền vào có thể thay đổi:
	+ parser.add_argument("--data_folder_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
	+ parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")
	+ parser.add_argument("--window_size", default = 1, type = int, help = "Window size for feature extraction SMA")
	+ parser.add_argument("--normalize", default = "Standard", type = str, help = "Normalization method, Standard or MinMax")
	+ parser.add_argument("--model_selected_feature", default = "None", type = str, help = "None, RFECV, SFS")
	+ parser.add_argument("--k_features", default = 11, type = int, help = "k of feature selected of SFS")
	+ parser.add_argument("--forward", default = False, type = bool, help = "True to use backward, False to use forward")
	+ parser.add_argument("--floating", default = True, type = bool, help = "True to use sfs with floating, False with no floating")
	+ parser.add_argument("--split", nargs='+', default=[] , type=int, help="the split of data example 2 6 to split data into 2 and 6 to extract feature")
	+ parser.add_argument("--estimator_RFECV", default='SVM', type=str, help="model for RFECV")
	+ parser.add_argument("--debug", default = 0, type = int, help="debug mode 0: no debug, 1: debug")
	+ parser.add_argument("--models_single", nargs='+', default=['LDA', 'SVM', 'RF','XGB'] , type=str, help="models to train")
	+ parser.add_argument("--models_mul", nargs='+', default=['MLP_Sklearn', 'MLP_Keras','TabNet'] , type=str, help="models to train")
- Kết quả sẽ được lưu vào kaggle/ouput/log/
