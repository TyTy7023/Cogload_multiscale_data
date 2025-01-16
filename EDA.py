import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

from sklearn.metrics import roc_curve, auc, roc_auc_score
    
class EDA:
    @staticmethod
    def draw_ROC_models_read_file(models, y_test,path):
        df = pd.read_csv(path)
        if path == '/kaggle/working/log/results_model.csv':
            # Xử lý để loại bỏ ký tự xuống dòng (\n)
            data_cleaned = df['Y Probs'].str.replace("\n", " ", regex=False)
            data_cleaned = data_cleaned.str.replace("[", "").str.replace("]", "")  # Loại bỏ dấu ngoặc vuông
            # Tách chuỗi và chuyển thành mảng số thực (float)
            y_prob = [np.array([float(x) for x in data_cleaned.iloc[0].split()])]
        else:
            # Chuyển trực tiếp thành mảng nếu không cần xử lý
            parsed_data = np.array(df['Y Probs'])
            y_prob = []
            for item in parsed_data:
                # Loại bỏ nháy đơn, nháy kép và dấu ngoặc vuông
                item_cleaned = item.strip("[]").replace('"', '').replace("'", "").split(', ')
                # Chuyển thành danh sách số thực
                prob_values = [float(x) for x in item_cleaned]
                y_prob.append(prob_values)
            y_prob = np.array(y_prob)  # Chuyển thành mảng NumPy 2D
            
        EDA.draw_ROC(f'/kaggle/working/log/remove/', y_test, y_prob, models)

    @staticmethod
    def draw_ROC(path, y_test, y_preds, model):
        plt.figure(figsize=(8, 8))
        for i, y_pred in enumerate(y_preds):
            if isinstance(y_test, list) and len(y_test) == len(y_preds):
                fpr, tpr, thresholds = roc_curve(y_test[i], y_pred)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model} fold({i})-(AUC = {roc_auc:.2f})')
            else: 
                fpr, tpr, thresholds = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model[i]} (AUC = {roc_auc:.2f})')

        # Đường chéo tham chiếu với AUC = 0.5
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # Thiết lập giới hạn của trục
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        # Thêm tiêu đề và nhãn trục
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')

        # Thêm chú thích (legend) để phân biệt các mô hình
        plt.legend(loc="lower right")

        # Hiển thị biểu đồ
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f"ROC-{model}"))
        plt.show()
    
    @staticmethod
    def draw_Bar(path, model, results, Type):
        data = {
            'Model': model,
            Type: results 
        }

        # Tạo DataFrame từ dữ liệu
        df = pd.DataFrame(data)

        # Chuyển đổi cột 'Accuracy' thành kiểu số thực
        df[Type] = df[Type].astype(float)

        # Vẽ biểu đồ barplot
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x='Model', y=Type, data=df, palette='pastel')
        plt.title('Algorithm Comparison')
        plt.ylabel(f'{Type} (Test)')

        # Thêm thông số trên các cột
        for p in barplot.patches:
            barplot.annotate(f'{p.get_height():.2f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            fontsize=12)

        plt.savefig(os.path.join(path, Type))
        plt.show()
    
    @staticmethod
    def draw_BoxPlot(path, model, results, Type):
        data = {
            'Model': model,
            Type: results 
        }

        # Tạo DataFrame từ dữ liệu
        df = pd.DataFrame(data)

        # Chuyển đổi cột 'Accuracy' thành kiểu số thực
        df[Type] = df[Type].astype(float)

        # Vẽ biểu đồ boxplot
        plt.figure(figsize=(10, 6))
        boxplot = sns.boxplot(x='Model', y=Type, data=df, palette='pastel')
        plt.title('Algorithm Comparison')
        plt.ylabel(f'{Type} (Test)')
        
        plt.savefig(os.path.join(path, Type))
        plt.show()

    @staticmethod
    def draw_LinePlot(path, model, results, Type):
        data = {
            'Feature': model,
            Type: results 
        }

        # Tạo DataFrame từ dữ liệu
        df = pd.DataFrame(data)

        # Chuyển đổi cột 'Accuracy' thành kiểu số thực
        df[Type] = df[Type].astype(float)
        df['Feature'] = df['Feature'].astype(str)
        # Vẽ biểu đồ boxplot
        plt.figure(figsize=(15, 6))
        line = sns.lineplot(x='Feature', y=Type, data=df, palette='pastel')
        
        # Thêm thông số trên các cột
        for p in line.patches:
            line.annotate(f'{p.get_height():.2f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            fontsize=12)
        plt.plot(df['Feature'], df[Type], marker='o')  # marker='o' thêm chấm tròn tại mỗi điểm    
        plt.title('Algorithm Comparison')
        plt.ylabel(f'{Type} (Test)')
        plt.xticks(rotation=90)
        
        plt.savefig(os.path.join(path, Type))
        plt.show()

