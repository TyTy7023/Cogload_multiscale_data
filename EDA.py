import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, roc_auc_score
    
class EDA:
    @staticmethod
    def draw_ROC(path, y_test, y_preds, model):
        plt.figure(figsize=(8, 8))
        for i, y_pred in enumerate(y_preds):
            if isinstance(y_test, list) and len(y_test) == len(y_preds):
                fpr, tpr, thresholds = roc_curve(y_test[i], y_pred)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model} fold({i})-(AUC = {roc_auc:.2f})')
            else: 
                y_test = y_test.argmax(axis=1)
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
        plt.show()
        plt.savefig(os.path.join(path, "ROC"))
    
    @staticmethod
    def draw_ACC(path, model, results, Type):
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
        barplot = sns.barplot(x='Model', y=Type, data=df, palette='viridis')
        plt.title('Algorithm Comparison')
        plt.ylabel(f'{Type} (Test)')
        plt.ylim(0, 1)  # Đặt giới hạn trục y từ 0 đến 1

        # Thêm thông số trên các cột
        for p in barplot.patches:
            barplot.annotate(f'{p.get_height():.2f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            fontsize=12)

        plt.show()
        plt.savefig(os.path.join(path, Type))


