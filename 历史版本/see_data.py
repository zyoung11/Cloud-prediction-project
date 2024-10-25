import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog
import tkinter as tk

def load_and_plot_data():
    # 弹出文件选择对话框
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if file_path:
        # 使用pandas读取CSV文件
        data = pd.read_csv(file_path)

        # 将数据列名转换为合适的Python变量名
        data.columns = ['avg_g_loss', 'avg_d_loss', 'val_loss', 'val_psnr', 'val_rae', 'val_rmse', 'val_ssim', 'val_ssim_mae', 'g_lr', 'd_lr']

        # 创建一个2x3的画布
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        # 绘制生成器、判别器和验证损失
        axs[0, 0].plot(data.index, data['avg_g_loss'], label='Generator Loss')
        axs[0, 0].plot(data.index, data['avg_d_loss'], label='Discriminator Loss')
        axs[0, 0].plot(data.index, data['val_loss'], label='Validation Loss')
        axs[0, 0].set_title('Generator, Discriminator, and Validation Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        # 绘制PSNR
        axs[0, 1].plot(data.index, data['val_psnr'])
        axs[0, 1].set_title('Validation PSNR')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('PSNR')

        # 绘制RAE 和 RMSE
        axs[0, 2].plot(data.index, data['val_rae'], label='Validation RAE')
        axs[0, 2].plot(data.index, data['val_rmse'], label='Validation RMSE')
        axs[0, 2].set_title('Validation RAE and RMSE')
        axs[0, 2].set_xlabel('Epoch')
        axs[0, 2].set_ylabel('Metric Value')
        axs[0, 2].legend()

        # 绘制SSIM
        axs[1, 0].plot(data.index, data['val_ssim'])
        axs[1, 0].set_title('Validation SSIM')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('SSIM')

        # 绘制SSIM_MAE
        axs[1, 1].plot(data.index, data['val_ssim_mae'])
        axs[1, 1].set_title('Validation SSIM_MAE')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('SSIM_MAE')

        # 绘制学习率
        axs[1, 2].plot(data.index, data['g_lr'], label='Generator Learning Rate')
        axs[1, 2].plot(data.index, data['d_lr'], label='Discriminator Learning Rate')
        axs[1, 2].set_title('Learning Rates')
        axs[1, 2].set_xlabel('Epoch')
        axs[1, 2].set_ylabel('Learning Rate')
        axs[1, 2].legend()

        # 调整子图间距
        plt.tight_layout()

        # 显示图表
        plt.show()

if __name__ == '__main__':
    load_and_plot_data()