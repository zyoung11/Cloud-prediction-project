import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QSizePolicy
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QIcon
import os
from matplotlib.ticker import ScalarFormatter

def load_image(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = os.path.join(sys._MEIPASS, relative_path)
    else:
        base_path = relative_path
    return base_path

class ModernButton2(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        # 设置按钮样式
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border-radius: 15px;
                padding: 10px 20px;
                font-size: 24px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        # 设置按钮大小策略为扩展，使其能够随着窗口大小变化而变化
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("训练数据可视化")
        self.setWindowIcon(QIcon(load_image('tubiao.ico')))
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.resize(1500, 800)

        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.button = ModernButton2("Load & Plot Data")
        self.button.clicked.connect(self.load_and_plot_data)
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.canvas = MplCanvas(self, width=18, height=12, dpi=100)
        layout.addWidget(self.canvas)

    def load_and_plot_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开CSV文件", "", "CSV文件 (*.csv)")
        if file_path:
            try:
                data = pd.read_csv(file_path)

                # 预期的列名
                data.columns = ['avg_g_loss', 'avg_d_loss', 'val_loss', 'val_psnr', 'val_rae', 'val_rmse',
                                'val_ssim', 'val_ssim_mae', 'val_real_score', 'val_fake_score', 'g_lr', 'd_lr']

                # 清除旧图表
                self.canvas.figure.clear()
                axs = [self.canvas.figure.add_subplot(2, 4, i + 1) for i in range(7)]

                # 第一行图表：PSNR, SSIM, SSIM_MAE, RAE and RMSE
                axs[0].plot(data.index, data['val_psnr'])
                axs[0].set_title('Validation PSNR')
                axs[0].set_xlabel('Epoch')
                axs[0].set_ylabel('PSNR')

                axs[1].plot(data.index, data['val_ssim'])
                axs[1].set_title('Validation SSIM')
                axs[1].set_xlabel('Epoch')
                axs[1].set_ylabel('SSIM')

                axs[2].plot(data.index, data['val_ssim_mae'])
                axs[2].set_title('Validation SSIM_MAE')
                axs[2].set_xlabel('Epoch')
                axs[2].set_ylabel('SSIM_MAE')

                axs[3].plot(data.index, data['val_rae'], label='Validation RAE')
                axs[3].plot(data.index, data['val_rmse'], label='Validation RMSE')
                axs[3].set_title('Validation RAE | Validation RMSE')
                axs[3].set_xlabel('Epoch')
                axs[3].set_ylabel('Metric Value')
                axs[3].legend()

                # 第二行图表：损失，学习率，真假分数
                axs[4].plot(data.index, data['avg_g_loss'], label='Generator Loss')
                axs[4].plot(data.index, data['avg_d_loss'], label='Discriminator Loss')
                axs[4].plot(data.index, data['val_loss'], label='Validation Loss')
                axs[4].set_title('Generator, Discriminator, Validation Loss')
                axs[4].set_xlabel('Epoch')
                axs[4].set_ylabel('Loss')
                axs[4].legend()

                # 绘制Learning Rates，并禁用科学计数法
                axs[5].plot(data.index, data['g_lr'], label='G LR')
                axs[5].plot(data.index, data['d_lr'], label='D LR')
                axs[5].set_title('Generator, Discriminator Learning Rates')
                axs[5].set_xlabel('Epoch')
                axs[5].set_ylabel('Learning Rate')
                axs[5].legend()
                
                # 禁用科学计数法，使用常规格式显示Learning Rates的纵坐标
                axs[5].yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
                axs[5].ticklabel_format(style='plain', axis='y')  # 禁用科学计数法

                # 真假分数图表
                axs[6].plot(data.index, data['val_real_score'], label='Real Score')
                axs[6].plot(data.index, data['val_fake_score'], label='Fake Score')
                axs[6].set_title('Real and Fake Score')
                axs[6].set_xlabel('Epoch')
                axs[6].set_ylabel('Score')
                axs[6].legend()

                # 调整子图间距
                self.canvas.figure.tight_layout()

                # 显示图表
                self.canvas.draw()
            except Exception as e:
                print(f"加载数据失败: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())