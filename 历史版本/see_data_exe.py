import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QSizePolicy
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QIcon
import os

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file_path:
            data = pd.read_csv(file_path)
            
            # 将数据列名转换为合适的Python变量名
            data.columns = ['avg_g_loss', 'avg_d_loss', 'val_loss', 'val_psnr', 'val_rae', 'val_rmse', 'val_ssim', 'val_ssim_mae', 'g_lr', 'd_lr']

            self.canvas.figure.clear()
            axs = [self.canvas.figure.add_subplot(2, 3, i + 1) for i in range(6)]

            # 绘制生成器、判别器和验证损失
            axs[0].plot(data.index, data['avg_g_loss'], label='Generator Loss')
            axs[0].plot(data.index, data['avg_d_loss'], label='Discriminator Loss')
            axs[0].plot(data.index, data['val_loss'], label='Validation Loss')
            axs[0].set_title('Generator, Discriminator, and Validation Loss')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')
            axs[0].legend()

            # 绘制PSNR
            axs[1].plot(data.index, data['val_psnr'])
            axs[1].set_title('Validation PSNR')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('PSNR')

            # 绘制RAE 和 RMSE
            axs[2].plot(data.index, data['val_rae'], label='Validation RAE')
            axs[2].plot(data.index, data['val_rmse'], label='Validation RMSE')
            axs[2].set_title('Validation RAE and RMSE')
            axs[2].set_xlabel('Epoch')
            axs[2].set_ylabel('Metric Value')
            axs[2].legend()

            # 绘制SSIM
            axs[3].plot(data.index, data['val_ssim'])
            axs[3].set_title('Validation SSIM')
            axs[3].set_xlabel('Epoch')
            axs[3].set_ylabel('SSIM')

            # 绘制SSIM_MAE
            axs[4].plot(data.index, data['val_ssim_mae'])
            axs[4].set_title('Validation SSIM_MAE')
            axs[4].set_xlabel('Epoch')
            axs[4].set_ylabel('SSIM_MAE')

            # 绘制学习率
            axs[5].plot(data.index, data['g_lr'], label='Generator Learning Rate')
            axs[5].plot(data.index, data['d_lr'], label='Discriminator Learning Rate')
            axs[5].set_title('Learning Rates')
            axs[5].set_xlabel('Epoch')
            axs[5].set_ylabel('Learning Rate')
            axs[5].legend()

            # 调整子图间距
            self.canvas.figure.tight_layout()

            # 显示图表
            self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())