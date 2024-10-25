import sys
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QVBoxLayout, QWidget, QSizePolicy, QHBoxLayout, QScrollArea, QFrame)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QIcon
import os
from matplotlib.ticker import ScalarFormatter
import numpy as np

def load_image(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = os.path.join(sys._MEIPASS, relative_path)
    else:
        base_path = relative_path
    return base_path

class ModernButton2(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border-radius: 15px;
                padding: 10px 20px;
                font-size: 24px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

class AspectRatioWidget(QWidget):
    def __init__(self, widget, aspect_ratio):
        super().__init__()
        self.aspect_ratio = aspect_ratio
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        self.setLayout(layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 锁定4:3的宽高比，即使窗口改变大小也不改变比例
        target_width = self.width()
        target_height = int(target_width / self.aspect_ratio)
        self.setFixedHeight(target_height)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        fig.set_tight_layout(True)
        self.setFixedSize(fig.bbox_inches.width * dpi, fig.bbox_inches.height * dpi)

    def redraw(self):
        self.figure.clear()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多模型训练数据对比可视化")
        self.setWindowIcon(QIcon(load_image('tubiao.ico')))
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: white;
            }
        """)
        self.resize(1200, 900)
        self.model_counter = 1
        self.first_load = True
        self.all_data = {}
        self.model_names = {}

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        outer_layout = QVBoxLayout(main_widget)

        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        self.add_button = ModernButton2("添加模型")
        button_layout.addWidget(self.add_button)
        outer_layout.addWidget(button_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        scroll.setWidget(container)

        self.create_canvases()
        self.add_button.clicked.connect(self.load_and_plot_data)

    def load_and_plot_data(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if folder_path:
            try:
                # 使用文件夹名称作为模型名称
                model_name = os.path.basename(folder_path)

                # 进入“训练日志”文件夹查找CSV文件
                logs_folder = os.path.join(folder_path, "训练日志")
                if not os.path.exists(logs_folder):
                    raise ValueError("未找到训练日志文件夹")

                # 查找“训练日志”文件夹中的所有CSV文件
                csv_files = [f for f in os.listdir(logs_folder) if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("训练日志文件夹中未找到CSV文件")

                # 按文件名中的数字部分排序，获取最新的文件
                csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_file = os.path.join(logs_folder, csv_files[-1])

                # 读取CSV文件
                data = pd.read_csv(latest_file)

                # 存储数据和模型名称
                self.model_names[self.model_counter] = model_name
                self.all_data[self.model_counter] = data
                self.model_counter += 1

                # 重新绘制所有图表
                self.plot_all_data()

            except Exception as e:
                print(f"加载数据失败: {e}")


    def resizeEvent(self, event):
        width = self.width()
        height = int(width * 3 / 4)
        self.resize(width, height)
        super().resizeEvent(event)

    def create_canvases(self):
        self.metric_canvases = {
            'psnr': MplCanvas(self),
            'ssim': MplCanvas(self),
            'ssim_mae': MplCanvas(self),
            'metrics': MplCanvas(self),
        }
        
        self.trend_canvases = {
            'loss': MplCanvas(self),
            'lr': MplCanvas(self),
            'score': MplCanvas(self),
        }
        
        for canvas in self.metric_canvases.values():
            wrapper = AspectRatioWidget(canvas, 4/3)
            wrapper.setMinimumWidth(400)
            self.main_layout.addWidget(wrapper)
            
        for canvas in self.trend_canvases.values():
            wrapper = AspectRatioWidget(canvas, 4/3)
            wrapper.setMinimumWidth(400)
            self.main_layout.addWidget(wrapper)

    def plot_all_data(self):
        if not self.all_data:
            return

        self.plot_bar_metrics()
        self.plot_trend_data()

    def plot_bar_metrics(self):
        # 更新绘制条形图的指标逻辑
        metrics = {
            'psnr': ('val_psnr', 'PSNR Comparison', 'max'),  # 显示最大值
            'ssim': ('val_ssim', 'SSIM Comparison', 'max'),  # 显示最大值
            'ssim_mae': ('val_ssim_mae', 'SSIM_MAE Comparison', 'min'),  # 显示最小值
            'metrics': (['val_rae', 'val_rmse'], 'RAE and RMSE Comparison', 'min')  # 显示最小值
        }

        for key, (cols, title, metric_type) in metrics.items():
            ax = self.metric_canvases[key].figure.clear()
            ax = self.metric_canvases[key].figure.add_subplot(111)

            if isinstance(cols, list):  # RAE 和 RMSE的组合图
                x = np.arange(len(self.all_data))
                width = 0.35

                rae_values = [data[cols[0]].min() for data in self.all_data.values()]
                rmse_values = [data[cols[1]].min() for data in self.all_data.values()]

                ax.bar(x - width/2, rae_values, width, label='RAE (Min)')
                ax.bar(x + width/2, rmse_values, width, label='RMSE (Min)')
                ax.set_xticks(x)
                ax.set_xticklabels([self.model_names[model_id] for model_id in self.all_data.keys()])
                ax.legend()

                # 自动设置y轴范围
                min_value = min(min(rae_values), min(rmse_values))
                max_value = max(max(rae_values), max(rmse_values))
                ax.set_ylim([min_value - 0.1 * abs(min_value), max_value + 0.1 * abs(max_value)])
            else:
                # 选择最大值或最小值
                if metric_type == 'max':
                    values = [data[cols].max() for data in self.all_data.values()]
                else:
                    values = [data[cols].min() for data in self.all_data.values()]

                ax.bar([self.model_names[model_id] for model_id in self.all_data.keys()], values)

                # 自动设置y轴范围
                min_value = min(values)
                max_value = max(values)
                ax.set_ylim([min_value - 0.1 * abs(min_value), max_value + 0.1 * abs(max_value)])

            ax.set_title(title)

            # 保持模型标签水平显示
            ax.set_xticklabels([self.model_names[model_id] for model_id in self.all_data.keys()])
            ax.tick_params(axis='x', labelrotation=0)  # 标签水平显示

            self.metric_canvases[key].figure.tight_layout()
            self.metric_canvases[key].draw()



    def plot_trend_data(self):
        trends = {
            'loss': {
                'cols': ['avg_g_loss', 'avg_d_loss', 'val_loss'],
                'labels': ['Generator Loss', 'Discriminator Loss', 'Validation Loss'],
                'title': 'Loss Comparison'
            },
            'lr': {
                'cols': ['g_lr', 'd_lr'],
                'labels': ['Generator LR', 'Discriminator LR'],
                'title': 'Learning Rate Comparison'
            },
            'score': {
                'cols': ['val_real_score', 'val_fake_score'],
                'labels': ['Real Score', 'Fake Score'],
                'title': 'Score Comparison'
            }
        }

        for key, info in trends.items():
            valid_data = self.all_data
            ax = self.trend_canvases[key].figure.clear()
            ax = self.trend_canvases[key].figure.add_subplot(111)

            for model_id, data in valid_data.items():
                for col, label in zip(info['cols'], info['labels']):
                    if col in data.columns:
                        ax.plot(data.index, data[col], label=f'{self.model_names[model_id]} - {label}')

            ax.set_title(info['title'])
            ax.set_xlabel('Epoch')
            ax.legend()

            if key == 'lr':
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
                ax.ticklabel_format(style='plain', axis='y')

            self.trend_canvases[key].figure.tight_layout()
            self.trend_canvases[key].draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
