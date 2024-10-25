import sys
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QVBoxLayout, QWidget, QSizePolicy, QHBoxLayout, QScrollArea)
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
        self.resize(900, 1200)
        self.model_counter = 1
        self.first_load = True
        self.all_data = {}
        self.model_names = {}
        self.display_mode = 'bar'  # 初始显示模式为条形图

        # 禁用全屏按钮
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowMaximizeButtonHint)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        outer_layout = QVBoxLayout(main_widget)

        # 创建按钮布局
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        
        # 添加“添加模型”按钮
        self.add_button = ModernButton2("添加模型")
        button_layout.addWidget(self.add_button)

        # 添加“切换显示模式”按钮
        self.toggle_button = ModernButton2("切换显示模式")
        button_layout.addWidget(self.toggle_button)

        # 添加“保存图标”按钮
        self.save_button = ModernButton2("保存图表")
        button_layout.addWidget(self.save_button)

        # 设置按钮宽度一致
        button_width = 300  # 可以根据需要调整宽度
        self.add_button.setFixedWidth(button_width)
        self.toggle_button.setFixedWidth(button_width)
        self.save_button.setFixedWidth(button_width)

        outer_layout.addWidget(button_widget)

        # 滚动区域设置
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        scroll.setWidget(container)

        # 创建图表区域
        self.create_canvases()

        # 连接按钮事件
        self.add_button.clicked.connect(self.load_and_plot_data)
        self.toggle_button.clicked.connect(self.toggle_display_mode)
        self.save_button.clicked.connect(self.save_plots)


    def load_and_plot_data(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if folder_path:
            try:
                model_name = os.path.basename(folder_path)
                logs_folder = os.path.join(folder_path, "训练日志")
                if not os.path.exists(logs_folder):
                    raise ValueError("未找到训练日志文件夹")

                csv_files = [f for f in os.listdir(logs_folder) if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("训练日志文件夹中未找到CSV文件")

                csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_file = os.path.join(logs_folder, csv_files[-1])

                data = pd.read_csv(latest_file)

                self.model_names[self.model_counter] = model_name
                self.all_data[self.model_counter] = data
                self.model_counter += 1

                self.plot_all_data()

            except Exception as e:
                print(f"加载数据失败: {e}")

    def resizeEvent(self, event):
        width = self.width()
        height = int(width * 4 / 3)
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

        if self.display_mode == 'bar':
            self.plot_bar_metrics()
        else:
            self.plot_trend_metrics()
        
        self.plot_trend_data()

    def toggle_display_mode(self):
        # 切换显示模式
        self.display_mode = 'trend' if self.display_mode == 'bar' else 'bar'
        self.plot_all_data()

    def plot_bar_metrics(self):
        metrics = {
            'psnr': ('val_psnr', 'PSNR Comparison', 'max'),
            'ssim': ('val_ssim', 'SSIM Comparison', 'max'),
            'ssim_mae': ('val_ssim_mae', 'SSIM_MAE Comparison', 'min'),
            'metrics': (['val_rae', 'val_rmse'], 'RAE and RMSE Comparison', 'min')
        }

        for key, (cols, title, metric_type) in metrics.items():
            ax = self.metric_canvases[key].figure.clear()
            ax = self.metric_canvases[key].figure.add_subplot(111)

            if isinstance(cols, list):
                x = np.arange(len(self.all_data))
                width = 0.35

                rae_values = [data[cols[0]].min() for data in self.all_data.values()]
                rmse_values = [data[cols[1]].min() for data in self.all_data.values()]

                ax.bar(x - width/2, rae_values, width, label='RAE (Min)')
                ax.bar(x + width/2, rmse_values, width, label='RMSE (Min)')
                ax.set_xticks(x)
                ax.set_xticklabels([self.model_names[model_id] for model_id in self.all_data.keys()])
                ax.legend()

                min_value = min(min(rae_values), min(rmse_values))
                max_value = max(max(rae_values), max(rmse_values))
                ax.set_ylim([min_value - 0.1 * abs(min_value), max_value + 0.1 * abs(max_value)])
            else:
                if metric_type == 'max':
                    values = [data[cols].max() for data in self.all_data.values()]
                else:
                    values = [data[cols].min() for data in self.all_data.values()]

                ax.bar([self.model_names[model_id] for model_id in self.all_data.keys()], values)

                min_value = min(values)
                max_value = max(values)
                ax.set_ylim([min_value - 0.1 * abs(min_value), max_value + 0.1 * abs(max_value)])

            ax.set_title(title)
            ax.set_xticklabels([self.model_names[model_id] for model_id in self.all_data.keys()])
            ax.tick_params(axis='x', labelrotation=0)

            self.metric_canvases[key].figure.tight_layout()
            self.metric_canvases[key].draw()

    def plot_trend_metrics(self):
        metrics = {
            'psnr': ('val_psnr', 'PSNR Trend', 'max'),
            'ssim': ('val_ssim', 'SSIM Trend', 'max'),
            'ssim_mae': ('val_ssim_mae', 'SSIM_MAE Trend', 'min'),
            'metrics': (['val_rae', 'val_rmse'], 'RAE and RMSE Trend', 'min')
        }

        for key, (cols, title, metric_type) in metrics.items():
            ax = self.metric_canvases[key].figure.clear()
            ax = self.metric_canvases[key].figure.add_subplot(111)

            for model_id, data in self.all_data.items():
                if isinstance(cols, list):
                    for col in cols:
                        ax.plot(data.index, data[col], label=f'{self.model_names[model_id]} - {col}')
                else:
                    ax.plot(data.index, data[cols], label=f'{self.model_names[model_id]} - {cols}')

            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
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

    def save_plots(self):
        # 创建保存文件夹
        save_dir = os.path.join(os.getcwd(), '数据图表')
        counter = 1
        while os.path.exists(save_dir):
            save_dir = os.path.join(os.getcwd(), f'数据图表({counter})')
            counter += 1

        os.makedirs(save_dir, exist_ok=True)

        # 保存每个图表
        for i, (key, canvas) in enumerate(self.metric_canvases.items(), start=1):
            file_path = os.path.join(save_dir, f'{key}_plot_{i}.png')
            canvas.figure.savefig(file_path)
            
        for i, (key, canvas) in enumerate(self.trend_canvases.items(), start=1):
            file_path = os.path.join(save_dir, f'{key}_plot_{i}.png')
            canvas.figure.savefig(file_path)

        print(f"图表已保存到文件夹: {save_dir}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())