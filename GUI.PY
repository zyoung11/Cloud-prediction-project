import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QLabel, QTextEdit, QSizePolicy)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from PyQt6.QtGui import QIcon
import shutil
import warnings

warnings.filterwarnings("ignore")
torch.set_num_threads(128)

class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, out_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        return x

class SeparationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparationModule, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.deconv1(x))
        x = self.leaky_relu(self.deconv2(x))
        x = self.leaky_relu(self.deconv3(x))
        x = self.leaky_relu(self.deconv4(x))
        x = self.leaky_relu(self.deconv5(x))
        return x

class XLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(XLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # 原始卷积层
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        # 添加可学习的门控机制
        self.gate_i = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))  # 输入门的可学习缩放参数
        self.gate_f = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))  # 遗忘门的可学习缩放参数
        self.gate_o = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))  # 输出门的可学习缩放参数

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 拼接输入和当前隐藏状态
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # 划分卷积的输出为 i, f, o, g
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 使用可学习的门控机制
        i = torch.sigmoid(cc_i * self.gate_i)
        f = torch.sigmoid(cc_f * self.gate_f)
        o = torch.sigmoid(cc_o * self.gate_o)
        g = torch.tanh(cc_g)

        # 更新细胞状态和隐藏状态
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    

class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers):
        super(Generator, self).__init__()
        self.fusion = FusionModule(input_channels, hidden_channels)
        self.num_layers = num_layers

        self.global_step = 0

        self.convlstm_cells = nn.ModuleList([XLSTMCell(hidden_channels, hidden_channels, (3, 3), True) for _ in range(num_layers)])
        self.separation = SeparationModule(hidden_channels, input_channels)

    def forward(self, x):
        batch_size, time_steps, _, height, width = x.size()
        h = [torch.zeros(batch_size, self.convlstm_cells[0].hidden_dim, height, width).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.convlstm_cells[0].hidden_dim, height, width).to(x.device) for _ in range(self.num_layers)]

        for t in range(time_steps):
            x_t = self.fusion(x[:, t])
            for l in range(self.num_layers):
                h[l], c[l] = self.convlstm_cells[l](x_t, (h[l], c[l]))
                x_t = h[l]

        output = self.separation(x_t)
        return output.unsqueeze(1)
    
    

def npy_to_png(source_dir, log_callback=None):
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 在当前目录下创建新的目标文件夹 'converted_images_temp'
    target_dir = os.path.join(current_dir, 'converted_images_temp')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有 .npy 文件
    npy_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npy')]
    for i, filename in enumerate(npy_files):  # 禁用控制台的进度条
        # 构建完整的文件路径
        npy_path = os.path.join(source_dir, filename)
        png_filename = filename.replace('.npy', '.png')
        png_path = os.path.join(target_dir, png_filename)

        # 加载 .npy 文件
        npy_data = np.load(npy_path)

        # 检查数据的形状并进行相应的处理
        if npy_data.ndim == 2:
            plt.imshow(npy_data, cmap='gray')
            if log_callback:
                log_callback(f"灰度图像: {filename}")
        elif npy_data.ndim == 3:
            if npy_data.shape[2] == 3:
                plt.imshow(npy_data)
                if log_callback:
                    log_callback(f"RGB图像: {filename}")
            elif npy_data.shape[2] == 4:
                plt.imshow(npy_data)
                if log_callback:
                    log_callback(f"RGBA图像: {filename}")
            else:
                if log_callback:
                    log_callback(f"Error: {filename} has an unsupported channel size ({npy_data.shape[2]}). Skipping...")
                continue
        else:
            if log_callback:
                log_callback(f"Error: {filename} has an unsupported number of dimensions ({npy_data.ndim}). Skipping...")
            continue

        # 保存为 .png 图像
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        if log_callback:
            log_callback(f"Converted {filename} to {png_filename}")

        # 更新进度
        if log_callback:
            progress = (i + 1) / len(npy_files) * 100
            log_callback(f"进度: {progress:.2f}%")

        # 强制刷新日志显示
        QApplication.processEvents()


def load_model(model_name, model_class, input_channels, hidden_units, num_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(sys, '_MEIPASS'):
        model_path = os.path.join(sys._MEIPASS, model_name)
    else:
        model_path = model_name
    model = model_class(input_channels=input_channels, hidden_channels=hidden_units, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def load_image(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = os.path.join(sys._MEIPASS, relative_path)
    else:
        base_path = relative_path
    return base_path


class ModernButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #808080;
            }
        """)

class ModernButton3(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #808080;
            }
        """)
        # 设置按钮大小策略为扩展，使其能够随着窗口大小变化而变化
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)


class DataModeGUI(QMainWindow):
    last_geometry = None  # 静态变量来存储最后的窗口位置和大小
    last_is_maximized = False  # 静态变量来存储窗口是否最大化

    def __init__(self):
        super().__init__()
        self.input_folder = ""
        self.output_folder = ""
        self.current_group = 0
        self.input_images = []
        self.selected_model = "10min"
        self.model_class = Generator
        self.input_channels = 1
        self.hidden_units = 64
        self.num_layers = 4
        self.models = {
            "10min": None,
            "30min": None,
            "1h": None,
            "2h": None,
            "3h": None
        }
        self.log_text = QTextEdit()
                # 延迟加载界面
        self.initUI()
        self.load_models()

    def initUI(self):
        self.setWindowTitle('AI云图预测 - 数据模式')
        self.setWindowIcon(QIcon(load_image('i.ico')))
        #self.setGeometry(100, 100, 800, 800)
        # 如果有保存的窗口状态，恢复窗口的大小和位置
        if DataModeGUI.last_geometry:
            self.setGeometry(DataModeGUI.last_geometry)
        else:
            self.setFixedSize(800, 800)
            self.center()

        # 恢复最大化状态
        if DataModeGUI.last_is_maximized:
            self.showMaximized()
        else:
            self.show() 

        # 设置背景颜色为深色主题
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        top_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()

        # 底部布局，放置按钮和日志
        bottom_left_layout = QVBoxLayout()
        bottom_right_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #3a3a3a; color: white; font-size: 14px; border-radius: 5px;")
        bottom_left_layout.addWidget(self.log_text)

        input_output_layout = QVBoxLayout()
        self.input_btn = ModernButton3("输入图像文件夹")
        self.input_btn.clicked.connect(self.select_input_folder)
        input_output_layout.addWidget(self.input_btn)

        self.output_btn = ModernButton3("保存图像文件夹")
        self.output_btn.setEnabled(False)
        self.output_btn.clicked.connect(self.select_output_folder)
        input_output_layout.addWidget(self.output_btn)

        self.start_btn = ModernButton3("开始预测")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_inference)
        input_output_layout.addWidget(self.start_btn)

        # 添加返回按钮
        self.back_btn = ModernButton("返回")
        self.back_btn.clicked.connect(self.go_back)
        input_output_layout.addWidget(self.back_btn)

        bottom_right_layout.addLayout(input_output_layout)

        bottom_layout.addLayout(bottom_left_layout, 3)
        bottom_layout.addLayout(bottom_right_layout, 1)

        main_layout.addLayout(top_layout, 2)
        main_layout.addLayout(bottom_layout, 1)
        main_widget.setLayout(main_layout)


    def load_models(self):
        # 加载每个模型
        for model_name in self.models.keys():
            model_file = f"{model_name}_模型.pth"
            model, device = load_model(model_file, self.model_class, self.input_channels, self.hidden_units, self.num_layers)
            self.models[model_name] = model
        self.device = device
        self.log("模型加载成功!")


    def select_input_folder(self):
        self.input_folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if self.input_folder:
            self.load_input_images()
            self.output_btn.setEnabled(True)


    def select_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if self.output_folder:
            self.output_btn.setEnabled(False)
            self.start_btn.setEnabled(True)


    def start_inference(self):
        self.start_btn.setEnabled(False)
        self.run_inference()


    def load_input_images(self):
        self.input_images = []
        converted_images_folder = None
        for file in sorted(os.listdir(self.input_folder)):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                self.input_images.append(os.path.join(self.input_folder, file))
            elif file.endswith('.npy'):
                if converted_images_folder is None:
                    # 第一次检测到 .npy 文件时，调用转换方法并获取转换后的文件夹路径
                    self.log("数据预处理中...")
                    QApplication.processEvents()
                    npy_to_png(self.input_folder, log_callback=self.log)    
                    converted_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'converted_images_temp')
                # 将转换后的 .png 文件添加到输入图像列表中
                png_file = file.replace('.npy', '.png')
                png_path = os.path.join(converted_images_folder, png_file)
                self.input_images.append(png_path)

        # 按6个一组分组
        self.input_images = [self.input_images[i:i + 6] for i in range(0, len(self.input_images), 6)]
        self.current_group = 0
        if not self.input_images:
            self.log("未找到可用的图像")
        elif len(self.input_images[0]) < 6:
            self.log("少于6张图像无法预测")
        else:
            self.log(f"载入 {len(self.input_images)}组 图像")


    def run_inference(self):
        for model_name in self.models.keys():
            self.selected_model = model_name
            self.log(f"开始使用 {model_name} 模型进行预测")
            
            for group_index, current_input in enumerate(self.input_images):
                if len(current_input) < 6:
                    self.log(f"组 {group_index}: 少于6张图像, 跳过")
                    continue

                input_images = [Image.open(img_path).convert('L') for img_path in current_input]
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                input_tensor = torch.stack([transform(img) for img in input_images]).unsqueeze(0)

                device = next(self.models[self.selected_model].parameters()).device
                input_tensor = input_tensor.to(device)

                model = self.models[self.selected_model]
                with torch.no_grad():
                    output = model(input_tensor)

                output_array = output[0, 0].cpu().numpy().squeeze()
                output_path = os.path.join(self.output_folder, f"{model_name}_prediction_{group_index}.png")
                Image.fromarray((output_array * 255).astype(np.uint8), mode='L').save(output_path)

                colored_output = self.apply_color_mapping(output_array)
                colored_output_path = os.path.join(self.output_folder, f"{model_name}_colored_prediction_{group_index}.png")
                plt.imsave(colored_output_path, colored_output)

                self.log(f"{model_name} 模型: 组 {group_index+1} 预测完成")
                # 强制刷新日志显示
                QApplication.processEvents()

            self.log(f"{model_name} 模型预测全部完成")

        self.log("所有模型预测完成!")


    def apply_color_mapping(self, image):
        colors = [
            (46 / 255, 4 / 255, 10 / 255), (96 / 255, 23 / 255, 27 / 255), (197 / 255, 36 / 255, 47 / 255),
            (240 / 255, 71 / 255, 51 / 255), (244 / 255, 109 / 255, 45 / 255), (248 / 255, 179 / 255, 53 / 255),
            (231 / 255, 231 / 255, 79 / 255), (209 / 255, 223 / 255, 76 / 255), (134 / 255, 196 / 255, 63 / 255),
            (93 / 255, 188 / 255, 71 / 255), (54 / 255, 170 / 255, 70 / 255), (56 / 255, 167 / 255, 74 / 255),
            (28 / 255, 64 / 255, 90 / 255), (36 / 255, 65 / 255, 135 / 255), (36 / 255, 134 / 255, 176 / 255),
            (69 / 255, 196 / 255, 209 / 255), (123 / 255, 207 / 255, 209 / 255), (205 / 255, 205 / 255, 205 / 255),
            (190 / 255, 190 / 255, 190 / 255), (152 / 255, 152 / 255, 152 / 255), (96 / 255, 96 / 255, 96 / 255),
            (67 / 255, 67 / 255, 67 / 255)
        ]

        custom_cmap = LinearSegmentedColormap.from_list("Custom22", colors, N=256) # 使用更多采样点创建连续颜色映射
        norm = mcolors.Normalize(vmin=np.percentile(image, 5), vmax=np.percentile(image, 95)) # 使用百分位数去噪，调整颜色差异

        # 应用颜色映射
        mapped_image = custom_cmap(norm(image))

        # 将归一化RGB转成uint8
        return (mapped_image[:, :, :3] * 255).astype(np.uint8)


    def log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())


    def closeEvent(self, event):
        # 检查并删除 'converted_images_temp' 文件夹
        folder_path = 'converted_images_temp'
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

        # 自动生成文件夹并分类存放照片
        if self.output_folder:
            for file_name in os.listdir(self.output_folder):
                if file_name.endswith('.png'):
                    # 判断是否为 colored 文件
                    if '_colored_' in file_name:
                        base_folder = os.path.join(self.output_folder, 'colored')
                    else:
                        base_folder = os.path.join(self.output_folder, 'non_colored')

                    # 提取时间段 (10min, 30min, etc.) 作为子文件夹名称
                    time_period = file_name.split('_')[0]  # 获取文件名中的时间段
                    time_folder = os.path.join(base_folder, time_period)

                    # 创建时间段子文件夹（如果不存在）
                    os.makedirs(time_folder, exist_ok=True)

                    # 移动文件到对应的时间段子文件夹
                    shutil.move(os.path.join(self.output_folder, file_name), os.path.join(time_folder, file_name))

    def go_back(self):
        DataModeGUI.last_geometry = self.geometry()
        DataModeGUI.last_is_maximized = self.isMaximized()

        self.start_screen = StartScreen()
        self.start_screen.show()
        self.close()

    def center(self):
        # 居中显示窗口
        screen_geometry = self.screen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def resizeEvent(self, event):
        # 记录窗口的大小和位置
        DataModeGUI.last_geometry = self.geometry()
        DataModeGUI.last_is_maximized = self.isMaximized()
        super().resizeEvent(event)

    def moveEvent(self, event):
        # 记录窗口的大小和位置
        DataModeGUI.last_geometry = self.geometry()
        super().moveEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    start_screen = StartScreen()
    start_screen.show()
    sys.exit(app.exec())