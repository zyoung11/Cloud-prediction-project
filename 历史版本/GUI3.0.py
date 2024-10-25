import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QLabel, QTextEdit, QSizePolicy)
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont, QPainter
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from tqdm.auto import tqdm
from pathlib import Path

torch.set_num_threads(128)

# 定义模型结构，确保与训练时的模型一致
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

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers):
        super(Generator, self).__init__()
        self.fusion = FusionModule(input_channels, hidden_channels)
        self.num_layers = num_layers
        self.convlstm_cells = nn.ModuleList([ConvLSTMCell(hidden_channels, hidden_channels, (3, 3), True) for _ in range(num_layers)])
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
    

class HoverButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_style = """
            border-radius: 20px;
            padding: 20px 40px;
            font-size: 20px;
            background-color: lightgray;
            border: 2px solid #CCCCCC;
        """
        self.hover_style = """
            background-color: #D3D3D3;
            border: 2px solid #AAAAAA;
        """
        self.setStyleSheet(self.default_style)
    
    def enterEvent(self, event):
        self.setStyleSheet(self.default_style + self.hover_style)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.setStyleSheet(self.default_style)
        super().leaveEvent(event)


def npy_to_png(source_dir, log_callback=None):
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 在当前目录下创建新的目标文件夹 'converted_images'
    target_dir = os.path.join(current_dir, 'converted_images')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有 .npy 文件
    npy_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npy')]
    for i, filename in enumerate(tqdm(npy_files, disable=True)):  # 禁用控制台的进度条
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


class InferenceGUI(QMainWindow):
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
        self.display_mode = 'side_by_side'  # 默认模式为左右对比
        self.initUI()
        self.load_models()


    def initUI(self):
        self.setWindowTitle('AI云图模型推演')
        self.setGeometry(100, 100, 2000, 1200)  # 调整窗口尺寸为原来的两倍

        # 设置背景颜色为 rgb(90, 106, 130)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(90, 106, 130))  # 深蓝灰色
        self.setPalette(palette)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        top_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()

        # 顶部布局，放置输入图和输出图
        self.image_container = QWidget()
        self.image_container.setMinimumSize(1600, 800)  # 调整容器尺寸为两倍
        self.image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout = QHBoxLayout(self.image_container)

        # 设置布局样式，使图像能够填满容器
        self.input_image_label = QLabel()
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_image_title = QLabel("原始图像")
        self.input_image_title.setAlignment(Qt.AlignCenter)
        self.input_image_title.setStyleSheet("color: white; font-size: 40px; font-weight: bold;")  # 调整字体大小
        input_image_layout = QVBoxLayout()
        input_image_layout.addWidget(self.input_image_title)
        input_image_layout.addWidget(self.input_image_label)
        input_image_layout.setAlignment(self.input_image_title, Qt.AlignTop)
        image_layout.addLayout(input_image_layout, 1)

        self.output_image_label = QLabel()
        self.output_image_label.setAlignment(Qt.AlignCenter)
        self.output_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_image_title = QLabel("预测图像")
        self.output_image_title.setAlignment(Qt.AlignCenter)
        self.output_image_title.setStyleSheet("color: white; font-size: 40px; font-weight: bold;")  # 调整字体大小
        output_image_layout = QVBoxLayout()
        output_image_layout.addWidget(self.output_image_title)
        output_image_layout.addWidget(self.output_image_label)
        output_image_layout.setAlignment(self.output_image_title, Qt.AlignTop)
        image_layout.addLayout(output_image_layout, 1)

        top_layout.addWidget(self.image_container)

        # 底部布局，放置按钮和日志
        bottom_left_layout = QVBoxLayout()
        bottom_right_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #4A4A4A; color: white; font-size: 20px;")  # 调整日志字体大小
        bottom_left_layout.addWidget(self.log_text)

        model_buttons_layout = QHBoxLayout()
        model_buttons = ["10min", "30min", "1h", "2h", "3h"]
        for model in model_buttons:
            btn = HoverButton(model)
            btn.setMinimumSize(180, 90)  # 设置按钮的最小尺寸
            btn.clicked.connect(lambda checked, m=model: self.select_model(m))
            model_buttons_layout.addWidget(btn)
        bottom_right_layout.addLayout(model_buttons_layout)

        input_output_layout = QVBoxLayout()
        self.input_btn = HoverButton("输入图像文件夹")
        self.input_btn.setMinimumSize(200, 100)  # 设置按钮的最小尺寸
        self.input_btn.setEnabled(False)  # 初始不可用状态
        self.input_btn.clicked.connect(self.select_input_folder)
        input_output_layout.addWidget(self.input_btn)

        self.output_btn = HoverButton("保存图像文件夹")
        self.output_btn.setMinimumSize(200, 100)  # 设置按钮的最小尺寸
        self.output_btn.setEnabled(False)  # 初始不可用状态
        self.output_btn.clicked.connect(self.select_output_folder)
        input_output_layout.addWidget(self.output_btn)

        self.start_btn = HoverButton("开始预测")
        self.start_btn.setMinimumSize(200, 100)  # 设置按钮的最小尺寸
        self.start_btn.setEnabled(False)  # 初始不可用状态
        self.start_btn.clicked.connect(self.start_inference)
        input_output_layout.addWidget(self.start_btn)

        self.toggle_mode_btn = HoverButton("切换对比模式")
        self.toggle_mode_btn.setMinimumSize(200, 100)  # 设置按钮的最小尺寸
        self.toggle_mode_btn.setEnabled(False)  # 初始不可用状态
        self.toggle_mode_btn.clicked.connect(self.toggle_display_mode)
        input_output_layout.addWidget(self.toggle_mode_btn)

        bottom_right_layout.addLayout(input_output_layout)

        navigation_layout = QHBoxLayout()
        self.prev_btn = HoverButton("←")
        self.prev_btn.setMinimumSize(100, 100)   # 设置按钮的最小尺寸
        self.prev_btn.setEnabled(False)  # 初始不可用状态
        self.prev_btn.clicked.connect(self.prev_group)
        navigation_layout.addWidget(self.prev_btn)

        self.next_btn = HoverButton("→")
        self.next_btn.setMinimumSize(100, 100)  # 设置按钮的最小尺寸
        self.next_btn.setEnabled(False)  # 初始不可用状态
        self.next_btn.clicked.connect(self.next_group)
        navigation_layout.addWidget(self.next_btn)

        bottom_right_layout.addLayout(navigation_layout)

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

    def select_model(self, model):
        self.selected_model = model
        self.log(f"预测时间: {model}")
        self.input_btn.setEnabled(True)  # 启用选择输入文件夹按钮
        
        # 如果已选择输入文件夹，则重新进行推理
        if self.input_folder:
            self.run_inference()

    def select_input_folder(self):
        self.input_folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if self.input_folder:
            self.load_input_images()
            self.input_btn.setEnabled(False)
            self.output_btn.setEnabled(True)

    def select_output_folder(self):
        self.output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if self.output_folder:
            self.output_btn.setEnabled(False)
            self.start_btn.setEnabled(True)

    def start_inference(self):
        self.start_btn.setEnabled(False)
        self.prev_btn.setEnabled(True)  # 启用上一组按钮
        self.next_btn.setEnabled(True)  # 启用下一组按钮
        self.toggle_mode_btn.setEnabled(True)  # 启用切换模式按钮
        self.run_inference()

    def toggle_display_mode(self):
        if self.display_mode == 'side_by_side':
            self.display_mode = 'overlay'
        else:
            self.display_mode = 'side_by_side'
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
                    npy_to_png(self.input_folder, log_callback=self.log)
                    converted_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'converted_images')
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
            self.update_navigation_buttons()

    def log(self, message):
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()

    def update_navigation_buttons(self):
        self.prev_btn.setEnabled(self.current_group > 0)
        self.next_btn.setEnabled(self.current_group < len(self.input_images) - 1)

    def prev_group(self):
        if self.current_group > 0:
            self.current_group -= 1
            self.update_navigation_buttons()
            self.run_inference()

    def next_group(self):
        if self.current_group < len(self.input_images) - 1:
            self.current_group += 1
            self.update_navigation_buttons()
            self.run_inference()

    def run_inference(self):
        if not self.input_images or not self.output_folder:
            self.log("请先选择文件夹")
            return

        current_input = self.input_images[self.current_group]
        if len(current_input) < 6:
            self.log("少于6张图像无法预测")
            return

        # 加载输入图像并进行预处理
        input_images = [Image.open(img_path).convert('L') for img_path in current_input]
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        input_tensor = torch.stack([transform(img) for img in input_images]).unsqueeze(0)

        # 获取模型所在设备
        device = next(self.models[self.selected_model].parameters()).device
        input_tensor = input_tensor.to(device)

        model = self.models[self.selected_model]
        with torch.no_grad():
            output = model(input_tensor)

        # 保存原始预测结果 (灰度图像)
        output_array = output[0, 0].cpu().numpy().squeeze()
        output_path = os.path.join(self.output_folder, f"prediction_{self.current_group}.png")
        Image.fromarray((output_array * 255).astype(np.uint8), mode='L').save(output_path)

        # 保存颜色映射后的图像
        colored_output = self.apply_color_mapping(output_array)
        colored_output_path = os.path.join(self.output_folder, f"colored_prediction_{self.current_group}.png")
        plt.imsave(colored_output_path, colored_output)

        if self.display_mode == 'overlay':
            # 创建叠加图像并显示在一个标签中
            overlay_image_path = self.create_overlay_image(current_input[-1], colored_output_path)
            self.display_image(overlay_image_path, self.output_image_label)
            self.input_image_label.clear()  # 清除另一个标签上的图像
        else:
            # 显示原始对比图像和生成图像
            self.display_image(current_input[-1], self.input_image_label)
            self.display_image(colored_output_path, self.output_image_label)

        self.log(f"{self.current_group}预测完成! 预测图保存到: {self.output_folder}")

    def display_image(self, image_path, label):
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def apply_color_mapping(self, image):
        colors = [
            (46 / 255, 4 / 255, 10 / 255), (96 / 255, 23 / 255, 27 / 255), (197 / 255, 36 / 255, 47 / 255),
            (240 / 255, 51 / 255, 35 / 255), (244 / 255, 109 / 255, 45 / 255), (248 / 255, 179 / 255, 53 / 255),
            (231 / 255, 231 / 255, 79 / 255), (209 / 255, 223 / 255, 76 / 255), (134 / 255, 196 / 255, 63 / 255),
            (93 / 255, 188 / 255, 71 / 255), (54 / 255, 170 / 255, 70 / 255), (56 / 255, 167 / 255, 74 / 255),
            (28 / 255, 64 / 255, 90 / 255), (36 / 255, 65 / 255, 135 / 255), (36 / 255, 134 / 255, 176 / 255),
            (69 / 255, 196 / 255, 209 / 255), (123 / 255, 207 / 255, 209 / 255), (205 / 255, 205 / 255, 205 / 255),
            (190 / 255, 190 / 255, 190 / 255), (152 / 255, 152 / 255, 152 / 255), (96 / 255, 96 / 255, 96 / 255),
            (67 / 255, 67 / 255, 67 / 255)
        ]
        custom_cmap = LinearSegmentedColormap.from_list("Custom22", colors, N=22)
        norm = mcolors.Normalize(vmin=image.min(), vmax=image.max())
        mapped_image = custom_cmap(norm(image))
        return (mapped_image[:, :, :3] * 255).astype(np.uint8)  # 只返回前3个通道（RGB），并转换为uint8类型

    def create_overlay_image(self, original_image_path, generated_image_path):
        original_image = Image.open(original_image_path).convert('RGBA')
        generated_image = Image.open(generated_image_path).convert('RGBA')

        # 调整生成图像的大小以匹配原始图像
        generated_image = generated_image.resize(original_image.size, Image.LANCZOS)

        # 去除生成图像中不明显的部分
        generated_array = np.array(generated_image)
        mask = (generated_array[:, :, 0] > 50) | (generated_array[:, :, 1] > 50) | (generated_array[:, :, 2] > 50)
        generated_array[~mask] = [0, 0, 0, 0]
        generated_array[mask, 3] = 90  # 降低透明度

        # 创建叠加图像
        overlay_image = Image.alpha_composite(original_image, Image.fromarray(generated_array))
        overlay_path = os.path.join(self.output_folder, f"overlay_{self.current_group}.png")
        overlay_image.save(overlay_path)

        return overlay_path

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = InferenceGUI()
    ex.show()
    sys.exit(app.exec_())

