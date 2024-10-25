import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QLabel, QTextEdit, QSizePolicy)
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from pathlib import Path

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

class InferenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_folder = ""
        self.output_folder = ""
        self.current_group = 0
        self.input_images = []
        self.selected_model = "10min"
        self.models = {
            "10min": None,
            "30min": None,
            "1h": None,
            "2h": None,
            "3h": None
        }
        self.initUI()
        self.load_models()

    def initUI(self):
        self.setWindowTitle('Model Inference GUI')
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
        self.input_image_title = QLabel("Original Image")
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
        self.output_image_title = QLabel("Generated Image")
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

        button_style = "border-radius: 20px; padding: 20px 40px; font-size: 20px;"  # 调整按钮样式，增加尺寸

        model_buttons_layout = QHBoxLayout()
        model_buttons = ["10min", "30min", "1h", "2h", "3h"]
        for model in model_buttons:
            btn = QPushButton(model)
            btn.setMinimumSize(180, 90)  # 设置按钮的最小尺寸
            btn.setStyleSheet(f"background-color: lightgray; {button_style}")
            btn.clicked.connect(lambda checked, m=model: self.select_model(m))
            model_buttons_layout.addWidget(btn)
        bottom_right_layout.addLayout(model_buttons_layout)

        input_output_layout = QVBoxLayout()
        self.input_btn = QPushButton("Select Input Folder")
        self.input_btn.setMinimumSize(200, 100)  # 设置按钮的最小尺寸
        self.input_btn.setStyleSheet(f"background-color: lightgray; {button_style}")
        self.input_btn.clicked.connect(self.select_input_folder)
        input_output_layout.addWidget(self.input_btn)

        self.output_btn = QPushButton("Select Output Folder")
        self.output_btn.setMinimumSize(200, 100)  # 设置按钮的最小尺寸
        self.output_btn.setStyleSheet(f"background-color: lightgray; {button_style}")
        self.output_btn.setEnabled(False)  # 初始不可用状态
        self.output_btn.clicked.connect(self.select_output_folder)
        input_output_layout.addWidget(self.output_btn)

        self.start_btn = QPushButton("Start Inference")
        self.start_btn.setMinimumSize(200, 100)  # 设置按钮的最小尺寸
        self.start_btn.setStyleSheet(f"background-color: lightgray; {button_style}")
        self.start_btn.setEnabled(False)  # 初始不可用状态
        self.start_btn.clicked.connect(self.start_inference)
        input_output_layout.addWidget(self.start_btn)

        bottom_right_layout.addLayout(input_output_layout)

        navigation_layout = QHBoxLayout()
        self.prev_btn = QPushButton("←")
        self.prev_btn.setMinimumSize(100, 100)   # 设置按钮的最小尺寸
        self.prev_btn.setStyleSheet(f"background-color: lightgray; {button_style}")
        self.prev_btn.setVisible(False)  # 初始隐藏
        self.prev_btn.clicked.connect(self.prev_group)
        navigation_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("→")
        self.next_btn.setMinimumSize(100, 100)  # 设置按钮的最小尺寸
        self.next_btn.setStyleSheet(f"background-color: lightgray; {button_style}")
        self.next_btn.setVisible(False)  # 初始隐藏
        self.next_btn.clicked.connect(self.next_group)
        navigation_layout.addWidget(self.next_btn)

        bottom_right_layout.addLayout(navigation_layout)

        bottom_layout.addLayout(bottom_left_layout, 3)
        bottom_layout.addLayout(bottom_right_layout, 1)

        main_layout.addLayout(top_layout, 2)
        main_layout.addLayout(bottom_layout, 1)
        main_widget.setLayout(main_layout)




    def load_models(self):
        for model_name in self.models.keys():
            model = Generator(input_channels=1, hidden_channels=64, num_layers=4)
            model.load_state_dict(torch.load(f"{model_name}_模型.pth"))
            model.eval()  # 设置模型为评估模式
            self.models[model_name] = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.log("Models loaded successfully.")

    def select_model(self, model):
        self.selected_model = model
        self.log(f"Selected model: {model}")

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
        self.start_btn.setVisible(False)
        self.prev_btn.setVisible(True)
        self.next_btn.setVisible(True)
        self.run_inference()

    def load_input_images(self):
        self.input_images = []
        for file in sorted(os.listdir(self.input_folder)):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                self.input_images.append(os.path.join(self.input_folder, file))
            elif file.endswith('.npy'):
                png_path = self.npy_to_png(os.path.join(self.input_folder, file))
                if png_path:
                    self.input_images.append(png_path)

        self.input_images = [self.input_images[i:i+6] for i in range(0, len(self.input_images), 6)]
        self.current_group = 0
        if not self.input_images:
            self.log("No valid images found in the input folder.")
        elif len(self.input_images[0]) < 6:
            self.log("Less than six images found. Cannot perform inference.")
        else:
            self.log(f"Loaded {len(self.input_images)} groups of images.")
            self.update_navigation_buttons()

    def npy_to_png(self, npy_path):
        try:
            npy_data = np.load(npy_path)
            if npy_data.ndim == 2:
                img = Image.fromarray(npy_data, mode='L')
                png_path = npy_path.replace('.npy', '.png')
                img.save(png_path)
            elif npy_data.ndim == 3 and npy_data.shape[2] in [3, 4]:
                img = Image.fromarray(npy_data)
                png_path = npy_path.replace('.npy', '.png')
                img.save(png_path)
            else:
                self.log(f"Unsupported .npy file: {npy_path}")
                return None
            return png_path
        except Exception as e:
            self.log(f"Error converting .npy to .png: {str(e)}")
            return None

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
            self.log("Please select input and output folders first.")
            return

        current_input = self.input_images[self.current_group]
        if len(current_input) < 6:
            self.log("Less than six images in the group. Cannot perform inference.")
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

        # 显示原始对比图像（最后一张输入图像）
        original_image_path = current_input[-1]
        self.display_image(original_image_path, self.input_image_label)

        # 保存原始预测结果 (灰度图像)
        output_array = output[0, 0].cpu().numpy().squeeze()
        output_path = os.path.join(self.output_folder, f"prediction_{self.current_group}.png")
        Image.fromarray((output_array * 255).astype(np.uint8), mode='L').save(output_path)

        # 保存颜色映射后的图像
        colored_output = self.apply_color_mapping(output_array)
        colored_output_path = os.path.join(self.output_folder, f"colored_prediction_{self.current_group}.png")
        plt.imsave(colored_output_path, colored_output)

        # 显示颜色映射后的图像
        self.display_image(colored_output_path, self.output_image_label)

        self.log(f"Inference completed for group {self.current_group}. Results saved to {self.output_folder}")

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

    def display_image(self, image_path, label):
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def log(self, message):
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = InferenceGUI()
    ex.show()
    sys.exit(app.exec_())
