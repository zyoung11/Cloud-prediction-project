
import torch
import torch.nn as nn
import warnings
import sys
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from utils import create_gif_and_video, clear_uploads_folder

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
    
# 加载模型的函数
def load_model(model_name, model_class, input_channels, hidden_units, num_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(input_channels=input_channels, hidden_channels=hidden_units, num_layers=num_layers).to(device)
    model_path = os.path.join(sys._MEIPASS, model_name) if hasattr(sys, '_MEIPASS') else model_name
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device)
    
    # 过滤掉不匹配的层
    model_state_dict = model.state_dict()
    for key in list(state_dict.keys()):
        if key not in model_state_dict or state_dict[key].shape != model_state_dict[key].shape:
            print(f"Skipping loading parameter {key} due to size mismatch.")
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def apply_color_mapping(image):
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
    
    custom_cmap = LinearSegmentedColormap.from_list("Custom22", colors, N=256)
    norm = Normalize(vmin=np.percentile(image, 5), vmax=np.percentile(image, 95))
    
    mapped_image = custom_cmap(norm(image))
    return (mapped_image[:, :, :3] * 255).astype(np.uint8)

def run_inference(models, model_files, upload_folder, output_folder, display_only=False):
    results = []  # 新增一个列表来存储结果
    non_colored_folder = os.path.join(output_folder, 'non_colored')
    colored_folder = os.path.join(output_folder, 'colored')
    
    for folder in [non_colored_folder, colored_folder]:
        os.makedirs(folder, exist_ok=True)

    files = os.listdir(upload_folder)
    files = [os.path.join(upload_folder, f) for f in sorted(files) if f.lower().endswith('.png')]
    
    grouped_images = [files[i:i + 6] for i in range(0, len(files), 6) if len(files[i:i + 6]) == 6]
    
    if not grouped_images:
        print('Not enough images to process')
        return []

    for model_name, params in model_files:
        print(f"Loading model: {model_name} with params: {params}")
        
        model_path = os.path.abspath(model_name)
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        try:
            model = models[model_name]
            device = next(model.parameters()).device
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        for index, current_input in enumerate(grouped_images):
            # 如果模型期望灰度图像
            input_images = [Image.open(img_path).convert('L') for img_path in current_input]
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            input_tensor = torch.stack([transform(img) for img in input_images]).unsqueeze(0)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)

            output_array = output[0, 0].cpu().numpy().squeeze()

            if not display_only:
                non_colored_output_path = os.path.join(non_colored_folder, f"{model_name}_prediction_{index}.png")
                Image.fromarray((output_array * 255).astype(np.uint8), mode='L').save(non_colored_output_path)

                colored_output = apply_color_mapping(output_array)
                colored_output_path = os.path.join(colored_folder, f"{model_name}_colored_prediction_{index}.png")
                plt.imsave(colored_output_path, colored_output)
            
            print(f"Processed group {index+1} with model {model_name}")
            
            # 将结果添加到列表中
            results.append({
                'model_name': model_name,
                'group_index': index,
                'non_colored_output_path': non_colored_output_path,
                'colored_output_path': colored_output_path
            })

            
    create_gif_and_video()
    clear_uploads_folder()
    return results  # 返回结果列表
    
