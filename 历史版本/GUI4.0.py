import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from tqdm.auto import tqdm
from pathlib import Path

torch.set_num_threads(128)

# 模型定义
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

def npy_to_png(source_dir, log_callback=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, 'converted_images')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    npy_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npy')]
    for i, filename in enumerate(tqdm(npy_files, disable=True)):
        npy_path = os.path.join(source_dir, filename)
        png_filename = filename.replace('.npy', '.png')
        png_path = os.path.join(target_dir, png_filename)

        npy_data = np.load(npy_path)

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

        plt.axis('off')
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        if log_callback:
            log_callback(f"Converted {filename} to {png_filename}")
            progress = (i + 1) / len(npy_files) * 100
            log_callback(f"进度: {progress:.2f}%")

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

class InferenceGUI(ctk.CTk):
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
        self.display_mode = 'side_by_side'
        self.initUI()
        self.load_models()

    def initUI(self):
        self.title('AI云图模型推演')
        self.geometry('2000x1200')

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=2)
        self.grid_rowconfigure(1, weight=1)

        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=1)
        top_frame.grid_rowconfigure(0, weight=1)

        self.input_image_frame = ctk.CTkFrame(top_frame)
        self.input_image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.input_image_label = ctk.CTkLabel(self.input_image_frame, text="原始图像")
        self.input_image_label.pack(pady=10)
        self.input_image = ctk.CTkLabel(self.input_image_frame, text="")
        self.input_image.pack(expand=True, fill="both")

        self.output_image_frame = ctk.CTkFrame(top_frame)
        self.output_image_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.output_image_label = ctk.CTkLabel(self.output_image_frame, text="预测图像")
        self.output_image_label.pack(pady=10)
        self.output_image = ctk.CTkLabel(self.output_image_frame, text="")
        self.output_image.pack(expand=True, fill="both")

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        bottom_frame.grid_columnconfigure(0, weight=3)
        bottom_frame.grid_columnconfigure(1, weight=1)

        self.log_text = ctk.CTkTextbox(bottom_frame, width=400, height=200)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        control_frame = ctk.CTkFrame(bottom_frame)
        control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        model_buttons_frame = ctk.CTkFrame(control_frame)
        model_buttons_frame.pack(fill="x", pady=10)
        model_buttons = ["10min", "30min", "1h", "2h", "3h"]
        for model in model_buttons:
            btn = ctk.CTkButton(model_buttons_frame, text=model, command=lambda m=model: self.select_model(m))
            btn.pack(side="left", padx=5)

        self.input_btn = ctk.CTkButton(control_frame, text="输入图像文件夹", command=self.select_input_folder)
        self.input_btn.pack(fill="x", pady=5)
        self.input_btn.configure(state="disabled")

        self.output_btn = ctk.CTkButton(control_frame, text="保存图像文件夹", command=self.select_output_folder)
        self.output_btn.pack(fill="x", pady=5)
        self.output_btn.configure(state="disabled")

        self.start_btn = ctk.CTkButton(control_frame, text="开始预测", command=self.start_inference)
        self.start_btn.pack(fill="x", pady=5)
        self.start_btn.configure(state="disabled")

        self.toggle_mode_btn = ctk.CTkButton(control_frame, text="切换对比模式", command=self.toggle_display_mode)
        self.toggle_mode_btn.pack(fill="x", pady=5)
        self.toggle_mode_btn.configure(state="disabled")

        nav_frame = ctk.CTkFrame(control_frame)
        nav_frame.pack(fill="x", pady=10)
        self.prev_btn = ctk.CTkButton(nav_frame, text="←", command=self.prev_group, width=50)
        self.prev_btn.pack(side="left", padx=5)
        self.prev_btn.configure(state="disabled")
        self.next_btn = ctk.CTkButton(nav_frame, text="→", command=self.next_group, width=50)
        self.next_btn.pack(side="right", padx=5)
        self.next_btn.configure(state="disabled")

    def load_models(self):
        for model_name in self.models.keys():
            model_file = f"{model_name}_模型.pth"
            model, device = load_model(model_file, self.model_class, self.input_channels, self.hidden_units, self.num_layers)
            self.models[model_name] = model
        self.device = device
        self.log("模型加载成功!")

    def select_model(self, model):
        self.selected_model = model
        self.log(f"预测时间: {model}")
        self.input_btn.configure(state="normal")
        if self.input_folder:
            self.run_inference()

    def select_input_folder(self):
        self.input_folder = ctk.filedialog.askdirectory(title="Select Input Folder")
        if self.input_folder:
            self.load_input_images()
            self.input_btn.configure(state="disabled")
            self.output_btn.configure(state="normal")

    def select_output_folder(self):
        self.output_folder = ctk.filedialog.askdirectory(title="Select Output Folder")
        if self.output_folder:
            self.output_btn.configure(state="disabled")
            self.start_btn.configure(state="normal")

    def start_inference(self):
        self.start_btn.configure(state="disabled")
        self.prev_btn.configure(state="normal")
        self.next_btn.configure(state="normal")
        self.toggle_mode_btn.configure(state="normal")
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
                    npy_to_png(self.input_folder, log_callback=self.log)
                    converted_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'converted_images')
                png_file = file.replace('.npy', '.png')
                png_path = os.path.join(converted_images_folder, png_file)
                self.input_images.append(png_path)

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
        self.log_text.insert(ctk.END, message + "\n")
        self.log_text.see(ctk.END)

    def update_navigation_buttons(self):
        self.prev_btn.configure(state="normal" if self.current_group > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_group < len(self.input_images) - 1 else "disabled")

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

        input_images = [Image.open(img_path).convert('L') for img_path in current_input]
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        input_tensor = torch.stack([transform(img) for img in input_images]).unsqueeze(0)

        device = next(self.models[self.selected_model].parameters()).device
        input_tensor = input_tensor.to(device)

        model = self.models[self.selected_model]
        with torch.no_grad():
            output = model(input_tensor)

        output_array = output[0, 0].cpu().numpy().squeeze()
        output_path = os.path.join(self.output_folder, f"prediction_{self.current_group}.png")
        Image.fromarray((output_array * 255).astype(np.uint8), mode='L').save(output_path)

        colored_output = self.apply_color_mapping(output_array)
        colored_output_path = os.path.join(self.output_folder, f"colored_prediction_{self.current_group}.png")
        plt.imsave(colored_output_path, colored_output)

        if self.display_mode == 'overlay':
            overlay_image_path = self.create_overlay_image(current_input[-1], colored_output_path)
            self.display_image(overlay_image_path, self.output_image)
            self.input_image.configure(image=None)
        else:
            self.display_image(current_input[-1], self.input_image)
            self.display_image(colored_output_path, self.output_image)

        self.log(f"{self.current_group}预测完成! 预测图保存到: {self.output_folder}")

    def display_image(self, image_path, label):
        image = Image.open(image_path)
        image.thumbnail((800, 800))  # Resize image while maintaining aspect ratio
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo

    def apply_color_mapping(self, image):
        colors = [
            (46/255, 4/255, 10/255), (96/255, 23/255, 27/255), (197/255, 36/255, 47/255),
            (240/255, 51/255, 35/255), (244/255, 109/255, 45/255), (248/255, 179/255, 53/255),
            (231/255, 231/255, 79/255), (209/255, 223/255, 76/255), (134/255, 196/255, 63/255),
            (93/255, 188/255, 71/255), (54/255, 170/255, 70/255), (56/255, 167/255, 74/255),
            (28/255, 64/255, 90/255), (36/255, 65/255, 135/255), (36/255, 134/255, 176/255),
            (69/255, 196/255, 209/255), (123/255, 207/255, 209/255), (205/255, 205/255, 205/255),
            (190/255, 190/255, 190/255), (152/255, 152/255, 152/255), (96/255, 96/255, 96/255),
            (67/255, 67/255, 67/255)
        ]
        custom_cmap = LinearSegmentedColormap.from_list("Custom22", colors, N=22)
        norm = mcolors.Normalize(vmin=image.min(), vmax=image.max())
        mapped_image = custom_cmap(norm(image))
        return (mapped_image[:, :, :3] * 255).astype(np.uint8)

    def create_overlay_image(self, original_image_path, generated_image_path):
        original_image = Image.open(original_image_path).convert('RGBA')
        generated_image = Image.open(generated_image_path).convert('RGBA')

        generated_image = generated_image.resize(original_image.size, Image.LANCZOS)

        generated_array = np.array(generated_image)
        mask = (generated_array[:, :, 0] > 50) | (generated_array[:, :, 1] > 50) | (generated_array[:, :, 2] > 50)
        generated_array[~mask] = [0, 0, 0, 0]
        generated_array[mask, 3] = 90

        overlay_image = Image.alpha_composite(original_image, Image.fromarray(generated_array))
        overlay_path = os.path.join(self.output_folder, f"overlay_{self.current_group}.png")
        overlay_image.save(overlay_path)

        return overlay_path

if __name__ == '__main__':
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = InferenceGUI()
    app.mainloop()