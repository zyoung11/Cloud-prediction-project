import sys
import os
from PyQt6.QtWidgets import QApplication
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap, Normalize
import shutil
import hashlib
import moviepy.editor as mpy
from datetime import datetime
import time
from flask import request
def load_input_images(files):
    input_images = []

    for file_path in sorted(files):
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            input_images.append(file_path)
            
    grouped_images = [input_images[i:i + 6] for i in range(0, len(input_images), 6)]

    if not grouped_images:
        print("未找到可用的图像")
    elif len(grouped_images[0]) < 6:
        print("少于6张图像无法预测")
    else:
        print(f"载入 {len(grouped_images)} 组图像")

    return grouped_images


def load_image(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = os.path.join(sys._MEIPASS, relative_path)
    else:
        base_path = relative_path
    return base_path


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


def clear_uploads_folder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_folder = os.path.join(script_dir, 'uploads')
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    print(f"uploads 文件夹路径: {uploads_folder}")

    if os.path.exists(uploads_folder) and os.path.isdir(uploads_folder):
        for filename in os.listdir(uploads_folder):
            file_path = os.path.join(uploads_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"已删除文件: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"已删除子文件夹: {file_path}")
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f"所有内容已从 {uploads_folder} 文件夹中删除")
    else:
        print("uploads 文件夹不存在或不是文件夹")

# 定义用户数据文件路径
USER_DATA_FILE = 'Login.txt'
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
IMATEST_FOLDER = 'imatest'

# 用户认证相关功能
def hash_user_data(username, password):
    user_data = username + password
    return hashlib.sha256(user_data.encode()).hexdigest()

def store_user_data(username, password):
    hashed_value = hash_user_data(username, password)
    with open(USER_DATA_FILE, 'a') as file:
        file.write(hashed_value + '\n')

def verify_user_data(username, password):
    hashed_value = hash_user_data(username, password)
    try:
        with open(USER_DATA_FILE, 'r') as file:
            stored_hashes = file.readlines()
        stored_hashes = [h.strip() for h in stored_hashes]
        return hashed_value in stored_hashes
    except FileNotFoundError:
        return False
    
'''



pip install pillow imageio[ffmpeg] moviepy




'''

def create_gif_and_video():
    # 定义路径
    upload_folder = 'uploads'
    colored_folder = 'outputs/colored'
    gif_folder = 'outputs/gif'
    mp4_folder = 'outputs/mp4'
    
    # 创建输出文件夹
    os.makedirs(gif_folder, exist_ok=True)
    os.makedirs(mp4_folder, exist_ok=True)
    
    # 获取并排序文件
    upload_files = sorted([f for f in os.listdir(upload_folder) if f.endswith('.png')])
    colored_files = sorted([f for f in os.listdir(colored_folder) if f.endswith('.png')])
    
    # 定义模型名称排序规则
    model_order = ['10min', '30min', '1h', '2h', '3h']
    
    # 按组处理
    num_groups = len(upload_files) // 6
    for group in range(num_groups):
        # 获取当前组的upload文件
        upload_group_files = upload_files[group * 6:(group + 1) * 6]
        upload_group_paths = [os.path.join(upload_folder, f) for f in upload_group_files]
        
        # 获取当前组的colored文件并按模型名称排序
        colored_group_files = [
            f for f in colored_files if f.endswith(f'_colored_prediction_{group}.png')
        ]
        colored_group_files.sort(key=lambda x: model_order.index(x.split('_')[0]))
        colored_group_paths = [os.path.join(colored_folder, f) for f in colored_group_files]
        
        # 确保 colored_group_paths 有五个文件（不同模型的预测结果）
        if len(colored_group_paths) != 5:
            print(f"Warning: Group {group} does not have 5 colored predictions, skipping.")
            continue
        
        # 合并路径
        all_images_paths = upload_group_paths + colored_group_paths
        
        # 读取图像并调整大小
        images = [Image.open(path).resize((256, 256)) for path in all_images_paths]
        
        # 创建GIF
        gif_path = os.path.join(gif_folder, f'Gif_{group}.gif')
        images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=500)
        
        # 创建MP4
        mp4_path = os.path.join(mp4_folder, f'Mp4_{group}.mp4')
        video_clip = mpy.ImageSequenceClip([np.array(img) for img in images], fps=2)
        video_clip.write_videofile(mp4_path, codec='libx264')
        
        print(f"Created GIF: {gif_path} and MP4: {mp4_path}")

def record_current_time():
    # 获取当前时间，并格式化为字符串
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 获取当前目录路径
    current_dir = os.getcwd()
    
    # 定义time.txt文件的路径
    time_txt_path = os.path.join(current_dir, 'time.txt')
    
    # 将当前时间写入time.txt文件
    with open(time_txt_path, 'w') as f:
        f.write(current_time)
    
    print(f"当前时间 {current_time} 已记录到 {time_txt_path}")


def move_folders_to_new_directory():
    # 获取当前脚本所在的目录
    current_dir = os.getcwd()
    
    # 定义outputs文件夹和四个子文件夹的路径
    outputs_dir = os.path.join(current_dir, 'outputs')
    folders_to_check = ['colored', 'gif', 'mp4', 'non_colored']
    folder_paths = [os.path.join(outputs_dir, folder) for folder in folders_to_check]
    
    # 检查四个文件夹是否存在
    if all(os.path.exists(folder_path) for folder_path in folder_paths):
        # 读取time.txt文件，获取新文件夹名称
        time_txt_path = os.path.join(current_dir, 'time.txt')
        if not os.path.exists(time_txt_path):
            print("time.txt 文件不存在")
            return
        
        with open(time_txt_path, 'r') as f:
            new_folder_name = f.read().strip()
        
        # 在outputs目录下创建一个新文件夹
        new_folder_path = os.path.join(outputs_dir, new_folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        
        # 将colored, gif, mp4, non_colored文件夹移动到新文件夹中
        for folder_path in folder_paths:
            folder_name = os.path.basename(folder_path)
            shutil.move(folder_path, os.path.join(new_folder_path, folder_name))
        
        print(f"文件夹已成功移动到 {new_folder_path}")
    else:
        print("输出文件夹中的某些文件夹不存在")


def npy_to_png(source_dir):
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 在当前目录下创建新的目标文件夹 'converted_images_temp'
    target_dir = os.path.join(current_dir, 'uploads')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有 .npy 文件
    npy_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npy')]
    for i, filename in enumerate(npy_files):  
        # 构建完整的文件路径
        npy_path = os.path.join(source_dir, filename)
        png_filename = filename.replace('.npy', '.png')
        png_path = os.path.join(target_dir, png_filename)

        # 加载 .npy 文件
        npy_data = np.load(npy_path)

        # 检查数据的形状并进行相应的处理
        if npy_data.ndim == 2:
            plt.imshow(npy_data, cmap='gray')
        elif npy_data.ndim == 3:
            if npy_data.shape[2] == 3:
                plt.imshow(npy_data)  # RGB 图像
            elif npy_data.shape[2] == 4:
                plt.imshow(npy_data)  # RGBA 图像
            else:
                print(f"Error: {filename} has an unsupported channel size ({npy_data.shape[2]}). Skipping...")
                continue
        else:
            print(f"Error: {filename} has an unsupported number of dimensions ({npy_data.ndim}). Skipping...")
            continue

        # 保存为 .png 图像
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 更新进度
        progress = (i + 1) / len(npy_files) * 100
        print(f"Converted {filename} to {png_filename}, Progress: {progress:.2f}%")
