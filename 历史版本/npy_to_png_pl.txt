import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm  # 导入 tqdm 用于进度条

def process_file(args):
    npy_path, target_dir = args
    try:
        # 加载 .npy 文件
        npy_data = np.load(npy_path)

        # 构建 .png 文件路径
        png_filename = os.path.basename(npy_path).replace('.npy', '.png')
        png_path = os.path.join(target_dir, png_filename)

        # 检查数据的形状并进行相应的处理
        if npy_data.ndim == 2:
            plt.imshow(npy_data, cmap='gray')
            print("灰度图像")
        elif npy_data.ndim == 3:
            if npy_data.shape[2] == 3:
                plt.imshow(npy_data)
                print("\nRGB图像\n")
            elif npy_data.shape[2] == 4:
                plt.imshow(npy_data)
                print("\nRGBA图像\n")
            else:
                print(f"Error: {npy_path} has an unsupported channel size ({npy_data.shape[2]}). Skipping...")
                return
        else:
            print(f"Error: {npy_path} has an unsupported number of dimensions ({npy_data.ndim}). Skipping...")
            return

        # 保存为 .png 图像
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Converted {os.path.basename(npy_path)} to {png_filename}")
    except Exception as e:
        print(f"Failed to process {npy_path}: {e}")

def npy_to_png(source_dir, num_workers):
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 在当前目录下创建新的目标文件夹 'converted_images'
    target_dir = os.path.join(current_dir, 'converted_images')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取所有 .npy 文件的路径
    npy_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.npy')]
    
    # 使用多进程处理，手动设置核心数
    with Pool(num_workers) as p:
        # 使用 tqdm 进度条包装 starmap 的输入参数
        list(tqdm(p.imap(process_file, [(npy_path, target_dir) for npy_path in npy_files]), total=len(npy_files)))

if __name__ == '__main__':
    # 假设图片文件夹名为 'npy_images'
    source_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_c_3')
    # 手动设置调用的核心数
    num_workers = 24  # 这里设置为24个核心
    npy_to_png(source_folder, num_workers)
