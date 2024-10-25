import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def npy_to_png(source_dir):
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 在当前目录下创建新的目标文件夹 'converted_images'
    target_dir = os.path.join(current_dir, 'converted_images')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有 .npy 文件
    for filename in tqdm(os.listdir(source_dir)):
        if filename.endswith('.npy'):
            # 构建完整的文件路径
            npy_path = os.path.join(source_dir, filename)
            png_filename = filename.replace('.npy', '.png')
            png_path = os.path.join(target_dir, png_filename)

            # 加载 .npy 文件
            npy_data = np.load(npy_path)

            # 检查数据的形状并进行相应的处理
            if npy_data.ndim == 2:
                # 灰度图像
                plt.imshow(npy_data, cmap='gray')
                print("\n灰度图像\n")
            elif npy_data.ndim == 3:
                if npy_data.shape[2] == 3:
                    # RGB 图像
                    plt.imshow(npy_data)
                    print("\nRGB图像\n")
                elif npy_data.shape[2] == 4:
                    # RGBA 图像
                    plt.imshow(npy_data)
                    print("\nRGBA图像\n")
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

            print(f"Converted {filename} to {png_filename}")

if __name__ == '__main__':
    # 假设图片文件夹名为 'npy_images'
    source_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_c_3')
    npy_to_png(source_folder)