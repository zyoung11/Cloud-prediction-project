import cv2
import os
import numpy as np
import shutil
from tqdm.auto import tqdm

# 文件夹路径
original_folder = 'image'
processed_temp_folder = 'processed_image_temp'
final_output_folder = 'processed_image'

# 创建文件夹（如果不存在）
if not os.path.exists(processed_temp_folder):
    os.makedirs(processed_temp_folder)

if not os.path.exists(final_output_folder):
    os.makedirs(final_output_folder)

# 获取“原图”文件夹中的所有图片
image_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.png') or f.endswith('.jpg')])

# 批量处理每张图片
for idx, image_file in tqdm(enumerate(image_files, 1), unit="张", colour="cyan"):
    # 读取图片
    image_path = os.path.join(original_folder, image_file)
    image = cv2.imread(image_path)

    # 获取图片的宽高
    height, width, _ = image.shape

    # 你提供的坐标，并将所有的 x 坐标向左偏移 1
    x_coords = [x - 1 for x in [4, 5, 90, 91, 176, 177, 262, 263, 348, 349, 434, 435, 520, 521, 606, 607, 692, 693, 778, 779]]
    y_coords = [86, 172, 258, 344, 430, 516, 602, 688, 774, 860]

    # 遍历x坐标，将相应的垂直线设为背景色（假设背景为黑色）
    for x in x_coords:
        if x >= 0 and x < width:  # 确保坐标在图片范围内
            image[:, x] = 0  # 将该列所有像素设为黑色

    # 遍历y坐标，将相应的水平线设为背景色
    for y in y_coords:
        if y < height:  # 确保坐标在图片范围内
            image[y, :] = 0  # 将该行所有像素设为黑色

    # 保存临时处理的图片
    temp_image_path = os.path.join(processed_temp_folder, f'{idx}.png')
    cv2.imwrite(temp_image_path, image)

    # 读取临时处理后的图片
    processed_image = cv2.imread(temp_image_path)

    # x 和 y 的坐标（去除网格后的区域），这些是图块的边界坐标
    x_coords = [4, 90, 176, 262, 348, 434, 520, 606, 692, 778, processed_image.shape[1]]
    y_coords = [86, 172, 258, 344, 430, 516, 602, 688, 774, processed_image.shape[0]]

    # 计算最终拼接图像的大小（每个块去掉1个像素的边缘）
    final_image_height = sum([(y_coords[i+1] - y_coords[i] - 2) for i in range(len(y_coords) - 1)])
    final_image_width = sum([(x_coords[i+1] - x_coords[i] - 2) for i in range(len(x_coords) - 1)])
    final_image = np.zeros((final_image_height, final_image_width, 3), dtype=np.uint8)

    # 当前放置块的初始坐标
    current_y = 0

    # 遍历y坐标提取图片块并去掉边缘像素
    for i in range(len(y_coords) - 1):
        current_x = 0
        for j in range(len(x_coords) - 1):
            # 提取每个图块
            block = processed_image[y_coords[i]:y_coords[i+1], x_coords[j]:x_coords[j+1]]
            
            # 删除上下左右各1个像素的边缘
            block_cropped = block[1:-1, 1:-1]
            
            # 将图块放置到最终图像中
            block_height = block_cropped.shape[0]
            block_width = block_cropped.shape[1]
            final_image[current_y:current_y + block_height, current_x:current_x + block_width] = block_cropped
            
            # 更新x坐标
            current_x += block_width
        
        # 更新y坐标
        current_y += block_height

    # 将最终图像转换为灰度图
    gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

    # 对灰度图进行反转（黑白对调）
    inverted_gray_image = cv2.bitwise_not(gray_image)

    # 保存反转后的灰度图
    final_gray_image_path = os.path.join(final_output_folder, f'{idx}_inverted_gray.png')
    cv2.imwrite(final_gray_image_path, inverted_gray_image)

# 删除“processed_image(临时文件)”文件夹及其内容
shutil.rmtree(processed_temp_folder)

print("所有图片处理完成并保存到 '处理图像' 文件夹中。")
