import cv2
import numpy as np

# 读取图片
image = cv2.imread('image.png')

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

# 保存结果
cv2.imwrite('processed_image.png', image)

# 读取图片
image = cv2.imread('processed_image.png')

# x 和 y 的坐标（去除网格后的区域），这些是图块的边界坐标
x_coords = [4, 90, 176, 262, 348, 434, 520, 606, 692, 778, image.shape[1]]
y_coords = [86, 172, 258, 344, 430, 516, 602, 688, 774, image.shape[0]]

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
        block = image[y_coords[i]:y_coords[i+1], x_coords[j]:x_coords[j+1]]
        
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

# 显示拼接后的图像
cv2.imshow('Final Image without Borders', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存拼接后的图像
cv2.imwrite('final_image_without_borders.png', final_image)
