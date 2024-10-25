import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_filenames = sorted([f for f in self.root_dir.glob('*.png') if f.is_file()])
        print(f"读取到 {len(self.image_filenames)} 张图像.")

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        return image, image_path.name

# 定义转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为256x256
    transforms.ToTensor(),           # 将PIL图像转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1, 1]范围
])

# 创建数据集
dataset = ImageDataset(root_dir="converted_images", transform=transform)

# 选取前12张图像
num_samples_to_show = 12
sample_indices = list(range(num_samples_to_show))

# 定义颜色
colors = [(46/255, 4/255, 10/255), 
         (96/255, 23/255, 27/255),
         (197/255, 36/255, 47/255),
         (240/255, 51/255, 35/255),
         (244/255, 109/255, 45/255),
         (248/255, 179/255, 53/255),
         (231/255, 231/255, 79/255),
         (209/255, 223/255, 76/255),
         (134/255, 196/255, 63/255),
         (93/255, 188/255, 71/255),
         (54/255, 170/255, 70/255),
         (56/255, 167/255, 74/255),
         (28/255, 64/255, 90/255),
         (36/255, 65/255, 135/255),
         (36/255, 134/255, 176/255),
         (69/255, 196/255, 209/255),
         (123/255, 207/255, 209/255),
         (205/255, 205/255, 205/255),
         (190/255, 190/255, 190/255),
         (152/255, 152/255, 152/255),
         (96/255, 96/255, 96/255),
         (67/255, 67/255, 67/255)]

# 创建颜色映射
custom_cmap_22 = LinearSegmentedColormap.from_list("Custom22", colors, N=22)
norm = mcolors.Normalize(vmin=-1, vmax=1)  # 根据你的数据调整vmin和vmax

# 创建图像显示
fig, axs = plt.subplots(3, 4, figsize=(15, 10))

for i, idx in enumerate(sample_indices):
    image, filename = dataset[idx]
    img_color = custom_cmap_22(norm(image.numpy().squeeze()))
    row, col = divmod(i, 4)
    axs[row, col].imshow(img_color)
    axs[row, col].axis('off')
    axs[row, col].set_title(filename, fontsize=10)

plt.tight_layout()
plt.show()