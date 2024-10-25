import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from torchvision.utils import save_image
from pytorch_msssim import ssim as pytorch_ssim

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\n设备: {device}\n\n")

class ImageToImageDataset(Dataset):
    def __init__(self, input_root_dir, label_root_dir, transform=None):
        self.input_root_dir = Path(input_root_dir)
        self.label_root_dir = Path(label_root_dir)
        self.transform = transform
        
        self.image_filenames = sorted([f for f in self.input_root_dir.glob('*') if f.is_file()])
        self.label_filenames = [self.label_root_dir / f.name for f in self.image_filenames]

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        input_image_path = self.image_filenames[idx]
        label_image_path = self.label_filenames[idx]
        
        input_image = Image.open(input_image_path).convert('L')  # L灰度, 或者RGB
        label_image = Image.open(label_image_path).convert('L')
        
        if self.transform:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)
        
        return input_image, label_image

# 指定训练和测试数据集的根目录
input_root_dir = Path("X_train")
label_root_dir = Path("y_true")

# 定义转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor()           # 将PIL图像转换为Tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 创建训练和测试数据集
train_dataset = ImageToImageDataset(input_root_dir=input_root_dir,
                                    label_root_dir=label_root_dir,
                                    transform=transform)

test_dataset = ImageToImageDataset(input_root_dir=Path('X_test'),
                                   label_root_dir=Path('y_test'),
                                   transform=transform)

# 创建 DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f'''\nNumber of samples in the dataset: {len(train_dataset)}\n''')

# 定义生成器(编码-解码结构)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # ConvLSTM(长短期记忆模型)
        self.convlstm = nn.LSTM(256, 256, batch_first=True)

        # 分离模块
        self.separation = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fusion(x)
        # 将输入调整为LSTM所需的3D tensor (batch_size, sequence_length, input_size)
        x = x.view(x.size(0), -1, 256).contiguous() # [batch_size, sequence_length, input_size]
        x, _ = self.convlstm(x)
        # 重新调整为与后续卷积层兼容的维度
        x = x.contiguous().view(x.size(0), 256, 256, 256)
        x = self.separation(x)
        return x


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, dim=(2, 3))
        return x   #.view(x.size(0), -1)  # 展平成 [batch_size, 1]



# 初始化生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定义损失函数和优化器
BCE_loss = nn.BCELoss()
MAE_loss = nn.L1Loss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.001)

def train(generator, discriminator, train_dataloader, optimizer_G, optimizer_D, BCE_loss, MAE_loss, epochs):
    for epoch in tqdm(range(epochs)):
        for i, (imgs, labels) in enumerate(train_dataloader):
            valid = torch.ones((imgs.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((imgs.size(0), 1), requires_grad=False).to(device)

            real_imgs = labels.to(device)
            z = imgs.to(device)

            # 判别器前向传播
            optimizer_D.zero_grad()
            real_output = discriminator(real_imgs)
            print(f"   Real Output Shape: {real_output.shape}, Valid Shape: {valid.shape}")  # 添加这行调试

            real_loss = BCE_loss(real_output, valid)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs.detach())
            print(f"   Fake Output Shape: {fake_output.shape}, Fake Shape: {fake.shape}")  # 添加这行调试

            fake_loss = BCE_loss(fake_output, fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # 生成器前向传播
            optimizer_G.zero_grad()
            g_loss = BCE_loss(discriminator(fake_imgs), valid) + MAE_loss(fake_imgs, real_imgs)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch}/{epochs}, D loss: {d_loss.item()}, G loss: {g_loss.item()}")


# 开始训练
train(generator, discriminator, train_dataloader, optimizer_G, optimizer_D, BCE_loss, MAE_loss, epochs=70)

# 保存模型
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

# 生成和保存预测结果
generator.eval()
with torch.no_grad():
    for i, (X, y) in enumerate(test_dataloader):
        X = X.to(device)
        pred = generator(X)
        save_image(pred, f'y_pred/{i+1}.jpg')

print("Predictions saved in y_pred folder.")


'''




1. nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
含义:
nn.Conv2d:这是一个二维卷积层,用于提取输入图像中的空间特征。它接受一个多通道输入并生成多通道输出。
1:表示输入通道数。在这个例子中,输入图像是单通道的(灰度图像)。
64:表示输出通道数。这意味着卷积操作将生成64个特征映射(feature maps)。
kernel_size=3:卷积核的大小是3x3。卷积核是一个小矩阵,在输入图像上滑动以提取特征。
stride=1:卷积核每次移动的步长为1,这意味着卷积核在输入图像上每次移动一个像素。
padding=1:在输入图像的边缘添加1个像素的填充,以保持输出特征图的大小与输入相同。即使卷积核在边缘上滑动,也能覆盖整个输入图像。
作用: 
这个卷积层会接受一个单通道的输入图像,并生成64个特征图。每个特征图捕捉到输入图像中的不同空间特征。

2. nn.LeakyReLU(0.2, inplace=True)
含义:
nn.LeakyReLU:这是一个激活函数。激活函数引入非线性,使得神经网络可以处理更复杂的数据模式。LeakyReLU是ReLU(Rectified Linear Unit)的一种变体。
0.2:表示负斜率系数。当输入值为负时,LeakyReLU会允许它通过,但乘以0.2的系数。这解决了即,ReLU输出为0时梯度消失的问题。
inplace=True:表示直接在输入上进行操作,节省内存。
作用: 
这个激活函数将被应用于卷积层的输出,以引入非线性。如果输入值为正,它会原样通过；如果为负,它会被乘以0.2。

3. nn.ConvLSTM2d(256, 256, kernel_size=3, stride=1, padding=1)
含义:
nn.ConvLSTM2d:这是一个二维卷积LSTM层。它结合了卷积操作和LSTM(长短期记忆)网络,用于处理带有时序信息的空间数据。它通常用于时空序列预测。
256:输入和输出通道数都是256。即,输入的特征图和输出的特征图的通道数保持不变。
kernel_size=3:卷积核的大小为3x3。
stride=1:步长为1。
padding=1:在输入特征图的边缘填充1个像素,确保输出大小与输入相同。
作用: 
这个卷积LSTM层将空间特征与时间信息结合起来处理。这对于时序数据的处理特别有效,比如在连续多帧的图像预测中。

4. nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
含义:
nn.ConvTranspose2d:这是一个转置卷积(反卷积)层,通常用于图像上采样,即增加图像的分辨率。
256:输入特征图的通道数为256。
128:输出特征图的通道数为128。
kernel_size=4:卷积核的大小为4x4。
stride=2:步长为2。这意味着输出的特征图的大小是输入的两倍。
padding=1:在输入特征图的边缘填充1个像素,使得输出的特征图尺寸更大。
作用: 
这个转置卷积层用于上采样,将低分辨率的特征图转换为更高分辨率的特征图。它在解码器部分特别常见,用于恢复原始图像大小。

5. nn.Tanh()
含义:
nn.Tanh:这是一个双曲正切激活函数。它将输出值压缩到y=[-1, 1]的范围内。
作用: 
这个激活函数通常用于生成器的输出层,确保生成的图像像素值在y=[-1, 1]之间。结合输入图像的预处理(如标准化到相同范围),它能有效地生成逼真的图像。

总结:
nn.Conv2d 用于提取空间特征。
nn.LeakyReLU 引入非线性,使网络更能适应复杂的数据模式。
nn.ConvLSTM2d 结合时空信息,适合时序数据预测。
nn.ConvTranspose2d 用于图像上采样,恢复图像分辨率。
nn.Tanh 压缩输出值,确保生成图像的像素值在特定范围内。







'''