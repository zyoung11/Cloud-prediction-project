import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import random
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim
from pytorch_msssim import ssim as pytorch_ssim
import torch.nn.functional as F


class ImageToImageDataset(Dataset):
    def __init__(self, input_root_dir, label_root_dir, sequence_length=8, transform=None):
        self.input_root_dir = Path(input_root_dir)
        self.label_root_dir = Path(label_root_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        
        self.image_filenames = sorted([f for f in self.input_root_dir.glob('*') if f.is_file()])
        self.label_filenames = [self.label_root_dir / f.name for f in self.image_filenames]

    def __len__(self):
        return len(self.image_filenames) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        input_sequence = []
        for i in range(self.sequence_length):
            input_image_path = self.image_filenames[idx + i]
            input_image = Image.open(input_image_path).convert('L')  # 转换为灰度图
            if self.transform:
                input_image = self.transform(input_image)
            input_sequence.append(input_image)
        
        label_image_path = self.label_filenames[idx + self.sequence_length - 1]
        label_image = Image.open(label_image_path).convert('L')  # 转换为灰度图
        if self.transform:
            label_image = self.transform(label_image)
        
        input_sequence = torch.stack(input_sequence, dim=0)
        return input_sequence, label_image

# 定义转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为256x256
    transforms.ToTensor(),           # 将PIL图像转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1, 1]范围
])

# 创建训练和测试数据集
train_dataset = ImageToImageDataset(input_root_dir=Path("X_train"),
                                    label_root_dir=Path("y_true"),
                                    sequence_length=8,
                                    transform=transform)

test_dataset = ImageToImageDataset(input_root_dir=Path('X_test'),
                                   label_root_dir=Path('y_test'),
                                   sequence_length=8,
                                   transform=transform)

# 创建 DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


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
        # x shape: (batch, time_steps, channels, height, width)
        batch_size, time_steps, _, height, width = x.size()
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.convlstm_cells[0].hidden_dim, height, width).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.convlstm_cells[0].hidden_dim, height, width).to(x.device) for _ in range(self.num_layers)]
        
        # Encoder
        for t in range(time_steps - 3):
            x_t = self.fusion(x[:, t])
            for l in range(self.num_layers):
                h[l], c[l] = self.convlstm_cells[l](x_t, (h[l], c[l]))
                x_t = h[l]
        
        # Decoder
        outputs = []
        for _ in range(3):  # Predict next 3 hours
            for l in range(self.num_layers):
                h[l], c[l] = self.convlstm_cells[l](x_t, (h[l], c[l]))
                x_t = h[l]
            output = self.separation(x_t)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)
    
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        
        self.fc1 = nn.Linear(16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.conv4(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)
    

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

    def forward(self, pred, target):
        mae_loss = F.l1_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1, size_average=True)
        return mae_loss + ssim_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, real_output, fake_output):
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss
    

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
generator = Generator(input_channels=1, hidden_channels=64, num_layers=4).to(device)
discriminator = Discriminator(input_channels=1).to(device)

# 初始化优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.001)

# 初始化损失函数
g_loss_fn = GeneratorLoss().to(device)
d_loss_fn = DiscriminatorLoss().to(device)

# 学习率调度器
g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=5, verbose=True)


def train_step(input_seq, target_seq):
    # 生成器前向传播
    gen_output = generator(input_seq)
    fake_input = torch.cat([input_seq[:, -3:], gen_output], dim=1)
    
    # 训练判别器
    discriminator.zero_grad()
    real_output = discriminator(torch.cat([input_seq[:, -3:], target_seq], dim=1))
    fake_output = discriminator(fake_input.detach())
    d_loss = d_loss_fn(real_output, fake_output)
    d_loss.backward()
    d_optimizer.step()

    # 训练生成器
    generator.zero_grad()
    g_loss = g_loss_fn(gen_output, target_seq)
    fake_output = discriminator(fake_input)
    adversarial_loss = d_loss_fn(fake_output, torch.ones_like(fake_output))
    total_g_loss = g_loss + 0.1 * adversarial_loss  # 可以调整对抗损失的权重
    total_g_loss.backward()
    g_optimizer.step()

    return g_loss.item(), d_loss.item()

def validate(val_loader):
    generator.eval()
    total_val_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            gen_output = generator(input_seq)
            val_loss = g_loss_fn(gen_output, target_seq)
            total_val_loss += val_loss.item()
    return total_val_loss / len(val_loader)

# 训练循环
num_epochs = 70
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0, 0
    
    for input_seq, target_seq in train_dataloader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        g_loss, d_loss = train_step(input_seq, target_seq)
        total_g_loss += g_loss
        total_d_loss += d_loss

    avg_g_loss = total_g_loss / len(train_dataloader)
    avg_d_loss = total_d_loss / len(train_dataloader)
    
    val_loss = validate(test_dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 更新学习率
    g_scheduler.step(val_loss)
    d_scheduler.step(val_loss)

    # 每5个epoch保存一次模型
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

    # 生成和保存预测结果
    generator.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = generator(X)
            save_image(pred, f'y_pred/{i+1}.jpg')