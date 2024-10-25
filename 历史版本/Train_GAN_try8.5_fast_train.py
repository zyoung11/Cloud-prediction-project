import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import numpy as np
from tqdm.auto import tqdm
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_pytorch
import pandas as pd
import warnings
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
seed = 2345 # 2345第一轮就有画面
set_seed(seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(128)
print(f"\n使用的设备: {device}\n")

class ImageToImageDataset(Dataset):
    def __init__(self, input_root_dir, label_root_dir, sequence_length, transform=None):
        self.input_root_dir = Path(input_root_dir)
        self.label_root_dir = Path(label_root_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        
        # 打印路径信息，确保路径正确
        print(f"输入路径: {self.input_root_dir}")
        print(f"标签路径: {self.label_root_dir}\n")

        self.image_filenames = sorted([f for f in self.input_root_dir.glob('*.png') if f.is_file()])
        self.label_filenames = [self.label_root_dir / f.name for f in self.image_filenames]

        # 打印读取到的文件数量
        print(f"读取到 {len(self.image_filenames)}张 训练图像.")
        print(f"读取到 {len(self.label_filenames)}张 标签图像.\n")

    def __len__(self):
        return len(self.image_filenames) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        input_sequence = []
        for i in range(self.sequence_length):
            input_image_path = self.image_filenames[idx + i]
            input_image = Image.open(input_image_path).convert('L')  # 转换为灰度图

            # print(f"Reading image: {input_image_path}")  # 打印文件路径
            
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

batch_size = int(input("批大小(batch_size): "))
print("")
sequence_length = int(input("序列长度(sequence_length): "))
print("")

train_dataset = ImageToImageDataset(input_root_dir=Path("X_train"),
                                    label_root_dir=Path("y_true"),
                                    sequence_length=sequence_length,
                                    transform=transform)

test_dataset = ImageToImageDataset(input_root_dir=Path('X_test'),
                                   label_root_dir=Path('y_test'),
                                   sequence_length=sequence_length,
                                   transform=transform)


# 创建 DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class KernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(KernelAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        attn = self.conv(x)
        attn = self.bn(attn)
        attn = self.activation(attn)
        attn = F.softmax(attn, dim=1)
        return x * attn

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(kernel_size//2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SpatialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers):
        super(Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.global_step = 0

        self.input_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.kernel_attention = KernelAttention(hidden_channels, hidden_channels, kernel_size=3)
        self.temporal_convs = nn.ModuleList([TemporalConvLayer(hidden_channels, hidden_channels, kernel_size=3) for _ in range(num_layers)])
        self.spatial_convs = nn.ModuleList([SpatialConvLayer(hidden_channels, hidden_channels, kernel_size=3) for _ in range(num_layers)])
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=1, batch_first=True)
        self.output_conv = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)

    def forward(self, x):
        batch_size, seq_length, _, height, width = x.size()
        x = x.view(batch_size * seq_length, self.input_channels, height, width)
        
        x = self.input_conv(x)
        x = self.kernel_attention(x)
        
        for temp_conv, spat_conv in zip(self.temporal_convs, self.spatial_convs):
            x = temp_conv(x)
            x = spat_conv(x)
        
        x = x.view(batch_size, seq_length, self.hidden_channels, height, width)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(batch_size * height * width, seq_length, self.hidden_channels)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = x.view(batch_size, height, width, self.hidden_channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.output_conv(x)
        return x.view(batch_size, self.input_channels, height, width)  

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.conv4(x))
        x = self.dropout(x)
        x = self.conv5(x)
        return torch.sigmoid(x)

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

    def forward(self, pred, target):
        # 确保 pred 和 target 有相同的维度
        if pred.dim() == 5:
            pred = pred.squeeze(1)  # 移除额外的维度
        if target.dim() == 5:
            target = target.squeeze(1)  # 移除额外的维度
        
        mae_loss = F.l1_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1, size_average=True)
        return mae_loss + mse_loss + ssim_loss

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

def select_file(title):
    root = tk.Tk()
    #root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(title=title, filetypes=[("PyTorch Model", "*.pth")])
    return file_path

# 获取用户输入是否微调模型
fine_tune = input("是否微调模型? (Y/N): ").upper() == "Y"
print(" ")

if fine_tune:
    print("请选择预训练的生成器模型文件。")
    pre_trained_gen_path = select_file("选择生成器模型")
    if not pre_trained_gen_path:
        print("未选择生成器模型文件，退出程序。")
        exit()
    
    print("请选择预训练的判别器模型文件。")
    pre_trained_disc_path = select_file("选择判别器模型")
    if not pre_trained_disc_path:
        print("未选择判别器模型文件，退出程序。")
        exit()

    print(f"选择的生成器模型文件: {pre_trained_gen_path}")
    print(f"选择的判别器模型文件: {pre_trained_disc_path}\n")

    # 加载预训练模型
    generator = Generator(input_channels=1, hidden_channels=64, num_layers=4).to(device)
    discriminator = Discriminator(input_channels=1).to(device)
    generator.load_state_dict(torch.load(pre_trained_gen_path))
    discriminator.load_state_dict(torch.load(pre_trained_disc_path))
else:
    # 从头开始训练
    generator = Generator(input_channels=1, hidden_channels=64, num_layers=4).to(device)
    discriminator = Discriminator(input_channels=1).to(device)

# 设置学习率
lr = float(input("学习率(默认0.001): "))
print("")

# 初始化优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=lr)

# 初始化损失函数
g_loss_fn = GeneratorLoss().to(device)
d_loss_fn = DiscriminatorLoss().to(device)

# 学习率调度器
'''
g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
'''
# 学习率调度器
g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, 
                                                         mode='min', 
                                                         factor=0.8, 
                                                         patience=4, 
                                                         threshold=0.01, 
                                                         threshold_mode='rel', 
                                                         cooldown=3, 
                                                         min_lr=0.0001, 
                                                         eps=0.0001, 
                                                         verbose=True)

d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, 
                                                         mode='min', 
                                                         factor=0.8, 
                                                         patience=4, 
                                                         threshold=0.01, 
                                                         threshold_mode='rel', 
                                                         cooldown=3, 
                                                         min_lr=0.0001, 
                                                         eps=0.0001, 
                                                         verbose=True)


# 在设置学习率之后，添加这段代码
gen_updates = int(input("请输入生成器对判别器的更新比例(2:1|输入2): "))
print(f"设置成功：判别器每更新 1 次, 生成器更新 {gen_updates} 次.\n")

def train_step(input_seq, target_seq):
    global gen_updates

    # 生成器前向传播
    gen_output = generator(input_seq)
    
    # 准备判别器的输入
    fake_input = gen_output.detach()  # 使用生成的输出作为假样本
    real_input = target_seq  # 使用目标序列作为真实样本

    # 训练判别器
    discriminator.zero_grad()
    real_output = discriminator(real_input)
    fake_output = discriminator(fake_input)
    d_loss = d_loss_fn(real_output, fake_output)
    d_loss.backward()
    d_optimizer.step()

    # 训练生成器
    generator.zero_grad()
    g_loss = g_loss_fn(gen_output, target_seq)
    fake_output = discriminator(gen_output)
    adversarial_loss = d_loss_fn(fake_output, torch.ones_like(fake_output))
    total_g_loss = g_loss + 0.1 * adversarial_loss
    
    # 梯度累积
    (total_g_loss / gen_updates).backward()
    
    if generator.global_step % gen_updates == 0:
        g_optimizer.step()
        generator.zero_grad()
    
    generator.global_step += 1

    return g_loss.item(), d_loss.item()


def validate(val_loader, generator, device, g_loss_fn):
    generator.eval()
    total_val_loss = 0
    total_psnr = 0
    total_rae = 0
    total_rmse = 0
    total_ssim = 0
    total_ssim_mae = 0
    num_samples = 0

    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            gen_output = generator(input_seq)
            
            # 确保 gen_output 和 target_seq 的维度匹配
            if gen_output.dim() == 4 and target_seq.dim() == 3:
                target_seq = target_seq.unsqueeze(1)
            elif gen_output.dim() == 3 and target_seq.dim() == 3:
                gen_output = gen_output.unsqueeze(1)
            
            val_loss = g_loss_fn(gen_output, target_seq)
            total_val_loss += val_loss.item()
            
            # 移除批次维度，只保留单个图像
            gen_output = gen_output.squeeze(0)
            target_seq = target_seq.squeeze(0)
            
            for i in range(gen_output.size(0)):
                output_img = gen_output[i].cpu()
                target_img = target_seq[i].cpu()

                # 确保图像是 2D 的
                output_img = output_img.squeeze()
                target_img = target_img.squeeze()

                # 计算PSNR
                psnr_value = psnr(target_img.numpy(), output_img.numpy(), data_range=1)
                total_psnr += psnr_value

                # 计算RAE
                rae = torch.sum(torch.abs(target_img - output_img)) / torch.sum(torch.abs(target_img))
                total_rae += rae.item()

                # 计算RMSE
                mse = torch.mean((target_img - output_img) ** 2)
                rmse = torch.sqrt(mse)
                total_rmse += rmse.item()

                # 计算SSIM
                ssim_value = ssim_pytorch(output_img.unsqueeze(0).unsqueeze(0), 
                                          target_img.unsqueeze(0).unsqueeze(0), 
                                          data_range=1, size_average=True).item()
                total_ssim += ssim_value

                # 计算SSIM结合MAE
                mae = torch.mean(torch.abs(target_img - output_img))
                ssim_mae = (1 - ssim_value) + mae.item()
                total_ssim_mae += ssim_mae

                num_samples += 1
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_psnr = total_psnr / num_samples
    avg_rae = total_rae / num_samples
    avg_rmse = total_rmse / num_samples
    avg_ssim = total_ssim / num_samples
    avg_ssim_mae = total_ssim_mae / num_samples
    
    return avg_val_loss, avg_psnr, avg_rae, avg_rmse, avg_ssim, avg_ssim_mae


model_time_str = input("预测时间: ") # 获取用户输入
base_dir = Path(model_time_str) # 创建文件夹结构
model_dir = base_dir / "模型文件"
log_dir = base_dir / "训练日志"
gen_model_dir = model_dir / "生成器"
disc_model_dir = model_dir / "判别器"

# 创建所有必要的文件夹
for dir in [base_dir, model_dir, log_dir, gen_model_dir, disc_model_dir]:
    dir.mkdir(parents=True, exist_ok=True)
print(f"已创建文件夹结构: {base_dir}\n")

image_counter = 1  # 初始化图像计数器

results = {
    "avg_g_loss": [],
    "avg_d_loss": [],
    "val_loss": [],
    "val_psnr": [],
    "val_rae": [],
    "val_rmse": [],
    "val_ssim": [],
    "val_ssim_mae": [],
    "g_lr": [],
    "d_lr": []
}


# 训练循环
num_epochs = int(input("轮数: "))

print('''
        生成器损失: avg_g_loss   |  判别器损失: avg_d_loss  |  验证集损失: val_loss  |  峰值信噪比: val_psnr 
        相对误差绝对值: val_rae  |  均方误差: val_rmse      |  结构相似性: val_ssim  |  结构相似性结合平均误差绝对值: val_ssim_mae
        当前生成器学习率: g_lr   |  当前判别器学习率: d_lr
      ''')

print("\n 开始训练! \n")


# 在主训练循环之前
best_val_loss = float('inf')
patience = 10  # 设定早停的耐心参数，即在多少个epoch没有改善时停止训练
no_improve_epochs = 0

# 在训练循环中
for epoch in tqdm(range(1, num_epochs+1), initial=1, total=num_epochs, desc="Epochs", unit="轮", colour="cyan"):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0, 0

    for input_seq, target_seq in tqdm(train_dataloader, desc="运算量", leave=False, colour="magenta"):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        g_loss, d_loss = train_step(input_seq, target_seq)
        total_g_loss += g_loss
        total_d_loss += d_loss

    avg_g_loss = total_g_loss / len(train_dataloader)
    avg_d_loss = total_d_loss / len(train_dataloader)
    
    val_loss, val_psnr, val_rae, val_rmse, val_ssim, val_ssim_mae = validate(test_dataloader, generator, device, g_loss_fn)

    print("\n")

    results["avg_g_loss"].append(avg_g_loss)
    results["avg_d_loss"].append(avg_d_loss)
    results["val_loss"].append(val_loss)
    results["val_psnr"].append(val_psnr)
    results["val_rae"].append(val_rae)
    results["val_rmse"].append(val_rmse)
    results["val_ssim"].append(val_ssim)
    results["val_ssim_mae"].append(val_ssim_mae)
    results["g_lr"].append(g_optimizer.param_groups[0]["lr"])
    results["d_lr"].append(d_optimizer.param_groups[0]["lr"])

    results_df = pd.DataFrame(data=results, index=range(1, len(results["avg_g_loss"]) + 1))
    print(f"{results_df}\n")

    # 每5个epoch或在最后一个epoch保存DataFrame
    if epoch % 5 == 0 or epoch == num_epochs:
        file_path = log_dir / f'{model_time_str}_training_data_epochs_{epoch}.csv'
        results_df.to_csv(file_path, index=False)
        print(f"训练数据已保存到 {file_path}")

    # 早停机制：如果验证集损失改善，则保存模型；否则增加未改善epoch计数
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        # 保存最佳模型
        torch.save(generator.state_dict(), gen_model_dir / f"{model_time_str}_best_generator.pth")
        torch.save(discriminator.state_dict(), disc_model_dir / f"{model_time_str}_best_discriminator.pth")
        print(f"已保存最佳模型 (epoch {epoch})")
    else:
        no_improve_epochs += 1
        print(f"验证集损失未改善 (连续 {no_improve_epochs} 次)")

    # 如果超过耐心值，终止训练
    if no_improve_epochs >= patience:
        print("验证集损失未改善，训练提前终止。")
        break

    # 更新学习率
    g_scheduler.step(val_loss)
    d_scheduler.step(val_loss)

    # 每5个epoch保存一次模型
    if epoch % 5 == 0 or epoch == num_epochs:
        # 保存生成器模型
        gen_model_path = gen_model_dir / f"{model_time_str}_生成器_模型_epoch_{epoch}.pth"
        print(f"\n生成器模型保存到: {gen_model_path}")
        torch.save(generator.state_dict(), gen_model_path)
        
        # 保存判别器模型
        disc_model_path = disc_model_dir / f"{model_time_str}_判别器_模型_epoch_{epoch}.pth"
        print(f"判别器模型保存到: {disc_model_path}\n")
        torch.save(discriminator.state_dict(), disc_model_path)

    # 生成和保存预测结果
    generator.eval()
    output_dir = Path('y_pred')
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            preds = generator(X)
            for j in range(preds.size(0)):
                save_image(preds[j, 0], output_dir / f'{image_counter}.png')
                image_counter += 1  # 增加计数器
                
    if epoch == num_epochs:
        break