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
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pytorch_msssim import SSIM
import math
from pytorch_msssim import ssim as ssim_pytorch

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

# 获取用户输入是否微调模型
print(" ")
root = Tk()
root.withdraw()
fine_tune = input("是否微调模型? (Y/N): ").upper() == "Y"

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
    transforms.ToTensor(),           # 将PIL图像转换为Tensor，同时归一化到[0, 1]
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

class Diffusion(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, time_steps, dropout=0.1):
        super(Diffusion, self).__init__()
        self.time_steps = time_steps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        
        # U-Net like architecture
        self.down1 = UNetBlock(input_channels + hidden_channels, hidden_channels, dropout)
        self.down2 = UNetBlock(hidden_channels, hidden_channels * 2, dropout)
        self.down3 = UNetBlock(hidden_channels * 2, hidden_channels * 4, dropout)
        
        self.middle = UNetBlock(hidden_channels * 4, hidden_channels * 4, dropout)
        
        self.up3 = UNetBlock(hidden_channels * 8, hidden_channels * 2, dropout, is_up=True)
        self.up2 = UNetBlock(hidden_channels * 4, hidden_channels, dropout, is_up=True)
        self.up1 = UNetBlock(hidden_channels * 2, hidden_channels, dropout, is_up=True)
        
        self.out = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, t):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.time_embed(t)  # shape: (batch_size, hidden_channels)
        t_emb = t_emb.view(batch_size, -1, 1, 1).expand(-1, -1, height, width)
        
        # Process each frame in the sequence
        outputs = []
        for i in range(seq_len):
            x_frame = x[:, i]  # (batch_size, channels, height, width)
            
            # Combine frame with time embedding
            x_t = torch.cat([x_frame, t_emb], dim=1)
            
            # U-Net like processing
            x1 = self.down1(x_t)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            
            x = self.middle(x3)
            
            x3 = self._check_sizes(x, x3)
            x = self.up3(torch.cat([x, x3], dim=1))
            
            x2 = self._check_sizes(x, x2)
            x = self.up2(torch.cat([x, x2], dim=1))
            
            x1 = self._check_sizes(x, x1)
            x = self.up1(torch.cat([x, x1], dim=1))
            
            outputs.append(x)
        
        # Stack outputs along sequence dimension
        x = torch.stack(outputs, dim=1)
        
        # 取最后一帧，并进行上采样
        x = x[:, -1]  # 形状应该是 [batch_size, channels, height, width]
        x = self.upsample(x)  # 上采样到原始大小
        
        return self.out(x)  # 返回上采样后的最后一帧预测
    

    def _check_sizes(self, x, y):
        if x.size(2) != y.size(2) or x.size(3) != y.size(3):
            y = F.interpolate(y, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return y
    

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, is_up=False):
        super(UNetBlock, self).__init__()
        self.is_up = is_up
        if self.is_up:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        if not is_up:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        if self.is_up:
            x = self.up(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation(x)
        if not self.is_up:
            x = self.downsample(x)
        return x


def diffusion_loss_fn(predicted, target, loss_type='l1'):
    if loss_type == 'l1':
        return F.l1_loss(predicted, target)
    elif loss_type == 'l2':
        return F.mse_loss(predicted, target)
    elif loss_type == 'huber':
        return F.smooth_l1_loss(predicted, target)
    else:
        raise ValueError("Unsupported loss type. Choose 'l1', 'l2', or 'huber'.")



class GeneratorLoss(nn.Module):
    def __init__(self, loss_type='l1', ssim_weight=0.1):
        super(GeneratorLoss, self).__init__()
        self.loss_type = loss_type
        self.ssim_weight = ssim_weight
        self.ssim = SSIM(data_range=1, size_average=True, channel=1)

    def forward(self, predicted, target):
        # 确保 predicted 和 target 都是 4D 张量
        if predicted.dim() == 3:
            predicted = predicted.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 确保空间维度匹配
        if predicted.shape[2:] != target.shape[2:]:
            predicted = F.interpolate(predicted, size=target.shape[2:], mode='bilinear', align_corners=False)

        if self.loss_type == 'l1':
            main_loss = F.l1_loss(predicted, target)
        elif self.loss_type == 'l2':
            main_loss = F.mse_loss(predicted, target)
        elif self.loss_type == 'huber':
            main_loss = F.smooth_l1_loss(predicted, target)
        else:
            raise ValueError("Unsupported loss type. Choose 'l1', 'l2', or 'huber'.")

        # 计算SSIM损失
        ssim_loss = 1 - self.ssim(predicted, target)

        # 组合损失
        total_loss = main_loss + self.ssim_weight * ssim_loss

        return total_loss



# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if fine_tune:
    # 让用户选择预训练的生成器和判别器模型
    pre_trained_gen_path = askopenfilename(title="选择生成器模型")

    # 加载预训练模型
    generator = Diffusion(input_channels=1, hidden_channels=64, output_channels=1, time_steps=1000).to(device)
    generator.load_state_dict(torch.load(pre_trained_gen_path))
else:
    # 从头开始训练
    generator = Diffusion(input_channels=1, hidden_channels=64, output_channels=1, time_steps=1000).to(device)


# 设置学习率
try: 
    lr = float(input("学习率(默认0.0001): "))
except:
    lr = 0.0001
    print("输入错误, 已将学习率设置为默认值 | lr = 0.0001")
print("")

# 选择优化器类型
optimizer_type = input("选择优化器类型 (adam/adamw/sgd, 默认adam): ").lower() or "adam"

# 初始化优化器
if optimizer_type == "adam":
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
elif optimizer_type == "adamw":
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
elif optimizer_type == "sgd":
    g_optimizer = torch.optim.SGD(generator.parameters(), lr=lr, momentum=0.9)
else:
    print("不支持的优化器类型，使用默认的 Adam 优化器")
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))

print(f"使用的优化器: {type(g_optimizer).__name__}\n")


# 学习率调度器
g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, 
                                                         mode='min', 
                                                         factor=0.8, 
                                                         patience=4, 
                                                         threshold=0.001, 
                                                         threshold_mode='rel', 
                                                         cooldown=3, 
                                                         min_lr=0.0001, 
                                                         eps=0.0001, 
                                                         verbose=True)


def train_step(input_seq, target_seq, t):
    generator.zero_grad()
    
    # 确保输入序列的形状正确
    if input_seq.dim() == 4:
        input_seq = input_seq.unsqueeze(1)  # 添加序列维度
    
    # 生成器前向传播
    gen_output = generator(input_seq, t)
    
    # 确保目标序列和生成器输出都是 4D 张量 (B, C, H, W)
    if target_seq.dim() == 3:
        target_seq = target_seq.unsqueeze(1)
    if gen_output.dim() == 3:
        gen_output = gen_output.unsqueeze(1)
    
    # 打印形状以进行调试
    #print(f"Generator output shape: {gen_output.shape}")
    #print(f"Target shape: {target_seq.shape}")
    
    # 计算损失
    g_loss = g_loss_fn(gen_output, target_seq)
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()


# 初始化损失函数
loss_type = input("选择损失函数类型 (l1/l2/huber, 默认l1): ").lower() or "l1"
print(" ")
ssim_weight = float(input("输入SSIM损失的权重 (0-1之间, 默认0.1): ") or 0.1)
g_loss_fn = GeneratorLoss(loss_type=loss_type, ssim_weight=ssim_weight).to(device)
print(f"\n使用的损失函数: {loss_type}, SSIM权重: {ssim_weight}\n")


def validate(val_loader, generator, device, g_loss_fn):
    generator.eval()
    total_val_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            t = torch.zeros(input_seq.size(0), device=device)  # 使用t=0进行最终预测
            gen_output = generator(input_seq, t)
            
            # 确保目标序列和生成器输出都是 4D 张量 (B, C, H, W)
            if target_seq.dim() == 3:
                target_seq = target_seq.unsqueeze(1)
            if gen_output.dim() == 3:
                gen_output = gen_output.unsqueeze(1)
            
            # 使用 GeneratorLoss 计算损失
            val_loss = g_loss_fn(gen_output, target_seq)
            total_val_loss += val_loss.item()
            
            # 计算 PSNR
            mse = F.mse_loss(gen_output, target_seq, reduction='mean').item()
            psnr = 10 * math.log10(1 / mse)
            total_psnr += psnr
            
            # 计算 SSIM
            ssim_val = ssim_pytorch(gen_output, target_seq, data_range=1, size_average=True).item()
            total_ssim += ssim_val
            
            num_samples += 1
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_val_loss, avg_psnr, avg_ssim


model_time_str = input("预测时间: ") # 获取用户输入
base_dir = Path(model_time_str) # 创建文件夹结构
model_dir = base_dir / "模型文件"
log_dir = base_dir / "训练日志"
gen_model_dir = model_dir / "生成器"


# 创建所有必要的文件夹
for dir in [base_dir, model_dir, log_dir, gen_model_dir]:
    dir.mkdir(parents=True, exist_ok=True)
print(f"已创建文件夹结构: {base_dir}\n")

image_counter = 1  # 初始化图像计数器

results = {
    "avg_g_loss": [],
    "val_loss": [],
    "val_psnr": [],
    "val_ssim": [],
    "g_lr": [],
}


# 训练循环
num_epochs = int(input("轮数: "))

print('''
        生成器损失: avg_g_loss   |  判别器损失: avg_d_loss  |  验证集损失: val_loss  |  峰值信噪比: val_psnr 
        相对误差绝对值: val_rae  |  均方误差: val_rmse      |  结构相似性: val_ssim  |  结构相似性结合平均误差绝对值: val_ssim_mae
        当前生成器学习率: g_lr 
      ''')

print("\n 开始训练! \n")


# 在主训练循环之前
best_val_loss = float('inf')
patience = 20  # 设定早停的耐心参数，即在多少个epoch没有改善时停止训练
no_improve_epochs = 0

# 在训练循环中
for epoch in tqdm(range(1, num_epochs+1), initial=1, total=num_epochs, desc="Epochs", unit="轮", colour="cyan"):
    generator.train()
    total_g_loss = 0

    for input_seq, target_seq in tqdm(train_dataloader, desc="运算量", leave=False, colour="magenta"):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
        # 打印输入和目标的形状
        #print(f"Input shape: {input_seq.shape}, Target shape: {target_seq.shape}")
            
        # 生成随机时间步
        t = torch.randint(0, generator.time_steps, (input_seq.size(0),), device=device)
            
        # 训练步骤
        g_loss = train_step(input_seq, target_seq, t)
        total_g_loss += g_loss


    avg_g_loss = total_g_loss / len(train_dataloader)
    print(f"Epoch {epoch}, Average Generator Loss: {avg_g_loss:.4f}")

    # 验证
    try:
        val_loss, val_psnr, val_ssim = validate(test_dataloader, generator, device, g_loss_fn)
        print(f"Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
    except RuntimeError as e:
        print(f"Error during validation: {e}")
        val_loss, val_psnr, val_ssim = float('inf'), 0, 0

    print("\n")

    results["avg_g_loss"].append(avg_g_loss)
    results["val_loss"].append(val_loss)
    results["val_psnr"].append(val_psnr)
    results["val_ssim"].append(val_ssim)
    results["g_lr"].append(g_optimizer.param_groups[0]["lr"])

    # 确保所有列表长度一致
    min_length = min(len(v) for v in results.values())
    results = {k: v[:min_length] for k, v in results.items()}

    # 创建DataFrame
    results_df = pd.DataFrame(data=results, index=range(1, min_length + 1))
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

    # 每5个epoch保存一次模型
    if epoch % 5 == 0 or epoch == num_epochs:
        # 保存生成器模型
        gen_model_path = gen_model_dir / f"{model_time_str}_生成器_模型_epoch_{epoch}.pth"
        print(f"\n生成器模型保存到: {gen_model_path}")
        torch.save(generator.state_dict(), gen_model_path)

    # 生成和保存预测结果
    generator.eval()
    output_dir = Path('y_pred')
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            try:
                X, y = X.to(device), y.to(device)
                t = torch.zeros(X.size(0), device=device)  # 使用t=0进行最终预测
                preds = generator(X, t)
                for j in range(preds.size(0)):
                    save_image(preds[j, 0], output_dir / f'{image_counter}.png')
                    image_counter += 1  # 增加计数器
            except RuntimeError as e:
                print(f"Error during prediction: {e}")
                print(f"Input shape: {X.shape}, Target shape: {y.shape}")
                continue

    if epoch == num_epochs:
        break