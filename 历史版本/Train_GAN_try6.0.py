import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import numpy as np
from tqdm.auto import tqdm, trange
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim
import torch.nn.functional as F

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
seed = 2345
set_seed(seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用的设备: {device}\n")

class ImageToImageDataset(Dataset):
    def __init__(self, data_dir, sequence_length, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        
        self.image_filenames = sorted([f for f in self.data_dir.glob('*.npy') if f.is_file()])

        print(f"读取到 {len(self.image_filenames)} 张图像.\n")

    def __len__(self):
        return len(self.image_filenames) - self.sequence_length - 2  # 减去推测未来1、2、3小时需要的帧数
    
    def __getitem__(self, idx):
        input_sequence = []
        for i in range(self.sequence_length):
            input_image_path = self.image_filenames[idx + i]
            input_image = np.load(input_image_path)  # 加载 .npy 图像
            input_image = Image.fromarray(input_image)  # 转换为 PIL 图像
            
            if self.transform:
                input_image = self.transform(input_image)
            else:
                input_image = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)  # 增加通道维度
            input_sequence.append(input_image)
        
        # 加载推测的三个时间点对应的标签
        label_1hr_path = self.image_filenames[idx + self.sequence_length]  # 未来1小时
        label_2hr_path = self.image_filenames[idx + self.sequence_length + 1]  # 未来2小时
        label_3hr_path = self.image_filenames[idx + self.sequence_length + 2]  # 未来3小时
        
        label_1hr = np.load(label_1hr_path)
        label_1hr = Image.fromarray(label_1hr)
        
        label_2hr = np.load(label_2hr_path)
        label_2hr = Image.fromarray(label_2hr)
        
        label_3hr = np.load(label_3hr_path)
        label_3hr = Image.fromarray(label_3hr)
        
        if self.transform:
            label_1hr = self.transform(label_1hr)
            label_2hr = self.transform(label_2hr)
            label_3hr = self.transform(label_3hr)
        else:
            label_1hr = torch.tensor(label_1hr, dtype=torch.float32).unsqueeze(0)
            label_2hr = torch.tensor(label_2hr, dtype=torch.float32).unsqueeze(0)
            label_3hr = torch.tensor(label_3hr, dtype=torch.float32).unsqueeze(0)
        
        input_sequence = torch.stack(input_sequence, dim=0)
        labels = torch.stack([label_1hr, label_2hr, label_3hr], dim=0)
        return input_sequence, labels

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
        batch_size, time_steps, _, height, width = x.size()
        
        h = [torch.zeros(batch_size, self.convlstm_cells[0].hidden_dim, height, width).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.convlstm_cells[0].hidden_dim, height, width).to(x.device) for _ in range(self.num_layers)]
        
        if time_steps < 3:
            raise ValueError(f"time_steps should be at least 3, but got {time_steps}")

        x_t = None  
        for t in range(time_steps):
            x_t = self.fusion(x[:, t])
            for l in range(self.num_layers):
                h[l], c[l] = self.convlstm_cells[l](x_t, (h[l], c[l]))
                x_t = h[l]

        # 解码器，预测接下来的三个时间点
        outputs = []
        for _ in range(3):  # 预测三个时间点
            for l in range(self.num_layers):
                h[l], c[l] = self.convlstm_cells[l](x_t, (h[l], c[l]))
                x_t = h[l]
            output = self.separation(x_t)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)  # 返回三个时间点的预测
    
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
        # 分别计算每个时间点的损失
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
    gen_output = generator(input_seq)
    fake_input = torch.cat([input_seq[:, -3:], gen_output], dim=1)
    
    real_input = torch.cat([input_seq[:, -3:], target_seq], dim=1)
    real_input = real_input.view(-1, *real_input.shape[2:])
    fake_input = fake_input.view(-1, *fake_input.shape[2:])

    discriminator.zero_grad()
    real_output = discriminator(real_input)
    fake_output = discriminator(fake_input.detach())
    d_loss = d_loss_fn(real_output, fake_output)
    d_loss.backward()
    d_optimizer.step()

    generator.zero_grad()
    g_loss = g_loss_fn(gen_output, target_seq)
    fake_output = discriminator(fake_input)
    adversarial_loss = d_loss_fn(fake_output, torch.ones_like(fake_output))
    total_g_loss = g_loss + 0.1 * adversarial_loss
    total_g_loss.backward()
    g_optimizer.step()

    return g_loss.item(), d_loss.item()

def validate(val_loader, generator, device, g_loss_fn):
    generator.eval()
    total_val_loss = 0
    total_psnr = 0
    total_rae = 0
    total_rmse = 0
    num_samples = 0

    val_loss_1hr = 0
    val_loss_2hr = 0
    val_loss_3hr = 0

    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            gen_output = generator(input_seq)
            
            val_loss = g_loss_fn(gen_output, target_seq)
            total_val_loss += val_loss.item()
            
            for t in range(gen_output.size(1)):  # 处理每个时间点
                gen_output_last = gen_output[:, t, :, :, :]
                target_image = target_seq[:, t, :, :, :]

                if t == 0:
                    val_loss_1hr += F.l1_loss(gen_output_last, target_image).item()
                elif t == 1:
                    val_loss_2hr += F.l1_loss(gen_output_last, target_image).item()
                elif t == 2:
                    val_loss_3hr += F.l1_loss(gen_output_last, target_image).item()

                for i in range(gen_output.size(0)):
                    output_img = gen_output_last[i].cpu().numpy().squeeze()
                    target_img = target_image[i].cpu().numpy().squeeze()

                    if output_img.ndim == 3:
                        output_img = output_img[0]  
                    if target_img.ndim == 3:
                        target_img = target_img[0]

                    psnr_value = psnr(target_img, output_img, data_range=1)
                    total_psnr += psnr_value

                    rae = np.sum(np.abs(target_img - output_img)) / np.sum(np.abs(target_img))
                    total_rae += rae

                    mse = np.mean((target_img - output_img) ** 2)
                    rmse = np.sqrt(mse)
                    total_rmse += rmse

                    num_samples += 1
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_psnr = total_psnr / num_samples
    avg_rae = total_rae / num_samples
    avg_rmse = total_rmse / num_samples
    
    return avg_val_loss, avg_psnr, avg_rae, avg_rmse, val_loss_1hr / len(val_loader), val_loss_2hr / len(val_loader), val_loss_3hr / len(val_loader)

data_dir = Path("data_c")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = ImageToImageDataset(data_dir=data_dir, sequence_length=6, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)

# 创建验证数据集和数据加载器
test_dataset = ImageToImageDataset(data_dir=data_dir, sequence_length=6, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

# 开始训练
num_epochs = 1
for epoch in trange(num_epochs, desc="Epochs"):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0, 0
    
    for input_seq, target_seq in tqdm(train_dataloader, desc="Training", leave=False):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        g_loss, d_loss = train_step(input_seq, target_seq)
        total_g_loss += g_loss
        total_d_loss += d_loss

    avg_g_loss = total_g_loss / len(train_dataloader)
    avg_d_loss = total_d_loss / len(train_dataloader)
    
    val_loss, val_psnr, val_rae, val_rmse, val_loss_1hr, val_loss_2hr, val_loss_3hr = validate(test_dataloader, generator, device, g_loss_fn)
    
    print(f"  训练轮次[{epoch+1}/{num_epochs}] | "
          f"生成器损失: {avg_g_loss:.4f} | 判别器损失: {avg_d_loss:.4f} | "
          f"验证集损失: {val_loss:.4f} | 峰值信噪比: {val_psnr:.4f} dB | "
          f"相对误差绝对值: {val_rae:.4f} | 均方根误差: {val_rmse:.4f} |"
          f" 1小时预测损失: {val_loss_1hr:.4f} | 2小时预测损失: {val_loss_2hr:.4f} | 3小时预测损失: {val_loss_3hr:.4f} |"
          f" 当前生成器学习率: {g_optimizer.param_groups[0]['lr']} |"
          f" 当前判别器学习率: {d_optimizer.param_groups[0]['lr']}")
    
    # 更新学习率
    g_scheduler.step(val_loss)
    d_scheduler.step(val_loss)

    # 每5个epoch保存一次模型
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        # 保存生成器模型
        gen_model_path = MODEL_PATH / f"生成器_气象云图预测模型_epoch_{epoch+1}.pth"
        print(f"\n生成器模型保存到: {gen_model_path}")
        torch.save(generator.state_dict(), gen_model_path)
        
        # 保存判别器模型
        disc_model_path = MODEL_PATH / f"判别器_气象云图预测模型_epoch_{epoch+1}.pth"
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
                for t in range(preds.size(1)):
                    save_image(preds[j, t], output_dir / f'epoch_{epoch+1}_batch_{i+1}_time_{t+1}_sample_{j+1}.jpg')





                    