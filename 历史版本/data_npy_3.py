import os
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import shutil

def preprocess_data(source_dir, train_ratio=0.8, sequence_length=6, forecast_hours=3):
    # 创建目标文件夹
    for folder in ['X_train', 'y_true', 'X_test', 'y_test', 'y_pred']:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # 获取所有.png文件并排序
    image_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.png')])
    total_images = len(image_files)

    # 计算训练集大小
    train_size = int(total_images * train_ratio)

    def process_sequence(start_idx, output_prefix):
        input_sequence = image_files[start_idx:start_idx+sequence_length]
        
        # 确保序列连续
        start_time = datetime.strptime(input_sequence[0][:-4], "%Y%m%d%H%M%S")
        if not all((datetime.strptime(f[:-4], "%Y%m%d%H%M%S") - start_time).total_seconds() / 600 == i 
                   for i, f in enumerate(input_sequence)):
            return False

        # 复制输入序列
        for i, file in tqdm(enumerate(input_sequence)):
            shutil.copy(os.path.join(source_dir, file), 
                        f'{output_prefix[0]}/{file}')

        # 复制目标
        for i in tqdm(range(forecast_hours)):
            target_idx = start_idx + sequence_length + i*6 - 1
            if target_idx < len(image_files):
                target_file = image_files[target_idx]
                shutil.copy(os.path.join(source_dir, target_file), 
                            f'{output_prefix[1]}/{target_file}')
            else:
                return False
        return True

    # 处理训练集
    train_count = 0
    for i in tqdm(range(0, train_size - sequence_length - forecast_hours * 6 + 1)):
        if process_sequence(i, ('X_train', 'y_true')):
            train_count += 1

    # 处理测试集
    test_count = 0
    for i in tqdm(range(train_size, total_images - sequence_length - forecast_hours * 6 + 1)):
        if process_sequence(i, ('X_test', 'y_test')):
            test_count += 1

    print(f"预处理完成! 训练样本: {train_count}, 测试样本: {test_count}")

# 使用示例
source_directory = 'data_png'  # 替换为您的源.png文件夹路径
preprocess_data(source_directory)