import os
import shutil
import re
from tqdm import tqdm

def get_sorted_files(directory):
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    return files

def ensure_divisible(directory, divisor):
    files = get_sorted_files(directory)
    while len(files) % divisor != 0:
        os.remove(os.path.join(directory, files.pop()))
    return files

def match_file_counts(dir1, dir2):
    files1 = get_sorted_files(dir1)
    files2 = get_sorted_files(dir2)
    
    while len(files1) > len(files2):
        os.remove(os.path.join(dir1, files1.pop()))
        files1 = get_sorted_files(dir1)
        
    while len(files2) > len(files1):
        os.remove(os.path.join(dir2, files2.pop()))
        files2 = get_sorted_files(dir2)

def rename_files_sequentially(directory):
    files = get_sorted_files(directory)
    for i, file in tqdm(enumerate(files, 1), desc=f"Renaming files in {directory}"):
        ext = os.path.splitext(file)[1]
        new_name = f"{i}{ext}"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

def adjust_time_interval(directory, interval):
    files = get_sorted_files(directory)
    for i in range(len(files) - 1, -1, -1):
        if i % interval != 0:
            os.remove(os.path.join(directory, files[i]))

def main(source_dir):
    # 询问用户希望的时间间隔
    print("请选择图像时间间隔：")
    print("1 = 10分钟")
    print("3 = 30分钟")
    print("6 = 1小时")
    print("12 = 2小时")
    print("18 = 3小时")
    print("30 = 5小时")
    time_interval = int(input("请输入对应的数字: "))

    # 询问用户希望的序列长度
    sequence_length = int(input("\n请输入序列长度 (sequence_length): "))

    # 创建目标文件夹
    for folder in ['X_train', 'x_test', 'y_true', 'y_test', 'y_pred']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 获取并排序源目录中的文件
    files = get_sorted_files(source_dir)

    # 计算训练集和测试集的分界点
    split_index = int(len(files) * 0.8)

    # 将文件复制到X_train和x_test文件夹
    for i, file in tqdm(enumerate(files), desc="Copying files"):
        if i < split_index:
            shutil.copy(os.path.join(source_dir, file), 'X_train')
        else:
            shutil.copy(os.path.join(source_dir, file), 'x_test')

    # 调整时间间隔
    for folder in ['X_train', 'x_test']:
        adjust_time_interval(folder, time_interval)

    # 为y_true和y_test准备标签
    for src_folder, target_folder in [('X_train', 'y_true'), ('x_test', 'y_test')]:
        src_files = get_sorted_files(src_folder)
        for i in tqdm(range(len(src_files) - sequence_length), desc=f"Preparing {target_folder}"):
            shutil.copy(os.path.join(src_folder, src_files[i + sequence_length]), 
                        os.path.join(target_folder, src_files[i]))

    # 删除X_train和x_test中多余的文件
    for folder in ['X_train', 'x_test']:
        files = get_sorted_files(folder)
        for file in tqdm(files[-sequence_length:], desc=f"Removing extra files from {folder}"):
            os.remove(os.path.join(folder, file))

    # 确保所有文件夹中的文件数量正确
    for folder in ['X_train', 'x_test', 'y_true', 'y_test']:
        ensure_divisible(folder, 1)  # 这里使用1是为了保留所有文件，但仍然调用函数以保持一致性

    # 使X_train和y_true文件夹中的文件数量相同
    match_file_counts('X_train', 'y_true')

    # 使x_test和y_test文件夹中的文件数量相同
    match_file_counts('x_test', 'y_test')

    # 按数字顺序重命名文件
    for folder in ['X_train', 'x_test', 'y_true', 'y_test']:
        rename_files_sequentially(folder)

    print(f"\n处理完成。")
    print(f"X_train 和 y_true 中的图像数量: {len(get_sorted_files('X_train'))}")
    print(f"x_test 和 y_test 中的图像数量: {len(get_sorted_files('x_test'))}")
    print(f"序列长度 (sequence_length): {sequence_length}")
    print(f"时间间隔: {time_interval * 10} 分钟")
    print(f"预测时间范围: {sequence_length * time_interval * 10} 分钟")

if __name__ == "__main__":
    source_directory = "converted_images"  # 替换为你的源文件夹路径
    main(source_directory)
    print("\n\n   数据处理完毕!\n\n")
