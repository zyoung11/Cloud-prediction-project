import os
import shutil
import re
from tqdm import tqdm

def get_sorted_files(directory):
    # 获取目录中的所有文件，并按文件名中的数字排序
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    return files

def ensure_divisible_by_six(directory):
    # 确保目录中的文件数量可以被6整除，如果不能则删除最后一个文件直到可以被6整除
    files = get_sorted_files(directory)
    while len(files) % 6 != 0:
        os.remove(os.path.join(directory, files.pop()))
    return files

def match_file_counts(dir1, dir2):
    # 使两个目录中的文件数量相同，删除多余的文件
    files1 = get_sorted_files(dir1)
    files2 = get_sorted_files(dir2)
    
    while len(files1) > len(files2):
        os.remove(os.path.join(dir1, files1.pop()))
        files1 = get_sorted_files(dir1)
        
    while len(files2) > len(files1):
        os.remove(os.path.join(dir2, files2.pop()))
        files2 = get_sorted_files(dir2)

def rename_files_sequentially(directory):
    # 按数字顺序重命名文件，从1开始
    files = get_sorted_files(directory)
    for i, file in tqdm(enumerate(files, 1)):
        ext = os.path.splitext(file)[1]
        new_name = f"{i}{ext}"
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

def main(source_dir):
    # 询问用户希望预测的时间范围
    print("请选择预测时间范围：")
    print("1 = 10分钟")
    print("3 = 30分钟")
    print("6 = 1小时")
    print("12 = 2小时")
    print("18 = 3小时")
    prediction_horizon = int(input("请输入对应的数字: "))

    # 创建目标文件夹
    for folder in tqdm(['X_train', 'x_test', 'y_true', 'y_test', 'y_pred']):
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 获取并排序源目录中的文件
    files = get_sorted_files(source_dir)

    # 计算训练集和测试集的分界点
    split_index = int(len(files) * 0.8)

    # 将文件复制到X_train和x_test文件夹
    for i, file in tqdm(enumerate(files)):
        if i < split_index:
            shutil.copy(os.path.join(source_dir, file), 'X_train')
        else:
            shutil.copy(os.path.join(source_dir, file), 'x_test')

    # 为y_true和y_test准备标签
    for src_folder, target_folder in tqdm([('X_train', 'y_true'), ('x_test', 'y_test')]):
        src_files = get_sorted_files(src_folder)
        for i in range(len(src_files) - prediction_horizon):
            shutil.copy(os.path.join(src_folder, src_files[i + prediction_horizon]), 
                        os.path.join(target_folder, src_files[i]))

    # 删除X_train和x_test中多余的文件
    for folder in ['X_train', 'x_test']:
        files = get_sorted_files(folder)
        for file in tqdm(files[-prediction_horizon:]):
            os.remove(os.path.join(folder, file))

    # 确保所有文件夹中的文件数量可以被6整除
    for folder in ['X_train', 'x_test', 'y_true', 'y_test']:
        ensure_divisible_by_six(folder)

    # 使X_train和y_true文件夹中的文件数量相同
    match_file_counts('X_train', 'y_true')

    # 使x_test和y_test文件夹中的文件数量相同
    match_file_counts('x_test', 'y_test')

    # 按数字顺序重命名文件
    for folder in ['X_train', 'x_test', 'y_true', 'y_test']:
        rename_files_sequentially(folder)

if __name__ == "__main__":
    source_directory = "converted_images"  # 替换为你的源文件夹路径
    main(source_directory)
    print("\n\n   数据处理完毕!\n\n")