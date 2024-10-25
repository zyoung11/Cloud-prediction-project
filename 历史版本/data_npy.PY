import os
import shutil
import glob


def copy_files(source_dir):
    # 确定目标文件夹的路径
    X_train_dir = os.path.join(os.getcwd(), 'X_train')
    y_true_dir = os.path.join(os.getcwd(), 'y_true')
    X_test_dir = os.path.join(os.getcwd(), 'X_test')
    y_test_dir = os.path.join(os.getcwd(), 'y_test')

    # 确保目标文件夹存在
    for dir_path in [X_train_dir, y_true_dir, X_test_dir, y_test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            os.makedirs("y_pred", exist_ok=True)

    # 获取源文件夹中的.npy文件并按数字顺序排序
    npy_files = sorted(glob.glob(os.path.join(source_dir, '*.npy')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # 当文件夹中还剩24个文件及更多时执行
    while len(npy_files) >= 24:
        # 文件1-6复制到X_train文件夹
        for i in range(6):
            shutil.copy(npy_files[i], X_train_dir)
        # 文件7-12复制到y_true文件夹
        for i in range(6, 12):
            shutil.copy(npy_files[i], y_true_dir)
        # 文件13-18复制到X_test文件夹
        for i in range(12, 18):
            shutil.copy(npy_files[i], X_test_dir)
        # 文件19-24复制到y_test文件夹
        for i in range(18, 24):
            shutil.copy(npy_files[i], y_test_dir)
        # 更新npy_files列表
        npy_files = npy_files[24:]

    npy_files_2 = sorted(glob.glob(os.path.join(X_test_dir, '*.npy')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))   
    while len(npy_files_2) >= 300:
        # 文件1-6移动到X_train文件夹
        for i in range(6):
            shutil.move(X_test_dir[i], X_train_dir)
        # 文件7-12移动到y_true文件夹
        for i in range(6, 12):
            shutil.move(y_test_dir[i], y_true_dir)     
        # 更新npy_files_4列表
        npy_files_2 = npy_files_2[300:]

def rename_files_in_order(source_dir):
    # 获取源文件夹中的.npy文件并按数字顺序排序
    npy_files = sorted(glob.glob(os.path.join(source_dir, '*.npy')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    # 按顺序重命名文件
    for i, file_path in enumerate(npy_files, start=1):
        new_name = f"{i}.npy"
        new_path = os.path.join(source_dir, new_name)
        os.rename(file_path, new_path)

if __name__ == "__main__":
    source_directory = os.path.join(os.getcwd(), 'npy_image')
    copy_files(source_directory)
    rename_files_in_order('X_train')
    rename_files_in_order('y_true')
    rename_files_in_order('X_test')
    rename_files_in_order('y_test')



