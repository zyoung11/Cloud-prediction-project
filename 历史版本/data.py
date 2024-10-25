import os
import shutil

def rename_images(directory):
    # 获取目录中的所有文件名，并按字母顺序排序
    files = sorted(os.listdir(directory))
    # 过滤出png文件
    png_files = [file for file in files if file.lower().endswith('.png')]

    # 重命名并移动文件
    for index, filename in enumerate(png_files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"{index}.png"
        new_path = os.path.join(directory, new_filename)
        
        # 移动并重命名文件
        shutil.move(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")
    
    # 检查文件数量是否为偶数
    if len(png_files) % 2 != 0:
        last_file_path = os.path.join(directory, f"{len(png_files)}.png")
        os.remove(last_file_path)
        print(f"Deleted {last_file_path}")
    return index

N = rename_images('png')
N = int(N)
N = N * 0.5
N = int(N)
'''
temp = N
temp = int(temp)
if N % 2 != 0:
    N -= 1
N = N * 0.5
N = int(N)
M = temp - N
'''
def make_folder():
    os.makedirs("X_train", exist_ok=True)
    os.makedirs("y_true", exist_ok=True)
    os.makedirs("y_test", exist_ok=True)
    os.makedirs("X_test", exist_ok=True)
    os.makedirs("y_pred", exist_ok=True)

make_folder()


def move_images(n, m):

    for i in range(1, n):
        if i % 2 != 0:
            source_path = os.path.join('png', f"{i}.png")
            destination_path = os.path.join("X_train", f"{i}.png")
            shutil.copy(source_path, destination_path)
        else:
            source_path = os.path.join('png', f"{i}.png")
            destination_path = os.path.join("y_true", f"{i}.png")
            shutil.copy(source_path, destination_path)

    for j in range(n, n+m):  # 使用不同的变量 j，并且从 n+1 开始
        if j % 2 != 0:
            source_path = os.path.join('png', f"{j}.png")
            destination_path = os.path.join("X_test", f"{j}.png")
            shutil.copy(source_path, destination_path)
        else:
            source_path = os.path.join('png', f"{j}.png")
            destination_path = os.path.join("y_test", f"{j}.png")
            shutil.copy(source_path, destination_path)

move_images(N, N)


def rename_images_2(directory):
    # 获取目录中的所有文件名，并按字母顺序排序
    files = sorted(os.listdir(directory))
    # 过滤出png文件
    png_files = [file for file in files if file.lower().endswith('.png')]

    # 重命名并移动文件
    for index, filename in enumerate(png_files, start=1000):
        old_path = os.path.join(directory, filename)
        new_filename = f"{index}.png"
        new_path = os.path.join(directory, new_filename)
        
        # 移动并重命名文件
        shutil.move(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")

rename_images_2('X_train')
rename_images_2('y_true')
rename_images_2('y_test')
rename_images_2('X_test')



