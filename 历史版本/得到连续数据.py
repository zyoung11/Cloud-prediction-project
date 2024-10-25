import os
from datetime import datetime, timedelta
import shutil

def find_missing_days_and_times(folder_path, interval_minutes=10):
    # 获取文件夹中所有.npy文件的文件名
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    # 将文件名解析为时间戳
    timestamps = [datetime.strptime(f[:-4], "%Y%m%d%H%M%S") for f in npy_files]

    # 获取所有天的集合
    all_days = set(timestamp.date() for timestamp in timestamps)

    # 生成2024年6月20日至2024年7月29日的所有天
    start_date = datetime(2024, 6, 20)
    end_date = datetime(2024, 7, 29)
    full_days = set(start_date.date() + timedelta(days=i) for i in range((end_date - start_date).days + 1))

    # 找出缺失的天
    missing_days = full_days - all_days

    # 找出某天缺失的时间点
    missing_times = {}
    for day in all_days:
        day_timestamps = sorted([timestamp for timestamp in timestamps if timestamp.date() == day])
        missing_times[day] = []

        for i in range(1, len(day_timestamps)):
            expected_time = day_timestamps[i-1] + timedelta(minutes=interval_minutes)
            while expected_time < day_timestamps[i]:
                missing_times[day].append(expected_time.strftime("%Y%m%d%H%M%S") + ".npy")
                expected_time += timedelta(minutes=interval_minutes)

    # 找出最长的连续时间段
    if not timestamps:
        return missing_days, missing_times, None

    longest_start = timestamps[0]
    longest_end = timestamps[0]
    current_start = timestamps[0]
    current_end = timestamps[0]
    skip_next_check = False

    for i in range(1, len(timestamps)):
        expected_time = current_end + timedelta(minutes=interval_minutes)
        
        if timestamps[i] == expected_time:
            current_end = timestamps[i]
            skip_next_check = False
        elif skip_next_check:
            current_start = timestamps[i]
            current_end = timestamps[i]
            skip_next_check = False
        elif timestamps[i] == expected_time + timedelta(minutes=interval_minutes):
            current_end = timestamps[i]
            skip_next_check = True
        else:
            if (current_end - current_start) > (longest_end - longest_start):
                longest_start = current_start
                longest_end = current_end
            current_start = timestamps[i]
            current_end = timestamps[i]
            skip_next_check = False

    # 检查最后一个连续段
    if (current_end - current_start) > (longest_end - longest_start):
        longest_start = current_start
        longest_end = current_end

    return missing_days, missing_times, (longest_start, longest_end)

def copy_and_fill_longest_period_files(folder_path, longest_period, target_folder, interval_minutes=10):
    if longest_period is None:
        print("没有找到连续的时间段。")
        return

    os.makedirs(target_folder, exist_ok=True)

    current_time = longest_period[0]
    end_time = longest_period[1]

    while current_time <= end_time:
        src_file = os.path.join(folder_path, f"{current_time.strftime('%Y%m%d%H%M%S')}.npy")
        if os.path.exists(src_file):
            shutil.copy(src_file, target_folder)
        else:
            # 如果文件缺失，复制下一个文件并重命名
            next_time = current_time + timedelta(minutes=interval_minutes)
            next_file = os.path.join(folder_path, f"{next_time.strftime('%Y%m%d%H%M%S')}.npy")
            if os.path.exists(next_file):
                dst_file = os.path.join(target_folder, f"{current_time.strftime('%Y%m%d%H%M%S')}.npy")
                shutil.copy(next_file, dst_file)

        current_time += timedelta(minutes=interval_minutes)

# 使用示例
folder_path = 'data_npy' 
target_folder = 'data_c'

missing_days, missing_times, longest_period = find_missing_days_and_times(folder_path)

if missing_days:
    print("缺失的天:")
    for day in sorted(missing_days):
        print(day)
else:
    print("没有缺失的天。")

print("\n缺失的时间点:")
for day, times in missing_times.items():
    if times:
        print(f"{day}:")
        for time in times:
            print(f"  {time}")

if longest_period:
    print(f"\n最长的连续时间段: 从 {longest_period[0]} 到 {longest_period[1]}")
    copy_and_fill_longest_period_files(folder_path, longest_period, target_folder)
    print(f"\n已将最长连续时间段的文件复制到 {target_folder} 文件夹中，并填补了缺失的文件。")
else:
    print("\n没有找到连续的时间段。")