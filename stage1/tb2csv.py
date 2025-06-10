import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def convert_event_to_csv(event_file, output_csv,tags=["charts/episodic_return"]):
    """将单个 events 文件转换为 CSV 文件"""
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # tags = event_acc.Tags()["scalars"]

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Wall time", "Step", "Value"])  # 写入 CSV 头部

        for tag in tags:
            events = event_acc.Scalars(tag)
            for event in events:
                writer.writerow([event.wall_time, event.step, event.value])

def batch_convert_events_to_csv(folder_path, output_folder):
    """批量转换文件夹中的所有 events 文件为 CSV 文件"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 创建输出文件夹

    # 遍历文件夹中的所有文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file_path = os.path.join(root, file)
                
                # 提取父文件夹名称作为 CSV 文件名
                parent_folder_name = os.path.basename(os.path.dirname(event_file_path))
                output_csv_path = os.path.join(output_folder, f"{parent_folder_name}.csv")

                print(f"正在转换: {event_file_path} -> {output_csv_path}")
                convert_event_to_csv(event_file_path, output_csv_path)

    print("转换完成！")



# 设置输入文件夹和输出文件夹
input_folder = "./runs"  # 替换为你的 events 文件夹路径
output_folder = "./tbcsv"  # 替换为你想保存 CSV 文件的路径

# 批量转换
batch_convert_events_to_csv(input_folder, output_folder)