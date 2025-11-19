from datetime import datetime
from progress.bar import Bar
import os
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from utils import *


def read_trace_files(filepath):
    # sample = {"id": "", "lo": 0.0, "la": 0.0, "datetime": [], "dur": 0}
    samples = {}
    # 遍历文件夹中的所有文件
    for file_path in Path(f"{dataset_path}\\{trace_path}\\{filepath}").glob("*.csv"):
        file_name = file_path.stem  # 获取文件名（不含扩展名）
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file, fieldnames=trace_dataset_headers)
            samples[file_name] = []
            csv_data = list(csv_reader)
            # total_steps = len(csv_data)
            # bar = Bar("Processing", max=total_steps)
            for data_line in csv_data:
                # print(data_line)
                dt = datetime.strptime(data_line["time"], "%Y-%m-%d %H:%M:%S")
                sample = {
                    "id": data_line["id"],
                    "lo": data_line["lo"],
                    "la": data_line["la"],
                    "datetime": [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second],
                    "dur": data_line["dur"],
                }
                if debugging_read_file:
                    print(sample)
                samples[file_name].append(sample)
                # bar.next()
                if debugging_read_file:
                    print(samples)
            # bar.finish()
        if debugging_read_file or test_one_file:
            return samples
            # data_dict[file_name] = list(csv_reader)
    print("data read into dict")
    return samples


def save_json(data, filepath):
    with open(f"{processed_path}\\{filepath}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
