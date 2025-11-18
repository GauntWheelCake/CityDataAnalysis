import folium
from folium.plugins import HeatMap
from datetime import datetime
import numpy as np
import math
from collections import defaultdict
from progress.bar import Bar
import os
import csv
import json
from pathlib import Path
from utils import *
from gen_functions import *


def create_map_data(samples):
    """
    将地图分割为1km×1km的网格，统计每个网格中出现过的不同人数

    参数:
    samples: 字典，包含所有轨迹数据

    返回:
    map_data: 字典，包含网格统计信息
    """

    # 地球半径（千米）
    EARTH_RADIUS_KM = 6371.0

    def lon_to_km(lon, lat):
        """将经度转换为千米"""
        return EARTH_RADIUS_KM * math.radians(lon) * math.cos(math.radians(lat))

    def lat_to_km(lat):
        """将纬度转换为千米"""
        return EARTH_RADIUS_KM * math.radians(lat)

    # 用于存储每个网格中不同用户的集合
    grid_people = defaultdict(set)

    # 遍历所有文件的数据
    for filename, traces in samples.items():
        for trace in traces:
            try:
                user_id = trace["id"]
                lon = float(trace["lo"])
                lat = float(trace["la"])

                # 将经纬度转换为千米
                x_km = lon_to_km(lon, lat)
                y_km = lat_to_km(lat)

                # 计算网格索引 (1km×1km网格)
                grid_x = int(x_km)  # 每1km一个网格
                grid_y = int(y_km)

                # 将用户ID添加到对应网格的集合中
                grid_key = (grid_x, grid_y)
                grid_people[grid_key].add(user_id)

            except (ValueError, KeyError) as e:
                print(f"数据格式错误: {e}, 跳过该记录")
                continue

    # 转换为最终的map_data格式
    map_data = {
        "grid_size_km": 1.0,
        "earth_radius_km": EARTH_RADIUS_KM,
        "grids": []
    }

    for (grid_x, grid_y), people_set in grid_people.items():
        grid_info = {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "people_count": len(people_set),
            "people_ids": list(people_set)  # 如果需要知道具体是哪些人
        }
        map_data["grids"].append(grid_info)

    # 按人数排序（可选）
    map_data["grids"].sort(key=lambda x: x["people_count"], reverse=True)

    print(f"总共统计了 {len(map_data['grids'])} 个网格")
    print(f"总人数: {sum(grid['people_count'] for grid in map_data['grids'])}")

    return map_data


def create_heatmap_folium(map_data, samples, output_file="heatmap.html"):
    """
    使用Folium创建交互式热力图
    """
    # 收集所有点的经纬度和权重
    heat_data = []

    for filename, traces in samples.items():
        for trace in traces:
            try:
                lon = float(trace["lo"])
                lat = float(trace["la"])
                heat_data.append([lat, lon, 1])  # [纬度, 经度, 权重]
            except (ValueError, KeyError):
                continue

    if not heat_data:
        print("没有有效数据生成热力图")
        return None

    # 计算中心点
    lats = [point[0] for point in heat_data]
    lons = [point[1] for point in heat_data]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # 创建地图
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # 添加热力图
    HeatMap(heat_data,
            min_opacity=0.2,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
            ).add_to(m)

    # 添加网格边界（可选）
    add_grid_boundaries(m, map_data)

    # 保存地图
    m.save(f"{dataset_path}\\{processed_path}\\{output_file}")
    print(f"热力图已保存为: {output_file}")
    return m


def add_grid_boundaries(map_obj, map_data):
    """
    在地图上添加网格边界（用于调试）
    """
    # 这里需要根据您的网格计算方式添加边界
    # 由于网格计算涉及复杂的地理转换，这里提供一个框架
    pass


def folium_main():
    six_six_samples = read_trace_files(six_six_path)
    if debugging_save_file:
        save_json(six_six_samples, six_six_path)

    map_data = create_map_data(six_six_samples)

    if debugging_save_file:
        save_json(map_data, f"{map_data_path}\\{six_six_path}")

    create_heatmap_folium(map_data, six_six_samples, f"{map_data_path}\\{six_six_path}.html")
