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
from ..utils import *
from ..gen_functions import *


def create_map_data(samples, grid_size_km=1.0):
    """
    将地图分割为指定边长的网格，统计每个网格中出现过的不同人数
    """
    # 地球半径（千米）
    EARTH_RADIUS_KM = 6371.0

    def lon_to_km(lon, lat):
        """将经度转换为千米"""
        return EARTH_RADIUS_KM * math.radians(lon) * math.cos(math.radians(lat))

    def lat_to_km(lat):
        """将纬度转换为千米"""
        return EARTH_RADIUS_KM * math.radians(lat)

    def km_to_lon(x_km, y_km):
        """将千米转换为经度"""
        lat = math.degrees(y_km / EARTH_RADIUS_KM)
        lon = math.degrees(x_km / (EARTH_RADIUS_KM * math.cos(math.radians(lat))))
        return lon, lat

    def km_to_lat(y_km):
        """将千米转换为纬度"""
        return math.degrees(y_km / EARTH_RADIUS_KM)

    # 用于存储每个网格中不同用户的集合
    grid_people = defaultdict(set)

    # 存储经纬度范围
    min_lon, max_lon = float('inf'), float('-inf')
    min_lat, max_lat = float('inf'), float('-inf')

    # 遍历所有文件的数据
    for filename, traces in samples.items():
        for trace in traces:
            try:
                user_id = trace["id"]
                lon = float(trace["lo"])
                lat = float(trace["la"])

                # 更新经纬度范围
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)

                # 将经纬度转换为千米
                x_km = lon_to_km(lon, lat)
                y_km = lat_to_km(lat)

                # 计算网格索引 (grid_size_km×grid_size_km网格)
                grid_x = int(x_km / grid_size_km)  # 每grid_size_km千米一个网格
                grid_y = int(y_km / grid_size_km)

                # 将用户ID添加到对应网格的集合中
                grid_key = (grid_x, grid_y)
                grid_people[grid_key].add(user_id)

            except (ValueError, KeyError) as e:
                print(f"数据格式错误: {e}, 跳过该记录")
                continue

    # 转换为最终的map_data格式
    map_data = {
        "grid_size_km": grid_size_km,
        "earth_radius_km": EARTH_RADIUS_KM,
        "geo_bounds": {
            "min_lon": min_lon,
            "max_lon": max_lon,
            "min_lat": min_lat,
            "max_lat": max_lat
        },
        "grids": []
    }

    for (grid_x, grid_y), people_set in grid_people.items():
        # 将网格坐标转换回经纬度（网格中心点）
        center_x_km = (grid_x + 0.5) * grid_size_km
        center_y_km = (grid_y + 0.5) * grid_size_km
        center_lon, center_lat = km_to_lon(center_x_km, center_y_km)

        grid_info = {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "people_count": len(people_set),
            "people_ids": list(people_set),  # 如果需要知道具体是哪些人
            "center_lon": center_lon,
            "center_lat": center_lat
        }
        map_data["grids"].append(grid_info)

    # 按人数排序（可选）
    map_data["grids"].sort(key=lambda x: x["people_count"], reverse=True)

    print(f"网格边长: {grid_size_km}km")
    print(f"总共统计了 {len(map_data['grids'])} 个网格")
    print(f"总人数: {sum(grid['people_count'] for grid in map_data['grids'])}")
    print(f"经纬度范围: 经度[{min_lon:.4f}, {max_lon:.4f}], 纬度[{min_lat:.4f}, {max_lat:.4f}]")

    return map_data


def create_heatmap_matplotlib(map_data, output_file="heatmap.png", use_log_norm=True):
    """
    使用Matplotlib创建静态热力图（带中国地图轮廓）
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    except ImportError:
        print("警告: 未安装cartopy库，无法显示地图轮廓")
        print("请安装: pip install cartopy")
        return create_heatmap_simple(map_data, output_file, use_log_norm)

    # 提取网格数据
    grid_data = {}
    for grid in map_data["grids"]:
        grid_x = grid["grid_x"]
        grid_y = grid["grid_y"]
        count = grid["people_count"]
        grid_data[(grid_x, grid_y)] = count

    if not grid_data:
        print("没有网格数据可显示")
        return

    # 创建网格矩阵
    x_coords = [coord[0] for coord in grid_data.keys()]
    y_coords = [coord[1] for coord in grid_data.keys()]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # 创建空的网格矩阵
    grid_matrix = np.zeros((max_y - min_y + 1, max_x - min_x + 1))

    # 填充数据
    for (x, y), count in grid_data.items():
        grid_matrix[y - min_y, x - min_x] = count

    # 获取网格大小
    grid_size_km = map_data.get("grid_size_km", 1.0)

    # 获取地理边界
    geo_bounds = map_data.get("geo_bounds", {})
    if not geo_bounds:
        print("警告: 没有地理边界信息，使用默认中国范围")
        # 中国大致范围
        lon_min, lon_max = 73, 135
        lat_min, lat_max = 18, 54
    else:
        lon_min, lon_max = geo_bounds["min_lon"], geo_bounds["max_lon"]
        lat_min, lat_max = geo_bounds["min_lat"], geo_bounds["max_lat"]

    # 扩展边界以便更好地显示
    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2
    lon_range = max(lon_max - lon_min, 5)  # 最小5度
    lat_range = max(lat_max - lat_min, 5)  # 最小5度

    # 创建带地图投影的图形
    fig = plt.figure(figsize=(12, 8))

    # 使用Plate Carree投影（简单的经纬度投影）
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置地图范围
    ax.set_extent([lon_center - lon_range / 1.5, lon_center + lon_range / 1.5,
                   lat_center - lat_range / 1.5, lat_center + lat_range / 1.5],
                  crs=ccrs.PlateCarree())

    # 添加地图特征
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # 添加中国省份边界
    try:
        # 尝试添加更详细的中国边界
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='gray')
    except:
        pass

    # 创建自定义颜色映射
    colors = ['darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    cmap = LinearSegmentedColormap.from_list('custom_heat', colors, N=256)

    # 将网格数据转换为经纬度坐标
    lons = []
    lats = []
    values = []

    for grid in map_data["grids"]:
        lons.append(grid["center_lon"])
        lats.append(grid["center_lat"])
        values.append(grid["people_count"])

    # 创建散点图热力图
    scatter = ax.scatter(lons, lats, c=values, cmap=cmap,
                         s=50, alpha=0.7, transform=ccrs.PlateCarree(),
                         norm=LogNorm() if use_log_norm else None)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('People Count (Log Scale)' if use_log_norm else 'People Count')

    # 添加网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.title(f'Population Distribution Heatmap ({grid_size_km}km×{grid_size_km}km grid)')

    # 保存图片
    plt.tight_layout()
    plt.savefig(f"{processed_path}\\{output_file}", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"带地图轮廓的热力图已保存为: {output_file}")


def create_heatmap_simple(map_data, output_file="heatmap.png", use_log_norm=True):
    """
    简化版热力图（没有地图轮廓）
    """
    # 提取网格数据
    grid_data = {}
    for grid in map_data["grids"]:
        grid_x = grid["grid_x"]
        grid_y = grid["grid_y"]
        count = grid["people_count"]
        grid_data[(grid_x, grid_y)] = count

    if not grid_data:
        print("没有网格数据可显示")
        return

    # 创建网格矩阵
    x_coords = [coord[0] for coord in grid_data.keys()]
    y_coords = [coord[1] for coord in grid_data.keys()]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # 创建空的网格矩阵
    grid_matrix = np.zeros((max_y - min_y + 1, max_x - min_x + 1))

    # 填充数据
    for (x, y), count in grid_data.items():
        grid_matrix[y - min_y, x - min_x] = count

    # 获取网格大小
    grid_size_km = map_data.get("grid_size_km", 1.0)

    # 创建热力图
    plt.figure(figsize=(12, 8))

    # 创建自定义颜色映射
    colors = ['darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    cmap = LinearSegmentedColormap.from_list('custom_heat', colors, N=256)

    # 使用对数归一化来增强低值区域的对比度
    if use_log_norm:
        # 避免对0取对数
        min_nonzero = np.min(grid_matrix[grid_matrix > 0]) if np.any(grid_matrix > 0) else 1
        norm = LogNorm(vmin=min_nonzero, vmax=np.max(grid_matrix))
        im = plt.imshow(grid_matrix, cmap=cmap, aspect='auto',
                        extent=[min_x, max_x + 1, min_y, max_y + 1],
                        origin='lower', norm=norm)
        colorbar_label = 'People Count (Log Scale)'
    else:
        # 使用线性归一化，但调整vmin和vmax
        vmin = 0
        vmax = np.percentile(grid_matrix, 95)  # 使用95%分位数作为最大值，避免极端值影响
        im = plt.imshow(grid_matrix, cmap=cmap, aspect='auto',
                        extent=[min_x, max_x + 1, min_y, max_y + 1],
                        origin='lower', vmin=vmin, vmax=vmax)
        colorbar_label = 'People Count'

    plt.colorbar(im, label=colorbar_label)
    plt.xlabel('Grid X Coordinate')
    plt.ylabel('Grid Y Coordinate')
    plt.title(f'Population Distribution Heatmap ({grid_size_km}km×{grid_size_km}km grid)')

    # 添加网格线
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.tight_layout()
    plt.savefig(f"{processed_path}\\{output_file}", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"静态热力图已保存为: {output_file}")


def create_heatmap_extreme_normalization(map_data, output_file="heatmap_extreme.png"):
    """
    使用更激进的归一化方法来处理极端不均匀的数据分布
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    except ImportError:
        print("警告: 未安装cartopy库，无法显示地图轮廓")
        return create_heatmap_simple(map_data, output_file)

    # 提取数据
    lons = []
    lats = []
    values = []

    for grid in map_data["grids"]:
        lons.append(grid["center_lon"])
        lats.append(grid["center_lat"])
        values.append(grid["people_count"])

    if not values:
        print("没有网格数据可显示")
        return

    # 获取网格大小
    grid_size_km = map_data.get("grid_size_km", 1.0)

    # 获取地理边界
    geo_bounds = map_data.get("geo_bounds", {})
    if not geo_bounds:
        lon_min, lon_max = 73, 135
        lat_min, lat_max = 18, 54
    else:
        lon_min, lon_max = geo_bounds["min_lon"], geo_bounds["max_lon"]
        lat_min, lat_max = geo_bounds["min_lat"], geo_bounds["max_lat"]

    # 扩展边界以便更好地显示
    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2
    lon_range = max(lon_max - lon_min, 5)
    lat_range = max(lat_max - lat_min, 5)

    # 创建带地图投影的图形
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置地图范围
    ax.set_extent([lon_center - lon_range / 1.5, lon_center + lon_range / 1.5,
                   lat_center - lat_range / 1.5, lat_center + lat_range / 1.5],
                  crs=ccrs.PlateCarree())

    # 添加地图特征
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # 对数据进行极端归一化处理
    values_array = np.array(values)

    # 方法1: 使用百分位数截断，只保留95%或99%的数据范围
    p_low = np.percentile(values_array, 5)  # 5%分位数
    p_high = np.percentile(values_array, 95)  # 95%分位数

    # 方法2: 使用更激进的对数变换
    # 将0值替换为很小的正数，避免对数计算错误
    values_nonzero = np.where(values_array == 0, 0.1, values_array)

    # 方法3: 使用平方根变换，比对数变换更温和但也能压缩高值
    values_sqrt = np.sqrt(values_nonzero)

    # 方法4: 使用分位数归一化
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal', random_state=0)
    values_quantile = qt.fit_transform(values_array.reshape(-1, 1)).flatten()

    # 选择一种归一化方法
    normalized_values = values_quantile  # 可以尝试不同的归一化方法

    # 创建自定义颜色映射
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
              '#4292c6', '#2171b5', '#08519c', '#08306b']
    cmap = LinearSegmentedColormap.from_list('extreme_heat', colors, N=256)

    # 根据网格大小调整点的大小
    if grid_size_km >= 50:
        point_size = 100
    elif grid_size_km >= 20:
        point_size = 50
    elif grid_size_km >= 10:
        point_size = 30
    elif grid_size_km >= 5:
        point_size = 15
    else:
        point_size = 8

    # 创建散点图热力图
    scatter = ax.scatter(lons, lats, c=normalized_values, cmap=cmap,
                         s=point_size, alpha=0.8, transform=ccrs.PlateCarree(),
                         edgecolors='black', linewidths=0.2)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Normalized People Count (Square Root Scale)')

    # 添加网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.title(f'Population Distribution Heatmap ({grid_size_km}km×{grid_size_km}km grid) - Extreme Normalization')

    # 保存图片
    plt.tight_layout()
    plt.savefig(f"{processed_path}\\{output_file}", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"极端归一化热力图已保存为: {output_file}")


def matplotlib_main():
    six_six_samples = read_trace_files(six_six_path)
    if debugging_save_file:
        save_json(six_six_samples, six_six_path)

    # 可以尝试不同的网格大小
    grid_sizes = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]  # 单位：千米

    for grid_size in grid_sizes:
        print(f"\n正在处理网格大小: {grid_size}km")
        map_data = create_map_data(six_six_samples, grid_size_km=grid_size)

        if debugging_save_file:
            save_json(map_data, f"{map_data_path}\\{six_six_path}_grid_{grid_size}km")

        create_heatmap_extreme_normalization(map_data, f"{map_data_path}\\{six_six_path}_grid_{grid_size}km_extreme.png")
