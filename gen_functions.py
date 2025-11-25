# form progress.bar import Bar # 原代码中有，保留
import csv
import json
from datetime import datetime
from pathlib import Path

# 需要额外导入 make_valid 和 unary_union
from shapely.geometry import shape, Point
from shapely.ops import unary_union
# --- 新增导入 ---
from shapely.prepared import prep
from shapely.validation import make_valid  # 需要 shapely >= 1.8.0

from utils import *


def load_china_boundary():
    """
    加载 GeoJSON 边界，自动修复无效几何体，并返回一个优化后的查询对象
    """
    geojson_path = "china_boundary.geojson"

    try:
        print(f"正在加载 {geojson_path} ...")
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        polygons = []
        for feature in data['features']:
            try:
                geom = shape(feature['geometry'])

                # --- 核心修复逻辑 ---
                if not geom.is_valid:
                    # 尝试方法 1: buffer(0) 技巧
                    geom = geom.buffer(0)

                    # 如果还是无效，尝试方法 2: make_valid (Shapely 1.8+)
                    if not geom.is_valid:
                        geom = make_valid(geom)

                # make_valid 可能会返回 GeometryCollection（混合几何体），我们需要提取出多边形
                if geom.geom_type == 'GeometryCollection':
                    clean_polys = []
                    for g in geom.geoms:
                        if g.geom_type in ['Polygon', 'MultiPolygon']:
                            clean_polys.append(g)
                    geom = unary_union(clean_polys)

                # 确保只添加多边形类型
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    polygons.append(geom)

            except Exception as e:
                print(f"跳过一个无效的 Feature: {e}")
                continue

        if not polygons:
            print("错误: 没有提取到有效的多边形数据。")
            return None

        print("正在合并并优化几何体（这可能需要几秒钟）...")
        # 合并为一个大的几何体
        china_boundary = unary_union(polygons)

        # 再次检查合并后的结果是否有效
        if not china_boundary.is_valid:
            china_boundary = china_boundary.buffer(0)

        # 使用 prep 进行预处理优化
        prepared_boundary = prep(china_boundary)

        print("成功加载并优化中国边界数据。")
        return prepared_boundary

    except FileNotFoundError:
        print(f"警告: 未找到 {geojson_path}，将跳过地理围栏过滤。")
        return None
    except Exception as e:
        print(f"加载边界数据出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# 全局加载一次边界数据，避免每次读取文件
china_boundary_prep = load_china_boundary()


def read_trace_files(filepath):
    samples = {}

    # 遍历文件夹中的所有文件
    search_path = Path(f"{dataset_path}\\{trace_path}\\{filepath}")
    if not search_path.exists():
        print(f"路径不存在: {search_path}")
        return {}

    for file_path in search_path.glob("*.csv"):
        file_name = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file, fieldnames=trace_dataset_headers)
            samples[file_name] = []
            csv_data = list(csv_reader)

            skipped_count = 0  # 统计被过滤的点

            for data_line in csv_data:
                try:
                    # 类型转换，确保是浮点数
                    lon = float(data_line["lo"])
                    lat = float(data_line["la"])

                    # --- 新增：地理围栏过滤逻辑 ---
                    if china_boundary_prep:
                        # 创建点对象
                        point = Point(lon, lat)
                        # contains 检查点是否在多边形内
                        if not china_boundary_prep.contains(point):
                            skipped_count += 1
                            continue  # 如果不在范围内，跳过当前循环，不添加到结果中
                    # ---------------------------

                    dt = datetime.strptime(data_line["time"], "%Y-%m-%d %H:%M:%S")
                    sample = {
                        "id": data_line["id"],
                        "lo": data_line["lo"],  # 保持原始字符串或存为 float 均可，建议后续统一转 float
                        "la": data_line["la"],
                        "datetime": [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second],
                        "dur": data_line["dur"],
                    }

                    if debugging_read_file:
                        print(sample)

                    samples[file_name].append(sample)

                except ValueError:
                    continue  # 跳过经纬度格式错误的行

            if skipped_count > 0:
                print(f"文件 {file_name}: 已过滤 {skipped_count} 个非中国境内的点。")

        if debugging_read_file or test_one_file:
            return samples

    print("data read into dict")
    return samples


def save_json(data, filepath):
    # 保持原样...
    with open(f"{processed_path}\\{filepath}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)