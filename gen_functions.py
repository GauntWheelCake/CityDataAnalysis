import json
import traceback
from pathlib import Path

import pandas as pd  # 引入pandas
# 引入 shapely 相关库
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid

from utils import *


# 1. 加载边界数据, 边界数据来源：https://datav.aliyun.com/portal/school/atlas/area_selector
def load_china_boundary():
    geojson_path = "F:\\city_data\\dataset\\city_dataset\\china_boundary.geojson"
    try:
        print(f"正在加载 {geojson_path} ...")
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        polygons = []
        for feature in data['features']:
            try:
                geom = shape(feature['geometry'])
                if not geom.is_valid:
                    geom = geom.buffer(0)
                    if not geom.is_valid:
                        geom = make_valid(geom)

                if geom.geom_type == 'GeometryCollection':
                    clean_polys = []
                    for g in geom.geoms:
                        if g.geom_type in ['Polygon', 'MultiPolygon']:
                            clean_polys.append(g)
                    geom = unary_union(clean_polys)

                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    polygons.append(geom)
            except Exception as e:
                continue

        if not polygons:
            return None

        print("正在合并并优化几何体...")
        china_boundary = unary_union(polygons)
        if not china_boundary.is_valid:
            china_boundary = china_boundary.buffer(0)

        # 预处理优化
        prepared_boundary = prep(china_boundary)
        print("成功加载并优化中国边界数据。")
        return prepared_boundary

    except FileNotFoundError:
        print(f"警告: 未找到 {geojson_path}，将跳过过滤。")
        return None
    except Exception as e:
        print(f"加载边界出错: {e}")
        traceback.print_exc()
        return None


# 全局变量
china_boundary_prep = load_china_boundary()


# 2. 核心修改：使用 Pandas 读取并过滤
def read_trace_files(filepath):
    samples = {}
    search_path = Path(f"{dataset_path}\\{trace_path}\\{filepath}")

    if not search_path.exists():
        print(f"路径不存在: {search_path}")
        return {}

    # 获取所有csv文件
    csv_files = list(search_path.glob("*.csv"))
    if not csv_files:
        return {}

    print(f"开始处理 {len(csv_files)} 个文件 (Pandas加速模式)...")

    for file_path in csv_files:
        file_name = file_path.stem
        try:
            # --- A. 极速读取 ---
            # usecols: 只读取需要的列，减少内存占用
            # header=0: 假设第一行是表头，如果是无表头文件请去掉此参数
            df = pd.read_csv(
                file_path,
                names=trace_dataset_headers,  # 使用 utils.py 里定义的表头
                header=None,  # 假设你的 CSV 第一行是标题，如果是数据请改为 None
                on_bad_lines='skip'  # 跳过损坏的行
            )

            # 确保经纬度是数字类型 (强制转换，非数字变 NaN)
            df['lo'] = pd.to_numeric(df['lo'], errors='coerce')
            df['la'] = pd.to_numeric(df['la'], errors='coerce')

            # 删除经纬度无效的行
            df.dropna(subset=['lo', 'la'], inplace=True)

            original_count = len(df)

            # --- B. 极速过滤 (核心优化) ---
            if china_boundary_prep:
                # 技巧：使用列表推导式 + zip 比 df.apply(axis=1) 快 10 倍以上
                # 我们直接把 lo 和 la 列打包，批量判断
                valid_mask = [
                    china_boundary_prep.contains(Point(lon, lat))
                    for lon, lat in zip(df['lo'], df['la'])
                ]

                # 应用过滤
                df = df[valid_mask]

                filtered_count = original_count - len(df)
                if filtered_count > 0:
                    print(f"[{file_name}] 过滤了 {filtered_count} 个境外点 (剩余 {len(df)})")

            # --- C. 数据格式化 ---
            if df.empty:
                continue

            # 转换时间格式 (Pandas 的 to_datetime 解析非常快)
            df['dt_obj'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
            df.dropna(subset=['dt_obj'], inplace=True)

            # 构造最终的 list of dicts 结构 (保持和你原有代码兼容)
            # 虽然 Pandas DataFrame 直接处理分析更快，但为了不改动你后面 heat_map 的代码，
            # 我们这里把 DataFrame 转回 sample 字典列表

            file_samples = []
            # 使用 itertuples 比 iterrows 快
            for row in df.itertuples(index=False):
                file_samples.append({
                    "id": row.id,
                    "lo": row.lo,  # 已经是 float
                    "la": row.la,  # 已经是 float
                    "datetime": [
                        row.dt_obj.year, row.dt_obj.month, row.dt_obj.day,
                        row.dt_obj.hour, row.dt_obj.minute, row.dt_obj.second
                    ],
                    "dur": row.dur
                })

            samples[file_name] = file_samples

        except Exception as e:
            print(f"处理文件 {file_name} 出错: {e}")
            # traceback.print_exc() # 调试时打开

    print("所有数据已读取并清洗完毕。")
    return samples


def save_json(data, filepath):
    # 保持原样
    with open(f"{processed_path}\\{filepath}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)