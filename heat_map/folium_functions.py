import datetime
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, MutableMapping, Tuple

import folium
import numpy as np
from folium.plugins import HeatMap

from gen_functions import read_trace_files, save_json
from utils import debugging_save_file, map_data_path, processed_path, six_six_path

try:
    from chinese_calendar import get_holiday_detail, is_workday
    HAS_CALENDAR = True
except ImportError:
    HAS_CALENDAR = False


EARTH_RADIUS_KM = 6371.0


def _lon_lat_to_km(lon: float, lat: float) -> Tuple[float, float]:
    """将经纬度转换为基于参考纬度缩放的公里坐标 (x, y)。"""

    x_km = EARTH_RADIUS_KM * math.radians(lon) * math.cos(math.radians(lat))
    y_km = EARTH_RADIUS_KM * math.radians(lat)
    return x_km, y_km


def _km_to_lon_lat(x_km: float, y_km: float) -> Tuple[float, float]:
    """将公里坐标反算为经纬度坐标 (lon, lat)。"""

    lat = math.degrees(y_km / EARTH_RADIUS_KM)
    lon = math.degrees(x_km / (EARTH_RADIUS_KM * math.cos(math.radians(lat))))
    return lon, lat


def get_date_type_name(date_str: str) -> str:
    """返回日期类型标签，用于 HTML 标题显示。"""

    try:
        dt = datetime.datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except Exception:
        return "未知日期"

    if not HAS_CALENDAR:
        return "工作日" if dt.weekday() < 5 else "周末"

    if is_workday(dt):
        return "工作日"

    is_holiday, holiday_name = get_holiday_detail(dt)
    if is_holiday and holiday_name:
        return f"节假日: {holiday_name}"
    return "周末"


def create_map_data(samples: MutableMapping[str, Iterable[dict]], grid_size_km: float = 1.0) -> Dict:
    """
    将轨迹数据聚合到固定大小的网格中，输出可重复使用的结构：

    返回字典结构示例：
    {
        "grid_size_km": 1.0,
        "earth_radius_km": 6371.0,
        "geo_bounds": {"min_lon": 0.0, "max_lon": 0.0, "min_lat": 0.0, "max_lat": 0.0},
        "grids": [
            {
                "grid_x": 123,
                "grid_y": 456,
                "people_count": 3,
                "people_ids": ["u1", "u2", "u3"],
                "center_lon": 120.1,
                "center_lat": 30.2,
            },
            ...
        ],
    }
    """

    grid_people: Dict[Tuple[int, int], set] = defaultdict(set)
    min_lon, max_lon = float("inf"), float("-inf")
    min_lat, max_lat = float("inf"), float("-inf")

    for traces in samples.values():
        for trace in traces:
            try:
                user_id = trace["id"]
                lon = float(trace["lo"])
                lat = float(trace["la"])
            except (TypeError, ValueError, KeyError) as exc:
                print(f"数据格式错误: {exc}, 跳过该记录")
                continue

            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)

            x_km, y_km = _lon_lat_to_km(lon, lat)
            grid_x = int(x_km / grid_size_km)
            grid_y = int(y_km / grid_size_km)
            grid_people[(grid_x, grid_y)].add(user_id)

    map_data = {
        "grid_size_km": grid_size_km,
        "earth_radius_km": EARTH_RADIUS_KM,
        "geo_bounds": {
            "min_lon": min_lon,
            "max_lon": max_lon,
            "min_lat": min_lat,
            "max_lat": max_lat,
        },
        "grids": [],
    }

    for (grid_x, grid_y), people_set in grid_people.items():
        center_x_km = (grid_x + 0.5) * grid_size_km
        center_y_km = (grid_y + 0.5) * grid_size_km
        center_lon, center_lat = _km_to_lon_lat(center_x_km, center_y_km)

        map_data["grids"].append(
            {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "people_count": len(people_set),
                "people_ids": list(people_set),
                "center_lon": center_lon,
                "center_lat": center_lat,
            }
        )

    map_data["grids"].sort(key=lambda grid: grid["people_count"], reverse=True)

    print(f"网格边长: {grid_size_km}km")
    print(f"总共统计了 {len(map_data['grids'])} 个网格")
    print(f"总人数: {sum(grid['people_count'] for grid in map_data['grids'])}")
    print(f"经纬度范围: 经度[{min_lon:.4f}, {max_lon:.4f}], 纬度[{min_lat:.4f}, {max_lat:.4f}]")

    return map_data


def _build_heat_data_from_grid(map_data: Dict) -> List[List[float]]:
    return [
        [grid["center_lat"], grid["center_lon"], grid["people_count"]]
        for grid in map_data.get("grids", [])
    ]


def _build_heat_data_from_samples(samples: MutableMapping[str, Iterable[dict]]) -> List[List[float]]:
    heat_data: List[List[float]] = []
    for traces in samples.values():
        for trace in traces:
            try:
                lon = float(trace["lo"])
                lat = float(trace["la"])
            except (TypeError, ValueError, KeyError):
                continue
            heat_data.append([lat, lon, 1])
    return heat_data


def create_heatmap_folium(
    map_data: Dict,
    samples: MutableMapping[str, Iterable[dict]],
    output_file: str = "heatmap.html",
    *,
    use_grid_data: bool = True,
    show_china_outline: bool = True,
    date_str: str = "",
    day_type: str = "",
):
    """创建交互式热力图。

    - ``use_grid_data=True``: 使用网格中心点 + 人数权重。
    - ``use_grid_data=False``: 使用原始轨迹点，每条记录权重为 1。
    """

    heat_data = (
        _build_heat_data_from_grid(map_data)
        if use_grid_data
        else _build_heat_data_from_samples(samples)
    )

    if not heat_data:
        print("没有有效数据生成热力图")
        return None

    lats = [point[0] for point in heat_data]
    lons = [point[1] for point in heat_data]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    geo_bounds = map_data.get("geo_bounds", {})
    bounds = (
        [[geo_bounds["min_lat"], geo_bounds["min_lon"]], [geo_bounds["max_lat"], geo_bounds["max_lon"]]]
        if geo_bounds
        else None
    )

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None # 不加载默认底图
    )

    if show_china_outline:
        # 第一个图层
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="OpenStreetMap",
            name="街道地图-OpenStreetMap",
            overlay=False,
            control=True,
        ).add_to(m)  # WGS-84 GPS坐标

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="卫星影像",
            overlay=False,
            control=True,
        ).add_to(m)

        # 添加高德街道图 (速度快，且适合中国大陆)
        folium.TileLayer(
            tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
            attr='高德地图',
            name='高德街道',
            overlay=False,
            control=True
        ).add_to(m) # GCJ-02 GPS坐标

    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=18,
        radius=15,
        blur=10,
        gradient={0.0: "blue", 0.5: "lime", 0.8: "yellow", 1.0: "red"},
    ).add_to(m)

    if use_grid_data:
        add_grid_boundaries(m, map_data)

    if bounds:
        m.fit_bounds(bounds)

    folium.LayerControl().add_to(m)

    grid_size_km = map_data.get("grid_size_km", 1.0)
    base_title = f"人口分布热力图 ({grid_size_km}km×{grid_size_km}km网格)"
    full_title = (
        f"{base_title}<br><span style='font-size:16px'>{date_str} {day_type}</span>"
        if date_str
        else base_title
    )

    title_html = f"""
                     <h3 align=\"center\" style=\"font-size:20px\"><b>{full_title}</b></h3>
                     """
    m.get_root().html.add_child(folium.Element(title_html))

    os.makedirs(f"{processed_path}", exist_ok=True)
    m.save(f"{processed_path}\\{output_file}")
    print(f"热力图已保存为: {output_file}")
    return m


def add_grid_boundaries(map_obj: folium.Map, map_data: Dict):
    """
    在地图上绘制网格边界，使用【分位数映射】(Quantile Mapping) 动态着色。

    原理：
    1. 统计所有非零网格的人数。
    2. 计算 20%, 40%, 60%, 80% 的分位数值。
    3. 根据数值落在哪个区间，分配对应的颜色。
    """

    grid_size_km = map_data.get("grid_size_km", 1.0)
    grids = map_data.get("grids", [])

    # --- 核心逻辑开始：计算分位数阈值 ---

    # 1. 提取有效数据
    # 我们只关心“有人的地方”内部的密度差异，所以排除 0 值。
    # 如果包含大量 0 值，会导致低分位数的阈值全部为 0，颜色区分度降低。
    valid_counts = [g["people_count"] for g in grids if g["people_count"] > 0]

    if not valid_counts:
        return

    # 2. 定义分位点 (0% - 20% - 40% - 60% - 80% - 100%)
    # 我们将数据切分为 5 等份
    quantiles = [20, 40, 60, 80]

    # 3. 使用 numpy 计算对应的具体数值阈值
    # 例如：如果 thresholds[3] 是 100，意味着 80% 的网格人数都小于等于 100
    thresholds = np.percentile(valid_counts, quantiles)

    # 打印出来方便调试，看看你的数据分布大概是什么样
    print(f"【分位数统计】总有数据网格数: {len(valid_counts)}")
    print(
        f"【颜色阈值】: 蓝色 < {thresholds[0]} <= 青色 < {thresholds[1]} <= 绿色 < {thresholds[2]} <= 橙色 < {thresholds[3]} <= 红色")

    # 4. 定义颜色方案 (从冷色到暖色)
    colors = [
        '#3b528b',  # Level 1: 深蓝 (最冷) - 对应底部 20%
        '#21918c',  # Level 2: 青色
        '#5ec962',  # Level 3: 绿色 (中等)
        '#fde725',  # Level 4: 黄色
        '#ff0000'  # Level 5: 红色 (最热) - 对应顶部 20%
    ]
    # --- 核心逻辑结束 ---

    for grid in grids:
        center_lat = grid["center_lat"]
        center_lon = grid["center_lon"]
        count = grid["people_count"]

        # 计算网格的矩形边界 (这部分逻辑保持不变)
        lat_degree_per_km = 1 / 110.574
        # 注意：这里加了个 max(..., 0.0001) 防止极地地区 cos 为 0 导致除零错误，虽然在中国不太可能发生
        lon_degree_per_km = 1 / (111.320 * max(math.cos(math.radians(center_lat)), 0.0001))

        half_lat = grid_size_km * lat_degree_per_km / 2
        half_lon = grid_size_km * lon_degree_per_km / 2

        bounds = [
            [center_lat - half_lat, center_lon - half_lon],
            [center_lat + half_lat, center_lon + half_lon],
        ]

        # --- 着色逻辑 ---
        if count == 0:
            # 没有人的网格：
            continue
            # 建议设为灰色且非常透明，或者直接不画（continue），这里为了保留网格感设为淡灰
            # color = "gray"
            # fill_opacity = 0.05
            # weight = 0  # 去掉边框，减少视觉干扰
        else:
            # 有人的网格：
            # np.searchsorted 会返回 count 应该插入 thresholds 的索引位置
            # 如果 count < 20%阈值，返回 0 -> 对应 colors[0] (深蓝)
            # 如果 count > 80%阈值，返回 4 -> 对应 colors[4] (红色)
            idx = np.searchsorted(thresholds, count)

            # 这里的 idx 有可能取到 len(thresholds)，即 4，正好对应 colors 的最后一个索引
            # 但为了安全起见（防止浮点数误差），限制一下最大索引
            idx = min(idx, len(colors) - 1)

            color = colors[idx]
            fill_opacity = 0.6  # 有数据的格子稍微不透明一点，突出显示
            weight = 0  # 【重要】去掉矩形的描边！
            # 如果不去掉，当地图缩小时，密密麻麻的黑色边框会把颜色完全遮住，变成一团黑。

        # 绘制矩形
        folium.Rectangle(
            bounds=bounds,
            color=color,  # 填充颜色
            fill=True,
            fill_opacity=fill_opacity,
            weight=weight,  # 边框宽度 (0表示无边框)
            popup=f"人数: {count}",
            # tooltip=f"人数: {count}"  # 鼠标悬停时直接显示数值，体验更好
        ).add_to(map_obj)

def create_comparison_heatmaps(
    samples: MutableMapping[str, Iterable[dict]], grid_sizes: Iterable[float] = (1.0, 5.0, 10.0, 20.0)
):
    """以不同网格粒度批量生成热力图，便于比对可视化效果。"""

    date_str = six_six_path[:10]
    day_type = get_date_type_name(date_str)

    for grid_size in grid_sizes:
        print(f"\n正在处理网格大小: {grid_size}km")
        map_data = create_map_data(samples, grid_size_km=grid_size)

        if debugging_save_file:
            save_json(map_data, f"{map_data_path}\\{six_six_path}_grid_{grid_size}km")

        create_heatmap_folium(
            map_data,
            samples,
            f"{map_data_path}\\{six_six_path}_grid_{grid_size}km.html",
            use_grid_data=True,
            show_china_outline=True,
            date_str=date_str,
            day_type=day_type,
        )

        if grid_size <= 5.0:
            create_heatmap_folium(
                map_data,
                samples,
                f"{map_data_path}\\{six_six_path}_raw_{grid_size}km.html",
                use_grid_data=False,
                show_china_outline=True,
                date_str=date_str,
                day_type=day_type,
            )


def folium_main():
    six_six_samples = read_trace_files(six_six_path)
    if debugging_save_file:
        save_json(six_six_samples, six_six_path)

    date_str = six_six_path[:10]
    day_type = get_date_type_name(date_str)

    print(f"当前处理日期: {date_str} ({day_type})")

    grid_sizes = [0.5]

    for grid_size in grid_sizes:
        map_data = create_map_data(six_six_samples, grid_size_km=grid_size)
        base_name = f"{date_str}_grid_{grid_size}km"

        if debugging_save_file:
            save_json(map_data, f"{map_data_path}\\{base_name}")

        create_heatmap_folium(
            map_data,
            six_six_samples,
            f"{map_data_path}\\{base_name}.html",
            use_grid_data=True,
            show_china_outline=True,
            date_str=date_str,
            day_type=day_type,
        )

