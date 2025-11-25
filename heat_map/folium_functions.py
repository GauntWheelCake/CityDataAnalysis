import datetime
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, MutableMapping, Tuple

import folium
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

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")

    if show_china_outline:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="卫星影像",
            overlay=False,
            control=True,
        ).add_to(m)

        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{y}/{x}.png",
            attr="OpenStreetMap",
            name="街道地图",
            overlay=False,
            control=True,
        ).add_to(m)

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
    """在地图上绘制网格边界，并用颜色标记人数级别。"""

    grid_size_km = map_data.get("grid_size_km", 1.0)

    for grid in map_data.get("grids", []):
        center_lat = grid["center_lat"]
        center_lon = grid["center_lon"]
        count = grid["people_count"]

        lat_degree_per_km = 1 / 110.574
        lon_degree_per_km = 1 / (111.320 * math.cos(math.radians(center_lat)))

        half_lat = grid_size_km * lat_degree_per_km / 2
        half_lon = grid_size_km * lon_degree_per_km / 2

        bounds = [
            [center_lat - half_lat, center_lon - half_lon],
            [center_lat + half_lat, center_lon + half_lon],
        ]

        if count == 0:
            color = "gray"
        elif count < 5:
            color = "blue"
        elif count < 10:
            color = "green"
        elif count < 20:
            color = "yellow"
        elif count < 50:
            color = "orange"
        else:
            color = "red"

        folium.Rectangle(
            bounds=bounds,
            color=color,
            fill=True,
            fill_opacity=0.2,
            weight=1,
            popup=f"人数: {count}",
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

    grid_sizes = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

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

