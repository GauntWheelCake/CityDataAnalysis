import os
import json
import numpy as np
import folium
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from typing import List, Tuple
from folium import Element

# 引入项目现有的工具函数和配置
from utils import dataset_path, trace_path, processed_path, map_data_path
from gen_functions import read_trace_files

# ==========================================
# 0. 配置区域
# ==========================================
# 日期筛选：如果设为 None，则读取所有文件夹
# 示例: "2024-10-01"
TARGET_START_DATE = "2024-10-01"
TARGET_END_DATE = "2024-10-01"

# 常量定义
EARTH_RADIUS_KM = 6371.0


# ==========================================
# 1. 数据加载 (包含自动日期检测)
# ==========================================
def is_date_in_range(folder_name: str, start_str: str, end_str: str) -> bool:
    if not start_str and not end_str: return True
    try:
        folder_date = datetime.strptime(folder_name[:10], "%Y-%m-%d")
        start = datetime.strptime(start_str, "%Y-%m-%d") if start_str else datetime.min
        end = datetime.strptime(end_str, "%Y-%m-%d") if end_str else datetime.max
        return start <= folder_date <= end
    except:
        return False


def load_data() -> Tuple[np.ndarray, List[dict], str, str]:
    """
    读取数据并返回：坐标数组, 元数据列表, 实际开始日期, 实际结束日期
    """
    base_dir = Path(dataset_path) / trace_path
    if not base_dir.exists():
        print(f"错误: 路径 {base_dir} 不存在")
        return np.array([]), [], "", ""

    coords = []
    meta_list = []
    loaded_dates = []  # 用于记录读取到了哪些天

    # 扫描并筛选文件夹
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    target_subdirs = [d for d in subdirs if is_date_in_range(d.name, TARGET_START_DATE, TARGET_END_DATE)]

    print(f"即将读取 {len(target_subdirs)} 个文件夹的数据...")

    for subdir in target_subdirs:
        print(f"  -> 读取: {subdir.name}")
        # 记录日期 (假设文件夹名格式为 YYYY-MM-DD-xxx)
        try:
            date_str = subdir.name[:10]
            # 简单的格式校验
            datetime.strptime(date_str, "%Y-%m-%d")
            loaded_dates.append(date_str)
        except:
            pass

        samples = read_trace_files(subdir.name)

        for _, traces in samples.items():
            for trace in traces:
                try:
                    lat, lon = float(trace["la"]), float(trace["lo"])
                    dt_list = trace["datetime"]
                    dt_str = f"{dt_list[0]}-{dt_list[1]:02d}-{dt_list[2]:02d} {dt_list[3]:02d}:{dt_list[4]:02d}:{dt_list[5]:02d}"

                    coords.append([lat, lon])
                    meta_list.append({"id": trace["id"], "datetime": dt_str})
                except:
                    continue

    print(f"数据加载完成，共 {len(coords)} 个点。")

    # 计算实际的时间范围
    if loaded_dates:
        loaded_dates.sort()
        real_start = loaded_dates[0]
        real_end = loaded_dates[-1]
    else:
        real_start, real_end = "未知日期", "未知日期"

    return np.array(coords), meta_list, real_start, real_end


# ==========================================
# 2. 聚类分析 (保持不变)
# ==========================================
def analyze_and_save(coords: np.ndarray, meta_list: List[dict], labels: np.ndarray, algo_name: str) -> List[dict]:
    print(f"正在分析 {algo_name} ...")

    df = pd.DataFrame(meta_list)
    df['lat'] = coords[:, 0]
    df['lon'] = coords[:, 1]
    df['cluster'] = labels

    valid_df = df[df['cluster'] != -1]

    cluster_summaries = []
    detailed_results = {}

    grouped = valid_df.groupby('cluster')

    for c_id, group in grouped:
        c_id = int(c_id)

        total = len(group)
        unique = group['id'].nunique()
        center_lat = group['lat'].mean()
        center_lon = group['lon'].mean()

        details = group[['id', 'datetime']].to_dict('records')

        detailed_results[f"cluster_{c_id}"] = {
            "cluster_id": c_id,
            "center": [center_lat, center_lon],
            "stats": {"total_traffic": total, "unique_users": unique},
            "user_details": details
        }

        cluster_summaries.append({
            "id": c_id,
            "lat": center_lat,
            "lon": center_lon,
            "total": total,
            "unique": unique
        })

    os.makedirs(map_data_path, exist_ok=True)
    out_json = f"{TARGET_START_DATE}/to/{TARGET_END_DATE}/{map_data_path}/clusters_{algo_name}_detailed.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    print(f"  -> [文件] 详细数据(含UID)已保存: {out_json}")

    return cluster_summaries


# ==========================================
# 3. Folium 可视化 (增加动态标题逻辑)
# ==========================================
def generate_folium_map(summaries: List[dict], algo_name: str, start_date: str, end_date: str):
    if not summaries: return

    # 计算中心点
    avg_lat = np.mean([s['lat'] for s in summaries])
    avg_lon = np.mean([s['lon'] for s in summaries])

    # 1. 初始化地图
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=6,
        tiles=None
    )

    # 2. 添加底图
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="街道地图 (OSM)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="卫星影像",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德地图',
        name='高德街道',
        overlay=False,
        control=True
    ).add_to(m)

    # 3. 绘制聚类点
    cluster_group = folium.FeatureGroup(name=f"Clusters ({algo_name})", show=True)

    for c in summaries:
        popup_html = f"""
        <div style="font-family: Arial; min-width: 150px;">
            <h4 style="margin-bottom:5px; color: #333;">Cluster {c['id']}</h4>
            <span style="font-size:12px; color:gray;">{algo_name}</span>
            <hr style="margin:5px 0; border: 0; border-top: 1px solid #ccc;">
            <b>总流量:</b> {c['total']}<br>
            <b>去重人数:</b> {c['unique']}<br>
            <b>中心坐标:</b> {c['lat']:.4f}, {c['lon']:.4f}
        </div>
        """

        radius = 5 + np.log1p(c['unique']) * 1.5
        color = '#3388ff' if 'KMeans' in algo_name else '#ff7800'

        folium.CircleMarker(
            location=[c['lat'], c['lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Cluster {c['id']}: {c['unique']} users"
        ).add_to(cluster_group)

    cluster_group.add_to(m)
    folium.LayerControl().add_to(m)

    # --- 4. 关键修改：生成动态日期标题 ---
    if start_date == end_date:
        date_display = f"{start_date}"
    else:
        date_display = f"{start_date} 至 {end_date}"

    # HTML 标题样式
    title_html = f"""
        <div style="
            position: fixed; 
            top: 10px; left: 50%; 
            transform: translateX(-50%);
            z-index: 9999; 
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px 20px;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
            text-align: center;
            font-family: 'Microsoft YaHei', sans-serif;
        ">
            <div style="font-size: 18px; font-weight: bold; color: #333;">聚类分析结果: {algo_name}</div>
            <div style="font-size: 14px; color: #555; margin-top: 5px;">统计时间: {date_display}</div>
        </div>
    """
    m.get_root().html.add_child(Element(title_html))

    os.makedirs(processed_path, exist_ok=True)
    out_html = f"{processed_path}/map_{algo_name}.html"
    m.save(out_html)
    print(f"  -> [地图] Folium地图已保存: {out_html}")


# ==========================================
# 4. 主程序
# ==========================================
def main():
    # 1. 读取数据 (获取实际的时间范围)
    coords, meta, real_start_date, real_end_date = load_data()
    if len(coords) == 0: return

    print(f"\n>>> 检测到的数据时间范围: {real_start_date} 到 {real_end_date}")

    # 2. KMeans (K=34)
    print("\n>>> 运行 KMeans (K=34)...")
    km = KMeans(n_clusters=34, random_state=42, n_init='auto')
    labels_km = km.fit_predict(coords)

    sums_km = analyze_and_save(coords, meta, labels_km, "KMeans_34")
    # 传入实际日期生成标题
    generate_folium_map(sums_km, "KMeans_34", real_start_date, real_end_date)

    # 3. DBSCAN (eps=25km, min=100)
    print("\n>>> 运行 DBSCAN (eps=25km)...")
    coords_rad = np.radians(coords)
    eps_rad = 25.0 / EARTH_RADIUS_KM
    db = DBSCAN(eps=eps_rad, min_samples=100, metric='haversine', algorithm='ball_tree')
    labels_db = db.fit_predict(coords_rad)

    sums_db = analyze_and_save(coords, meta, labels_db, "DBSCAN_25km")
    generate_folium_map(sums_db, "DBSCAN_25km", real_start_date, real_end_date)

    print("\n全部完成！请查看 processed_samples 目录下的 HTML 文件。")


if __name__ == "__main__":
    main()