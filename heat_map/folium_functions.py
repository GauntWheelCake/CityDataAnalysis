import folium
from folium.plugins import HeatMap
from gen_functions import *


def create_map_data(samples, grid_size_km=1.0):
    """
    将地图分割为指定边长的网格，统计每个网格中出现过的不同人数

    参数:
    samples: 字典，包含所有轨迹数据
    grid_size_km: 网格边长，单位千米，默认为1.0km

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


def create_heatmap_folium(map_data, samples, output_file="heatmap.html", use_grid_data=True, show_china_outline=True):
    """
    使用Folium创建交互式热力图

    参数:
    map_data: 网格数据
    samples: 原始轨迹数据
    output_file: 输出文件名
    use_grid_data: 是否使用网格数据（True）或原始数据（False）
    show_china_outline: 是否显示中国地图轮廓
    """

    # 根据数据源选择热力图数据
    if use_grid_data:
        # 使用网格数据
        heat_data = []
        for grid in map_data["grids"]:
            # 使用网格中心点坐标
            lat = grid["center_lat"]
            lon = grid["center_lon"]
            # 使用人数作为权重，可以应用归一化
            weight = grid["people_count"]
            heat_data.append([lat, lon, weight])
    else:
        # 使用原始数据
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

    # 计算中心点和地图范围
    lats = [point[0] for point in heat_data]
    lons = [point[1] for point in heat_data]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # 获取地理边界
    geo_bounds = map_data.get("geo_bounds", {})
    if geo_bounds:
        min_lon, max_lon = geo_bounds["min_lon"], geo_bounds["max_lon"]
        min_lat, max_lat = geo_bounds["min_lat"], geo_bounds["max_lat"]
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    else:
        bounds = None

    # 创建地图 - 使用OpenStreetMap作为底图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # 如果需要显示中国地图轮廓，可以添加其他底图
    if show_china_outline:
        # 添加多个底图选项
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='卫星影像',
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles='https://{s}.tile.openstreetmap.org/{z}/{y}/{x}.png',
            attr='OpenStreetMap',
            name='街道地图',
            overlay=False,
            control=True
        ).add_to(m)

        # 添加中国边界图层（需要GeoJSON数据）
        # 这里可以添加中国的GeoJSON边界文件
        # china_geojson = "path/to/china.geojson"
        # folium.GeoJson(china_geojson, name="中国边界").add_to(m)

    # 添加热力图
    HeatMap(heat_data,
            min_opacity=0.3,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={0.0: 'blue', 0.5: 'lime', 0.8: 'yellow', 1.0: 'red'}
            ).add_to(m)

    # 添加网格边界（可选）
    if use_grid_data:
        add_grid_boundaries(m, map_data)

    # 设置地图范围
    if bounds:
        m.fit_bounds(bounds)

    # 添加图层控制
    folium.LayerControl().add_to(m)

    # 添加标题和说明
    grid_size_km = map_data.get("grid_size_km", 1.0)
    title_html = f'''
                 <h3 align="center" style="font-size:20px"><b>人口分布热力图 ({grid_size_km}km×{grid_size_km}km网格)</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 保存地图
    os.makedirs(f"{processed_path}", exist_ok=True)
    m.save(f"{processed_path}\\{output_file}")
    print(f"热力图已保存为: {output_file}")
    return m


def add_grid_boundaries(map_obj, map_data):
    """
    在地图上添加网格边界
    """
    grid_size_km = map_data.get("grid_size_km", 1.0)

    # 为每个网格添加边界
    for grid in map_data["grids"]:
        center_lat = grid["center_lat"]
        center_lon = grid["center_lon"]
        count = grid["people_count"]

        # 计算网格的近似大小（度）
        # 注意：这是一个近似值，实际网格大小会随纬度变化
        lat_degree_per_km = 1 / 110.574  # 1km约等于0.009度纬度
        lon_degree_per_km = 1 / (111.320 * math.cos(math.radians(center_lat)))  # 1km约等于的经度

        # 计算网格边界
        half_lat = grid_size_km * lat_degree_per_km / 2
        half_lon = grid_size_km * lon_degree_per_km / 2

        # 创建矩形边界
        bounds = [
            [center_lat - half_lat, center_lon - half_lon],
            [center_lat + half_lat, center_lon + half_lon]
        ]

        # 根据人数设置颜色
        if count == 0:
            color = 'gray'
        elif count < 5:
            color = 'blue'
        elif count < 10:
            color = 'green'
        elif count < 20:
            color = 'yellow'
        elif count < 50:
            color = 'orange'
        else:
            color = 'red'

        # 添加矩形
        folium.Rectangle(
            bounds=bounds,
            color=color,
            fill=True,
            fill_opacity=0.2,
            weight=1,
            popup=f"人数: {count}"
        ).add_to(map_obj)


def create_comparison_heatmaps(samples, grid_sizes=[1.0, 5.0, 10.0, 20.0]):
    """
    创建多个不同网格大小的热力图进行比较
    """
    for grid_size in grid_sizes:
        print(f"\n正在处理网格大小: {grid_size}km")
        map_data = create_map_data(samples, grid_size_km=grid_size)

        if debugging_save_file:
            save_json(map_data, f"{map_data_path}\\{six_six_path}_grid_{grid_size}km")

        # 创建使用网格数据的热力图
        create_heatmap_folium(
            map_data,
            samples,
            f"{map_data_path}\\{six_six_path}_grid_{grid_size}km.html",
            use_grid_data=True,
            show_china_outline=True
        )

        # 创建使用原始数据的热力图（可选）
        if grid_size <= 5.0:  # 对于较小的网格，原始数据可能过于密集
            create_heatmap_folium(
                map_data,
                samples,
                f"{map_data_path}\\{six_six_path}_raw_{grid_size}km.html",
                use_grid_data=False,
                show_china_outline=True
            )


def folium_main():
    six_six_samples = read_trace_files(six_six_path)
    if debugging_save_file:
        save_json(six_six_samples, six_six_path)

    grid_sizes = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]  # 单位：千米
    for grid_size in grid_sizes:
        map_data = create_map_data(six_six_samples, grid_size_km=grid_size)

        if debugging_save_file:
            save_json(map_data, f"{map_data_path}\\{six_six_path}_grid_{grid_size}km")

        create_heatmap_folium(
            map_data,
            six_six_samples,
            f"{map_data_path}\\{six_six_path}_grid_{grid_size}km.html",
            use_grid_data=True,
            show_china_outline=True
        )

    # 方法2: 创建多个网格大小的比较
    # create_comparison_heatmaps(six_six_samples, grid_sizes=[1.0, 5.0, 10.0, 20.0, 50.0])
