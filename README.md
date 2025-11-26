# City Data Analysis

## 快速开始
- `python main.py`：根据 `utils.function` 生成 Folium/Matplotlib 热力图，默认跑 `folium_main()`。
- 数据/输出路径在 `utils.py`，轨迹预处理逻辑在 `gen_functions.py`。

## 聚类分析（新增）
- **DBSCAN**（自动确定簇数，haversine 距离）：
  *Linux/macOS（bash）*
  ```bash
  python - <<'PY'
  from trace_graph.clustering import demo_dbscan
  demo_dbscan(eps_km=80, min_samples=10, max_points=20000)
  ```
  *Windows PowerShell*
  ```powershell
  python -c "from trace_graph.clustering import demo_dbscan; demo_dbscan(eps_km=80, min_samples=10, max_points=20000)"
  ```
- **KMeans**（指定簇数，适合 30–40 个“城市”）：
  *Linux/macOS（bash）*
  ```bash
  python - <<'PY'
  from trace_graph.clustering import demo_kmeans
  demo_kmeans(n_clusters=35)
  PY
  ```
  *Windows PowerShell*
  ```powershell
  python -c "from trace_graph.clustering import demo_kmeans; demo_kmeans(n_clusters=35)"
  ```

结果会写入 `map_data/<日期>_<算法>_clusters.json`，字段包含簇编号、人数、中心经纬度，可作为后续构图或可视化的输入。若遇到 DBSCAN 内存不足，可减小 `eps_km`、调小 `max_points`（默认 20000 会做随机下采样），或改用 KMeans。

### 中国有多少个城市？
691个城市。
23个省，5个自治区，4个直辖市，2个特别行政区
总计34个。