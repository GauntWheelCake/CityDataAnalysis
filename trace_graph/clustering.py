"""地理聚类与示例流程。

提供两类聚类：
- DBSCAN（haversine 距离，按公里设定 eps，适合自动找到簇数并保留噪声）。
- KMeans（指定簇数，自动给出中心点，可用于想要 30–40 个城市级簇的场景）。

默认示例读取 ``utils.six_six_path`` 下的数据并将结果保存到 ``map_data`` 目录，
方便和现有可视化结果对齐。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from gen_functions import read_trace_files, save_json
from utils import map_data_path, six_six_path

EARTH_RADIUS_KM = 6371.0


@dataclass
class ClusterSummary:
    """聚类后每个簇的统计信息。"""

    label: int
    count: int
    center_lon: float
    center_lat: float


def _collect_coordinates(samples: MutableMapping[str, Iterable[dict]]) -> Tuple[np.ndarray, List[dict]]:
    """从样本中提取经纬度数组和元数据列表。"""

    coords: List[Tuple[float, float]] = []
    meta: List[dict] = []

    for traces in samples.values():
        for trace in traces:
            try:
                lon = float(trace["lo"])
                lat = float(trace["la"])
            except (TypeError, ValueError, KeyError):
                continue

            coords.append((lat, lon))
            meta.append({"id": trace.get("id"), "datetime": trace.get("datetime")})

    return np.array(coords), meta


def _summarize_clusters(coords_deg: np.ndarray, labels: np.ndarray) -> Dict[int, ClusterSummary]:
    """根据标签计算簇中心和计数，忽略噪声标签 -1。"""

    summaries: Dict[int, ClusterSummary] = {}
    for label in set(labels):
        if label == -1:
            continue

        mask = labels == label
        cluster_points = coords_deg[mask]
        center_lat = float(cluster_points[:, 0].mean())
        center_lon = float(cluster_points[:, 1].mean())

        summaries[label] = ClusterSummary(
            label=label,
            count=int(mask.sum()),
            center_lon=center_lon,
            center_lat=center_lat,
        )

    return summaries


def cluster_dbscan(
    samples: MutableMapping[str, Iterable[dict]], *, eps_km: float = 80.0, min_samples: int = 10
) -> Tuple[np.ndarray, Dict[int, ClusterSummary]]:
    """使用 Haversine 距离的 DBSCAN 做地理聚类。

    参数说明：
    - ``eps_km``: 两点被视为同一邻域的最大距离（公里），可按想要的“城市直径”调整。
    - ``min_samples``: 核心点最小邻居数，控制簇的稠密度。

    返回 ``(labels, summaries)``，其中 ``labels`` 与输入样本顺序一一对应。
    """

    coords_deg, _ = _collect_coordinates(samples)
    if coords_deg.size == 0:
        return np.array([]), {}

    coords_rad = np.radians(coords_deg)
    eps = eps_km / EARTH_RADIUS_KM

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(coords_rad)

    return labels, _summarize_clusters(coords_deg, labels)


def cluster_kmeans(
    samples: MutableMapping[str, Iterable[dict]], *, n_clusters: int = 35, random_state: int = 42
) -> Tuple[np.ndarray, Dict[int, ClusterSummary]]:
    """用 KMeans 聚类经纬度，适合直接指定簇数（如 30–40 个城市）。"""

    coords_deg, _ = _collect_coordinates(samples)
    if coords_deg.size == 0:
        return np.array([]), {}

    mean_lat = coords_deg[:, 0].mean()
    lon_scale = max(math.cos(math.radians(mean_lat)), 0.0001)
    scaled = coords_deg.copy()
    scaled[:, 1] = scaled[:, 1] * lon_scale

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(scaled)

    centers_scaled = model.cluster_centers_
    centers = centers_scaled.copy()
    centers[:, 1] = centers[:, 1] / lon_scale

    summaries: Dict[int, ClusterSummary] = {}
    for label, (center_lat, center_lon) in enumerate(centers):
        count = int(np.sum(labels == label))
        summaries[label] = ClusterSummary(
            label=label,
            count=count,
            center_lon=float(center_lon),
            center_lat=float(center_lat),
        )

    return labels, summaries


def save_cluster_result(
    summaries: Dict[int, ClusterSummary], *, algorithm: str, extra_info: dict | None = None
) -> str:
    """把聚类结果保存到 ``map_data`` 目录，返回保存的文件名（不含扩展名）。"""

    clusters = [
        {
            "label": summary.label,
            "count": summary.count,
            "center_lon": summary.center_lon,
            "center_lat": summary.center_lat,
        }
        for summary in summaries.values()
    ]

    payload = {"algorithm": algorithm, "clusters": clusters}
    if extra_info:
        payload.update(extra_info)

    filename = f"{map_data_path}\\{six_six_path}_{algorithm}_clusters"
    save_json(payload, filename)
    return filename


def demo_dbscan(*, eps_km: float = 80.0, min_samples: int = 10) -> Dict[int, ClusterSummary]:
    """读取默认日期数据，跑 DBSCAN，并落盘结果。"""

    samples = read_trace_files(six_six_path)
    labels, summaries = cluster_dbscan(samples, eps_km=eps_km, min_samples=min_samples)

    noise = int(np.sum(labels == -1)) if labels.size else 0
    save_cluster_result(
        summaries,
        algorithm=f"dbscan_{eps_km}km_{min_samples}",
        extra_info={"noise_count": noise, "eps_km": eps_km, "min_samples": min_samples},
    )

    print(f"DBSCAN 完成: 生成 {len(summaries)} 个簇，噪声点 {noise} 个。")
    return summaries


def demo_kmeans(*, n_clusters: int = 35) -> Dict[int, ClusterSummary]:
    """读取默认日期数据，跑 KMeans，并落盘结果。"""

    samples = read_trace_files(six_six_path)
    labels, summaries = cluster_kmeans(samples, n_clusters=n_clusters)

    save_cluster_result(
        summaries, algorithm=f"kmeans_{n_clusters}", extra_info={"n_clusters": n_clusters}
    )

    print(f"KMeans 完成: 生成 {len(summaries)} 个簇（目标 {n_clusters}）。")
    return summaries


__all__ = [
    "ClusterSummary",
    "cluster_dbscan",
    "cluster_kmeans",
    "demo_dbscan",
    "demo_kmeans",
    "save_cluster_result",
]
