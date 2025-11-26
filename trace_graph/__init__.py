"""Trace graph utilities (聚类等)."""

# Prefer relative import so ``trace_graph`` works whether the project is on ``PYTHONPATH``
# or installed as a package. On Windows PowerShell the previous absolute import
# (``from clustering import ...``) could not be resolved, causing
# ``ModuleNotFoundError: No module named 'clustering'``. The fallback keeps
# compatibility if the package is executed in unusual contexts.
try:  # pragma: no cover - defensive import path handling
    from .clustering import (  # type: ignore
        ClusterSummary,
        cluster_dbscan,
        cluster_kmeans,
        demo_dbscan,
        demo_kmeans,
        save_cluster_result,
    )
except ModuleNotFoundError:  # pragma: no cover
    # Fallback for edge cases where ``trace_graph`` is imported as a flat module.
    from clustering import (  # type: ignore
        ClusterSummary,
        cluster_dbscan,
        cluster_kmeans,
        demo_dbscan,
        demo_kmeans,
        save_cluster_result,
    )

__all__ = [
    "ClusterSummary",
    "cluster_dbscan",
    "cluster_kmeans",
    "demo_dbscan",
    "demo_kmeans",
    "save_cluster_result",
]
