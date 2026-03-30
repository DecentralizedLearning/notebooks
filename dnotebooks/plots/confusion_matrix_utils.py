from typing import Callable

import numpy as np
import logging
from networkx import Graph
from networkx.algorithms import descendants_at_distance

from dengine.analysis import ExperimentConfusionMatrix


def reduce_n_hop_normalized_confusion_matrices(
    n: int,
    graph: Graph,
    confusion_matrices: ExperimentConfusionMatrix,
    reduction_fun: Callable[[np.ndarray], np.ndarray]
) -> ExperimentConfusionMatrix:
    devices_n_hops_neighbors_mean_cf = []

    if len(graph.nodes) == 0:
        return ExperimentConfusionMatrix(
            devices_confusion_matrices=confusion_matrices.confusion_matrix(normalize=True)
        )

    for device in graph.nodes:
        k_hop_neighbors = descendants_at_distance(graph, device, n)
        try:
            cfs = [confusion_matrices.device_view(X).confusion_matrix(normalize=True) for X in k_hop_neighbors]
        except Exception:
            logging.warning(f"No confusion found for {device}")
        mean_cfs = reduction_fun(np.stack(cfs))
        devices_n_hops_neighbors_mean_cf.append(mean_cfs)
    return ExperimentConfusionMatrix(
        devices_confusion_matrices=np.stack(devices_n_hops_neighbors_mean_cf)
    )


def get_n_hop_normalized_confusion_matrices_mean(
    n: int,
    graph: Graph,
    confusion_matrices: ExperimentConfusionMatrix
) -> ExperimentConfusionMatrix:
    return reduce_n_hop_normalized_confusion_matrices(
        n,
        graph,
        confusion_matrices,
        lambda x: np.mean(x, axis=0)
    )


def _confidence_interval_length(X: np.ndarray, z: float) -> np.ndarray:   # Compute the length of the interval
    n = X.shape[0]
    Xv = np.sqrt(X.var(axis=0, ddof=1))
    return ((Xv * z) / np.sqrt(n))


def _confidence_interval_length_student_95(X: np.ndarray) -> np.ndarray:
    return _confidence_interval_length(X, z=1.96)


def get_n_hop_normalized_confusion_matrices_confidence_interval(
    n: int,
    graph: Graph,
    confusion_matrices: ExperimentConfusionMatrix
) -> ExperimentConfusionMatrix:
    return reduce_n_hop_normalized_confusion_matrices(
        n,
        graph,
        confusion_matrices,
        _confidence_interval_length_student_95
    )
