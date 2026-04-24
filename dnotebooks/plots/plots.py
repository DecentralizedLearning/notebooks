from __future__ import annotations
from typing import List, Optional, Sequence, Callable, Dict, Literal, Tuple
from collections import defaultdict
from pathlib import Path
from enum import Enum

from ipywidgets import Output
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib import colors, cm
from networkx.algorithms import descendants_at_distance
from networkx import Graph
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sn
import traceback

from dengine.analysis import (
    ExperimentConfusionMatrix,
    ExperimentPartitions,
)

from dnotebooks.utils import RegexColorDict

from .confusion_matrix_utils import (
    get_n_hop_normalized_confusion_matrices_mean,
    get_n_hop_normalized_confusion_matrices_confidence_interval,
    _confidence_interval_length_student_95
)
from .base import Plot


_CONFUSION_MATRIX_HEATMAP_PRECISION = '.3f'
_MAX_DEVICES_PLOT_LABELS = 5


def moving_average(X, p: float):
    """https://stackoverflow.com/a/14314054"""
    ret = np.cumsum(X, dtype=float)
    n = int(len(X) * p)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def confusion_matrix_heatmap(
    confusion_matrix: np.ndarray,
    ax: Axes,
):
    classes, _ = confusion_matrix.shape
    df_cm = pd.DataFrame(
        confusion_matrix,
        index=[i for i in range(classes)],
        columns=[i for i in range(classes)]
    )
    heatmap_default_args = {
        'cbar': False,
        'annot': True,
        'fmt': _CONFUSION_MATRIX_HEATMAP_PRECISION,
        'ax': ax
    }
    if np.min(confusion_matrix) < 0:
        rdgn = sn.diverging_palette(30, 260, s=100, center='dark', as_cmap=True)
        heatmap = sn.heatmap(
            df_cm,
            cmap=rdgn,
            vmin=-1, vmax=1,
            **heatmap_default_args
        )
    else:
        heatmap = sn.heatmap(df_cm, **heatmap_default_args)

    heatmap.set_xticks(np.arange(classes) + 0.5)
    heatmap.set_yticks(np.arange(classes) + 0.5)

    heatmap.set_xticklabels([str(i) for i in range(classes)], rotation=0)
    heatmap.set_yticklabels([str(i) for i in range(classes)], rotation=0)

    heatmap.set(xlabel='Predicted', ylabel='Ground Truth')
    return heatmap


# def _confusion_matrix_heatmap(
#     confusion_matrix: np.ndarray,
#     ax: Axes,
#     row_indices: list[int] | None = None,
#     col_indices: list[int] | None = None,
# ):
#     classes, _ = confusion_matrix.shape

#     row_indices = row_indices if row_indices is not None else list(range(classes))
#     col_indices = col_indices if col_indices is not None else list(range(classes))

#     subset = confusion_matrix[np.ix_(row_indices, col_indices)]

#     df_cm = pd.DataFrame(
#         subset,
#         index=row_indices,
#         columns=col_indices,
#     )
#     heatmap_default_args = {
#         'cbar': False,
#         'annot': True,
#         'fmt': _CONFUSION_MATRIX_HEATMAP_PRECISION,
#         'ax': ax
#     }
#     rdgn = sn.diverging_palette(30, 260, s=100, center='dark', as_cmap=True)
#     heatmap = sn.heatmap(
#         df_cm,
#         # cmap=rdgn,
#         vmin=0, vmax=1,
#         **heatmap_default_args
#     )

#     heatmap.set_xticks(np.arange(len(col_indices)) + 0.5)
#     heatmap.set_yticks(np.arange(len(row_indices)) + 0.5)

#     heatmap.set_xticklabels([str(i) for i in col_indices], rotation=0)
#     heatmap.set_yticklabels([str(i) for i in row_indices], rotation=0)

#     heatmap.set(xlabel='Predicted', ylabel='Ground Truth')
#     return heatmap


# def confusion_matrix_heatmap(
#     confusion_matrix: np.ndarray,
#     ax: Axes,
#     row_range: Tuple[int, int] | None = (94, 95),
#     col_range: Tuple[int, int] | None = (0, 9),

#     # row_range: Tuple[int, int] | None = (20, 21),
#     # col_range: Tuple[int, int] | None = (69, 78),
# ):
#     row_indices = list(range(row_range[0], row_range[1])) if row_range else None
#     col_indices = list(range(col_range[0], col_range[1])) if col_range else None
#     return _confusion_matrix_heatmap(
#         confusion_matrix, ax,
#         row_indices, col_indices
#     )

def partitions_histogram(
    partitions: ExperimentPartitions,
    ax: Axes,
    devices: Sequence[int] = [],
):
    if (len(devices) == 0) or (len(devices) > _MAX_DEVICES_PLOT_LABELS):
        if len(devices) == 0:
            ax.set_title('Train dataset distribution')
        else:
            ax.set_title(f'Train dataset distribution for >{len(devices)} devices')

        devices = partitions.devices()
        ax.bar(
            partitions.targets(),
            partitions.train(),
            alpha=.2,
        )
        ax.set_yscale('log')
        return

    total_width = .5
    bar_width = total_width / len(devices)
    targets = partitions.targets().astype(int)
    for i, dev_id in enumerate(devices):
        ax.bar(
            (i * bar_width) + targets,
            partitions.train(dev_id),
            label=dev_id,
            width=bar_width,
        )

    ax.set_xticks(targets + (total_width / 2))
    ax.set_xticklabels(partitions.targets())
    ax.set_yscale('log')
    ax.set_title('Devices class samples distribution')

    if len(devices) > 10:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend()


def plot_f1(
    ax: Axes,
    confusion_matrices: ExperimentConfusionMatrix,
    label_idx: int,
    devices: Sequence[int] = [],
    **kwargs
):
    round_f1 = confusion_matrices.f1(label_idx)
    plot_metric_over_time(ax, round_f1, devices, **kwargs)


def plot_recall(
    ax: Axes,
    confusion_matrices: ExperimentConfusionMatrix,
    label_idx: int,
    devices: Sequence[int] = [],
    **kwargs
):
    round_recall = confusion_matrices.recall(label_idx)
    plot_metric_over_time(ax, round_recall, devices, **kwargs)


def plot_precision(
    ax: Axes,
    confusion_matrices: ExperimentConfusionMatrix,
    label_idx: int,
    devices: Sequence[int] = [],
    **kwargs
):
    round_precision = confusion_matrices.precision(label_idx)
    plot_metric_over_time(ax, round_precision, devices, **kwargs)


def plot_accuracy(
    ax: Axes,
    confusion_matrices: ExperimentConfusionMatrix,
    devices: Sequence[int] = [],
    **kwargs
):
    round_accuracies = confusion_matrices.accuracy()
    plot_metric_over_time(ax, round_accuracies, devices, **kwargs)


def plot_metric_over_time(
    ax: Axes,
    X: np.ndarray,
    devices: Sequence[int] = [],
):
    rounds = X.shape[1]
    plot_has_many_devices = len(devices) > _MAX_DEVICES_PLOT_LABELS
    if len(X) == 0:
        return

    if len(devices):
        color = "#0d47a1" if plot_has_many_devices else None
        alpha = .2 if plot_has_many_devices else .8
        mean_linestyle = ':'
        accuracies_view = zip(
            X[devices],
            devices,
            [color] * rounds
        )
    else:
        color = "#0d47a1"
        alpha = .2
        mean_linestyle = '-'
        accuracies_view = zip(
            X,
            [None] * rounds,
            [color] * rounds
        )

    for idx, (acc, label, c) in enumerate(accuracies_view):
        ax.plot(range(rounds), acc, alpha=alpha, color=c, label=label)
        ax.set_ylim((0, 1.2))

    ax.plot(
        range(rounds),
        X.mean(axis=0),
        color="red",
        label="mean",
        linestyle=mean_linestyle,
    )
    if not plot_has_many_devices:
        ax.legend()


def plot_partition_distribution(
    partitions: ExperimentPartitions,
    ax: Axes,
    selection: Optional[List[int]] = None
):
    data = partitions.train_df().copy()
    if selection:
        data = data.iloc[selection]
    data["Total Samples"] = data.values.sum(axis=1)
    mask = np.zeros_like(data.values)
    mask[:, -1] = True
    heatmap = sn.heatmap(
        data,
        annot=True,
        fmt='d',
        ax=ax,
        cbar=False,
        norm=LogNorm(),
        cmap='Blues',
        mask=mask
    )
    sn.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt='d',
        cbar=False,
        norm=LogNorm(),
        cmap='gray_r',
        mask=np.logical_not(mask),
    )

    heatmap.set(xlabel='Class', ylabel='PAIV-ID')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)


def plot_scatterplot_partition_distribution(
    partitions: ExperimentPartitions,
    ax: Axes,
    selection: Optional[List[int]] = None
):
    df = partitions.train_df().copy()
    if selection:
        df = df.iloc[selection]

    df.index.name = 'nodeid'
    df = df.reset_index().melt(id_vars='nodeid', var_name='digit', value_name='count')

    # Set scatterplot sizes
    sorted_unique_count = sorted(df['count'].unique())
    if len(sorted_unique_count) > 1:
        count_size_idx = df['count'].apply(lambda x: sorted_unique_count.index(x))
        rescaled_count_size_idx = count_size_idx / (len(sorted_unique_count) - 1)
        scatterplot_sizes = rescaled_count_size_idx * 100 + 15
        df['scatterplot_size'] = scatterplot_sizes
    else:
        df['scatterplot_size'] = 50

    # Generate unique colors for each nodeid
    unique_nodes = df['digit'].unique()
    colors = plt.colormaps["tab20"](np.linspace(0, 1, len(unique_nodes)))
    color_map = dict(zip(unique_nodes, colors))
    df['color'] = df['digit'].map(color_map)

    # Scatter
    for nodeid in unique_nodes:
        node_data = df[df['digit'] == nodeid]
        ax.scatter(
            x=node_data['nodeid'],
            y=node_data['digit'],
            s=node_data['scatterplot_size'],
            c=[node_data['color'].iloc[0]],
            alpha=1.,
            edgecolors='w',
        )
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Class ID")

    # Legend
    unique_count_and_sizes = (
        df[["count", "scatterplot_size"]]
        .drop_duplicates()
        .sort_values("count")
    )
    for _, (cnt, sz) in unique_count_and_sizes.iterrows():
        ax.scatter([], [], s=sz, c='black', alpha=1., label=int(cnt))
    ax.legend(
        title="Count Sizes",
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        fancybox=False,
        shadow=True,
        ncol=5
    )


def plot_n_hops_confusion_matrix_mean_heatmap(
    ax: Axes,
    confusion_matrices: ExperimentConfusionMatrix,
    graph: Graph,
    selection: Optional[List[int]] = None,
    hops: int = 0,
    nodes_reduction_fun: Callable[[np.ndarray, int], np.ndarray] = np.mean
):
    cf = get_n_hop_normalized_confusion_matrices_mean(
        hops,
        graph,
        confusion_matrices.epoch_view([-1])
    )
    if selection is not None and len(selection) > 0:
        cf = cf.device_view(selection)

    m = nodes_reduction_fun(cf.confusion_matrix(normalize=False), 0).squeeze()
    ax.set_title(
        _confusion_matrix_description(
            confusion_matrices,
            selection,
            hops,
            nodes_reduction_fun=nodes_reduction_fun
        )
    )
    confusion_matrix_heatmap(m, ax)


def plot_n_hops_confusion_matrix_heatmap_delta(
    ax: Axes,
    confusion_matrices: ExperimentConfusionMatrix,
    graph: Graph,
    selection: Optional[List[int]] = None,
    hops: int = 0,
):
    confusion_matrices = confusion_matrices.epoch_view([-1])
    nhop_cf = get_n_hop_normalized_confusion_matrices_mean(
        hops,
        graph,
        confusion_matrices
    )
    if selection is not None and len(selection) > 0:
        nhop_cf = nhop_cf.device_view(selection)
        confusion_matrices = confusion_matrices.device_view(selection)

    cf_mean = np.mean(confusion_matrices.confusion_matrix(), axis=0).squeeze()
    n_hop_cf_mean = np.mean(nhop_cf.confusion_matrix(), axis=0).squeeze()
    delta = cf_mean - n_hop_cf_mean

    ax.set_title(
        _confusion_matrix_description(
            confusion_matrices,
            selection,
            hops,
            prefix='Confusion matrix delta'
        )
    )
    confusion_matrix_heatmap(delta, ax)


def plot_n_hops_confusion_matrix_confidence_interval_heatmap(
    ax: Axes,
    confusion_matrices: ExperimentConfusionMatrix,
    graph: Graph,
    selection: Optional[List[int]] = None,
    hops: int = 0,
):
    confusion_matrices = confusion_matrices.epoch_view([-1])
    ci = get_n_hop_normalized_confusion_matrices_confidence_interval(
        hops,
        graph,
        confusion_matrices
    )
    if selection is not None and len(selection) > 0:
        ci = ci.device_view(selection)

    m = np.mean(ci.confusion_matrix(), axis=0).squeeze()
    ax.set_title(
        _confusion_matrix_description(
            confusion_matrices,
            selection,
            hops,
            prefix='Confusion matrix mean confidence interval'
        )
    )
    confusion_matrix_heatmap(m, ax)


def _confusion_matrix_description(
    confusion_matrices: ExperimentConfusionMatrix,
    selection: Optional[List[int]] = None,
    hops: int = 0,
    prefix: str = "confusion matrix",
    nodes_reduction_fun: Callable[[np.ndarray, int], np.ndarray] = np.mean
) -> str:
    descr = ""
    if desc := getattr(confusion_matrices, 'description', None):
        descr = f"{desc}\n"

    if (selection is None) or (len(selection) == 0):
        descr += f"{nodes_reduction_fun.__name__} {prefix} for the whole network"
    elif len(selection) == 1:
        descr += f"{prefix} for device {selection[0]}"
    elif len(selection) > _MAX_DEVICES_PLOT_LABELS:
        descr += f"{nodes_reduction_fun.__name__} {prefix} for >{len(selection)} devices"
    else:
        descr += f"{nodes_reduction_fun.__name__} {prefix} for devices {selection}"

    descr += f" at epoch {confusion_matrices.rounds}"

    if hops > 0:
        descr += f" ({hops}-hops neighbors)"

    return descr


class PartitionsHistogram(Plot):
    def __init__(
        self,
        confusion_matrices: Optional[ExperimentConfusionMatrix],
        partitions: ExperimentPartitions,
        interactive_graph: InteractiveGraph,
        output_directory: Path = Path('res/')
    ):
        self._output_dir = output_directory
        self._confusion_matrices = confusion_matrices
        self._partitions = partitions
        self._graph = interactive_graph

        self.fig, _ = plt.subplots()
        self._ax = self.fig.axes[0]

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(
            self._output_dir / 'hist.svg',
            transparent=True,
            bbox_inches='tight',
            pad_inches=0
        )

    def draw(self):
        self._ax.cla()
        partitions_histogram(
            self._partitions,
            self._ax,
            self._graph.selection_with_neighbors
        )


class MetricPlot(Plot):
    class SupportedPlots(Enum):
        accuracy = 'accuracy'
        class_precision = 'precision'
        class_recall = 'recall'
        class_f1 = 'f1'

    PLOT_MAPPING = {
        SupportedPlots.class_precision.value: plot_precision,
        SupportedPlots.class_recall.value: plot_recall,
        SupportedPlots.accuracy.value: plot_accuracy,
        SupportedPlots.class_f1.value: plot_f1
    }

    def __init__(
        self,
        confusion_matrices: ExperimentConfusionMatrix,
        interactive_graph: InteractiveGraph,
        output_directory: Path = Path('res/'),
        custom_colors: Optional[RegexColorDict] = None
    ):
        self._custom_colors = custom_colors if custom_colors else RegexColorDict()
        self._output_dir = output_directory

        self._graph = interactive_graph
        self._metric = self.SupportedPlots.accuracy
        self._metric_class_label = None

        self.fig, _ = plt.subplots()
        plt.show()
        self._ax = self.fig.axes[0]

        self._confusion_matrices = confusion_matrices

        self._epoch = self._confusion_matrices.rounds - 1
        self._ylimit = None

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value
        self.draw()

    @property
    def ylimit(self) -> Optional[tuple]:
        return self._ylimit

    @ylimit.setter
    def ylimit(self, value: int | tuple):
        if isinstance(value, int):
            value = (0, value)
        self._ylimit = value
        self.draw()

    @property
    def metric(self) -> SupportedPlots:
        return self._metric

    @metric.setter
    def metric(self, value: SupportedPlots):
        self._metric = value
        self.draw()

    @property
    def metric_class_label(self) -> Optional[int]:
        return self._metric_class_label

    @metric_class_label.setter
    def metric_class_label(self, value: int):
        self._metric_class_label = value
        self.draw()

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(
            self._output_dir / 'plot.svg',
            transparent=True,
            bbox_inches='tight',
            pad_inches=0
        )

    def draw(self):
        self._ax.cla()
        confusion_matrices = self._confusion_matrices.truncate_at_epoch_view(self.epoch)

        if self.metric == self.SupportedPlots.accuracy:
            plot_accuracy(
                self._ax,
                confusion_matrices,
                self._graph.selection_with_neighbors,
            )
            return

        if self.metric_class_label is None:
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._ax.text(
                .5,
                .5,
                'Set class label to visualize this plot \n',
                ha='center',
                va='center',
                bbox=dict(fc='blue', alpha=.5)
            )
            return

        self._ax.set_title(f'{self.metric.value} class "{self.metric_class_label}"')
        self.PLOT_MAPPING[self.metric.value](
            ax=self._ax,
            confusion_matrices=confusion_matrices,
            devices=self._graph.selection,
            label_idx=self.metric_class_label,
        )


class MultiExperimentMetricComparisonPlot(MetricPlot):
    class SupportedAggregation(Enum):
        min_max_mean = 'min_max_mean'
        confidence_interval = 'confidence_interval'

    class SupportedPlots(Enum):
        accuracy = 'accuracy'
        class_precision = 'precision'
        class_recall = 'recall'
        class_f1 = 'f1'

    PLOT_MAPPING = {
        SupportedPlots.class_precision.value: plot_precision,
        SupportedPlots.class_recall.value: plot_recall,
        SupportedPlots.accuracy.value: plot_accuracy,
        SupportedPlots.class_f1.value: plot_f1
    }

    def __init__(
        self,
        confusion_matrices: Dict[str, List[ExperimentConfusionMatrix]],
        output_directory: Path = Path('res/'),
        smoothing: float = 0,
        custom_colors: Optional[RegexColorDict] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        external_legend: bool = False,
        linestyles: Dict[str, Optional[str]] = defaultdict(None),
    ):
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._external_legend = external_legend
        self._linestyles = linestyles

        self._custom_colors = custom_colors if custom_colors else RegexColorDict()
        self._output_dir = output_directory

        self._aggregation = self.SupportedAggregation.confidence_interval
        self._metric = self.SupportedPlots.accuracy
        self._metric_class_label = None

        self.fig, _ = plt.subplots()
        plt.show()
        self._ax = self.fig.axes[0]

        self._confusion_matrices = confusion_matrices
        self._smoothing = smoothing

        epochs = [
            cf.epochs for exp_seeds in self._confusion_matrices.values()
            for cf in exp_seeds
        ]
        self._epoch = (0, max([x for x in epochs if x > 1]))
        self._ylimit = None

    @property
    def epoch(self) -> tuple:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int | tuple):
        if isinstance(value, int):
            value = (0, value)
        self._epoch = value
        self.draw()

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value: float):
        self._smoothing = value
        self.draw()

    @property
    def metric(self) -> SupportedPlots:
        return self._metric

    @metric.setter
    def metric(self, value: SupportedPlots):
        self._metric = value
        self.draw()

    @property
    def aggregation(self) -> SupportedAggregation:
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value: SupportedAggregation):
        self._aggregation = value
        self.draw()

    @property
    def metric_class_label(self) -> Optional[int]:
        return self._metric_class_label

    @metric_class_label.setter
    def metric_class_label(self, value: int):
        self._metric_class_label = value
        self.draw()

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.fig.savefig(
            self._output_dir / 'experiments_comparison_plot.png',
            transparent=True,
            bbox_inches='tight',
            pad_inches=0,
            dpi=250
        )
        self.fig.savefig(
            self._output_dir / 'experiments_comparison_plot.svg',
            transparent=True,
            bbox_inches='tight',
            pad_inches=0
        )

    def draw(self):
        self._ax.cla()

        # Something went wrong, show error message
        if (
            (self.metric.name != self.SupportedPlots.accuracy.name)
            and (self.metric_class_label is None)
        ):
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._ax.text(
                .5,
                .5,
                'Set class label to visualize this plot \n',
                ha='center',
                va='center',
                bbox=dict(fc='blue', alpha=.5)
            )
            return

        max_time = [x[0].time_data for x in self.confusion_matrices.values()]
        max_time = [np.max(x) for x in max_time if x is not None]
        max_time = max(max_time) if len(max_time) > 0 else None

        for label, seeds_confusion_matrices in self.confusion_matrices.items():
            Y = stack_mean_node_metric(
                seeds_confusion_matrices,
                self.epoch[0], self.epoch[1],
                self.metric.value,
                self.metric_class_label
            )

            if self._aggregation.value == self.SupportedAggregation.min_max_mean.value:
                Y = Y.mean(axis=0)
                Y_upper_bound = Y.max(axis=0)
                Y_lower_bound = Y.min(axis=0)
            else:
                Y_cf = _confidence_interval_length_student_95(Y)
                Y = Y.mean(axis=0)
                Y_upper_bound = Y + Y_cf
                Y_lower_bound = Y - Y_cf

            if self._smoothing:
                Y = moving_average(Y, self.smoothing)
                Y_upper_bound = moving_average(Y_upper_bound, self.smoothing)
                Y_lower_bound = moving_average(Y_lower_bound, self.smoothing)

            X = reduce_time_components(
                seeds_confusion_matrices,
                self.epoch[0], self.epoch[1]
            )[:len(Y)]

            linestyle = self._linestyles[label]
            c = self._custom_colors.get(label)
            self._ax.fill_between(X, Y_lower_bound, Y_upper_bound, color=c, alpha=.1)

            if self.ylimit:
                self._ax.set_ylim(ymin=self.ylimit[0], ymax=self.ylimit[1])
            self._ax.plot(X, Y, color=c, label=label, alpha=1, linestyle=linestyle)

        if self._xlabel:
            self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self.metric.name)
        if self._title:
            self._ax.set_title(self._title)

        if self._external_legend:
            self._ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            self._ax.legend()

    @property
    def title(self) -> str:
        title = self.metric.value
        if self.metric.name != self.SupportedPlots.accuracy.name:
            title += f' class "{self.metric_class_label}"'
        if self.smoothing > 0:
            title += f'(smoothing {self.smoothing})'
        title = f"{self._title} ({title})"
        return title

    @property
    def confusion_matrices(self) -> Dict[str, List[ExperimentConfusionMatrix]]:
        confusion_matrices = {}
        for label, exp_seeds in self._confusion_matrices.items():
            exp_seeds_confusion_matrices = []
            for ith_seed_confusion_matrix in exp_seeds:
                cf = ith_seed_confusion_matrix.truncate_at_epoch_view(self.epoch)
                exp_seeds_confusion_matrices.append(cf)
            confusion_matrices[label] = exp_seeds_confusion_matrices
        return confusion_matrices


def reduce_time_components(
    experiments: List[ExperimentConfusionMatrix],
    start: int,
    end: int,
) -> np.ndarray:
    cf = experiments[0]
    if cf.time_data is not None:
        X = cf.time_data[start:end]
        assert np.all(X[:] == X[0])
        X = X[0]
    else:
        X = np.array(range(start, end), dtype=int)
    return X


def stack_mean_node_metric(
    experiments: List[ExperimentConfusionMatrix],
    start: int,
    end: int,
    metric: Literal['accuracy', 'precision', 'recall', 'f1'],
    class_label=None
) -> np.ndarray:
    exps_metric: List[np.ndarray] = []
    for cf in experiments:
        if metric == "accuracy":
            X = cf.accuracy()
        else:
            assert class_label is not None
            if metric == "precision":
                X = cf.precision(class_label)
            elif metric == "recall":
                X = cf.recall(class_label)
            elif metric == "f1":
                X = cf.f1(class_label)

        if X.shape[-1] == 1:
            X = np.zeros((1, end - start)) + X

        if len(experiments) == 1:
            return X

        exps_metric.append(
            np.mean(X, axis=0)
        )
    return np.stack(exps_metric)


class ConfusionMatrixHeatmap(MetricPlot):
    class Reductions(Enum):
        mean = 'mean'
        max = 'max'
        min = 'min'

    REDUCTION_FUNCTION_MAPPING = {
        Reductions.max: np.max,
        Reductions.min: np.min,
        Reductions.mean: np.mean
    }

    def __init__(
        self,
        confusion_matrices: ExperimentConfusionMatrix,
        partitions: ExperimentPartitions,
        interactive_graph: InteractiveGraph,
        output_directory: Path = Path('res/')
    ):
        self._output_dir = output_directory
        self._selection_hops = 0
        self._output_filename_postfix = "confusion_matrix"

        self._graph = interactive_graph
        self._fig, _ = plt.subplots()
        plt.show()
        self._reduction_function = np.mean
        self._ax = self._fig.axes[0]

        self._confusion_matrices = confusion_matrices
        self._partitions = partitions

        self._epoch = self._confusion_matrices.rounds - 1

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value
        self.draw()

    @property
    def reduction_function(self) -> Callable:
        return self._reduction_function

    @reduction_function.setter
    def reduction_function(self, value: Reductions):
        self._reduction_function = self.REDUCTION_FUNCTION_MAPPING[value]
        self.draw()

    @property
    def selection_hops(self):
        return self._selection_hops

    @selection_hops.setter
    def selection_hops(self, n: int):
        self._selection_hops = n
        self.draw()

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)

        fname = _confusion_matrix_description(self._confusion_matrices, nodes_reduction_fun=self._reduction_function)
        self._fig.savefig(
            self._output_dir / f'{fname}.svg',
            transparent=True,
            bbox_inches='tight',
            pad_inches=0
        )

    def draw(self):
        confusion_matrices = self._confusion_matrices.truncate_at_epoch_view(self._epoch)
        self._ax.cla()

        plot_n_hops_confusion_matrix_mean_heatmap(
            self._ax,
            confusion_matrices,
            self._graph.graph,
            list(self._graph.selection),
            self.selection_hops,
            nodes_reduction_fun=self.reduction_function
        )


class ConfusionMatrixHeatmapDelta(ConfusionMatrixHeatmap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ax.set_title('Confusion matrix delta heatmap')
        self._output_filename_postfix = "confusion_matrix_delta"

    def draw(self):
        self._ax.cla()

        if (len(self._graph.selection) == 0) or (self.selection_hops == 0):
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._ax.text(
                .5,
                .5,
                'You are seing this because either no node is selected \n'
                'or because the selection hops are set to zero. Set \n'
                'these values to have this plot drawn.',
                ha='center',
                va='center',
                bbox=dict(fc='blue', alpha=.5)
            )
            return

        self._ax.set_visible(True)
        self._ax.set_visible(True)

        plot_n_hops_confusion_matrix_heatmap_delta(
            self._ax,
            self._confusion_matrices,
            self._graph.graph,
            list(self._graph._selected),
            self.selection_hops
        )


class ConfusionMatrixHeatmapConfidenceInterval(ConfusionMatrixHeatmap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ax.set_title('Confusion matrix confidence interval heatmap')
        self._output_filename_postfix = "confusion_matrix_ci"

    def draw(self):
        self._ax.cla()

        if (len(self._graph.selection) == 0) or (self.selection_hops == 0):
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._ax.text(
                .5,
                .5,
                'You are seing this because either no node is selected \n'
                'or because the selection hops are set to zero. Set \n'
                'these values to have this plot drawn.',
                ha='center',
                va='center',
                bbox=dict(fc='blue', alpha=.5)
            )
            return

        self._ax.set_visible(True)
        self._ax.set_visible(True)

        plot_n_hops_confusion_matrix_confidence_interval_heatmap(
            self._ax,
            self._confusion_matrices,
            self._graph.graph,
            list(self._graph._selected),
            self.selection_hops
        )


class InteractiveGraph(MetricPlot):
    GRAPH_EDGELIST = "networkx/graph.edgelist"

    class SupportedNetworkXNodeColors(Enum):
        default = ''
        accuracy = 'accuracy'
        f1 = 'f1'
        rescaled_accuracy = 'rescaled accuracy'
        training_set_size = 'training set size'

    def __init__(
        self,
        confusion_matrices: Optional[ExperimentConfusionMatrix],
        partitions: ExperimentPartitions,
        graph: Graph,
        selection_hops: int = 0,
        output_directory: Path = Path('res/'),
        callback: Optional[Callable[[], None]] = None,
    ):
        self._output_dir = output_directory
        self._node_color = self.SupportedNetworkXNodeColors.default
        self._selection_hops = selection_hops
        self._callback = callback
        self._metric_class_label = None

        self._fig_graph, _ = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 0.1]})
        plt.show()
        self._exception_output = Output()
        self._ax_networkx_graph = self._fig_graph.axes[0]
        self._ax_cmap = self._fig_graph.axes[1]

        self.graph = graph
        self._cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["#ff8a65", "#4fc3f7"])

        self._G = graph
        if len(self._G.nodes) == 0:
            self._G.add_nodes_from([x for x in partitions.devices()])
        self._selected = set()
        self._confusion_matrices = confusion_matrices
        self._partitions = partitions

        if self._confusion_matrices:
            self._epoch = self._confusion_matrices.rounds - 1
        else:
            self._epoch = 0

        self._fig_graph.canvas.mpl_connect('button_press_event', self._onclick_print_execpt)

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value
        self.draw()

    @property
    def selection(self):
        return [x for x in list(self._selected)]

    @property
    def selection_with_neighbors(self):
        selection = set(self.selection)
        if len(self.graph.edges) == 0:
            return list(selection)

        for src in self.selection:
            selection.update(
                descendants_at_distance(self.graph, src, self.selection_hops)
            )
        return list(selection)

    @property
    def figure(self) -> Figure:
        return self._fig_graph

    @property
    def colorby(self):
        return self._node_color

    @colorby.setter
    def colorby(self, colorby: SupportedNetworkXNodeColors):
        self._node_color = colorby
        self.draw()

    @property
    def selection_hops(self):
        return self._selection_hops

    @selection_hops.setter
    def selection_hops(self, n: int):
        self._selection_hops = n
        self.draw()

    @property
    def supported_networkX_node_colors(self):
        supported_colors = []
        for e in self.SupportedNetworkXNodeColors:
            supported_colors.append(e)

        return supported_colors

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._fig_graph.savefig(
            self._output_dir / 'graph.svg',
            transparent=True,
            bbox_inches='tight',
            pad_inches=0
        )

    def draw(self):
        self._ax_networkx_graph.cla()
        self._ax_networkx_graph.axis('off')
        pos = nx.spring_layout(self._G, iterations=200, k=0.6, seed=3)
        self._nodes = nx.draw_networkx_nodes(
            self._G,
            pos=pos,
            node_color=self._color_map(),  # type: ignore
            edgecolors=self._edge_colors(),  # type: ignore
            ax=self._ax_networkx_graph
        )
        nx.draw_networkx_labels(self._G, pos=pos, ax=self._ax_networkx_graph, labels=self.get_labels())
        nx.draw_networkx_edges(self._G, pos=pos, ax=self._ax_networkx_graph)

        if self.colorby.value == self.SupportedNetworkXNodeColors.training_set_size.value:
            self._ax_networkx_graph.set_title(self.colorby.value)
        elif self.colorby.value == self.SupportedNetworkXNodeColors.default.value:
            pass
        else:
            self._ax_networkx_graph.set_title(
                f"{self.colorby.value} at epoch {self.epoch}"
            )

    def get_labels(self):
        if self.colorby.value == self.SupportedNetworkXNodeColors.training_set_size.value:
            p = (
                self._partitions.train_df().sum(axis=1)
            )
            return {id: v for (id, v) in enumerate(p.values.tolist())}

        return None

    def _edge_colors(self) -> Optional[List[Optional[str]]]:
        if (
            (self._selection_hops == 0) or
            (len(self._selected) != 1)
        ):
            return

        cmap: List[Optional[str]] = ['#ffffff'] * len(self._G.nodes)
        neighbors = descendants_at_distance(
            self._G,
            list(self._selected)[0],
            self._selection_hops
        )
        for nid in neighbors:
            cmap[nid] = '#dd2c00'
        return cmap

    def _node_color_values(self):
        self._ax_cmap.clear()
        self._ax_cmap.set_axis_off()

        if (
            (self.colorby.value == self.SupportedNetworkXNodeColors.default.value)
        ):
            return np.ones(len(self._partitions.devices()))

        if self.colorby.value == self.SupportedNetworkXNodeColors.training_set_size.value:
            color_metric = self._partitions.train_df().values.sum(axis=1).astype(np.float16)
            max_accuracy = max(color_metric)
            color_metric /= max_accuracy
            return color_metric

        if self._confusion_matrices is None:
            raise ValueError('Missing experiment confusion matrix')

        confusion_matrices = self._confusion_matrices.truncate_at_epoch_view(self.epoch)

        if self.colorby.value == self.SupportedNetworkXNodeColors.accuracy.value:
            return confusion_matrices.accuracy()[:, -1]

        if (
            (self.colorby.value == self.SupportedNetworkXNodeColors.f1.value)
            and self.metric_class_label
        ):
            norm = colors.Normalize(vmin=0, vmax=1)
            sm = cm.ScalarMappable(cmap=self._cmap, norm=norm)
            self.figure.colorbar(sm, cax=self._ax_cmap)
            self._ax_cmap.set_axis_on()
            return confusion_matrices.f1(self.metric_class_label).mean(axis=-1)

        if self.colorby.value == self.SupportedNetworkXNodeColors.rescaled_accuracy.value:
            color_metric = confusion_matrices.accuracy()[:, -1]
            min_v = np.min(color_metric)
            max_v = np.max(color_metric)
            color_metric = (color_metric - min_v) / (max_v - min_v)
            return color_metric

        raise ValueError(f'Unsupported colorby value: {self.colorby}')

    def _color_map(self) -> List[str]:
        color_map = []

        color_metric = self._node_color_values()

        for id, data in self._G.nodes(data=True):
            data['title'] = str(id)
            data['label'] = str(id)
            if id in self._selected:
                color = '#4caf50'
            else:
                try:
                    device_accuracy = color_metric[id]
                    rgb_color = self._cmap(device_accuracy)
                    color = colors.rgb2hex(rgb_color)  # type: ignore
                except IndexError:
                    color = '#ffffff'
            color_map.append(color)
        return color_map

    def _onclick_print_execpt(self, *args, **kwargs):
        with self._exception_output:
            try:
                self._onclick(*args, **kwargs)
            except Exception as e:
                traceback.print_stack()
                print(e)

    def _onclick(self, event, *args, **kwargs):
        contains, data = self._nodes.contains(event)
        if not contains:
            return
        id = int(data['ind'][0])
        if id in self._selected:
            self._selected.remove(id)
        else:
            self._selected.add(id)

        if self._callback:
            self._callback()

        self.draw()

    def reset_selection(self):
        self._selected = set()
        self.draw()
