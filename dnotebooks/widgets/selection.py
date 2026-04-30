
from __future__ import annotations

from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import logging
import os

import numpy as np
from ipywidgets import Layout, HBox, VBox, Dropdown, Checkbox, Label
from ipyfilechooser import FileChooser

from dengine.analysis import (
    find_confusion_matrices,
    ExperimentMetric,
    ExperimentConfusionMatrix,
    ExperimentConfusionMatrixDelta
)
from dnotebooks.utils import RegexColorDict

from .miscellaneous import LabelPP, MultiCheckboxWidget, MultiCheckboxWithLimitWidget


@dataclass
class ExperimentGroupMetrics:
    metric_name: str
    group_name: str
    metrics: List[ExperimentMetric]

    def aggregate_mean(self) -> ExperimentMetric:
        m0 = self.metrics[0]
        if len(self.metrics) == 1:
            return ExperimentConfusionMatrix(
                devices_confusion_matrices=m0.data,
                description=m0.description
            )
        stacked_means = np.stack([m.mean() for m in self.metrics])
        return ExperimentConfusionMatrix(
            devices_confusion_matrices=stacked_means,
            description=m0.description,
        )


@dataclass
class ExperimentGroup:
    name: str
    paths: List[Path]

    def __getitem__(self, index: int) -> Path:
        return self.paths[index]

    def load_group_metric_files(self) -> Dict[str, ExperimentGroupMetrics]:
        metrics: Dict[str, ExperimentGroupMetrics] = {}
        for run_output_path in self.paths:
            for metric_name, nodes_metric_files in find_confusion_matrices(run_output_path).items():
                if metric_name not in metrics:
                    metrics[metric_name] = ExperimentGroupMetrics(metric_name, self.name, [])
                metrics[metric_name].metrics.append(
                    ExperimentMetric(nodes_metric_files)
                )
        return metrics


def _load_experiment_group(root: Path) -> Dict[str, ExperimentGroup]:
    root_str = str(root)
    experiments_cfgs = glob(root_str + '/**/config.yaml', recursive=True)

    experiment_map: Dict[str, ExperimentGroup] = {}
    for cfg in experiments_cfgs:
        exp_name = Path(cfg).parent.name
        if exp_name not in experiment_map:
            experiment_map[exp_name] = ExperimentGroup(name=exp_name, paths=[])
        experiment_map[exp_name].paths.append(Path(cfg).parent)
    return experiment_map


def StyledExperimentSelectionWidget(
        root: Path = Path("."),
        limit: Optional[int] = None,
        style_options: Optional[List[str]] = None,
) -> Tuple[VBox, Callable[[], List[Tuple[ExperimentGroup, str]]]]:
    """
    Multi-experiment selection widget with style dropdown per experiment.

    Args:
        root: Root directory to search for experiments
        limit: Optional limit on number of selections
        style_options: List of style options for the dropdown (default: ['solid', 'dashed', 'dotted'])

    Returns:
        Tuple of (widget, getter_function)
        - getter_function returns List[Tuple[Path, style]] ordered by selection time
    """
    if style_options is None:
        style_options = ['solid', 'dashed', 'dotted']

    # Track experiments and their widgets
    experiment_rows: Dict[str, Tuple[ExperimentGroup, HBox, Checkbox, Dropdown]] = {}
    selection_order: OrderedDict[str, int] = OrderedDict()  # name -> selection_timestamp
    selection_counter = [0]  # Mutable counter for tracking selection order
    experiments_container = VBox([])

    def _create_experiment_row(name: str, experiment_group: ExperimentGroup) -> Tuple[HBox, Checkbox, Dropdown]:
        """Create a row with checkbox and style dropdown for an experiment."""
        checkbox = Checkbox(
            value=False,
            description=name,
            indent=False,
            layout=Layout(width='600px')
        )
        count_label = Label(f"({len(experiment_group.paths)} files)", layout=Layout(width='100px'))
        style_dropdown = Dropdown(
            options=style_options,
            value=style_options[0],
            layout=Layout(width='150px'),
            disabled=True  # Disabled until checkbox is selected
        )

        def on_checkbox_change(change):
            if change['new']:  # Selected
                style_dropdown.disabled = False
                if name not in selection_order:
                    selection_order[name] = selection_counter[0]
                    selection_counter[0] += 1
                # Check limit
                if limit and len([n for n in selection_order if experiment_rows[n][2].value]) > limit:
                    checkbox.value = False
                    return
            else:  # Deselected
                style_dropdown.disabled = True
                if name in selection_order:
                    del selection_order[name]

        checkbox.observe(on_checkbox_change, names='value')

        row = HBox([
            checkbox,
            count_label,
            Label('Style:', layout=Layout(width='50px')),
            style_dropdown
        ], layout=Layout(align_items='center', gap='10px'))

        return row, checkbox, style_dropdown

    def _file_chooser_widget_change(*args):
        nonlocal experiment_rows, selection_order

        value = file_chooser_wg.selected_path
        if not value:
            return

        experiments_map = _load_experiment_group(Path(value))

        # Clear previous experiments
        experiment_rows = {}
        selection_order = OrderedDict()
        selection_counter[0] = 0

        # Create rows for each experiment
        rows = []
        for name, group in sorted(experiments_map.items()):
            row, checkbox, dropdown = _create_experiment_row(name, group)
            experiment_rows[name] = (group, row, checkbox, dropdown)
            rows.append(row)

        experiments_container.children = rows

    file_chooser_wg = FileChooser(str(root))
    layout = Layout(justify_content='flex-start', gap='5em')
    file_chooser_widget_box = HBox([
        LabelPP("Select experiments lookup directory:", width="25em"),
        file_chooser_wg
    ], layout=layout)

    def get_selection_paths() -> List[Tuple[ExperimentGroup, str]]:
        """
        Get selected paths with their styles, ordered by selection time.

        Returns:
            List of (Path, style) tuples in selection order
        """
        # Collect selected experiments with their order
        selected: List[Tuple[float, ExperimentGroup, str]] = []
        for name in experiment_rows:
            group, row, checkbox, dropdown = experiment_rows[name]
            if checkbox.value:
                order = selection_order.get(name, float('inf'))
                selected.append((order, group, dropdown.value))  # type: ignore

        # Sort by selection order and return
        selected.sort(key=lambda x: x[0])
        return [(path, style) for _, path, style in selected]

    file_chooser_wg.register_callback(_file_chooser_widget_change)

    return VBox([
        file_chooser_widget_box,
        experiments_container,
    ]), get_selection_paths


def MultiExperimentSelection(
        root: Path = Path("."),
        limit: Optional[int] = None,
) -> Tuple[VBox, Callable[[], List[Path]]]:
    def _file_chooser_widget_change(*args):
        value = file_chooser_wg.selected_path
        if not value:
            return
        experiments_cfgs = glob(value + '/**/config.yaml', recursive=True)
        experiments = {
            Path(cfg).parent.name: Path(cfg).parent for cfg in experiments_cfgs
        }
        experiment_selection_dropdown.update_options(experiments)

    file_chooser_wg = FileChooser(str(root))
    layout = Layout(justify_content='flex-start', gap='5em')
    file_chooser_widget_box = HBox([
        LabelPP("Select experiments lookup directory:", width="25em"),  # type: ignore
        file_chooser_wg
    ], layout=layout)

    if limit:
        experiment_selection_dropdown = MultiCheckboxWithLimitWidget(
            selection_limit=limit
        )
    else:
        experiment_selection_dropdown = MultiCheckboxWidget()

    def get_selection_paths():
        selection = experiment_selection_dropdown.get_selection()
        return [Path(x) for x in selection]

    file_chooser_wg.register_callback(_file_chooser_widget_change)
    return VBox([
        file_chooser_widget_box,
        experiment_selection_dropdown.widget(),
    ]), get_selection_paths


def RegexColorDictFileSelection(
    description: str = "Select yaml colors file:"
):
    def get_selection():
        if file_chooser.value and os.path.exists(file_chooser.value):
            return RegexColorDict(yaml_file_path=Path(file_chooser.value))
        return RegexColorDict()

    layout = Layout(justify_content='flex-start', gap='5em')
    file_chooser = FileChooser()
    widget = HBox([
        LabelPP(description, width="25em"),  # type: ignore
        file_chooser
    ], layout=layout)

    return widget, get_selection


CONFUSION_MATRIX_REL_PATH = "metrics/"
GRAPH_EDGELIST = "networkx/graph.edgelist"


def ConfusionMatrixPartitionSelection(
    root: Path = Path("."),
    description: str = "Select confusion matrix:"
):
    def get_selection():
        return confusion_matrix_selection.value

    dropdown_options = find_confusion_matrices(root)
    dropdown_options["-"] = None  # type: ignore
    layout = Layout(justify_content='flex-start', gap='5em')
    confusion_matrix_selection = Dropdown(
        options=dropdown_options.items(),
        layout=Layout(),
        value=None
    )
    widget = HBox([
        LabelPP(description, width="25em"),  # type: ignore
        confusion_matrix_selection
    ], layout=layout)

    return widget, get_selection


def ConfusionMatrixPartitionDeltaSelection(experiments: List[Path]):
    def get_confusion_matrix():
        confusion_matrices = ExperimentConfusionMatrix(get_confusion_matrix_selection())
        if not confusion_matrices:
            return

        delta_confusion_matrices_selection = get_confusion_matrix_delta_selection()
        if delta_confusion_matrices_selection:
            delta_confusion_matrices = ExperimentConfusionMatrix(delta_confusion_matrices_selection)
            print("Computing the confusion matrix delta")
            min_epoch = min(confusion_matrices.epochs, delta_confusion_matrices.epochs)
            confusion_matrices = ExperimentConfusionMatrixDelta(
                confusion_matrices.truncate_at_epoch_view(min_epoch),
                delta_confusion_matrices.truncate_at_epoch_view(min_epoch)
            )
        return confusion_matrices

    if len(experiments) == 1:
        core_experiment = experiments[0]
        experiment_delta = core_experiment
    elif len(experiments) == 2:
        core_experiment = experiments[0]
        experiment_delta = experiments[1]
    else:
        raise ValueError("Only two experiments are supported")

    confusion_matrix_widget, get_confusion_matrix_selection = ConfusionMatrixPartitionSelection(
        core_experiment
    )
    confusion_matrix_delta_widget, get_confusion_matrix_delta_selection = ConfusionMatrixPartitionSelection(
        experiment_delta,
        description="Select delta: "
    )

    return VBox([
        confusion_matrix_widget,
        confusion_matrix_delta_widget
    ]), get_confusion_matrix


def ConfusionMatrixPartitionMultiSelection(
    experiments_groups: List[ExperimentGroup],
    description: str = "Select confusion matrix:"
):
    def get_selection() -> Optional[Dict[str, ExperimentConfusionMatrix]]:
        if confusion_matrix_selection.value is None:
            return
        matrices = {}
        selection: List[ExperimentGroupMetrics] = confusion_matrix_selection.value
        for group_metrics in selection:
            try:
                matrices[group_metrics.group_name] = group_metrics.aggregate_mean()
            except Exception as e:
                logging.error(f"Unable to load: {group_metrics}: \n{e}")
        return matrices

    dropdown_options: Dict[str, List[ExperimentGroupMetrics]] = defaultdict(list)
    for group in experiments_groups:
        for metric_name, group_metrics in group.load_group_metric_files().items():
            dropdown_options[metric_name].append(group_metrics)

    layout = Layout(justify_content='flex-start', gap='5em')
    confusion_matrix_selection = Dropdown(
        options=dropdown_options.items(),
        layout=Layout(),
        value=None
    )
    widget = HBox([
        LabelPP(description, width="25em"),  # type: ignore
        confusion_matrix_selection
    ], layout=layout)

    return widget, get_selection
