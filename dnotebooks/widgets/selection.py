
from __future__ import annotations

from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable
import logging
import os

from ipywidgets import Layout, HBox, VBox, Dropdown, Checkbox, Label
from ipyfilechooser import FileChooser

from dengine.analysis import (
    find_confusion_matrices,
    ExperimentConfusionMatrix,
    ExperimentConfusionMatrixDelta
)
from dnotebooks.utils import RegexColorDict

from .miscellaneous import LabelPP, MultiCheckboxWidget, MultiCheckboxWithLimitWidget


def StyledExperimentSelectionWidget(
        root: Path = Path("."),
        limit: Optional[int] = None,
        style_options: Optional[List[str]] = None,
) -> Tuple[VBox, Callable[[], List[Tuple[Path, str]]]]:
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
    experiment_rows: Dict[str, Tuple[Path, HBox, Checkbox, Dropdown]] = {}
    selection_order: OrderedDict[str, int] = OrderedDict()  # name -> selection_timestamp
    selection_counter = [0]  # Mutable counter for tracking selection order
    experiments_container = VBox([])

    def _create_experiment_row(name: str, path: Path) -> Tuple[HBox, Checkbox, Dropdown]:
        """Create a row with checkbox and style dropdown for an experiment."""
        checkbox = Checkbox(
            value=False,
            description=name,
            indent=False,
            layout=Layout(width='600px')
        )

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
            Label('Style:', layout=Layout(width='50px')),
            style_dropdown
        ], layout=Layout(align_items='center', gap='10px'))

        return row, checkbox, style_dropdown

    def _file_chooser_widget_change(*args):
        nonlocal experiment_rows, selection_order

        value = file_chooser_wg.selected_path
        if not value:
            return

        experiments_cfgs = glob(value + '/**/config.yaml', recursive=True)
        experiments_map = {
            Path(cfg).parent.name: Path(cfg).parent for cfg in experiments_cfgs
        }

        # Clear previous experiments
        experiment_rows = {}
        selection_order = OrderedDict()
        selection_counter[0] = 0

        # Create rows for each experiment
        rows = []
        for name, path in sorted(experiments_map.items()):
            row, checkbox, dropdown = _create_experiment_row(name, path)
            experiment_rows[name] = (path, row, checkbox, dropdown)
            rows.append(row)

        experiments_container.children = rows

    file_chooser_wg = FileChooser(str(root))
    layout = Layout(justify_content='flex-start', gap='5em')
    file_chooser_widget_box = HBox([
        LabelPP("Select experiments lookup directory:", width="25em"),
        file_chooser_wg
    ], layout=layout)

    def get_selection_paths() -> List[Tuple[Path, str]]:
        """
        Get selected paths with their styles, ordered by selection time.

        Returns:
            List of (Path, style) tuples in selection order
        """
        # Collect selected experiments with their order
        selected = []
        for name in experiment_rows:
            path, row, checkbox, dropdown = experiment_rows[name]
            if checkbox.value:
                order = selection_order.get(name, float('inf'))
                selected.append((order, path, dropdown.value))

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
    experiments: List[Path],
    description: str = "Select confusion matrix:"
):
    def get_selection():
        if confusion_matrix_selection.value is None:
            return
        matrices = {}
        for (exp, x) in zip(experiments, confusion_matrix_selection.value):
            try:
                matrices[exp] = ExperimentConfusionMatrix(x)
            except Exception as e:
                logging.error(f"Unable to load: {exp}: \n{e}")
        return matrices

    dropdown_options = defaultdict(list)
    for exp in experiments:
        exp_cf_files = find_confusion_matrices(exp)
        for key, key_files in exp_cf_files.items():
            dropdown_options[key].append(key_files)

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
