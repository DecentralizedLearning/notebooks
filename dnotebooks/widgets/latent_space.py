from __future__ import annotations

from pathlib import Path
import functools
from collections import defaultdict
from typing import Optional
import traceback

from ipywidgets import HBox, VBox, HTML, Dropdown, Button, Output
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, Subset
import markdown
import matplotlib.pyplot as plt
import numpy as np
import torch

from dengine.analysis import Experiment
from dengine.utils.utils import model_on_device_context, get_output_in_production
from dengine.dataset import SupervisedDataset
from dengine.graph import NXGraph

from dnotebooks.plots import InteractiveGraph

from .miscellaneous import LabelPP


def _get_module_by_path(model, path):
    return functools.reduce(getattr, path.split("."), model)


def _get_layer_activations(
    net: Module,
    D: SupervisedDataset,
    layer: str,
    batch_size: int = 128,
) -> np.ndarray:
    activation = defaultdict(list)

    def get_activation(name):
        def hook(model, input, output):
            if output is None:
                return
            activation[name].append(output)
        return hook

    # Access nested module
    target_module = _get_module_by_path(net, layer)
    handle = target_module.register_forward_hook(get_activation(layer))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with model_on_device_context(net, torch.device(device)):
        get_output_in_production(
            net,
            D,
            None,
            batch_size=batch_size
        ).squeeze()
    handle.remove()

    return torch.concat(activation[layer]).cpu().numpy()


class LayerActivationDashboard:
    def __init__(
        self,
        experiment: Experiment,
    ):
        self._experiment = experiment
        self._output_dir = Path('res/')

        assert isinstance(experiment.network_graph, NXGraph)
        self.interactive_graph = InteractiveGraph(
            confusion_matrices=None,
            partitions=experiment.partitions,
            graph=experiment.network_graph.nx_graph,
            output_directory=self._output_dir,
            callback=self._render_figures
        )

        layer_keys = [name for name, module in self._get_model('0').named_modules()]
        self._layer_selection_dropdown = Dropdown(
            options=layer_keys
        )

    def _get_model(self, id: str):
        return self._experiment.training_engine.clients[id].model

    def _generate_widget(self):
        descr_label_size = '12em'
        widget = [
            HTML(markdown.markdown("<hr>")),
            HTML(markdown.markdown("<hr>")),
            HBox([
                LabelPP("Layer selection", width=descr_label_size),
                self._layer_selection_dropdown
            ]),
            HTML(markdown.markdown("<hr>")),
            HTML(markdown.markdown("<hr>")),
        ]
        return VBox(widget)

    def _render_figures(self):
        self.interactive_graph.draw()

    def render(self):
        self._render_figures()
        return self._generate_widget()

    def get_activations(self, D: SupervisedDataset, model: Optional[Module] = None):
        if len(self.interactive_graph.selection) == 0:
            raise ValueError("No nodes have been selected, unable to compute activations")
        if len(self.interactive_graph.selection) > 1:
            raise ValueError("Only one node is supported, unable to compute activations")
        selection_id = str(self.interactive_graph.selection[0])
        return _get_layer_activations(
            net=model or self._get_model(selection_id),
            D=D,
            layer=self._layer_selection_dropdown.value
        )


def _get_labels(D: Subset | Dataset) -> np.ndarray:
    if isinstance(D, Subset):
        return D.dataset.targets[D.indices]  # type: ignore
    else:
        return D.targets  # type: ignore


def _get_data(D: Subset | Dataset) -> Tensor:
    if isinstance(D, Subset):
        return D.dataset.data[D.indices]  # type: ignore
    else:
        return D.data  # type: ignore


class UmapInteractivePlot:
    def __init__(
        self,
        D: SupervisedDataset,
        activations,
        labels: dict[int, str] | None = None,
        title: str = "",
        highlight_labels: set[int] | None = None,
        output_directory: Path = Path('res/'),
    ):
        self._output_dir = output_directory

        self._fig, _ = plt.subplots()
        plt.show()
        self._ax = self._fig.axes[0]
        self._title = title
        self._inset_ax = None
        self._highlight_idx_global = None
        self._distance_threshold = 0.1
        self._embedding = activations
        self._D = D
        self._highlight_labels = highlight_labels or set()

        self._exception_output = Output()

        self._label_names = labels or {}
        self._data = _get_data(D)
        self._labels = _get_labels(D)
        self._unique_labels = np.unique(self._labels)
        self._fig.canvas.mpl_connect('button_press_event', self._onclick_print_execpt)

    def _label_text(self, label):
        if label in self._label_names:
            return f"{self._label_names[label]} ({label})"
        return f"{label}"

    def draw(self, highlight_idx=None, img=None):
        self._ax.cla()

        for label in reversed(self._unique_labels):
            mask = (self._labels == label)
            is_highlighted = not self._highlight_labels or label in self._highlight_labels

            self._ax.scatter(
                self._embedding[mask, 0],
                self._embedding[mask, 1],
                label=self._label_text(label),
                alpha=0.3 if is_highlighted else 0.08,
                edgecolors="red" if (label < 0) else "none",
                color="gray" if not is_highlighted else None,
                s=32
            )

        if highlight_idx is not None:
            self._ax.scatter(
                self._embedding[highlight_idx, 0],
                self._embedding[highlight_idx, 1],
                facecolors='none',
                edgecolors='red',
                linewidths=1,
                s=36,
                zorder=10
            )

            if self._inset_ax:
                self._inset_ax.remove()

            cls = int(self._labels[highlight_idx])
            label_text = self._label_text(cls)
            self._inset_ax = self._ax.inset_axes([0.65, 0.65, 0.3, 0.3])
            self._inset_ax.imshow(img, cmap='gray')
            self._inset_ax.set_title(f'Idx: {highlight_idx} ({label_text})', fontsize=8)
            self._inset_ax.axis('off')
        else:
            if self._inset_ax:
                self._inset_ax.remove()
                self._inset_ax = None

        self._ax.set_aspect('equal', 'datalim')
        self._ax.legend()
        self._ax.set_axis_off()
        self._ax.set_title(self._title)
        self._fig.canvas.draw_idle()

    def _onclick_print_execpt(self, *args, **kwargs):
        with self._exception_output:
            try:
                self._onclick(*args, **kwargs)
            except Exception as e:
                traceback.print_stack()
                print(e)

    def _onclick(self, event):
        if event.inaxes != self._ax:
            return

        min_dist = float('inf')
        closest_idx = None

        for label in self._unique_labels:
            mask = (self._labels == label)
            if (mask == 0).all():
                continue

            emb_masked = self._embedding[mask]
            distances = ((emb_masked[:, 0] - event.xdata) ** 2 + (emb_masked[:, 1] - event.ydata) ** 2)

            local_idx = distances.argmin()
            dist = distances[local_idx].item()

            if dist < min_dist:
                min_dist = dist
                closest_idx = torch.arange(len(self._data))[mask][local_idx]

        if min_dist > self._distance_threshold:
            self.draw(highlight_idx=None, img=None)
            return

        img = self._data[closest_idx].cpu().numpy()
        self.draw(highlight_idx=closest_idx, img=img)

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self._highlight_labels:
            highlight_str = "_".join(self._label_text(li) for li in sorted(self._highlight_labels))
            filename = f"{self._title}_{highlight_str}.png"
        else:
            filename = f"{self._title}.png"

        self._ax.set_title(self._title)
        self._fig.savefig(filename, bbox_inches="tight", pad_inches=0, transparent=True, dpi=300)


class UmapWidget:
    def __init__(
        self,
        D: SupervisedDataset,
        activations,
        labels: dict[int, str] | None = None,
        title: str = "",
        highlight_labels: set[int] | None = None,
    ):
        self._output_dir = Path('res/')
        self._umap_plot = UmapInteractivePlot(
            D=D,
            activations=activations,
            labels=labels,
            title=title,
            highlight_labels=highlight_labels,
            output_directory=self._output_dir
        )
        self._dump_button = Button(description='Save all figures')
        self._dump_button.on_click(self.dump)

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._umap_plot.dump()

    def _control_dashboard_widget(self):
        dashboard_controls_widget = []
        dashboard_controls_widget.extend([
            HTML(markdown.markdown("<hr>")),
            self._dump_button,
            HTML(markdown.markdown("<hr>")),
        ])
        return VBox(dashboard_controls_widget)

    def _render_figures(self):
        self._umap_plot.draw()

    def render(self):
        self._render_figures()
        return self._control_dashboard_widget()
