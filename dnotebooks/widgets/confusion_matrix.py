from typing import Optional, List, Dict, Sequence
from collections import defaultdict
from pathlib import Path

import markdown
from ipywidgets import (
    Dropdown, IntSlider, IntRangeSlider,
    Button, FloatSlider, FloatRangeSlider,
    Layout, HBox, VBox, HTML
)
from networkx import Graph

from dengine.analysis import (
    ExperimentConfusionMatrix,
    ExperimentPartitions,
)
from dengine.graph import NXGraph

from dnotebooks.utils import RegexColorDict
from dnotebooks.plots import (
    MetricPlot,
    ConfusionMatrixHeatmap,
    PartitionsHistogram,
    InteractiveGraph,
    MultiExperimentMetricComparisonPlot
)

from .miscellaneous import LabelPP, set_epoch_slider_widget


class ExperimentDashboard:
    def __init__(
        self,
        experiment_root_path: Path,
        confusion_matrices: Optional[ExperimentConfusionMatrix],
        partitions: ExperimentPartitions,
        graph: Graph,
    ):
        self._output_dir = Path('res/')
        self._confusion_matrices = confusion_matrices
        self._partitions = partitions
        assert isinstance(graph, NXGraph)
        self.graph = graph

        self.interactive_graph = InteractiveGraph(
            confusion_matrices=confusion_matrices,
            partitions=partitions,
            graph=self.graph.nx_graph,
            output_directory=self._output_dir,
            callback=self._render_figures
        )
        self._training_set_histogram = PartitionsHistogram(
            confusion_matrices=confusion_matrices,
            partitions=partitions,
            interactive_graph=self.interactive_graph
        )
        self._confusion_matrix_plots: List[MetricPlot] = []
        if self._confusion_matrices:
            self._confusion_matrix_plots.extend([
                MetricPlot(
                    confusion_matrices=self._confusion_matrices,
                    interactive_graph=self.interactive_graph,
                ),
                ConfusionMatrixHeatmap(
                    confusion_matrices=self._confusion_matrices,
                    partitions=self._partitions,
                    interactive_graph=self.interactive_graph
                ),
                # ConfusionMatrixHeatmapDelta(
                #     confusion_matrices=self._confusion_matrices,
                #     partitions=self._partitions,
                #     interactive_graph=self.interactive_graph
                # ),
                # ConfusionMatrixHeatmapConfidenceInterval(
                #     confusion_matrices=self._confusion_matrices,
                #     partitions=self._partitions,
                #     interactive_graph=self.interactive_graph
                # )
            ])

        self._reset_selection_button = Button(
            description='Reset Selection',
            layout=Layout(width='auto', height='32px')
        )
        self._reset_selection_button.on_click(self._on_reset_selection_click)

        self._dump_button = Button(description='Save all figures')
        self._dump_button.on_click(self.dump)

        self._hops_slider = IntSlider(
            max=5,
            value=0,
        )
        self._hops_slider.observe(
            self._on_hops_slider_change,
            'value'
        )

    def _on_reset_selection_click(self, *args):
        self.interactive_graph.reset_selection()
        self._render_figures()

    def _on_hops_slider_change(self, event: Dict):
        value = event["new"]
        self.interactive_graph.selection_hops = value
        for cm_plot in self._confusion_matrix_plots:
            if isinstance(cm_plot, ConfusionMatrixHeatmap):
                cm_plot.selection_hops = value
        self._render_figures()

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive_graph.dump()
        self._training_set_histogram.dump()
        for cm_plot in self._confusion_matrix_plots:
            cm_plot.dump()

    def _control_dashboard_widget(self):
        if not self._confusion_matrices:
            descr_label_size = '12em'
            dashboard_controls_widget = VBox([
                HTML(markdown.markdown("<hr>")),
                HTML(markdown.markdown("<hr>")),
                HTML(markdown.markdown("### Graph")),
                HBox([
                    LabelPP("Nodes color", width=descr_label_size),
                    networkX_node_color_widget(self.interactive_graph)
                ]),
                HBox([
                    self._reset_selection_button,
                ]),
                HTML(markdown.markdown("<hr>")),
                HTML(markdown.markdown("<hr>")),
            ])
            return dashboard_controls_widget

        epoch_slider_widget = set_epoch_slider_widget(
            self._confusion_matrices,
            [
                self.interactive_graph,
                *self._confusion_matrix_plots
            ]
        )
        descr_label_size = '12em'

        dashboard_controls_widget = [
            HTML(markdown.markdown("<hr>")),
            HTML(markdown.markdown("<hr>")),
            HBox([
                LabelPP("Epoch", width=descr_label_size),
                epoch_slider_widget
            ]),
            HTML(markdown.markdown("### Graph")),
            HBox([
                LabelPP("Nodes color", width=descr_label_size),
                networkX_node_color_widget(self.interactive_graph)
            ]),
            HBox([
                self._reset_selection_button,
            ]),
        ]
        if len(self._confusion_matrix_plots) > 0:
            dashboard_controls_widget.extend([
                HTML(markdown.markdown("### Plots")),
                HBox([
                    LabelPP("Metric", width=descr_label_size),
                    metric_plot_selection_widget(
                        metric_plots=[*self._confusion_matrix_plots, self.interactive_graph],
                        confusion_matrix=self._confusion_matrices
                    ),
                ]),
            ])

        if len(self._confusion_matrix_plots) > 0:
            dashboard_controls_widget.extend([
                HTML(markdown.markdown("### Confusion Matrices")),
                HBox([
                    LabelPP("Hops", width=descr_label_size),
                    self._hops_slider,
                ]),
                HBox([
                    LabelPP("Nodes reduction fun", width=descr_label_size),
                    confusion_matrix_reduction_dropdown_widget(self._confusion_matrix_plots)
                ]),
                HTML(markdown.markdown("<hr>")),
                self._dump_button,
                HTML(markdown.markdown("<hr>")),
            ])
        return VBox(dashboard_controls_widget)

    def _render_figures(self):
        self.interactive_graph.draw()
        self._training_set_histogram.draw()
        for cm_plot in self._confusion_matrix_plots:
            cm_plot.draw()

    def render(self):
        self._render_figures()
        return self._control_dashboard_widget()


def metric_plot_selection_widget(
    metric_plots: Sequence[MetricPlot],
    confusion_matrix: ExperimentConfusionMatrix
) -> HBox:
    def _class_dropdown_widget_change(*args):
        value = plot_metric_dropdown_wg.value
        if value == MetricPlot.SupportedPlots.accuracy.value:
            class_label_dropdown_wg.layout.visibility = 'hidden'
        else:
            class_label_dropdown_wg.layout.visibility = 'visible'
            class_label_dropdown_wg.options = confusion_matrix.class_indexes()
        for mp in metric_plots:
            mp.metric = MetricPlot.SupportedPlots(plot_metric_dropdown_wg.value)
            mp.draw()

    def on_class_label_change(*args):
        for mp in metric_plots:
            mp.metric_class_label = int(class_label_dropdown_wg.value)
            mp.draw()

    plot_metric_dropdown_wg = Dropdown(
        options=[(e.value, e.value) for e in MetricPlot.SupportedPlots],
        value=MetricPlot.SupportedPlots.accuracy.value,
    )
    plot_metric_dropdown_wg.observe(_class_dropdown_widget_change, 'value')

    class_label_dropdown_wg = Dropdown(options=[])
    class_label_dropdown_wg.layout.visibility = 'hidden'
    class_label_dropdown_wg.observe(
        on_class_label_change,
        'value'
    )
    return HBox([
        plot_metric_dropdown_wg,
        class_label_dropdown_wg,
    ])


def multi_experiment_metric_comparison_plot_widget(
    metric_plots: Sequence[MultiExperimentMetricComparisonPlot],
    confusion_matrix: ExperimentConfusionMatrix
) -> VBox:
    def _class_dropdown_widget_change(*args):
        for mp in metric_plots:
            mp.aggregation = MultiExperimentMetricComparisonPlot.SupportedAggregation(
                plot_aggregation_dropdown_wg.value
            )

    current_box = metric_plot_selection_widget(metric_plots, confusion_matrix)
    plot_aggregation_dropdown_wg = Dropdown(
        options=[(e.value, e.value) for e in MultiExperimentMetricComparisonPlot.SupportedAggregation],
        value=MultiExperimentMetricComparisonPlot.SupportedAggregation.confidence_interval.value,
    )
    plot_aggregation_dropdown_wg.observe(_class_dropdown_widget_change, 'value')

    return VBox([
        current_box,
        plot_aggregation_dropdown_wg
    ])


def confusion_matrix_reduction_dropdown_widget(
    confusion_matrix_heatmap: Sequence[MetricPlot],
):
    def _on_dropdown_widget_change(*args):
        for cmh in confusion_matrix_heatmap:
            if not isinstance(cmh, ConfusionMatrixHeatmap):
                continue
            cmh.reduction_function = ConfusionMatrixHeatmap.Reductions(
                confusion_matrix_node_reduction_fun_wg.value
            )

    confusion_matrix_node_reduction_fun_wg = Dropdown(
        options=[(e.name, e.value) for e in ConfusionMatrixHeatmap.Reductions],
        value=ConfusionMatrixHeatmap.Reductions.mean.value,
    )
    confusion_matrix_node_reduction_fun_wg.observe(
        _on_dropdown_widget_change,
        'value',
    )

    return confusion_matrix_node_reduction_fun_wg


def networkX_node_color_widget(dynamic_graph: InteractiveGraph):

    def _on_dropdown_widget_change(*args):
        dynamic_graph.colorby = InteractiveGraph.SupportedNetworkXNodeColors(wd.value)

    wd = Dropdown(
        options=[(e.value, e.value) for e in dynamic_graph.supported_networkX_node_colors],
        value=InteractiveGraph.SupportedNetworkXNodeColors.default.value,
    )

    wd.observe(_on_dropdown_widget_change, 'value')
    return wd


class MultiExperimentMetricsComparisonDashboard:
    def __init__(
        self,
        confusion_matrices: Dict[str, List[ExperimentConfusionMatrix]],
        custom_colors: Optional[RegexColorDict] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        linestyles: Dict[str, Optional[str]] = defaultdict(None),
        external_legend: bool = False,
    ):
        self._output_dir = Path('res/')
        self._confusion_matrices = confusion_matrices

        self._confusion_matrix_plot = None
        self._confusion_matrix_delta_plot = None
        self._confusion_matrix_ci_plot = None
        self._metric_plot = MultiExperimentMetricComparisonPlot(
            confusion_matrices=self._confusion_matrices,
            custom_colors=custom_colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            external_legend=external_legend,
            linestyles=linestyles
        )

        self._dump_button = Button(description='Save all figures')
        self._dump_button.on_click(self.dump)

        self._epoch_slider = IntRangeSlider(
            min=0,
            value=(0, self._metric_plot.epoch[1] - 1),
            max=self._metric_plot.epoch[1] - 1,
        )
        self._epoch_slider.observe(
            self._on_epoch_slider_change,
            'value'
        )

        self._smoothing_slider = FloatSlider(min=0, value=0, max=1., step=.01)
        self._smoothing_slider.observe(
            self._on_smoothing_slider_change,
            'value'
        )

        self._y_axis_limit = FloatRangeSlider(min=0, value=(0, 1), max=1., step=.01)
        self._y_axis_limit.observe(
            self._on_y_limit_change,
            'value'
        )

    def dump(self, *args, **kwargs):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metric_plot.dump()

    def _on_smoothing_slider_change(self, event: Dict):
        value = event["new"]
        self._metric_plot.smoothing = value

    def _on_epoch_slider_change(self, event: Dict):
        self._metric_plot.epoch = event["new"]

    def _on_y_limit_change(self, event: Dict):
        self._metric_plot.ylimit = event["new"]

    def _control_dashboard_widget(self):
        descr_label_size = '12em'

        return VBox([
            HTML(markdown.markdown("<hr>")),
            HTML(markdown.markdown("<hr>")),
            HBox([
                LabelPP("Epoch", width=descr_label_size),
                self._epoch_slider
            ]),
            HBox([
                LabelPP("Smoothing", width=descr_label_size),
                self._smoothing_slider
            ]),
            HBox([
                LabelPP("Y Axis Limit", width=descr_label_size),
                self._y_axis_limit
            ]),
            HTML(markdown.markdown("### Plots")),
            HBox([
                LabelPP("Metric", width=descr_label_size),
                multi_experiment_metric_comparison_plot_widget(
                    metric_plots=[self._metric_plot],
                    confusion_matrix=list(self._confusion_matrices.values())[0][0]
                ),
            ]),
            HTML(markdown.markdown("<hr>")),
            self._dump_button,
            HTML(markdown.markdown("<hr>")),
        ])

    def _render_figures(self):
        self._metric_plot.draw()

    def render(self):
        self._render_figures()
        return self._control_dashboard_widget()
