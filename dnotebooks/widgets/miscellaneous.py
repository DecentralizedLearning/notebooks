from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Any, Dict, Union, Optional

from ipywidgets import Layout, IntRangeSlider, Label
import ipywidgets as widgets

from dengine.analysis import ExperimentConfusionMatrix


class LabelPP(Label):
    def __init__(self, value=None, width: Optional[str | int] = None, **kwargs):
        css_layout = Layout(width=width)
        super().__init__(value, layout=css_layout, **kwargs)


def set_epoch_slider_widget(confusion_matrices: ExperimentConfusionMatrix, widgets: List):
    return int_slider_with_setter_callback(
        widgets=widgets,
        property_name="epoch",
        min=0,
        max=confusion_matrices.rounds - 1,
        value=(0, confusion_matrices.rounds - 1),
    )


def int_slider_with_setter_callback(
    widgets: List,
    property_name: str,
    **kwargs
):
    def on_slider_change(*args):
        for wg in widgets:
            setattr(wg, property_name, slider_wg.value)
    slider_wg = IntRangeSlider(**kwargs)
    slider_wg.observe(on_slider_change, 'value')
    return slider_wg


@dataclass
class _MultiCheckboxEntry:
    value: Any
    checkbox: widgets.Checkbox


class MultiCheckboxWidget:
    def __init__(
        self,
        descriptions: Union[List, Dict[str, Any]] = {},
        max_height='200px',
        max_width="100%"
    ):
        self.max_width = max_width
        self.options_dict: Dict[str, _MultiCheckboxEntry] = {}
        self.callback: Optional[Callable] = None

        self.search_widget = widgets.Text(placeholder="Search")
        self.select_all_checkbox = widgets.Checkbox(
            description="Select All",
            value=False,
            indent=False,
            layout=widgets.Layout(width=self.max_width)
        )

        self.options_widget = widgets.VBox([], layout=widgets.Layout(
            overflow='auto',
            max_height=max_height,
            width=max_width
        ))
        self.multi_select = widgets.VBox([self.search_widget, self.select_all_checkbox, self.options_widget])

        self.search_widget.observe(self.on_text_change, names='value')
        self.select_all_checkbox.observe(self.on_select_all_change, names='value')

        self.update_options(descriptions)

    def on_text_change(self, change):
        search_input = change['new']
        if search_input == '':
            new_options = [self.options_dict[description] for description in self.options_dict.keys()]
        else:
            new_options = [
                self.options_dict[description]
                for description in self.options_dict.keys()
                if search_input in description.lower()
            ]
        self.options_widget.children = [x.checkbox for x in new_options]

    def on_select_all_change(self, change):
        select_all = change['new']
        if select_all:
            checkboxes = [e for e in self.options_widget.children]
        else:
            checkboxes = [e.checkbox for e in self.options_dict.values()]
        for e in checkboxes:
            e.value = select_all

    def update_options(self, descriptions: Union[List, Dict[str, Any]]):
        if not isinstance(descriptions, dict):
            descriptions = {key: key for key in descriptions}

        options_dict = {}
        for key, value in descriptions.items():
            checkbox = widgets.Checkbox(
                description=key,
                value=False,
                layout=widgets.Layout(width=self.max_width, description_width=self.max_width)
            )
            options_dict[key] = _MultiCheckboxEntry(value, checkbox)
            checkbox.observe(
                lambda *args, key=key, **kwargs: self.on_checkbox_change(*args, key=key, **kwargs),
                names='value'
            )
        self.options_dict = options_dict

        self.options_widget.children = [
            self.options_dict[description].checkbox
            for description in descriptions
        ]

    def on_checkbox_change(self, event, key):
        if not event["new"]:
            self.select_all_checkbox.value = False

        all_selected = all(
            ith_entry.checkbox.value
            for ith_key, ith_entry in self.options_dict.items()
            if ith_key != key
        )
        if all_selected:
            self.select_all_checkbox.value = True

        if self.callback:
            self.callback()

    def widget(self):
        return self.multi_select

    def get_selection(self):
        return [
            e.value
            for e in self.options_dict.values()
            if e.checkbox.value
        ]


class MultiCheckboxWithLimitWidget(MultiCheckboxWidget):
    def __init__(
        self,
        descriptions: List | Dict[str, Any] = {},
        max_height='200px',
        max_width="100%",
        selection_limit: int = 1,
    ):
        super().__init__(descriptions, max_height, max_width)
        self.multi_select = widgets.VBox([self.search_widget, self.options_widget])
        self._selection_limit = selection_limit
        self._queue = []

    def on_checkbox_change(self, event, key: str):
        if not event["new"]:
            try:
                self._queue.remove(key)
            except ValueError:
                pass
        else:
            if len(self._queue) == self._selection_limit:
                try:
                    popped_key = self._queue.pop(0)
                    self.options_dict[popped_key].checkbox.value = False
                except IndexError:
                    pass
            self._queue.append(key)

        if self.callback:
            self.callback()
