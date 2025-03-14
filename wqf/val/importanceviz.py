#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module provides a feature importance visualizer."""

from typing import Any
from typing import Literal
from typing import Sequence

import xgboost
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from xgboost import Booster

from wqf.interface.constants import VID_DEP
from wqf.interface.constants import VID_MDT
from wqf.interface.plot import Plot
from wqf.xgb import registry


class ImportanceVisualizer(Plot):
    """A feature importance visualizer."""

    _model: Booster
    """The forecast model."""

    def __init__(self, name: str = "default"):
        """
        Creates a new analyzer.

        :param name: The name of the forecast model to analyze.
        """
        self._name = name
        self._model = registry().model(name)
        self._model.feature_names = self._feature_names()

    def visualize(self, feature_count: int | None = None):
        """
        Visualizes feature importance.

        :param feature_count: The number of features to visualize.
        """
        self._plot("cover", (0.0, 1.0e07), feature_count)
        self._plot("gain", (0.0, 1.0e06), feature_count)
        self._plot("total_gain", (0.0, 1.0e07), feature_count)
        self._plot("total_cover", (0.0, 1.0e09), feature_count)
        self._plot("weight", (0.0, 1.0e03), feature_count)

    def plot(
        self,
        data: None = None,
        xlabel: str | None = None,
        ylabel: str | None = "feature",
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
        title: str | None = "Feature importance",
        fn: str | None = None,
        show: bool = False,
        *,
        bar_height: Any = 0.4,
        feature_count: int = 12,
        importance_type: Literal[
            "cover", "gain", "total_cover", "total_gain", "weight"
        ] = "total_gain",
        show_grid: bool = False,
        show_values: bool = True,
        values_format: str = "{v:,.1f}",
    ) -> Figure:
        fig, ax = plt.subplots()
        xgboost.plot_importance(
            self._model,
            ax=ax,
            height=bar_height,
            title=title,
            xlim=xlim,
            xlabel=xlabel,
            ylabel=ylabel,
            importance_type=importance_type,
            max_num_features=feature_count,
            grid=show_grid,
            show_values=show_values,
            values_format=values_format,
        )
        if fn is not None:
            fig.savefig(f"{fn}.pdf", bbox_inches="tight")
        if show:
            fig.show()
        plt.close()
        return fig

    def _plot(
        self,
        importance_type: Literal[
            "cover", "gain", "total_cover", "total_gain", "weight"
        ],
        xlim: tuple | None = None,
        feature_count: int | None = None,
    ):
        """Generates an importance plot."""
        fig = self.plot(
            xlabel=f"{importance_type.replace('_', ' ')} (arbitrary units)",
            ylabel="feature",
            xlim=xlim,
            fn=f"{self._name}_{importance_type}",
            feature_count=feature_count,
            importance_type=importance_type,
        )
        fig.clear()

    def _feature_names(
        self, static: Sequence[str] = (VID_DEP, VID_MDT)
    ) -> Sequence[str]:
        """Returns feature names suitable for an importance plot."""
        labels: list[str] = []
        for feature_name in self._model.feature_names:
            day, v = feature_name.split("_")

            if v in static:
                labels.append(f"{v}")
            else:
                labels.append(f"{v}[{ImportanceVisualizer._t(day)}]")
        return labels

    @staticmethod
    def _t(day: str) -> str:
        """Returns the time string for a given day string."""
        d = day[1:] if "-" in day and day[2:] != "0" else day[2:]
        return r"$t_{" + d + "}$"


if __name__ == "__main__":
    ImportanceVisualizer().visualize(feature_count=12)
