#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This Python package provides the FC-WQ forecast model specification
file registry.
"""

from importlib import resources
from pathlib import Path

import yaml
from xgboost import Booster

__version__ = "2025.1.0"
"""The version of the forecast model registry."""


class Registry:
    """A simple registry for forcast model specification files."""

    _config: dict[str:str]
    """The mapping from model name to model specification file."""

    def __init__(self):
        """
        Creates a new registry instance.
        """
        with resources.path(
            "wqf.xgb.config", "wqf.xgb.config.yml"
        ) as resource:
            with open(resource) as r:
                self._config = yaml.safe_load(r).get("models", {})

    @property
    def default_name(self) -> str:
        """Returns the default model name."""
        return self._config["default"]

    @property
    def names(self) -> list[str]:
        """Returns the model names registered."""
        return list(self._config)

    def file(self, name: str) -> Path:
        """
        Returns the path to the model specification file associated with
        the model name supplied as argument.

        :param name: The model name.
        :return: The path to the model specification file associated with
        the model name supplied as argument.
        """
        with resources.path("wqf.xgb.config", self._config[name]) as resource:
            return resource

    def model(self, name: str) -> Booster:
        """
        Returns the model associated with the model name supplied
        as argument.

        :param name: The model name.
        :return: The model associated with the model name supplied
        as argument.
        """
        return Booster(model_file=self.file(name))

    def __contains__(self, name: str) -> bool:
        return name in self._config

    def __str__(self) -> str:
        """Returns a string representation of the registry."""
        return "{{{0}}}".format(str(self.names)[1:-2].replace("'", ""))


_registry = Registry()
"""The forecast model registry."""


def registry() -> Registry:
    """Returns the forecast model registry."""
    return _registry
