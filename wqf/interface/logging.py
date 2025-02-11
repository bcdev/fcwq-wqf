#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module defines the logging interface."""

from abc import ABC
from abc import abstractmethod
from typing import Literal


class Logging(ABC):
    """The Logging interface."""

    ERROR: Literal["error"] = "error"
    """The name of the @code ERROR log level."""
    WARNING: Literal["warning"] = "warning"
    """The name of the @c WARNING log level."""
    INFO: Literal["info"] = "info"
    """The name of the @c INFO log level."""
    DEBUG: Literal["debug"] = "debug"
    """The name of the @c DEBUG log level."""

    @abstractmethod
    def debug(self, msg: str, *args, **kwargs):
        """
        Issues a debugging message to `stdout`.

        :param msg: The message text.
        """

    @abstractmethod
    def info(self, msg: str, *args, **kwargs):
        """
        Issues a general informational message to `stdout`.

        :param msg: The message text.
        """

    @abstractmethod
    def warning(self, msg: str, *args, **kwargs):
        """
        Issues a warning message, when an exceptional condition has
        occurred but the processor was able to continue.

        :param msg: The message text.
        """

    @abstractmethod
    def error(self, msg: str, *args, **kwargs):
        """Issues an error message, when an exceptional condition has
        occurred and the processor was not able to continue.

        :param msg: The message text.
        """

    def is_enabled(
        self, level: Literal["debug", "info", "warning", "error"]
    ) -> bool:
        """
        Returns true if logging is enabled for a given level of interest,
        and returns false otherwise.

        :param level: The log level of interest.
        :return: True if logging is enabled, false otherwise.
        """
