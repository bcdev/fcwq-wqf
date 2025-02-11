#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module defines the signal handling interface."""

from abc import ABC
from abc import abstractmethod


class SignalHandling(ABC):
    """Interface for a callable object to handle signals."""

    @abstractmethod
    def __call__(self, signal_number: int, frame):
        """
        Handles the signal.

        :param signal_number: The signal number to be handled.
        :param frame: The current stack frame.
        """
