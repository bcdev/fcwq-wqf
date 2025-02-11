#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides signal handlers.
"""

import signal

from typing_extensions import override

from .interface.signalhandling import SignalHandling


class AbortHandler(SignalHandling):
    """
    To handle abnormal process termination.

    This handler raises a `RuntimeError`.
    """

    @override
    def __call__(self, signal_number: int, frame):  # noqa: D102
        if signal_number == signal.SIGABRT:
            raise RuntimeError("abnormal process termination (SIGABRT)")


class KeyboardInterruptHandler(SignalHandling):
    """To handle interrupt from keyboard (CTRL + C).

    This handler replicates the default action, which is to
    raise a `KeyboardInterrupt`.
    """

    @override
    def __call__(self, signal_number: int, frame):  # noqa: D102
        if signal_number == signal.SIGINT:
            raise KeyboardInterrupt("keyboard interrupt by user (SIGINT)")


class TerminationRequestHandler(SignalHandling):
    """To handle program termination requests.

    This handler raises a `RuntimeError`.
    """

    @override
    def __call__(self, signal_number: int, frame):  # noqa: D102
        if signal_number == signal.SIGTERM:
            raise RuntimeError("process termination request sent (SIGTERM)")
