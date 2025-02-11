#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides callbacks used for logging and timing
computations.
"""

import time
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from dask.callbacks import Callback
from typing_extensions import override

from .logger import get_logger


class NodeFilter(metaclass=ABCMeta):
    """Interface for filtering named graph nodes."""

    @abstractmethod
    def accept(self, key: tuple[str, int, ...] | str) -> bool:
        """
        Returns true, if the given key is accepted and returns false if not.

        :param key: The key representing the Dask node.
        :return: True or false.
        """


class AcceptAlgorithmsOnly(NodeFilter):
    """A filter to accept nodes representing WQF algorithms only."""

    @override
    def accept(self, key: tuple[str, int, ...] | str) -> bool:
        return key[0].startswith("wqf")


class AcceptAll(NodeFilter):
    """A filter accepting all nodes."""

    @override
    def accept(self, key: tuple[str, int, ...] | str) -> bool:
        return True


class RejectAll(NodeFilter):
    """A filter rejecting all nodes."""

    @override
    def accept(self, key: tuple[str, int, ...] | str) -> bool:
        return False


class SelectiveCallback(Callback):
    """
    An abstract selective callback, which is effective for
    selected nodes of interest only.
    """

    def __init__(self, node_filter: NodeFilter = RejectAll()):
        """
        Creates a new callback instance.

        :param node_filter: A node filter to select nodes of interest.
        """
        super().__init__()
        self._node_filter = node_filter

    def _accept(self, key):
        """This method does not belong to public API."""
        return self._node_filter.accept(key)

    def _pretask(self, key, dask, state):
        """This method belongs to dask API."""
        if self._accept(key):
            self._pretask_impl(key, dask, state)

    def _posttask(self, key, result, dask, state, worker_id):
        """This method belongs to dask API."""
        if self._accept(key):
            self._posttask_impl(key, result, dask, state, worker_id)

    @abstractmethod
    def _pretask_impl(self, key, dask, state):
        """This method is called by dask API."""

    @abstractmethod
    def _posttask_impl(self, key, result, dask, state, worker_id):
        """This method is called by dask API."""


class AlgorithmMonitor(SelectiveCallback):
    """A callback for monitoring algorithm computations."""

    def __init__(self):
        """Creates a new callback instance."""
        super().__init__(AcceptAlgorithmsOnly())

    @override
    def _pretask_impl(self, key, dask, state):
        logger = get_logger()
        logger.debug(f"starting computing: {key}")

    @override
    def _posttask_impl(self, key, result, dask, state, worker_id):
        logger = get_logger()
        logger.debug(f"finished computing: {key} {worker_id}")


class AlgorithmTimer(SelectiveCallback):
    """
    A callback for timing the computation of algorithms.

    The timer measures CPU time, which corresponds to elapsed
    time in synchronous mode only. Time spent on re-computations,
    which may happen occasionally, is not taken into account.
    """

    _started: dict
    _stopped: dict

    def __init__(self):
        """Creates a new callback instance."""
        super().__init__(AcceptAlgorithmsOnly())
        self._started = {}
        self._stopped = {}

    @override
    def _pretask_impl(self, key, dask, state):
        self.start(key)

    @override
    def _posttask_impl(self, key, result, dask, state, worker_id):
        self.stop(key)

    @property
    def amassed_times(self) -> list:
        """
        Returns the amassed time (seconds) spent on computing each node.

        :return: The amassed time (seconds) spent on computing each node.
        """
        accu: dict = {}
        for key in self._stopped.keys():
            if self._accept(key):
                name = self._node_name(key)
                secs = self.process_time(key)
                if name not in accu:
                    accu[name] = secs
                else:
                    accu[name] = accu[name] + secs
        return sorted(accu.items(), key=lambda item: item[1], reverse=True)

    def start(self, key: tuple[str, int, ...] | str):
        """
        Starts the timer associated with a given key.

        :param key: The key.
        """
        if key not in self._started:
            self._started[key] = time.perf_counter()

    def stop(self, key: tuple[str, int, ...] | str):
        """
        Stops the timer associated with a given key.

        :param key: The key.
        """
        if key not in self._stopped:
            self._stopped[key] = time.perf_counter()

    def process_time(self, key: tuple[str, int, ...] | str):
        """
        Returns the process time associated with a given key.

        :param key: The key.
        :return: The process time (seconds) associated with the key.
        """
        if key in self._stopped and key in self._started:
            return self._stopped[key] - self._started[key]
        return np.nan

    @staticmethod
    def _node_name(key: tuple[str, int, ...]):
        """This method does not belong to public API."""
        return key[0][: key[0].find("-")]


class StatusLogger(SelectiveCallback):
    """
    A callback for logging status in terms of computational
    tasks pending and completed.
    """

    def __init__(self, n: int = 100):
        """
        Creates a new callback instance.

        If the number of computational tasks in a subgraph is considered
        low, no progress is logged for that subgraph.

        :param n: The number of progress messages logged per subgraph.
        """
        super().__init__(AcceptAll())
        self._n = n

    @override
    def _pretask_impl(self, key, dask, state):
        pass

    @override
    def _posttask_impl(self, key, result, dask, state, worker_id):
        self._show_progress(self._computation_status(state))

    def _computation_status(self, state):
        """This method does not belong to public API."""
        i = self._stopped_count(state)
        k = self._pending_count(state) + i
        m = k // self._n
        if m < 10:
            return None
        if i < k and i % m > 0:
            return None
        return i, k

    @staticmethod
    def _pending_count(state):
        """This method does not belong to public API."""
        return sum(len(state[k]) for k in ["ready", "waiting", "running"])

    @staticmethod
    def _stopped_count(state):
        """This method does not belong to public API."""
        return len(state["finished"])

    @staticmethod
    def _show_progress(info: tuple[int, int]):
        """This method does not belong to public API."""
        if info is not None:
            get_logger().info(
                f"computation status: {info[0] :4d} ({info[1] :4d})"
            )
