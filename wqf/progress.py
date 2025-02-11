#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module provides a context for progress display."""


class Progress:
    """
    The context for progress display.

    This implementation is a template and does not do anything yet.
    """

    def __init__(self, show_progress: bool = False):
        """
        Creates a new context.

        :param show_progress: If true, progress is shown.
        """
        self._show_progress = show_progress

    def __enter__(self):
        """This method belongs to dask API."""
        if self._show_progress:
            from dask.diagnostics import ProgressBar

            self._progress_bar = ProgressBar(minimum=1.0, dt=1.0)
            self._progress_bar.register()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """This method belongs to dask API."""
        if self._show_progress:
            self._progress_bar.unregister()
