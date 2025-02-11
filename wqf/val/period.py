#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module defines a plain time period class."""

from xarray import DataArray

from ..interface.constants import VID_TIM


class Period:
    """A plain time period."""

    def __init__(self, beg: int, end: int | None = None):
        """
        Creates a new period.

        :param beg: The beginning of the period (year).
        :param end: The inclusive end of the period (year).
        """
        self._beg = beg
        self._end = beg if end is None else end

    def slice(self, a: DataArray) -> DataArray:
        """
        Slices the data array supplied as argument.

        :param a: The data array.
        :return: The slice of the data array which is covered by this
        time period.
        """
        return a.sel(
            {VID_TIM: slice(f"{self._beg}-01-01", f"{self._end}-12-31")}
        )

    def str(self, sep: str) -> str:
        """
        Returns a string representation of this period.

        :param sep: To separate beginning and end of the period.
        :return: A string in the form `YYYY`{sep}YYYY`
        """
        return (
            f"{self._beg}"
            if self._end == self._beg
            else f"{self._beg}{sep}{self._end}"
        )

    @property
    def lim(self) -> tuple[int, int]:
        """
        Returns a tuple representation of this period.

        :return: A tuple of the beginning and the exclusive end
        of the period.
        """
        return self._beg, self._end + 1

    def __str__(self) -> str:
        """Returns a string representation of this period."""
        return self.str(" - ")
