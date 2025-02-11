#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module provides loggers."""

import logging
import platform
import sys
import time
from typing import Literal
from typing import TextIO

from typing_extensions import override

from .interface.logging import Logging


class _DefaultLogger(Logging):
    """
    The default logger.
    """

    _levels = {
        Logging.DEBUG: logging.DEBUG,
        Logging.INFO: logging.INFO,
        Logging.WARNING: logging.WARNING,
        Logging.ERROR: logging.CRITICAL,
    }

    def __init__(
        self,
        processor: str,
        version: str,
        hostname: str,
        level: Literal["debug", "info", "warning", "error"] = Logging.INFO,
        out: TextIO = sys.stdout,
        err: TextIO = sys.stderr,
    ):
        """! Creates a new instance of this class.

        :param processor: The processor name.
        :param version: The processor version.
        :param hostname: The hostname.
        :param level: The log level.
        :param out: The stream to use for usual log messages.
        :param err: The stream to use for error log messages.
        """
        # map error messages to standard critical messages
        logging.addLevelName(logging.CRITICAL, "E")
        logging.addLevelName(logging.WARNING, "W")
        logging.addLevelName(logging.INFO, "I")
        logging.addLevelName(logging.DEBUG, "D")

        self._logger = logging.getLogger(processor)
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

        formatter = self._formatter(hostname, processor, version)
        self._logger.addHandler(self._handler(out, formatter))
        self._logger.addHandler(
            self._handler(err, formatter, Logging.WARNING)
        )
        self._logger.setLevel(self._levels[level])

    @staticmethod
    def _formatter(hostname, processor, version):
        """This method does not belong to public API."""
        fmt = (
            f"%(asctime)s.%(msecs)03d000Z {hostname} {processor} {version}"
            f" [%(process)d] [%(levelname)s] %(message)s"
        )
        formatter = logging.Formatter(fmt, "%Y-%m-%dT%H:%M:%S")
        formatter.converter = time.gmtime
        return formatter

    def _handler(self, stream, formatter, level=None):
        """This method does not belong to public API."""
        handler = logging.StreamHandler(stream=stream)
        handler.setFormatter(formatter)
        if level is not None:
            handler.setLevel(self._levels[level])
        return handler

    @override
    def debug(self, msg: str, *args, **kwargs):  # noqa: D102
        self._logger.debug(msg, *args, **kwargs)

    @override
    def info(self, msg: str, *args, **kwargs):  # noqa: D102
        self._logger.info(msg, *args, **kwargs)

    @override
    def warning(self, msg: str, *args, **kwargs):  # noqa: D102
        self._logger.error(msg, *args, **kwargs)

    @override
    def error(self, msg: str, *args, **kwargs):  # noqa: D102
        self._logger.critical(msg, *args, **kwargs)

    @override
    def is_enabled(  # noqa: D102
        self, level: Literal["debug", "info", "warning", "error"]
    ) -> bool:
        return self._logger.isEnabledFor(self._levels[level])


class _SilentLogger(Logging):
    """
    A silent logger.

    Does not issue any messages.
    """

    @override
    def debug(self, msg: str, *args, **kwargs):  # noqa: D102
        pass

    @override
    def info(self, msg: str, *args, **kwargs):  # noqa: D102
        pass

    @override
    def warning(self, msg: str, *args, **kwargs):  # noqa: D102
        pass

    @override
    def error(self, msg: str, *args, **kwargs):  # noqa: D102
        pass

    @override
    def is_enabled(  # noqa: D102
        self, level: Literal["debug", "info", "warning", "error"]
    ) -> bool:
        return False


_logger: Logging = _SilentLogger()
"""The logger instance (silent by default)"""


def set_logger(
    processor: str,
    version: str,
    hostname: str | None = None,
    level: Literal["debug", "info", "warning", "error", "off"] = "warning",
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
):
    """
    Configures the logger.

    This method shall be called once before the logger instance is used.

    :param hostname: The hostname.
    :param processor: The processor name.
    :param version: The processor version.
    :param level: The log level.
    :param out: The stream to use for usual log messages.
    :param err: The stream to use for error log messages.
    """
    global _logger

    if hostname is None:
        hostname = platform.node()

    match level:
        case "off":
            _logger = _SilentLogger()
        case _:
            # noinspection PyTypeChecker
            _logger = _DefaultLogger(
                processor, version, hostname, level, out, err
            )


def get_logger() -> Logging:
    """
    Returns the logger.

    :return: The logger.
    """
    global _logger
    return _logger
