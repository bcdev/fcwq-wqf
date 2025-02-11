#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides the runner class to run the WQF processor."""

import sys
import traceback
from argparse import ArgumentError
from argparse import ArgumentParser
from argparse import Namespace
from cProfile import Profile
from io import StringIO
from pstats import Stats
from typing import TextIO

import dask

from .callbacks import AlgorithmMonitor
from .callbacks import AlgorithmTimer
from .callbacks import StatusLogger
from .interface.exitcodes import ExitCodes
from .interface.logging import Logging
from .interface.processing import Processing
from .logger import get_logger
from .logger import set_logger


class Runner:
    """
    Runs a processor.

    Takes care of parsing command line arguments, setting up the logger,
    configuring Dask, running (and optionally profiling) the processor,
    handling exceptions, terminating Dask, and calling system exit with
    an exit code.
    """

    _parser: ArgumentParser
    """The command line parser."""
    _processor: Processing
    """The processor."""
    _profiler: Profile | None
    """The profiler."""
    _monitor: AlgorithmMonitor
    """The callback for monitoring algorithms."""
    _status_logger: StatusLogger
    """The callback for progress logging."""
    _timer: AlgorithmTimer
    """The callback for timing algorithms."""

    def __init__(self, processor: Processing, parser: ArgumentParser):
        """
        Creates a new instance of this class.

        :param processor: The processor to be run.
        :param parser: The parser to parse processing command line arguments.
        """
        self._parser = parser
        self._processor = processor
        self._profiler = None
        self._monitor = AlgorithmMonitor()
        self._status_logger = StatusLogger()
        self._timer = AlgorithmTimer()

    def run(
        self,
        args: list[str] | None = None,
        out: TextIO = sys.stdout,
        err: TextIO = sys.stderr,
    ) -> int:
        """
        Runs the processor. Reads the default processor configuration,
        reads an optional custom processor configuration, parses command
        line arguments, sets up the logger, configures Dask, runs (and
        optionally profiles) the processor, handles exceptions, and returns
        an exit code.

        :param args: An optional list of arguments for use in unit tests.
        :param out: The stream to use for usual log messages.
        :param err: The stream to use for error log messages.
        :return: The exit code.
        """
        processor_name = self._processor.get_name()
        processor_version = self._processor.get_version()
        set_logger(processor_name, processor_version, out=out, err=err)
        exit_code = ExitCodes.SUCCESS

        try:
            config = self._processor.get_default_config()
            args = self._parser.parse_args(args, Namespace(**config))
            set_logger(
                processor_name,
                processor_version,
                level=args.log_level,
                out=out,
                err=err,
            )
            self._init_dask(args)
            self._init_profiler(args)
            self._run_processor(args)
            self._stop_profiler(args)
        except ArgumentError as e:
            arg = e.argument_name
            if arg is not None:
                get_logger().error(f"argument error: {arg}")
            else:
                get_logger().error(f"{e}")
            exit_code = ExitCodes.FAILURE_ARGUMENT_ERROR
            self._report_stack_traces(args)
        except AssertionError as e:
            get_logger().error(f"an assertion failed: {e}")
            exit_code = ExitCodes.FAILURE_ASSERTION
            self._report_stack_traces(args)
        except KeyboardInterrupt as e:
            get_logger().error(f"{e}")
            exit_code = ExitCodes.FAILURE_KEYBOARD_INTERRUPT
            self._report_stack_traces(args)
        except MemoryError as e:
            get_logger().error(f"a memory error occurred: {e}")
            exit_code = ExitCodes.FAILURE_KEYBOARD_INTERRUPT
            self._report_stack_traces(args)
        except OSError as e:
            get_logger().error(f"an operating system error occurred: {e}")
            exit_code = ExitCodes.FAILURE_OS_ERROR
            self._report_stack_traces(args)
        except RuntimeError as e:
            get_logger().error(f"{e}")
            exit_code = ExitCodes.FAILURE_RUNTIME_ERROR
            self._report_stack_traces(args)
        except SystemError as e:
            get_logger().error(f"a system error occurred: {e}")
            exit_code = ExitCodes.FAILURE_SYSTEM_ERROR
            self._report_stack_traces(args)
        except SystemExit as e:
            if e.code > 0:
                get_logger().error(f"system exit raised ({e})")
                exit_code = ExitCodes.FAILURE_SYSTEM_EXIT
                self._report_stack_traces(args)
        except Exception as e:
            get_logger().error(f"an error occurred: {e}")
            exit_code = ExitCodes.FAILURE_UNSPECIFIC
            self._report_stack_traces(args)

        return exit_code

    @staticmethod
    def _init_dask(args):
        """This method does not belong to public API."""
        get_logger().debug("starting initializing Dask")
        # noinspection PyTestUnpassedFixture
        dask.config.set({"temporary-directory": args.tmpdir})
        if args.prof is not None:
            args.mode = "synchronous"
        match args.mode:
            case "synchronous":
                dask.config.set(scheduler="synchronous")
            case "multithreading":
                dask.config.set(scheduler="threads")
                if isinstance(args.workers, int):
                    dask.config.set(num_workers=args.workers)
        get_logger().debug("finished initializing Dask")

    def _init_profiler(self, args):
        """This method does not belong to public API."""
        if args.prof is not None:
            self._profiler = Profile()
            self._profiler.enable()

    def _stop_profiler(self, args):
        """This method does not belong to public API."""
        if self._profiler is not None:
            self._profiler.disable()
            get_logger().info(
                f"starting writing profiler report: {args.prof}"
            )
            stream = StringIO()
            report = Stats(self._profiler, stream=stream)
            report.sort_stats("time")
            report.print_stats()
            with open(args.prof, "w") as f:
                f.write(stream.getvalue())
            get_logger().info("finished writing profiler report")

    def _run_processor(self, args):
        """This method does not belong to public API."""
        if get_logger().is_enabled(Logging.DEBUG):
            self._register(self._monitor)
            self._register(self._timer)
        if get_logger().is_enabled(Logging.INFO):
            self._register(self._status_logger)
        try:
            get_logger().info("starting running processor")
            self._timer.start("processor")
            self._processor.run(args)
            self._timer.stop("processor")
            get_logger().info("finished running processor")
        finally:
            self._unregister(self._status_logger)
            self._unregister(self._timer)
            self._unregister(self._monitor)
        for name, secs in self._timer.amassed_times:
            get_logger().debug(f"amassed time (seconds): {secs :8.3f} {name}")
        secs = self._timer.process_time("processor")
        get_logger().info(f"elapsed time (seconds): {secs :8.3f}")

    @staticmethod
    def _report_stack_traces(args):
        """This method does not belong to public API."""
        if isinstance(args, Namespace) and args.stack_traces is True:
            traceback.print_exc(file=sys.stderr)

    @staticmethod
    def _register(callback):
        """This method does not belong to public API."""
        callback.register()

    @staticmethod
    def _unregister(callback):
        """This method does not belong to public API."""
        try:
            callback.unregister()
        except KeyError:  # ignored
            pass
