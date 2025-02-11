#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module defines the exit codes returned by the WQF processor.
"""


class ExitCodes:
    """
    This class defines the exit codes returned by the WQF processor.
    """

    SUCCESS = 0
    """
    Indicates a successful and complete processing.
    """

    FAILURE_ARGUMENT_ERROR = 128
    """
    Indicates severe processing failure due to an invalid command line
    argument.
    """

    FAILURE_ASSERTION = 129
    """
    Indicates severe processing failure due to a failed assertion.
    """

    FAILURE_KEYBOARD_INTERRUPT = 130
    """
    Indicates severe processing failure due to keyboard interrupt.
    """

    FAILURE_MEMORY_ERROR = 131
    """
    Indicates severe processing failure due to an out of memory
    error.
    """

    FAILURE_OS_ERROR = 132
    """
    Indicates severe processing failure due to an operating system
    error.
    """

    FAILURE_RUNTIME_ERROR = 133
    """
    Indicates severe processing failure due to a runtime error.
    """

    FAILURE_SYSTEM_ERROR = 134
    """Indicates severe processing failure due to a system error.
    """

    FAILURE_SYSTEM_EXIT = 135
    """
    Indicates severe processing failure due to system exit.
    """

    FAILURE_UNSPECIFIC = 255
    """
    Indicates severe processing failure due to an unspecific reason.
    """
