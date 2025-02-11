#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module provides a writer factory."""

from typing import Any

from .interface.writing import Writing


class WriterFactory:
    """The writer factory."""

    @staticmethod
    def create_writer(
        aws: bool = False,
        config: dict[str:Any] | None = None,
        progress: bool = False,
    ) -> Writing:
        """
        A static factory method to create a new writer instance.

        :param aws: Returns an AWS team store writer, if `True`, and a file
        system writer, if `False`.
        :param config: An optional configuration for the file system writer.
        :param progress: To display a progress bar on the console, if
        supported by the writer.
        :return: A new writer instance.
        """
        import os
        from .aws import AWS
        from .writer import Writer

        if aws:
            return AWS(
                bucket=os.environ["S3_USER_STORAGE_BUCKET"],
                key=os.environ["S3_USER_STORAGE_KEY"],
                secret=os.environ["S3_USER_STORAGE_SECRET"],
            )
        else:
            return Writer(config, progress=progress)
