#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""This module provides a reader factory."""

from typing import Any

from .interface.reading import Reading


class ReaderFactory:
    """The reader factory."""

    @staticmethod
    def create_reader(
        aws: bool = False, config: dict[str:Any] | None = None
    ) -> Reading:
        """
        A static factory method to create a new reader instance.

        :param aws: Returns an AWS team store reader, if `True`, and a file
        system reader, if `False`.
        :param config: An optional configuration for the file system reader.
        :return: A new reader instance.
        """
        import os
        from .aws import AWS
        from .reader import Reader

        if aws:
            return AWS(
                bucket=os.environ["S3_USER_STORAGE_BUCKET"],
                key=os.environ["S3_USER_STORAGE_KEY"],
                secret=os.environ["S3_USER_STORAGE_SECRET"],
            )
        else:
            return Reader(config)
