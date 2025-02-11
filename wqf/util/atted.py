#  Copyright (c) Brockmann Consult GmbH, 2024
#  License: MIT

"""
This module provides a utility to edit dataset attributes in place.
"""

import os
import uuid
from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser

from xcube.core.store import new_data_store


def _new_team_store():
    """Returns a new team store."""
    return new_data_store(
        "s3",
        root=os.environ["S3_USER_STORAGE_BUCKET"],
        storage_options={
            "anon": False,
            "key": os.environ["S3_USER_STORAGE_KEY"],
            "secret": os.environ["S3_USER_STORAGE_SECRET"],
        },
    )


def _new_file_store():
    """Returns a new file store."""
    return new_data_store("file", root=".")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="wqf-atted",
        description="This command edits the global attributes of a dataset"
        "in place.",
        epilog="Copyright (c) Brockmann Consult GmbH, 2024. License: MIT",
        exit_on_error=True,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--acknowledge-copernicus-service",
        help="acknowledge use of data provided by the Copernicus Service",
        action="store_true",
        default=False,
        required=False,
        dest="acknowledge_copernicus_service",
    )
    parser.add_argument(
        "--acknowledge-ospar",
        help="acknowledge use of data provided by the OSPAR Commission",
        action="store_true",
        default=False,
        required=False,
        dest="acknowledge_ospar",
    )
    parser.add_argument("data_id", help="the dataset identifier")
    args = parser.parse_args()

    team_store = _new_team_store()
    file_store = _new_file_store()

    data = team_store.open_data(args.data_id)
    data.attrs.clear()
    data.attrs["Conventions"] = "CF-1.11"
    acknowledgements = ""
    if args.acknowledge_copernicus_service:
        acknowledgements = (
            f"{acknowledgements}"
            f"Contains modified Copernicus Service information 2016, 2017, "
            f"2018, 2019, 2020. "
        )
    if args.acknowledge_ospar:
        acknowledgements = (
            f"{acknowledgements}"
            f"Contains unpublished chlorophyll concentration data provided "
            f"by the OSPAR Commission 2016, 2017, 2018, 2019, 2020. Public "
            f"OSPAR data are available at OSPAR Data and Information "
            f"Management System, https://odims.ospar.org/."
        )
    data.attrs["acknowledgements"] = acknowledgements
    data.attrs["creator"] = "Brockmann Consult GmbH"
    data.attrs["funding"] = (
        "ESA ITT for Future EO-1 EO Science for Society Permanently Open "
        "Call for Proposals, ESA Ref. ESA AO/1-10468/20/I-FvO"
    )
    data.attrs["license"] = (
        "CC BY-NC-ND 4.0, https://creativecommons.org/licenses/by-nc-nd/4.0/"
    )
    data.attrs["uuid"] = f"{uuid.uuid4()}"

    file_store.write_data(data, args.data_id, replace=True)
    team_store.write_data(
        file_store.open_data(args.data_id), args.data_id, replace=True
    )
