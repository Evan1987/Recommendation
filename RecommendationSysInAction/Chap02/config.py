""" Config file for this chapter."""

import os


DATA_SOURCE = ""


def _config_validity():
    if not os.path.exists(DATA_SOURCE):
        raise IOError(f"`{DATA_SOURCE}` not exist.")
    return True


assert _config_validity(), "Some config items are invalid, please check."
