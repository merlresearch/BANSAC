# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------
# filter out fails
def filter_outlier_estimates(
    data: pd.DataFrame,
    angle_limit_degrees: float = None,
) -> pd.DataFrame:

    angle_limit = 90 * np.pi / 360
    list_methods = list(data.keys())
    _data = copy.deepcopy(data)
    for method in list_methods:
        _data = _data[_data[method] < angle_limit]

    return _data


# ----------------------------------------------------------------------------------
# stats we want to show
def print_stat(data: pd.DataFrame, info: str = "Undefined") -> None:

    list_methods = list(data.keys())
    if info == "time":
        print(":=> Computation time [avg(std)]")
        for method in list_methods:
            print(
                " '"
                + method
                + "': "
                + str(round(1000 * data.describe()[method]["mean"], 2))
                + "("
                + str(round(1000 * data.describe()[method]["std"], 2))
                + ") "
            )
    else:
        print(":=> " + info + " errors [avg(std)]")
        for method in list_methods:
            print(
                " '"
                + method
                + "': "
                + str(round(data.describe()[method]["mean"] * 180 / np.pi, 2))
                + "("
                + str(round(data.describe()[method]["std"] * 180 / np.pi, 2))
                + ")"
            )


# ----------------------------------------------------------------------------------
# filter out fails -- homography
def filter_outlier_estimates_h(
    data: pd.DataFrame,
    error: int,
) -> pd.DataFrame:

    list_methods = list(data.keys())
    _data = copy.deepcopy(data)
    for method in list_methods:
        _data = _data[_data[method] < error]

    return _data


# ----------------------------------------------------------------------------------
# stats we want to show -- homography
def print_stat_h(data: pd.DataFrame, info: str = "Undefined") -> None:

    list_methods = list(data.keys())
    if info == "time":
        print(":=> Computation time [avg(std)]")
        for method in list_methods:
            print(
                " '"
                + method
                + "': "
                + str(round(1000 * data.describe()[method]["mean"], 2))
                + "("
                + str(round(1000 * data.describe()[method]["std"], 2))
                + ") "
            )
    else:
        print(":=> " + info + " errors [avg(std)]")
        for method in list_methods:
            print(
                " '"
                + method
                + "': "
                + str(round(data.describe()[method]["mean"], 2))
                + "("
                + str(round(data.describe()[method]["std"], 2))
                + ")"
            )
