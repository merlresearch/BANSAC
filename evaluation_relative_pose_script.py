# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -------------------------------------------------------------------------------------
# Import packages
import argparse
import copy
import enum
import pickle
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import tqdm
from tabulate import tabulate

from extra_functions import *

#
from utils import *

# -------------------------------------------------------------------------------------
# methods
cv_keys = {}

cv_keys_method = {}
cv_keys_method["RANSAC"] = cv2.SAMPLING_UNIFORM
cv_keys_method["NAPSAC"] = cv2.SAMPLING_NAPSAC
cv_keys_method["P-NAPSAC"] = cv2.SAMPLING_PROGRESSIVE_NAPSAC
cv_keys_method["PROSAC"] = cv2.SAMPLING_PROSAC
cv_keys_method["BANSAC"] = cv2.SAMPLING_BANSAC
cv_keys_method["P-BANSAC"] = cv2.SAMPLING_PBANSAC

cv_keys_method_lo = {}
cv_keys_method_lo["RANSAC_LO"] = cv2.SAMPLING_UNIFORM
cv_keys_method_lo["NAPSAC_LO"] = cv2.SAMPLING_NAPSAC
cv_keys_method_lo["P-NAPSAC_LO"] = cv2.SAMPLING_PROGRESSIVE_NAPSAC
cv_keys_method_lo["PROSAC_LO"] = cv2.SAMPLING_PROSAC
cv_keys_method_lo["BANSAC_LO"] = cv2.SAMPLING_BANSAC
cv_keys_method_lo["P-BANSAC_LO"] = cv2.SAMPLING_PBANSAC

cv_keys = {**cv_keys_method, **cv_keys_method_lo}

# -------------------------------------------------------------------------------------
# Paths for datasets
datasets_directory = "data/"

# -------------------------------------------------------------------------------------
all_sequences = {
    "essential_fundamental": [
        "brandenburg_gate",
        "buckingham_palace",
        "colosseum_exterior",
        "grand_place_brussels",
        "notre_dame_front_facade",
        "palace_of_westminster",
        "pantheon_exterior",
        "prague_old_town_square",
        "sacre_coeur",
        "st_peters_square",
        "taj_mahal",
        "temple_nara_japan",
        "trevi_fountain",
        "westminster_abbey",
    ],
    "homography": ["EVD", "HPatchesSeq"],
}

# -------------------------------------------------------------------------------------
# Parsers
parser = argparse.ArgumentParser()
parser.add_argument(
    "--type",
    type=str,
    choices=["evaluate", "results", "all"],
    default="all",
    help="Either run the get the number for a particular sequence (option 'evaluate') or get the numbers ('results')",
)
parser.add_argument(
    "--sequence",
    type=str,
    choices=[
        "brandenburg_gate",
        "buckingham_palace",
        "colosseum_exterior",
        "grand_place_brussels",
        "notre_dame_front_facade",
        "palace_of_westminster",
        "pantheon_exterior",
        "prague_old_town_square",
        "sacre_coeur",
        "st_peters_square",
        "taj_mahal",
        "temple_nara_japan",
        "trevi_fountain",
        "westminster_abbey",
    ],
    default="sacre_coeur",
    help="The sequence to run the experiments. The default is 'sacre_coeur'",
)
parser.add_argument(
    "--number_pairs",
    type=int,
    default=None,
    help="Number of selected pairs from the dataset. The code will get this number of pair results starting from the beggining of the dataset. Default is running all.",
)
parser.add_argument(
    "--problem",
    type=str,
    choices=["essential", "fundamental"],
    default="fundamental",
    help="Select the problem to be solved",
)

# -------------------------------------------------------------------------------------
# Problems
def is_problem_essential(problem: str) -> bool:
    return problem == "essential"


def is_problem_fundamental(problem: str) -> bool:
    return problem == "fundamental"


# -------------------------------------------------------------------------------------
# mAA auxiliary functions to compute the numbers
def calc_mAA_pose(MAEs: np.array, ths: np.array = np.linspace(1.0, 10, 100)) -> float:
    acc = []
    for th in ths:
        A = (MAEs <= th * 3.1415 / 180).astype(np.float32).mean()
        acc.append(A)
    return np.array(acc).mean()


# auxiliary function to computer the time avg
def get_time_avg_eval(data: pd.DataFrame, method: str) -> float:
    time = round(1000 * data.describe()[method]["mean"], 4)
    return time


# get all the resulta
def show_results_relative_pose(data_rotation, data_translation, data_time, maa_thresholds):

    data = []
    for item, methods in enumerate([list(cv_keys_method.keys()), list(cv_keys_method_lo.keys())]):
        if item == 0:
            print("Results WITHOUT Local Optimization")
        else:
            print("\nResults WITH Local Optimization")
        print("-------------------------------------")

        data = []
        for method in methods:

            header = ["Method"]
            data_row = [method]
            compute_time = get_time_avg_eval(data_time, method)
            for thr in maa_thresholds:
                maa_rot_thr = calc_mAA_pose(data_rotation[method], np.linspace(1.0, thr, 100))
                maa_trans_thr = calc_mAA_pose(data_translation[method], np.linspace(1.0, thr, 100))
                header.append("mAA(R," + str(thr) + ")")
                data_row.append(maa_rot_thr)
                header.append("mAA(t," + str(thr) + ")")
                data_row.append(maa_trans_thr)
            header.append("Time")
            data_row.append(compute_time)
            data.append(data_row)

        print(tabulate(data, headers=header))


# -------------------------------------------------------------------------------------
# single evaluation script
def eval_sample(input: tuple) -> tuple:

    key, m, ms, dR, dT, K1, K2, problem = input

    _m = copy.deepcopy(m)
    _ms = copy.deepcopy(ms)

    # all these methods required ordered data
    # in our case we only need the weights
    if key == "PROSAC" or key == "PROSAC_LO" or key == "P-NAPSAC" or key == "P-NAPSAC_LO":
        sort_index = np.argsort(_ms)
        _ms = _ms[sort_index]
        sort_index_matrix = np.array(sort_index).reshape(len(sort_index), 1) * np.array([1, 1, 1, 1]).reshape(1, 4)
        _m = np.take_along_axis(_m, sort_index_matrix, axis=0)

    _good_matches = _ms < 0.85
    _pts1 = _m[_good_matches, :2]  # coordinates in image 1
    _pts2 = _m[_good_matches, 2:]  # coordinates in image 2

    _p1n = normalize_keypoints(_pts1, K1)
    _p2n = normalize_keypoints(_pts2, K2)

    # -------------------------------------------------------------------------------------
    # usac general settings
    params = cv2.UsacParams()
    params.score = cv2.SCORE_METHOD_RANSAC
    params.loMethod = cv2.LOCAL_OPTIM_NULL
    dist = np.array([0, 0, 0, 0])
    # settings for local optimization
    if (
        key == "RANSAC_LO"
        or key == "PROSAC_LO"
        or key == "BANSAC_LO"
        or key == "P-BANSAC_LO"
        or key == "P-NAPSAC_LO"
        or key == "NAPSAC_LO"
    ):
        params.loMethod = cv2.LOCAL_OPTIM_INNER_AND_ITER_LO

    # settings for P-BANSAC
    # we need to send the weights
    if key == "P-BANSAC" or key == "P-BANSAC_LO":
        params.weights = 1 - _ms[_good_matches]

    if is_problem_fundamental(problem):
        # estimate fundamental matrix
        params.maxIterations = 10000
        params.confidence = 0.999
        params.threshold = 0.5
        params.sampler = cv_keys[key]
        s_time = time.time()
        estimate, mask = cv2.findFundamentalMat(_pts1, _pts2, params)
        e_time = time.time()

    elif is_problem_essential(problem):
        # estimate essential matrix
        params.maxIterations = 1000
        params.confidence = 0.999
        params.threshold = 1e-3
        params.sampler = cv_keys[key]
        s_time = time.time()
        estimate, mask = cv2.findEssentialMat(_p1n, _p2n, np.eye(3), np.eye(3), dist, dist, params)
        e_time = time.time()

    # compute evaluation
    compute_time = e_time - s_time

    # NAPSAC fails sometimes
    if estimate is None or mask is None:
        print("Method " + key + " failed!")
        return key, np.pi, np.pi, compute_time, [0] * len(_p1n), params.maxIterations

    # Evaluation of the fundamental matrix estimation
    # is similar to the essential one. Meaning that we need
    # to convert from essential to fundamental matrix
    if is_problem_fundamental(problem):
        estimate = get_E_from_F(estimate, K1, K2)

    inliers = mask[0 : _pts1.shape[0]]  # Get inlier mask: 0.0 is outliers, 1.0 is inlier
    weights = mask[_pts1.shape[0] : mask.shape[0] - 1]  # Get probabilities of all points
    iterations = int(mask[-1, 0])  # Get number of iterations
    error_rot, error_trans = eval_essential_matrix(_p1n, _p2n, estimate, dR, dT)

    return key, error_rot, error_trans, compute_time, list(inliers).count(1), iterations


# -------------------------------------------------------------------------------------
# run evaluation
def evaluate(sequence: str, pairs: int, problem: str) -> None:

    print(":=> Loading data...")
    # load data

    matches = load_h5(f"{datasets_directory}/{sequence}/matches.h5")
    F_gt = load_h5(f"{datasets_directory}/{sequence}/Fgt.h5")
    E_gt = load_h5(f"{datasets_directory}/{sequence}/Egt.h5")
    matches_scores = load_h5(f"{datasets_directory}/{sequence}/match_conf.h5")
    K1_K2 = load_h5(f"{datasets_directory}/{sequence}/K1_K2.h5")
    R = load_h5(f"{datasets_directory}/{sequence}/R.h5")
    T = load_h5(f"{datasets_directory}/{sequence}/T.h5")

    # number of time pairs are repeated
    repete_pairs = 5

    # data dicts
    est_rot = {}
    est_trans = {}
    t_est = {}
    inliner = {}
    iter = {}
    for key in list(cv_keys.keys()):
        est_rot[key] = []
        est_trans[key] = []
        t_est[key] = []
        inliner[key] = []
        iter[key] = []

    # in case we do not run every sample
    # the the first number of pairs
    if pairs != None:
        keys = list(F_gt.keys())
        keys = keys[:pairs]
        values = [F_gt[k] for k in keys]
        samples = dict(zip(keys, values)).items()
    else:
        samples = F_gt.items()

    # get solutions
    print(":=> Running evaluation:")
    with Pool() as pool:
        for k, F in tqdm.tqdm(samples):

            img_id1 = k.split("-")[0]
            img_id2 = k.split("-")[1]

            m = matches[k]
            ms = matches_scores[k]

            K1 = K1_K2[img_id1 + "-" + img_id2][0][0]
            K2 = K1_K2[img_id1 + "-" + img_id2][0][1]

            R1 = R[img_id1]
            R2 = R[img_id2]
            T1 = T[img_id1]
            T2 = T[img_id2]
            dR = np.dot(R2, R1.T)
            dT = T2 - np.dot(dR, T1)

            if len(ms[ms < 0.85]) <= 25:
                print("WARN: Not enough data!")
                continue

            items = [(key, m, ms, dR, dT, K1, K2, problem) for key in list(cv_keys.keys()) * repete_pairs]
            for result in pool.map(eval_sample, items):
                (
                    key,
                    iteration_error_rot,
                    iteration_error_trans,
                    iteration_time,
                    iteration_inliers,
                    iteration_iterations,
                ) = result
                # essential matrix estimation
                est_rot[key].append(iteration_error_rot)
                est_trans[key].append(iteration_error_trans)
                t_est[key].append(iteration_time)
                inliner[key].append(iteration_inliers)
                iter[key].append(iteration_iterations)

    # save results!
    experiments_dicts = est_rot, est_trans, t_est, inliner, iter
    file = problem + "_" + sequence + "_" + str(pairs) + ".pkl"
    file_open = open(file, "wb")
    print(":=> Saving results: file " + file)
    pickle.dump(experiments_dicts, file_open)
    file_open.close()


# -------------------------------------------------------------------------------------
# get the numbers
def results(sequence: str, pairs: int, problem: str, maa_accuracy_list: list = [1, 5, 10]) -> None:

    print("Settings")
    print("------------------------")
    print("Problem=" + problem + ";")
    print("Sequence=" + sequence + "; ")
    print("Number of pairs=" + str(pairs))
    print(" ")

    # loading data
    file = problem + "_" + sequence + "_" + str(pairs) + ".pkl"

    file_open = open(file, "rb")
    experiments_dicts = pickle.load(file_open)
    file_open.close()

    if is_problem_fundamental(problem) or is_problem_essential(problem):
        # get the data
        estimation_rotation, estimation_translation, computational_time, _, _ = experiments_dicts
        # rotation errors
        rotation_dataframe = pd.DataFrame(estimation_rotation)
        # translation errors
        translation_dataframe = pd.DataFrame(estimation_translation)
        # time
        time_dataframe = pd.DataFrame(computational_time)
        # show results
        show_results_relative_pose(rotation_dataframe, translation_dataframe, time_dataframe, maa_accuracy_list)


# -------------------------------------------------------------------------------------
# main
def main() -> None:

    args = parser.parse_args()
    sequence = args.sequence
    pairs = args.number_pairs
    problem = args.problem

    if args.type == "evaluate" or args.type == "all":
        evaluate(sequence, pairs, problem)
    if args.type == "results" or args.type == "all":
        results(sequence, pairs, problem)


if __name__ == "__main__":
    main()
