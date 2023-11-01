# Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -------------------------------------------------------------------------------------
# Import packages
import argparse
import copy
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

ERROR = 99999999

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
    choices=["EVD", "HPatchesSeq"],
    default="HPatchesSeq",
    help="The sequence to run the experiments. The default is 'sacre_coeur'",
)
parser.add_argument(
    "--number_pairs",
    type=int,
    default=None,
    help="Number of selected pairs from the dataset. The code will get this number of pair results starting from the beggining of the dataset. Default is running all, which we suggest since the dataset is small.",
)

# -------------------------------------------------------------------------------------
# mAA auxiliary functions to compute the numbers
def calc_mAA_pose(MAEs: np.array, ths: np.array = np.linspace(1.0, 10, 100)) -> float:
    acc = []
    for th in ths:
        A = (MAEs <= th).astype(np.float32).mean()
        acc.append(A)
    return np.array(acc).mean()


# auxiliary function to computer the time avg
def get_time_avg_eval(data: pd.DataFrame, method: str) -> float:
    time = round(1000 * data.describe()[method]["mean"], 4)
    return time


# get all the resulta
def show_results_homography(data_homography, data_time, maa_thresholds):

    data = []
    for item, methods in enumerate([list(cv_keys_method.keys()), list(cv_keys_method_lo.keys())]):
        if item == 0:
            print("Results WITHOUT Local Optimization")
        else:
            print("\nResults WITH Local Optimization")
        print("-------------------------------------")

        data = []
        for method in methods:

            compute_time = get_time_avg_eval(data_time, method)
            header = ["Method"]
            data_row = [method]
            for thr in maa_thresholds:
                maa_h_thr = calc_mAA_pose(data_homography[method], np.linspace(1.0, thr, 100))
                header.append("mAA(H," + str(thr) + ")")
                data_row.append(maa_h_thr)
            header.append("Time")
            data_row.append(compute_time)
            data.append(data_row)

        print(tabulate(data, headers=header))
        print(" ")


# -------------------------------------------------------------------------------------
# single evaluation script
def eval_sample(input: tuple) -> tuple:

    key, m, ms, H_gt, h, w = input

    _m = copy.deepcopy(m)
    _ms = copy.deepcopy(ms)

    # all these methods required ordered data
    # in our case we only need the weights
    if key == "PROSAC" or key == "PROSAC_LO" or key == "P-NAPSAC" or key == "P-NAPSAC_LO":
        sort_index = np.argsort(_ms)
        _ms = _ms[sort_index]
        sort_index_matrix = np.array(sort_index).reshape(len(sort_index), 1) * np.array([1, 1, 1, 1]).reshape(1, 4)
        _m = np.take_along_axis(_m, sort_index_matrix, axis=0)

    _good_matches = _ms < 0.6
    src_pts = _m[_good_matches, :2]
    dst_pts = _m[_good_matches, 2:]

    # usac general settings
    params = cv2.UsacParams()
    # P-BANSAC needs the pre-computed weights
    if key == "P-BANSAC":
        params.weights = 1 - _ms[_good_matches]

    # NAPSAC does not work without this options
    if key == "NAPSAC" or key == "P-NAPSAC":
        params.neighborsSearch = cv2.NEIGH_FLANN_RADIUS

    params.score = cv2.SCORE_METHOD_RANSAC
    params.loMethod = cv2.LOCAL_OPTIM_NULL
    params.maxIterations = 1000
    params.confidence = 0.999
    params.threshold = 1.0
    params.sampler = cv_keys[key]

    # as in the ef case, we have to have two options here since
    # napsac does not work with usac.params
    s_h_time = time.time()
    H, mask = cv2.findHomography(src_pts, dst_pts, params)
    e_h_time = time.time()

    # when methods fail
    if H is None or mask is None:
        return key, ERROR, ERROR, ERROR, ERROR

    # save and output the results
    h_time = e_h_time - s_h_time
    h_inliers = mask[0 : src_pts.shape[0]]  # Get inlier mask: 0.0 is outliers, 1.0 is inlier
    h_iterations = int(mask[-1, 0])  # Get number of iterations
    #
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    # Ground truth transformation
    dst_GT = cv2.perspectiveTransform(pts, H_gt)
    #
    error_points = dst - dst_GT
    error = 0
    for point in error_points:
        point_array = point.reshape((2,))
        error = error + (point_array[0] ** 2 + point_array[1] ** 2) ** (1 / 2)

    return key, error / 4, h_time, list(h_inliers).count(1), h_iterations


# -------------------------------------------------------------------------------------
# run evaluation
def evaluate(seq: str, pairs: int) -> None:

    print(":=> Loading data...")
    # load data
    DIR = "data/homography"
    dataset = seq
    split = "val"
    matches = load_h5(f"{DIR}/{dataset}/{split}/matches.h5")
    Hgt = load_h5(f"{DIR}/{dataset}/{split}/Hgt.h5")
    matches_scores = load_h5(f"{DIR}/{dataset}/{split}/match_conf.h5")

    # for this case we repeat each pair 10 times
    repete_pairs = 10

    # dict for keeping results
    h_est = {}
    t_est = {}
    iter_est = {}
    inlier_est = {}
    for key in list(cv_keys.keys()):
        h_est[key] = []
        t_est[key] = []
        iter_est[key] = []
        inlier_est[key] = []

    # run compute solutions
    print(":=> Running evaluation:")
    with Pool() as pool:
        # go over the image pairs
        for k, H_gt in tqdm.tqdm(Hgt.items()):

            # we need to load the images because the w and h
            # are needed in evaluation
            if dataset == "HPatchesSeq":
                m = matches[k]
                ms = matches_scores[k].reshape(-1)
                img1_fname = f"{DIR}/{dataset}/{split}/imgs/{k[:-4]}/1.ppm"
                # img2_fname = f'{DIR}/{dataset}/{split}/imgs/{k[:-4]}/{k[-1]}.ppm'
                img1 = cv2.cvtColor(cv2.imread(img1_fname), cv2.COLOR_BGR2RGB)
            elif dataset == "EVD":
                m = matches[k]
                ms = matches_scores[k].reshape(-1)
                img1_fname = f"{DIR}/{dataset}/{split}/imgs/1/" + k.split("-")[0] + ".png"
                # img2_fname = f'{DIR}/{dataset}/{split}/imgs/2/' + k.split('-')[0] + '.png'
                img1 = cv2.cvtColor(cv2.imread(img1_fname), cv2.COLOR_BGR2RGB)
            else:
                print("ERR: Invalid dataset")
                exit(0)

            if len(ms[ms < 0.6]) <= 25:
                print("Not enough data!")
                continue

            # getting results
            h, w, __ = img1.shape
            items = [(key, m, ms, H_gt, h, w) for key in list(cv_keys.keys()) * repete_pairs]
            for result in pool.map(eval_sample, items):
                key, h_est_error, t_est_error, inl_est_error, iter_est_error = result
                h_est[key].append(h_est_error)
                t_est[key].append(t_est_error)
                iter_est[key].append(iter_est_error)
                inlier_est[key].append(inl_est_error)

    # save data
    experiments_dicts = (h_est, t_est, iter_est, inlier_est)
    file = "homography_" + seq + "_" + str(pairs) + ".pkl"
    print(":=> Saving results: " + file)
    file_open = open(file, "wb")
    pickle.dump(experiments_dicts, file_open)
    file_open.close()


# -------------------------------------------------------------------------------------
# get the numbers
def results(sequence: str, pairs: int, maa_accuracy_list: list = [1, 5, 10], cut_off_error: int = 250) -> None:

    # loading data
    file = "homography_" + sequence + "_" + str(pairs) + ".pkl"
    file_open = open(file, "rb")
    experiments_dicts = pickle.load(file_open)
    file_open.close()

    # get the data
    estimation_homography, computational_time, _, _ = experiments_dicts
    # homography errors
    homography_dataframe = pd.DataFrame(estimation_homography)
    # time
    time_dataframe = pd.DataFrame(computational_time)
    # filter out fail cases
    for method in list(cv_keys.keys()):
        homography_dataframe = homography_dataframe[homography_dataframe[method] < ERROR]
        time_dataframe = time_dataframe[time_dataframe[method] < ERROR]
    # show results
    show_results_homography(homography_dataframe, time_dataframe, maa_accuracy_list)


# -------------------------------------------------------------------------------------
# main
def main() -> None:

    args = parser.parse_args()
    sequence = args.sequence
    pairs = args.number_pairs

    if args.type == "evaluate" or args.type == "all":
        evaluate(sequence, pairs)
    if args.type == "results" or args.type == "all":
        results(sequence, pairs)


if __name__ == "__main__":
    main()
