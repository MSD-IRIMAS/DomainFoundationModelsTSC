import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import random
import os
from utils.ucr_names import ucr_names_by_type


def get_strs(dir_path):

    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            if "baselines" not in dir:
                if "file_too_long" in dir:
                    return None
                return dir.split("_")


def get_results_ensemble(exp_name, clf_name):

    if os.path.exists(
        "exps/" + exp_name + "/" + clf_name + "/results/results_ens.csv"
    ) and os.path.exists(
        "exps/" + exp_name + "/" + clf_name + "/results/results_baselines_ens.csv"
    ):
        return pd.read_csv(
            "exps/" + exp_name + "/" + clf_name + "/results/results_ens.csv"
        ), pd.read_csv(
            "exps/" + exp_name + "/" + clf_name + "/results/results_baselines_ens.csv"
        )

    df_results_ens = pd.DataFrame(columns=["dataset", "accuracy-ens"])

    datasets = get_strs(dir_path="exps/" + exp_name + "/" + clf_name + "/results/")

    if datasets is None:
        datasets = ucr_names_by_type[exp_name]
        dir_datasets = "dataset_names_(file_too_long)"

    else:
        dir_datasets = "_".join(datasets)

    path_datasets = (
        "exps/" + exp_name + "/" + clf_name + "/results/" + dir_datasets + "/"
    )
    paths_ensembles = path_datasets + "ensembles/"

    for dataset in datasets:

        df = pd.read_csv(paths_ensembles + "/" + dataset + "/metrics.csv")

        score = df["accuracy"][0]

        df_results_ens.loc[len(df_results_ens)] = {
            "dataset": dataset,
            "accuracy-ens": score,
        }

    df_results_ens.to_csv(
        "exps/" + exp_name + "/" + clf_name + "/results/results_ens.csv", index=False
    )

    df_baselines_ens = pd.DataFrame(columns=["dataset", "accuracy-ens"])

    path_baselines_ens = "exps/" + exp_name + "/" + clf_name + "/results/baselines_ens/"

    for dataset in datasets:

        df = pd.read_csv(path_baselines_ens + dataset + "/metrics.csv")

        score = df["accuracy"][0]

        df_baselines_ens.loc[len(df_baselines_ens)] = {
            "dataset": dataset,
            "accuracy-ens": score,
        }

    df_baselines_ens.to_csv(
        "exps/" + exp_name + "/" + clf_name + "/results/results_baselines_ens.csv",
        index=False,
    )

    return df_results_ens, df_baselines_ens


if __name__ == "__main__":

    exp_name = "test"
    clf_name = "HInception"

    df_results_ens, df_baselines_ens = get_results_ensemble(
        exp_name=exp_name, clf_name=clf_name
    )
