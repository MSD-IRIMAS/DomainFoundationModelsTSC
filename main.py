import numpy as np
import pandas as pd
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import load_datasets, create_directory, plot_datasets
from utils.ucr_names import ucr_names

from classifiers.HInception import HINCEPTION
from classifiers.HInception_baseline import HINCEPTION_BASELINE

from sklearn.metrics import accuracy_score


def get_dir_name_(list_of_datasets):

    if list_of_datasets is None:
        return "all_ucr/"
    else:
        return "_".join(list_of_datasets) + "/"


def get_list_of_datasets_(list_of_datasets):

    if list_of_datasets is None:
        return ucr_names
    return list_of_datasets


@hydra.main(config_name="config.yaml", config_path="config")
def main(args: DictConfig):

    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    xtrains, ytrains, xtests, ytests, xtrain_pretext, ytrain_pretext = load_datasets(
        list_file_names=args.list_of_datasets
    )

    list_classes = [len(np.unique(ytrain)) for ytrain in ytrains]
    list_of_length_TS = [int(x.shape[1]) for x in xtrains]

    output_dir_classifier = args.classifier + "/"
    create_directory(output_dir_classifier)

    output_dir_results = output_dir_classifier + args.output_dir + "/"
    create_directory(output_dir_results)

    if args.run_pretext_finetune:

        try:
            output_dir_datasets = output_dir_results + get_dir_name_(
                list_of_datasets=args.list_of_datasets
            )
            create_directory(output_dir_datasets)
        except OSError:
            output_dir_datasets = output_dir_results + "dataset_names_(file_too_long)/"
            create_directory(output_dir_datasets)

        plot_datasets(
            xtrains=xtrains,
            list_of_datasets=get_list_of_datasets_(args.list_of_datasets),
            output_dir=output_dir_datasets,
        )

        ypreds = []

        for _run_pretext in range(args.runs_pretext):

            output_dir_run_pretext = (
                output_dir_datasets + "pretext_run_" + str(_run_pretext) + "/"
            )
            create_directory(output_dir_run_pretext)

            if args.classifier == "HInception":

                clf = HINCEPTION(
                    output_dir=output_dir_run_pretext,
                    list_of_datasets=get_list_of_datasets_(args.list_of_datasets),
                    list_of_n_classes=list_classes,
                    list_of_length_TS=list_of_length_TS,
                    depth_pretext=args.depth_pretext,
                    depth=args.depth,
                    batch_size_pretext=args.batch_size_pretext,
                    batch_size=args.batch_size,
                    n_epochs_pretext=args.n_epochs_pretext,
                    n_epochs=args.n_epochs,
                )

            if args.train_pretext:
                clf.fit_pretext(xtrains=xtrain_pretext, ytrains=ytrain_pretext)

            ypreds.append(
                clf.fit_and_predict_models(
                    xtrains=xtrains,
                    ytrains=ytrains,
                    xtests=xtests,
                    ytests=ytests,
                    n_runs=args.runs_fine_tune,
                    train_models=args.train_finetune,
                )
            )

        ypreds_per_data = [
            np.zeros(shape=(len(ytest), len(np.unique(ytest)))) for ytest in ytests
        ]

        output_dir_pretext_ens = output_dir_datasets + "ensembles/"
        create_directory(output_dir_pretext_ens)

        for d in range(len(ypreds[0])):

            output_dir_dataset_ens = (
                output_dir_pretext_ens
                + get_list_of_datasets_(args.list_of_datasets)[d]
                + "/"
            )
            create_directory(output_dir_dataset_ens)

            for _run_pretext in range(args.runs_pretext):
                ypreds_per_data[d] += np.asarray(ypreds[_run_pretext][d])

            ypreds_per_data[d] /= 1.0 * args.runs_pretext
            acc = accuracy_score(
                y_true=ytests[d],
                y_pred=np.argmax(ypreds_per_data[d], axis=1),
                normalize=True,
            )

            df_ens = pd.DataFrame(columns=["accuracy"])
            df_ens.loc[len(df_ens)] = {"accuracy": acc}

            df_ens.to_csv(output_dir_dataset_ens + "metrics.csv", index=False)

    if args.run_baseline:

        output_dir_baseline = output_dir_results + "baselines/"
        create_directory(output_dir_baseline)

        output_dir_baseline_ens = output_dir_results + "baselines_ens/"
        create_directory(output_dir_baseline_ens)

        for d, dataset_name in enumerate(get_list_of_datasets_(args.list_of_datasets)):

            xtrain = xtrains[d]
            ytrain = ytrains[d]

            xtest = xtests[d]
            ytest = ytests[d]

            length_TS = int(xtrain.shape[1])
            n_classes = len(np.unique(ytrain))

            ypred = np.zeros(shape=(len(ytest), n_classes))

            for _run_baseline in range(args.runs_baseline):

                output_dir_baseline_run = (
                    output_dir_baseline + "baseline_run_" + str(_run_baseline) + "/"
                )
                create_directory(output_dir_baseline_run)

                output_dir_baseline_dataset = (
                    output_dir_baseline_run + dataset_name + "/"
                )
                create_directory(output_dir_baseline_dataset)

                if args.classifier == "HInception":

                    clf = HINCEPTION_BASELINE(
                        output_dir=output_dir_baseline_dataset,
                        length_TS=length_TS,
                        n_classes=n_classes,
                        depth=args.depth,
                        batch_size=args.batch_size_baseline,
                        n_epochs=args.n_epochs_baseline,
                    )

                if args.train_baseline:
                    clf.fit(xtrain=xtrain, ytrain=ytrain)

                ypred = ypred + clf.predict(xtest=xtest, ytest=ytest)

            ypred = ypred / (1.0 * args.runs_baseline)

            ypred = np.argmax(ypred, axis=1)

            acc = accuracy_score(y_true=ytest, y_pred=ypred, normalize=True)

            output_dir_baseline_ens_dataset = (
                output_dir_baseline_ens + dataset_name + "/"
            )
            create_directory(output_dir_baseline_ens_dataset)

            df = pd.DataFrame(columns=["accuracy"])
            df.loc[len(df)] = {"accuracy": acc}

            df.to_csv(output_dir_baseline_ens_dataset + "metrics.csv", index=False)


if __name__ == "__main__":

    create_directory("exps/")
    main()
