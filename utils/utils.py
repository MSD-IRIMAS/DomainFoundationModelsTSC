import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder


def load_data(file_name):

    folder_path = "/pretext-code/ucr_archive/"
    folder_path += file_name + "/"

    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"

    if os.path.exists(test_path) <= 0:
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest


def znormalisation(x):

    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


def encode_labels(y):

    labenc = LabelEncoder()

    return labenc.fit_transform(y)


def create_directory(directory_path):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def load_datasets(list_file_names):

    xtrains = []
    ytrains = []

    xtests = []
    ytests = []

    n_samples = 0

    for file_name in list_file_names:

        xtrain, ytrain, xtest, ytest = load_data(file_name=file_name)

        xtrain = znormalisation(xtrain)
        xtest = znormalisation(xtest)

        ytrain = encode_labels(ytrain)
        ytest = encode_labels(ytest)

        xtrains.append(xtrain)
        ytrains.append(ytrain)

        xtests.append(xtest)
        ytests.append(ytest)

        n_samples = n_samples + len(ytrain)

    xtrain_pretext = []
    ytrain_pretext = np.zeros(shape=(n_samples), dtype=np.int32)

    counter = 0

    for i, file_name in enumerate(list_file_names):

        for j in range(len(ytrains[i])):

            xtrain_pretext.append(xtrains[i][j])
            ytrain_pretext[counter] = i

            counter = counter + 1

    return xtrains, ytrains, xtests, ytests, xtrain_pretext, ytrain_pretext


def plot_datasets(xtrains, list_of_datasets, output_dir):

    plt.figure(figsize=(20, 10))

    colors = ["blue", "red", "green", "purple"]

    for i, xtrain in enumerate(xtrains):

        ns = np.random.randint(low=0, high=len(xtrain), size=4)

        for j, n in enumerate(ns):
            plt.plot(xtrain[n], lw=3, color=colors[j])

        plt.savefig(output_dir + list_of_datasets[i] + ".pdf")

        plt.cla()

    plt.clf()
