import numpy as np
import os
import pickle
from sklearn import decomposition


class PCA:
    def __init__(self, output_dim, input_dim):
        self.W = None
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.decomposer = decomposition.PCA(n_components=output_dim)

    def fit(self, X):
        self.decomposer.fit(X)

    def transform(self, X):
        return self.decomposer.transform(X)

    def get_transform_matrix(self):
        return self.decomposer.components_


def make_single_matrix(dataset):
    X = None
    y = np.array([])
    for key, data_ in dataset.items():
        X_entry = None
        tag_key = list(data_["data"].keys())[2:]
        for tag in tag_key:
            if X_entry is None:
                X_entry = data_["data"][tag]
            else:
                X_entry = np.hstack([X_entry, data_["data"][tag]])

        if X is None:
            X = X_entry
        else:
            X = np.vstack([X, X_entry])

        y = np.concatenate([y, data_["labels"]])

    return y, X


def transform_individual(data, pca):
    X_entry = None
    tag_key = list(data["data"].keys())[2:]
    for tag in tag_key:
        if X_entry is None:
            X_entry = data["data"][tag]
        else:
            X_entry = np.hstack([X_entry, data["data"][tag]])

    X_entry = pca.transform(X_entry)
    return X_entry


train_data_file = "data/train_data.pkl"
test_data_file = "data/test_data.pkl"
post_pca_train_data_file = "data/train_data_pca.pkl"
post_pca_test_data_file = "data/test_data_pca.pkl"

pca_result_file = "data/pca_result.pkl"

if __name__ == "__main__":
    with open(train_data_file, "rb") as f:
        data = pickle.load(f)

    y, X = make_single_matrix(data)

    pca = PCA(10, X.shape[1])
    pca.fit(X)
    X_pca = pca.transform(X)
    print(X_pca.shape)
    W_pca = pca.get_transform_matrix()

    with open(pca_result_file, "wb") as f:
        pickle.dump({"W": W_pca}, f)

    for pre_process, post_processed in [(train_data_file, post_pca_train_data_file),
                                        (test_data_file, post_pca_test_data_file)]:
        with open(pre_process, "rb") as f:
            data = pickle.load(f)

        data_reduced = {}
        for key, data_ in data.items():
            X_entry = transform_individual(data_, pca)
            data_reduced[key] = {"X_pca": X_entry}
            data_reduced[key].update(data_)
            # print(X_entry.shape)

        with open(post_processed, "wb") as f:
            pickle.dump(data_reduced, f)
