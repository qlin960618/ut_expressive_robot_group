import pickle

import pgmpy.models
import pgmpy.inference
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd

import numpy as np


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


class BDN:
    def __init__(self, X_sample, y_sample, depth=10):
        self.Xs_train = None
        self.ys_train = None
        self.train_data = None
        self.num_features = X_sample.shape[1]
        self.depth = depth
        self.model = pgmpy.models.DynamicBayesianNetwork()

        nodes = [f'X{i}' for i in range(self.num_features)]
        self.node_names = nodes+['Y']
        print(self.node_names)
        # Add nodes
        for t in range(depth):
            self.model.add_nodes_from((node, t) for node in self.node_names)
        # print("model nodes: ", self.model.nodes())

        # add edges
        edges = []
        for i in range(self.num_features):
            edges.append((('X' + str(i) + '_0', 0), ('Y_0', 0)))
        for t in range(1, depth):
            for i in range(self.num_features):
                for j in range(self.num_features):
                    if i != j:
                        edges.append((('X' + str(i) , t - 1), ('X' + str(j), t)))
        self.model.add_edges_from(edges)
        self.model.initialize_initial_state()
        self.model.check_model()
        # print("model edges: ", self.model.edges())

    def train(self, Xs, ys):
        self.Xs_train, self.ys_train = Xs, ys

        for X, y in zip(Xs, ys):
            # samples = np.hstack((X, [y]))
            samples = np.hstack((X, y.reshape(-1, 1)))
            # for t in range(0, self.depth):
            #     samples = np.hstack((samples, np.hstack((X, y.reshape(-1, 1)))))
            frame_names = []
            # for t in range(self.depth):
            frame_names += [name for name in self.node_names]
            samples = pd.DataFrame(samples, columns=frame_names)

            self.model.fit(samples, estimator="MLE")

            # print("model: ", self.model)
            # mle = MaximumLikelihoodEstimator(model=self.model, data=samples)
            # self.train_data = samples

    def predict(self, X):

        self.model.predict(X)
        # print("model_inference: ", self.model_inference)



train_data_file = "data/train_data_pca.pkl"
test_data_file = "data/test_data_pca.pkl"

if __name__ == "__main__":
    train = load_pickle(train_data_file)

    Xs = []
    ys = []
    for key, value in train.items():
        Xs.append(value["X_pca"])
        ys.append(value["labels"])

    bdn_model = BDN(Xs[0], ys[0])
    bdn_model.train(Xs, ys)
    exit()

    test = load_pickle(test_data_file)

    Xs_test = []
    ys_truth = []
    ys_pred = []
    for key, value in test.items():
        X_test = value["X_pca"]
        y_truth = value["labels"]
        # y_pred = bdn_model.predict(X_test)
