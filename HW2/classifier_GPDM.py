import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        return data


train_data_file = "data/train_data_pca.pkl"
test_data_file = "data/test_data_pca.pkl"  # Add the test data file path here

if __name__ == "__main__":
    train_data = load_pickle(train_data_file)

    # Initialize lists to store features and labels
    X_train = []
    y_train = []

    # Iterate through each trace file
    for trace_key, trace_data in train_data.items():
        # Extract relevant data from trace_data
        # Modify this part according to the structure of your trace data
        # states = trace_data['states']
        states = trace_data['X_pca']
        class_labels = trace_data['labels']

        # Append features and labels to X_train and y_train
        X_train.extend(states)  # Modify if using observations as features
        y_train.extend(class_labels)

    # Initialize and train the classifier (Random Forest as an example)
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train, y_train)

    # Optionally, evaluate the classifier on the training data
    train_predictions = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("Training Accuracy:", train_accuracy)

    test_data = load_pickle(test_data_file)

    # Initialize lists to store features and labels
    y_truth = []
    y_pred = []

    # Iterate through each trace file
    for trace_key, trace_data in train_data.items():
        # Extract relevant data from trace_data
        # Modify this part according to the structure of your trace data
        # states = trace_data['states']
        states = trace_data['X_pca']
        class_labels = trace_data['labels']

        test_pred = classifier.predict(states)
        print("dataset: ", trace_key, " -->ground truth", np.mean(class_labels), "predicted", np.mean(test_pred))
        y_truth.append(int(class_labels[0]))
        y_pred.append(int(np.round(np.mean(test_pred), 0)))

    print(y_truth)
    print(y_pred)
    conf_mat = metrics.confusion_matrix(y_truth, y_pred)
    print(conf_mat)
