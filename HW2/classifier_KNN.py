import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Mapping of class labels to their corresponding indices
y_value_lookup = {
    "COE": 0,
    "TRE": 1,
    "NEE": 2,
    "JOE": 3
}

# List of class names
class_names = list(y_value_lookup.keys())

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        return data

train_data_file = "data/train_data_pca.pkl"
test_data_file = "data/test_data_pca.pkl"

if __name__ == "__main__":
    train_data = load_pickle(train_data_file)

    X_train = []
    y_train = []

    for trace_key, trace_data in train_data.items():
        states = trace_data['X_pca']
        class_labels = trace_data['labels']
        X_train.extend(states)
        y_train.extend(class_labels)

    classifier = KNeighborsClassifier(n_neighbors=4)
    classifier.fit(X_train, y_train)

    train_predictions = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("Training Accuracy:", train_accuracy)

    test_data = load_pickle(test_data_file)

    y_truth = []
    y_pred = []

    for trace_key, trace_data in test_data.items():
        states = trace_data['X_pca']
        class_labels = trace_data['labels']

        test_pred = classifier.predict(states)
        pred =  np.mean(test_pred[-60:])
        print("dataset: ", trace_key, " -->ground truth", np.mean(class_labels), "predicted",pred)
        y_truth.append(int(class_labels[0]))
        y_pred.append(int(pred))

    print(y_truth)
    print(y_pred)

    conf_mat = metrics.confusion_matrix(y_truth, y_pred)
    print(conf_mat)
    print(classification_report(y_truth, y_pred, target_names=class_names, digits=4))

    # Plotting the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
