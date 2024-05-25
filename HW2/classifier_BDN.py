import pickle


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


train_data_file = "data/train_data_pca.pkl"

if __name__ == "__main__":
    train = load_pickle(train_data_file)

    print(train)
