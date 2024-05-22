import pandas
import numpy as np
import os
import glob
from queue import Queue
import pickle
import random

def read_trc(file_path):
    pass


# JOE: Joy
# TRE: Tiredness
# NEE: Neutral

def get_entry_list(dir_name):
    file_paths = glob.glob(os.path.join(dir_name, "*.trc"))

    file_names = []
    class_labels = []
    participant_ids = []
    entry_ids = []
    for file in file_paths:
        file_name = os.path.basename(file)
        file_names.append(file_name)
        # participant_id , class_label, entry_id
        field = file_name.split(".")[0]
        id = int(field[-2:])
        field = field[:-2]
        class_ = field[-3:]
        field = field[:-3]
        participant_id = field
        participant_ids.append(participant_id)
        class_labels.append(class_)
        entry_ids.append(id)

    return {file_name: {"participant_id": participant_id, "class_label": class_, "entry_id": id_, "full_path": file}
            for file_name, participant_id, class_, id_, file in
            zip(file_names, participant_ids, class_labels, entry_ids, file_paths)}


def read_file(path):
    text_queue = Queue()
    with open(path, "r") as f:
        for line in f:
            text_queue.put(line)
    # check file type
    line = text_queue.get()
    # if line.find("PathFileType") != -1:
    #     fields = line.strip().split("\t")
    #     if not "Kaydara" == fields[-1]:
    #         print("File type is Kaydara, got ", fields)
    #     assert "Kaydara" == fields[-1], "File type is not Kaydara"

    meta_fields_txt = text_queue.get()
    meta_fields_data_txt = text_queue.get()
    meta_fields = meta_fields_txt.strip().split("\t")
    meta_fields_data = meta_fields_data_txt.strip().split("\t")
    meta_data = {field: data for field, data in zip(meta_fields, meta_fields_data)}
    for key, value in meta_data.items():
        try:
            meta_data[key] = int(value)
        except ValueError:
            pass

    # print(meta_data)

    data_header_txt = text_queue.get()
    data_header = data_header_txt.strip("\n").split("\t")
    # drop empty string
    data_header = [x for x in data_header if x]
    print(data_header)

    entry_header_txt = text_queue.get()
    entry_header = entry_header_txt.strip("\n").split("\t")

    # assert len(data_header) == meta_data["NumMarkers"] + 2, (f"Data header length is not correct, "
    #                                                          f"got {len(data_header)} expected "
    #                                                          f"{meta_data['NumMarkers'] + 2}")
    # print(entry_header)
    # assert len(entry_header) == meta_data["NumMarkers"] * 3 + 2, (f"Entry header length is not correct, "
    #                                                               f"got {len(entry_header)} expected "
    #                                                               f"{meta_data['NumMarkers'] * 3 + 2}")
    _ = text_queue.get()

    single_frame_index = 2
    # remove participant tag
    for i in range(meta_data["NumMarkers"]):
        data_header[i+single_frame_index] = data_header[i+single_frame_index].split("_")[1]

    data_raw = []

    while not text_queue.empty():
        line = text_queue.get()
        entries = line.strip("\n").split("\t")
        f_entries = []
        for x in entries:
            try:
                f_entries.append(float(x))
            except ValueError:
                pass
        data_raw.append(f_entries)

    data_raw = np.array(data_raw)

    data = {}
    for i in range(single_frame_index):
        data[data_header[i]] = data_raw[:, i]

    for i in range(meta_data["NumMarkers"]):
        data[data_header[i + single_frame_index]] = data_raw[:, i + single_frame_index:i + single_frame_index + 3]

    return data, meta_data


def zero_offset(data):
    for key, value in data.items():
        if key == "Frame#":
            continue
        if key == "Time":
            continue
        data[key] = value - value[0]
    return data


def append_first_order(data_, rate):  # velocity
    delta_t = 1.0 / rate
    keys = list(data_.keys())
    for key in keys:
        if key == "Frame#":
            continue
        if key == "Time":
            continue
        value = data_[key]
        diff = np.diff(value, axis=0)
        diff = np.concatenate([value[0].reshape(1, -1), diff], axis=0) / delta_t
        data_[key + "_1order"] = diff
    return data_


def append_second_order(data_, rate):  # acceleration
    delta_t = 1.0 / rate
    keys = list(data_.keys())
    for key in keys:
        if key == "Frame#":
            continue
        if key == "Time":
            continue
        if key.find("_1order") != -1:
            continue
        value = data_[key]
        diff = np.diff(value, axis=0)
        diff = np.concatenate([value[0].reshape(1, -1), diff], axis=0) / delta_t
        diff = np.diff(diff, axis=0)
        diff = np.concatenate([value[0].reshape(1, -1), diff], axis=0) / delta_t
        data_[key + "_2order"] = diff
    return data_


def convert_magnitude(data_):
    keys = list(data_.keys())
    for key in keys:
        if key == "Frame#":
            continue
        if key == "Time":
            continue
        value = data_[key]
        data_[key] = np.linalg.norm(value, axis=1)
    return data_

y_value_lookup = {
    "COE": 0,
    "TRE": 1,
    "NEE": 2,
    "JOE": 3
}
"""
TRE = sad
COE = angry
NEE = neutral
JOE = happy
"""


save_dir = "./data"

root_dir = "/Users/quentinlin/Nextcloud/TokyoUniversity/Doctorate/2024 S12/Expressive_robot_control/mocap_data/Emotions_Walk_College_de_France/MotionCaptureData trc"
if __name__ == "__main__":
    file_infos = get_entry_list(root_dir)
    datas = {}
    for file_key, file_info in file_infos.items():
        # print(file_key, file_info)
        data, metadata = read_file(file_info["full_path"])
        data = zero_offset(data)
        data = append_first_order(data, rate=metadata["DataRate"])
        data = append_second_order(data, rate=metadata["DataRate"])
        # convert to magnitude only
        # data = convert_magnitude(data)
        metadata.update(file_info)
        datas[file_key] = {"data": data, "metadata": metadata}
        datas[file_key]["labels"] = np.ones([data["Frame#"].shape[0]]) * y_value_lookup[file_info["class_label"]]
        # break

    # make train, test, split
    train_test_ratio = 0.8
    all_keys = list(datas.keys())
    all_keys = random.sample(all_keys, len(all_keys))
    train_keys = all_keys[:int(len(all_keys) * train_test_ratio)]
    test_keys = all_keys[int(len(all_keys) * train_test_ratio):]

    train_datas = {key: datas[key] for key in train_keys}
    test_datas = {key: datas[key] for key in test_keys}

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_datas, f)
    with open(os.path.join(save_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_datas, f)

