import pandas
import numpy as np
import os
import glob


def read_trc(file_path):
    pass


def get_entry_list(dir_name):
    file_list = glob.glob(os.path.join(dir_name, "*.trc"))




root_dir = "/Users/quentinlin/Nextcloud/TokyoUniversity/Doctorate/2024 S12/Expressive_robot_control/mocap_data/Emotions_Walk_College_de_France/MotionCaptureData trc"
if __name__ == "__main__":
    file_path = get_entry_list(root_dir)

    print(file_path)
    # df = read_trc(file_path)
    # print(df.head())
