import numpy as np
import math
import os
import logging

# logger init
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import dqrobotics as dql
from dqrobotics.interfaces.json11 import DQ_JsonReader
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH
import numpy as np

import json


class DHRobot(DQ_SerialManipulatorDH):
    def __init__(self, json_path=None):
        # Standard of VS050

        if json_path is None or not os.path.isfile(json_path):
            raise ValueError("robot.json not specified")

        try:
            with open(json_path) as j_f:
                jdata = json.load(j_f)

        except Exception as e:
            raise ValueError("DH loading file read error")

        reader = DQ_JsonReader()

        if jdata['robot_type'] == "DQ_SerialManipulatorDH":
            self.source_robot = reader.get_serial_manipulator_dh_from_json(json_path)
        elif jdata['robot_type'] == "DQ_SerialManipulatorDenso":
            self.source_robot = reader.get_serial_manipulator_denso_from_json(json_path)
        else:
            raise ValueError("json parameter type definition error: " + str(type))

        dh_matrix = np.vstack([self.source_robot.get_thetas(),
                               self.source_robot.get_ds(),
                               self.source_robot.get_as(),
                               self.source_robot.get_alphas(),
                               self.source_robot.get_types()])

        super().__init__(dh_matrix)
        self.set_lower_q_limit(self.source_robot.get_lower_q_limit())
        self.set_upper_q_limit(self.source_robot.get_upper_q_limit())
        self.set_lower_q_dot_limit(self.source_robot.get_lower_q_dot_limit())
        self.set_upper_q_dot_limit(self.source_robot.get_upper_q_dot_limit())

    # @staticmethod
    def source_kinematics(self):
        return self.source_robot


#############################for testing only################################
def main():
    jpath = "./denso_vs050_DH_test.json"
    # jpath="./denso_vs050_denso_11U483.json"
    robot = VS050Robot(jpath)
    logger.info(str(robot))
    logger.info(str(robot.fkm([0, 0, 0, 0, 0, 0])))


if __name__ == '__main__':
    main()
#############################for testing only################################
