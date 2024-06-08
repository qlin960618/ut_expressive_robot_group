import time
from enum import Enum

import dqrobotics as dql
from dqrobotics.interfaces.vrep import DQ_VrepInterface
import numpy as np
from numpy.linalg import pinv

from scipy.linalg import block_diag
import math
import os
import logging
from dh_vrep_robot import Robot_VrepInterface
from tools.trajectory_t_ellipse import TrajTranslationEllipse

from dqrobotics.solvers import DQ_QuadprogSolver
from dqrobotics.robot_modeling import DQ_Kinematics
from tools.ui_interface import ControlWindowLauncher

import traceback

# logger init
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ControlObjective(Enum):
    TRANSLATION = 0
    ROTATION = 1
    POSE = 2


JSON_PATH = os.path.join(os.path.dirname(__file__), "dh_vrep_robot/franka_emika_panda_robot.json")
VREP_ROBOT_NAME = "Franka_"

CONFIG = {
    # "vrep_ip": "10.198.113.186",
    # "vrep_ip": "192.168.10.103",
    "vrep_ip": "127.0.0.1",
    "vrep_port": 20000,

    "robot_name": VREP_ROBOT_NAME,

    "x_name": "x1",
    "xd_name": "xd1",

    "control_tau": 0.01,
    "control_alpha": 0.99,
    "control_objb": 0.0001,
    "control_gain": 1.0,
    "effector_t": [0, 0, 0.1],
    "effector_r": [1, 0, 0, 0],
    "null_space_gain": 1.0,

    "ui_slider_range": [-3, 3],
    "ui_num_slider": 7,

    "control_objective": ControlObjective.ROTATION,

}


def invariant_pose_error(x, xd):
    ex_1_minus_norm = np.linalg.norm(dql.vec8(dql.conj(x) * xd - 1))
    ex_1_plus_norm = np.linalg.norm(dql.vec8(dql.conj(x) * xd + 1))
    if ex_1_plus_norm < ex_1_minus_norm:
        ex = dql.conj(x) * xd + 1
    else:
        ex = dql.conj(x) * xd - 1
    return ex


def closest_invariant_rotation_error(r, rd):
    ex_1_minus_norm = np.linalg.norm(dql.vec4(dql.conj(r) * rd - 1))
    ex_1_plus_norm = np.linalg.norm(dql.vec4(dql.conj(r) * rd + 1))
    if ex_1_plus_norm < ex_1_minus_norm:
        ex = dql.conj(r) * rd + 1
    else:
        ex = dql.conj(r) * rd - 1
    return ex


def translation_error(t, td):
    return t - td


def get_objective_jacobian(Jx, x: dql.DQ, xd: dql.DQ, control_objective: ControlObjective):
    if control_objective is ControlObjective.POSE:
        return dql.haminus8(xd) @ dql.C8() @ Jx
    elif control_objective is ControlObjective.TRANSLATION:
        return DQ_Kinematics.translation_jacobian(Jx, x)
    elif control_objective is ControlObjective.ROTATION:
        return dql.haminus4(dql.rotation(xd)) @ dql.C4() @ DQ_Kinematics.rotation_jacobian(Jx)
    else:
        raise ValueError("Control objective not supported")


def get_objective_error_vec(x: dql.DQ, xd: dql.DQ, control_objective: ControlObjective):
    if control_objective is ControlObjective.POSE:
        return -invariant_pose_error(x, xd).vec8()
    elif control_objective is ControlObjective.TRANSLATION:
        return -translation_error(dql.translation(x), dql.translation(xd)).vec4()
    elif control_objective is ControlObjective.ROTATION:
        return -closest_invariant_rotation_error(dql.rotation(x), dql.rotation(xd)).vec4()
    else:
        raise ValueError("Control objective not supported")


def pose_to_entty_dq(pose, geometry):
    if geometry in ["line", "line_angle"]:
        line_l = dql.Ad(dql.rotation(pose), dql.k_)
        line_l_dq = line_l + dql.E_ * (dql.cross(dql.translation(pose), line_l))
        return line_l_dq
    elif geometry == "joint":
        return pose
    else:
        raise ValueError(f"null space geometry: {geometry} not supported")


def get_variable_boundaries_inequality(q, q_plus, q_minus):
    n_dims = len(q)
    W = np.vstack([-np.eye(n_dims), np.eye(n_dims)])
    w = np.concatenate([-(q_minus - q), (q_plus - q)])

    return W, w


def main(config):
    robot = Robot_VrepInterface(JSON_PATH, config["robot_name"])

    vrep_interface = DQ_VrepInterface()
    vrep_interface.connect(config["vrep_ip"], config["vrep_port"], 100, 10)

    robot.set_vrep_interface(vrep_interface)

    robot.apply_vrep_reference_frame()

    robot.set_x_and_xd_name(config["x_name"], config["xd_name"])

    # robot.apply_effector_from_current_x()
    effector = dql.DQ(config["effector_r"]) + 0.5 * dql.E_ * dql.DQ(config["effector_t"]) * dql.DQ(config["effector_r"])
    robot.set_effector(effector)
    logger.info("robot_effector: " + str(robot.get_effector()))

    logger.info("Setting up UI")
    ui = ControlWindowLauncher(config["ui_num_slider"], [tuple(config['ui_slider_range'])] * config["ui_num_slider"],
                               self_centering=0)

    robot_q_plus = robot.get_upper_q_limit()
    robot_q_minus = robot.get_lower_q_limit()

    #########################################
    # setup trajectory
    #########################################
    traj = TrajTranslationEllipse(
        duration=10,  # duration of the trajectory
        sampling_time=config["control_tau"],  # sampling time of the trajectory
        center=dql.DQ([0, 0, 1]),  # center of the ellipse
        normal=dql.DQ([0, 0, 1]),  # normal of the ellipse
        radius_a= 0.5,  # radius along the major axis
        radius_b= 0.1,  # radius along the minor axis
        start_vector=dql.DQ([0, 1, 0]),  # start vector of the trajectory
    )

    vrep_interface.start_simulation()

    # joint_initial = np.array([0, 0, 0, -np.pi / 2.0, 0, np.pi / 2.0, 0])
    joint_initial = robot.get_joint_positions()

    def update_robot(q):
        robot.send_joint_positions(q)
        robot_x = robot.fkm(q)
        robot.send_x_pose(robot_x)

    try:
        time.sleep(1)
        robot.send_joint_positions(joint_initial)
        x_init = robot.fkm(joint_initial)
        time.sleep(1)
        robot.send_xd_pose(x_init)
        dims = robot.get_dim_configuration_space()
        solver = DQ_QuadprogSolver()
        solver.set_equality_constraints_tolerance(dql.DQ_threshold)
        robot_q = robot.get_joint_positions()

        while ui.is_alive():
            xd = robot.get_xd_pose()
            robot_x = robot.fkm(robot_q)

            Jx = robot.pose_jacobian(robot_q)

            # Ws_limit = []
            # ws_limit = []
            # for i, (robot, q, q_plus, q_minus) in enumerate(
            #         zip([robot], [robot_q], [robot_q_plus], [robot_q_minus])
            # ):
            #     W, w = get_variable_boundaries_inequality(q, q_plus, q_minus)
            #     Ws_limit.append(W)
            #     ws_limit.append(w)
            # W_joint_limit = block_diag(*Ws_limit)
            # w_joint_limit = np.concatenate(ws_limit)

            #########################################
            # Objective
            #########################################
            J_obj = get_objective_jacobian(Jx, robot_x, xd, config['control_objective'])

            err_vec = get_objective_error_vec(robot_x, xd, config['control_objective'])

            u_null = np.array(ui.get_values())  # get nullspace values from UI

            try:
                ua_ = (pinv(J_obj) @ (config["control_gain"] * err_vec)
                       + config["null_space_gain"] * (np.eye(dims) - pinv(J_obj) @ J_obj) @ u_null)
                robot_q = robot_q + ua_ * config["control_tau"]
            except Exception as e:
                print(e)
                traceback.print_exc()

            update_robot(robot_q)

            time.sleep(config['control_tau'])


    except KeyboardInterrupt:
        logger.warning("exit on KeyboardInterrupt")
    except Exception as e:
        print(e)
        vrep_interface.stop_simulation()
        vrep_interface.disconnect()
        raise e

    # vrep_interface.stop_simulation()
    vrep_interface.disconnect_all()
    logger.info("simulation stopped and disconnected")


if __name__ == "__main__":
    main(CONFIG)
