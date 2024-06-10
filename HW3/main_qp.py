import time
from enum import Enum
from typing import List

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
    "gaze_target_name": "gaze_target",

    "control_tau": 0.01,
    "control_gain": 5.0,
    "control_damping": 0.0001,
    "effector_t": [0, 0, 0.1],
    "effector_r": [1, 0, 0, 0],
    "null_space_gain": 1.0,
    "null_rotation_gain": 10.0,

    "ui_slider_range": [-1, 1],
    "ui_num_slider": 3,

    "control_objective": ControlObjective.TRANSLATION,

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


def apply_secondary_maintaince_equality(ua_primary, Jx, x, xd, keeping_types: List[ControlObjective]):
    # setup equality constraints for task maintenance
    W_eq_keeping = np.zeros([0, Jx.shape[1]])
    w_eq_keeping = np.zeros([0])
    if ControlObjective.TRANSLATION in keeping_types:
        Jt = get_objective_jacobian(Jx, x, xd, ControlObjective.TRANSLATION)
        W_eq_keeping = np.vstack([W_eq_keeping, Jt])
        w_eq_keeping = np.hstack([w_eq_keeping, Jt @ ua_primary])
    if ControlObjective.ROTATION in keeping_types:
        Nr = get_objective_jacobian(Jx, x, xd, ControlObjective.ROTATION)
        W_eq_keeping = np.vstack([W_eq_keeping, Nr])
        w_eq_keeping = np.hstack([w_eq_keeping, Nr @ ua_primary])
    if ControlObjective.POSE in keeping_types:
        Nx = get_objective_jacobian(Jx, x, xd, ControlObjective.POSE)
        W_eq_keeping = np.vstack([W_eq_keeping, Nx])
        w_eq_keeping = np.hstack([w_eq_keeping, Nx @ ua_primary])
    return W_eq_keeping, w_eq_keeping


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


def get_objective_feed_forward_vec(ff: dql.DQ, x: dql.DQ, control_objective: ControlObjective):
    if control_objective is ControlObjective.POSE:
        return dql.vec8(dql.conj(x) * ff)
    elif control_objective is ControlObjective.TRANSLATION:
        return -dql.vec4(ff)
    elif control_objective is ControlObjective.ROTATION:
        return dql.vec4(dql.conj(dql.rotation(x)) * ff)
    else:
        raise ValueError("Control objective not supported")


def get_variable_boundaries_inequality(q, q_plus, q_minus):
    n_dims = len(q)
    W = np.vstack([-np.eye(n_dims), np.eye(n_dims)])
    w = np.concatenate([-(q_minus - q), (q_plus - q)])

    return W, w


def get_gaze_rotation(robot_x, target_t):
    # calculate gaze rotation align z axis with robot t to target_t
    robot_t = dql.translation(robot_x)
    robot_r = dql.rotation(robot_x)
    gaze_vec = (target_t - robot_t).normalize().vec3()
    z_vec = dql.Ad(robot_r, dql.k_).vec3()
    rot_axis = np.cross(z_vec, gaze_vec)
    angle = np.arccos(np.dot(z_vec, gaze_vec) / (np.linalg.norm(z_vec) * np.linalg.norm(gaze_vec)))
    return (np.cos(angle / 2) + np.sin(angle / 2) * dql.DQ(rot_axis).normalize()) * robot_r


def PAD_to_JV_conversion(pad_val, robot_q, joint_limits, rd, t):  # dim: 1,7
    q_lower, q_upper = joint_limits
    hi = (q_upper - q_lower) / 2 * 0.5
    ceta_v0_i = (q_upper + q_lower) / 2
    ceta_0i = robot_q

    w = np.pi * np.array([1, 2, 3, 1, 2, 1, 2])
    wjs = [np.pi * np.array([1, 2, 3, 1, 2, 1, 2]), np.pi * np.array([1, 2, 3, 1, 2, 1, 2]) * 2]
    aj = [3, 2]
    bj = [6, 5]

    # P: Pleasure
    # A: Arousal
    # D: Dominance
    J = (1 - pad_val[0]) / 2
    V = (1 + pad_val[1]) / 2
    G = (1 + pad_val[2]) / 2

    # G
    w_gaze = 2 * np.pi / 5.0
    a_gaze = np.radians(60)
    gaze_ang = (1 - G) * a_gaze * np.sin(w_gaze * t)
    r_gaze = np.cos(gaze_ang / 2) + dql.j_ * np.sin(gaze_ang / 2)

    # P
    phi = J * np.sum([(aj[j_] * np.sin(wj * t) + bj[j_] * np.cos(wj * t)) for j_, wj in enumerate(wjs)])

    # V
    ceta_target = (1 - V) * ceta_0i + V * (ceta_v0_i + hi * np.sin(w * t + phi))

    return ceta_target - robot_q, r_gaze * rd


def main(config):
    robot = Robot_VrepInterface(JSON_PATH, config["robot_name"])
    solver = DQ_QuadprogSolver()

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
                               self_centering=None,
                               slider_labels=["Pleasure", "Arousal", "Dominance"])

    robot_q_plus = robot.get_upper_q_limit()
    robot_q_minus = robot.get_lower_q_limit()

    #########################################
    # setup trajectory
    #########################################
    traj = TrajTranslationEllipse(
        duration=20,  # duration of the trajectory
        sampling_time=config["control_tau"],  # sampling time of the trajectory
        center=dql.DQ([0, 0, 0.7]),  # center of the ellipse
        normal=dql.DQ([0, 0, 1]),  # normal of the ellipse
        radius_a=0.1,  # radius along the major axis
        radius_b=0.3,  # radius along the minor axis
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

        interation = 0

        while ui.is_alive():
            interation += 1
            # xd = robot.get_xd_pose()
            robot_x = robot.fkm(robot_q)
            td, td_dot = traj.get_setpoint(interation * config["control_tau"])

            target_t = vrep_interface.get_object_translation(config["gaze_target_name"])
            # calculate gaze rotation allinge z axis with robot t to target_t
            """
            def get_gaze_rotation(robot_x, target_t)\
            """
            rd_gaze = get_gaze_rotation(robot_x, target_t)

            xd = rd_gaze + 0.5 * dql.E_ * td * rd_gaze

            vrep_interface.set_object_translation(config["xd_name"], td)

            Jx = robot.pose_jacobian(robot_q)

            Ws_limit = []
            ws_limit = []
            for i, (robot, q, q_plus, q_minus) in enumerate(
                    zip([robot], [robot_q], [robot_q_plus], [robot_q_minus])
            ):
                W, w = get_variable_boundaries_inequality(q, q_plus, q_minus)
                Ws_limit.append(W)
                ws_limit.append(w)
            W_joint_limit = block_diag(*Ws_limit)
            w_joint_limit = np.concatenate(ws_limit)

            #########################################
            # Objective
            #########################################
            dims = robot.get_dim_configuration_space()
            J_obj = get_objective_jacobian(Jx, robot_x, xd, config['control_objective'])

            err_vec = get_objective_error_vec(robot_x, xd, config['control_objective'])

            ff_vec = get_objective_feed_forward_vec(td_dot, robot_x, config['control_objective'])

            objb = config["control_damping"] ** 2
            H_obj = 2 * (J_obj.T @ J_obj + objb * np.eye(dims))
            f_obj = 2 * (-err_vec * config["control_gain"] + ff_vec).T @ J_obj

            P_null = (np.eye(dims) - np.linalg.pinv(J_obj) @ J_obj)

            W_ineq = np.vstack([W_joint_limit])
            w_ineq = np.hstack([w_joint_limit])
            ua_primary = solver.solve_quadratic_program(
                H_obj,
                f_obj,
                W_ineq,
                w_ineq,
                None, None
            )

            pad_val = np.array(ui.get_values())  # get nullspace values from UI
            # def PAD_to_JV_conversion(pad_val, t) -> np.ndarray: # dim: 1,7
            #     # P: Pleasure
            #     # A: Arousal
            #     # D: Dominance
            #     # θmi t = 1−V θ0i + V[θV0i + hisin(ωt + φi)]
            #     pass
            #
            u_null, rd_gaze = PAD_to_JV_conversion(
                pad_val=pad_val,
                robot_q=robot_q,
                joint_limits=[robot_q_minus, robot_q_plus],
                rd=rd_gaze,
                t=interation * config["control_tau"],
            )

            J_rot = get_objective_jacobian(Jx, robot_x, rd_gaze, ControlObjective.ROTATION)
            err_rot_vec = get_objective_error_vec(robot_x, rd_gaze, ControlObjective.ROTATION)
            vrep_interface.set_object_rotation(config["xd_name"], rd_gaze)
            H_rot = 2 * P_null @ (J_rot.T @ J_rot) @ P_null.T
            f_rot = 2 * (-err_rot_vec * config["null_rotation_gain"]).T @ J_rot @ P_null

            P_null = P_null @ (np.eye(dims) - np.linalg.pinv(J_rot) @ J_rot)

            H_joint_null = 2 * P_null @ P_null.T
            f_joint_null = 2 * (config["null_space_gain"] * -u_null.T) @ P_null

            print(u_null)
            W_eq, w_eq = apply_secondary_maintaince_equality(ua_primary, Jx, robot_x, xd, [config['control_objective']])

            try:
                ua_ = solver.solve_quadratic_program(
                    H_obj+H_rot+H_joint_null,
                    f_obj+f_rot+f_joint_null,
                    W_ineq,
                    w_ineq,
                    W_eq,
                    w_eq
                )
            except Exception as e:
                print("Secondary control failed")
                ua_ = ua_primary
                traceback.print_exc()

            robot_q = robot_q + ua_ * config["control_tau"]

            update_robot(robot_q)

            time.sleep(config['control_tau'])

    except KeyboardInterrupt:
        logger.warning("exit on KeyboardInterrupt")
    except Exception as e:
        print(e)
        vrep_interface.disconnect_all()
        raise e

    # vrep_interface.stop_simulation()
    vrep_interface.disconnect_all()
    logger.info("simulation stopped and disconnected")


if __name__ == "__main__":
    main(CONFIG)
