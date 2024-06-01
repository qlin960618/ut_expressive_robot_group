import time
from enum import Enum

import dqrobotics as dql
from dqrobotics.interfaces.vrep import DQ_VrepInterface
import numpy as np
from scipy.linalg import block_diag
import math
import os
import logging
from dh_vrep_robot import Robot_VrepInterface

from dqrobotics.solvers import DQ_QuadprogSolver
from dqrobotics.robot_modeling import DQ_Kinematics
from tools.ui_interface import ControlWindowLauncher

import traceback

# logger init
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    "control_gain": 10.0,
    "effector_t": [0, 0, 0.1],
    "effector_r": [1, 0, 0, 0],
    "null_space_gain": 10.0,

    "enable_exploration": True,


    #------------------- Select one only -------------------
    # line to line angle
    "null_sapce_geometry": "line_angle",
    "null_space_entity_target": "VFI_static_sphere",
    "null_space_entity_source": "panda_null_source_line",
    "null_space_entity_joint": 4,
    "null_space_range": [-3, 3],
    #--------------------------------------
    # joint
    # "null_sapce_geometry": "joint",
    # "null_space_entity_target" : None,
    # "null_space_range": [-10, 10],


}


class KeepingType(Enum):
    TRANSLATION = 1
    ROTATION = 2
    FULL = 3


def apply_secondary_maintaince_equality(ua_primary, Jx, Jr, Jt, keeping_types):
    # setup equality constraints for task maintenance
    W_eq_keeping = np.zeros([0, Jx.shape[1]])
    w_eq_keeping = np.zeros([0])
    if KeepingType.TRANSLATION in keeping_types:
        W_eq_keeping = np.vstack([W_eq_keeping, Jt])
        w_eq_keeping = np.hstack([w_eq_keeping, Jt @ ua_primary])
    if KeepingType.ROTATION in keeping_types:
        W_eq_keeping = np.vstack([W_eq_keeping, Jr])
        w_eq_keeping = np.hstack([w_eq_keeping, Jr @ ua_primary])
    if KeepingType.FULL in keeping_types:
        W_eq_keeping = np.vstack([W_eq_keeping, Jx])
        w_eq_keeping = np.hstack([w_eq_keeping, Jx @ ua_primary])
    return W_eq_keeping, w_eq_keeping


def pose_to_entty_dq(pose, geometry):
    if geometry in ["line", "line_angle"]:
        line_l = dql.Ad(dql.rotation(pose), dql.k_)
        line_l_dq = line_l + dql.E_ * (dql.cross(dql.translation(pose), line_l))
        return line_l_dq
    elif geometry == "joint":
        return pose
    else:
        raise ValueError(f"null space geometry: {geometry} not supported")


def get_null_space_dimensions(geometry):
    if geometry == "line":
        nullspace_dimensions = 8
    elif geometry == "line_angle":
        nullspace_dimensions = 1
    elif geometry == "joint":
        nullspace_dimensions = 7
    else:
        raise ValueError(f"null space geometry: {geometry} not supported")
    return nullspace_dimensions


def get_null_space_jacobian_runtime(robot: DQ_Kinematics, q: np.ndarray, null_space_parameter: dict, null_target_x=None):
    dim_extension = robot.get_dim_configuration_space()-null_space_parameter["to_joint"]-1
    if null_space_parameter["geometry"] == "line":
        robot_x = robot.fkm(q, null_space_parameter["to_joint"])
        Jx = robot.pose_jacobian(q, null_space_parameter["to_joint"])
        Jx = np.hstack([Jx, np.zeros([8, dim_extension])])
        J_X_n = dql.haminus8(null_space_parameter["tf"]) @ Jx
        x_null = robot_x * null_space_parameter["tf"]
        line_l = dql.Ad(dql.rotation(x_null), dql.k_)
        line_l_dq = line_l + dql.E_ * (dql.cross(dql.translation(x_null), line_l))
        return DQ_Kinematics.line_jacobian(J_X_n, line_l_dq, dql.k_)
    elif null_space_parameter["geometry"] == "line_angle":
        robot_x = robot.fkm(q, null_space_parameter["to_joint"])
        Jx = robot.pose_jacobian(q, null_space_parameter["to_joint"])
        Jx = np.hstack([Jx, np.zeros([8, dim_extension])])
        J_X_n = dql.haminus8(null_space_parameter["tf"]) @ Jx
        x_null = robot_x * null_space_parameter["tf"]
        line_l = dql.Ad(dql.rotation(x_null), dql.k_)
        line_l_dq = line_l + dql.E_ * (dql.cross(dql.translation(x_null), line_l))
        target_l = dql.Ad(dql.rotation(null_target_x), dql.k_)
        target_l_dq = target_l + dql.E_ * (dql.cross(dql.translation(null_target_x), target_l))
        robot_Jline = DQ_Kinematics.line_jacobian(J_X_n, line_l_dq, dql.k_)
        return DQ_Kinematics.line_to_line_angle_jacobian(robot_Jline, line_l_dq, target_l_dq)

    elif null_space_parameter["geometry"] == "joint":
        return np.eye(robot.get_dim_configuration_space())
    else:
        raise ValueError("null space geometry not supported")


def get_null_sapce_entity_init(vrep_interface, robot, null_space_entity_name, null_space_entity_joint, geometry):
    if geometry == "joint":
        return {
            "geometry": geometry,
            "to_joint": robot.get_dim_configuration_space() - 1
        }
    entity_x = vrep_interface.get_object_pose(null_space_entity_name)
    joint_x = robot.fkm(robot.get_joint_positions(), null_space_entity_joint)
    tf = dql.conj(joint_x) * entity_x
    null_space_parameter = {
        "geometry": geometry,
        "tf": tf,
        "to_joint": null_space_entity_joint
    }

    return null_space_parameter


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
    nullspace_dimensions = get_null_space_dimensions(config["null_sapce_geometry"])
    ui = ControlWindowLauncher(nullspace_dimensions, [tuple(config['null_space_range'])] * nullspace_dimensions,
                               self_centering=0)

    robot_q_plus = robot.get_upper_q_limit()
    robot_q_minus = robot.get_lower_q_limit()

    vrep_interface.start_simulation()

    # joint_initial = np.array([0, 0, 0, -np.pi / 2.0, 0, np.pi / 2.0, 0])
    joint_initial = robot.get_joint_positions()

    # get runtime information for null space target parameter
    null_space_parameter = get_null_sapce_entity_init(vrep_interface, robot,
                                                      config.get("null_space_entity_target", None),
                                                      config.get("null_space_entity_joint", None),
                                                      config["null_sapce_geometry"])
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
            Jt = DQ_Kinematics.translation_jacobian(Jx, robot_x)
            Jr = DQ_Kinematics.rotation_jacobian(Jx)
            Nr = dql.haminus4(dql.rotation(xd)) @ dql.C4() @ Jr
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
            err_t1 = translation_error(dql.translation(robot_x), dql.translation(xd)).vec4()
            err_r1 = closest_invariant_rotation_error(dql.rotation(robot_x), dql.rotation(xd)).vec4()
            objb = config["control_objb"]
            alpha = config["control_alpha"]
            gain = config["control_gain"]

            H_objb = objb * np.eye(dims)
            H = 2 * (alpha * Jt.T @ Jt + (1.0 - alpha) * Nr.T @ Nr) + H_objb
            f = 2 * (alpha * err_t1.T @ Jt + (1.0 - alpha) * err_r1.T @ Nr) * gain

            W_ineq = np.vstack([W_joint_limit])
            w_ineq = np.hstack([w_joint_limit])
            W_eq = np.zeros([0, dims])
            w_eq = np.zeros([0])
            try:
                ua_ = solver.solve_quadratic_program(
                    H, f,
                    W_ineq, w_ineq, W_eq, w_eq,
                )

                if config["enable_exploration"]:
                    try:
                        u_null = np.array(ui.get_values())  # get nullspace values from UI

                        """
                        This method doesnt guarantee that VFI remain valid. need to use quadprog if vfi needed to be kept
                        """
                        # copy old constraints matrix
                        W_ineq_old, w_ineq_old, W_eq_old, w_eq_old = W_ineq.copy(), w_ineq.copy(), W_eq.copy(), w_eq.copy()
                        H_old = H
                        f_old = f

                        #########################################
                        if "null_space_entity_target" in config and config["null_space_entity_target"] is not None:
                            null_ref_x = vrep_interface.get_object_pose(config["null_space_entity_target"])
                        else:
                            null_ref_x = None

                        J_null = get_null_space_jacobian_runtime(robot, robot_q, null_space_parameter, null_ref_x)
                        # print(J_null)
                        # print(J_null.shape)
                        Px = (np.eye(dims) - np.linalg.pinv(Jx) @ Jx) @ J_null.T  # Nullspace projection matrix

                        # print(Px)
                        # Pt = np.eye(dims) - np.linalg.pinv(Jt) @ Jt
                        # Pr = np.eye(dims) - np.linalg.pinv(Nr) @ Nr

                        # print("Px", Px.shape, "J_null", J_null.shape, "u_null", u_null.shape)
                        gain_sec = config["null_space_gain"]

                        f = f_old + gain_sec * u_null @ Px.T
                        H = H_old + gain_sec * Px @ Px.T

                        # f = 2 * (alpha * (err_t1.T + Px @ J_null.T @ u_null) @ Jt + (1.0 - alpha) * err_r1.T @ Nr) * gain

                        # W_ineq_old, w_ineq_old, W_eq_old, w_eq_old

                        W_eq_keeping, w_eq_keeping = (
                            apply_secondary_maintaince_equality(ua_, Jx, Jr, Jt, [KeepingType.FULL]))
                        W_ineq = np.vstack([W_ineq_old])
                        w_ineq = np.hstack([w_ineq_old])
                        W_eq = np.vstack([W_eq_old, W_eq_keeping])
                        w_eq = np.hstack([w_eq_old, w_eq_keeping])
                        ua_ = solver.solve_quadratic_program(
                            H, f,
                            W_ineq, w_ineq, W_eq, w_eq,
                        )
                        # logger.info(f"Nullspace exploration: {ua_}")
                    except ValueError:
                        logger.warning("Nullspace exploration failed")
                        traceback.print_exc()
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
