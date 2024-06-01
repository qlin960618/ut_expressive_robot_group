from .DHRobot import DHRobot
from dqrobotics.interfaces.vrep import DQ_VrepInterface
from dqrobotics.robot_modeling import DQ_Kinematics
import dqrobotics as dql


class Robot_VrepInterface(DHRobot):
    def __init__(self, json_path, vrep_robot_name):
        super().__init__(json_path)
        self.vrep_name_prefix, self.vrep_name_suffix = vrep_robot_name.split('_')
        # self.vrep_name_suffix = self.vrep_name_suffix
        # self.vrep_name_prefix = self.vrep_name_prefix + "_"

        self.vrep_robot_ref_name = None
        self.vrep_robot_joint_names = None
        self.vrep_robot_ref_frame_dq = None
        self.vrep_x_name = None
        self.vrep_xd_name = None
        self.vrep_interface = None

        self._process_vrep_robot_obj_name()

    def _process_vrep_robot_obj_name(self):
        # print(base_name)
        n_dims = self.get_dim_configuration_space()
        self.vrep_robot_ref_name = self.vrep_name_prefix + "_reference_frame" + self.vrep_name_suffix
        self.vrep_robot_joint_names = []
        for i in range(n_dims):
            self.vrep_robot_joint_names.append(
                self.vrep_name_prefix + "_joint{:d}".format(i + 1) + self.vrep_name_suffix)

    def set_vrep_interface(self, vrep_interface: DQ_VrepInterface):
        assert isinstance(vrep_interface, DQ_VrepInterface), "vrep_interface is not a DQ_VrepInterface"
        self.vrep_interface = vrep_interface

    ##################################################
    # Wrapped call to robot modeling
    ##################################################

    ##################################################
    # initialization related
    ##################################################
    def apply_vrep_reference_frame(self):
        assert self.vrep_interface is not None, "vrep_interface is not yet set"
        self.vrep_robot_ref_frame_dq = self.vrep_interface.get_object_pose(self.vrep_robot_ref_name)
        self.set_reference_frame(self.vrep_robot_ref_frame_dq)

    def set_x_and_xd_name(self, x_name, xd_name):
        self.vrep_x_name = x_name
        self.vrep_xd_name = xd_name

    def get_reference_frame(self):
        assert self.vrep_interface is not None, "vrep_interface is not yet set"
        return self.vrep_interface.get_object_pose(self.vrep_robot_ref_name)

    def apply_effector_from_current_x(self):
        assert self.vrep_interface is not None, "vrep_interface is not yet set"
        assert self.vrep_x_name is not None, "x_name is not set"
        x_ = self.vrep_interface.get_object_pose(self.vrep_x_name)
        current_ee = self.get_effector()
        fkm_x = self.fkm()
        ee = dql.conj(fkm_x * dql.conj(current_ee)) * x_
        return self.set_effector(ee)

    def get_offset_from_effector(self, vrep_name):
        assert self.vrep_interface is not None, "vrep_interface is not yet set"
        x_ = self.vrep_interface.get_object_pose(vrep_name)
        fkm_x = self.fkm()
        tf = dql.conj(fkm_x) * x_
        return tf

    ##################################################
    # robot modeling related
    ##################################################
    def fkm(self, qs=None, to_nth_joint=None):
        if qs is None:
            qs = self.get_joint_positions()
        if to_nth_joint is None:
            return super().fkm(qs)
        else:
            return super().fkm(qs, to_nth_joint)

    ##################################################
    # quick tools
    ##################################################
    def send_xd_pose(self, xd_):
        assert self.vrep_interface is not None, "vrep_interface is not yet set"
        assert self.vrep_xd_name is not None, "xd_name is not set"
        return self.vrep_interface.set_object_pose(self.vrep_xd_name, xd_)

    def send_x_pose(self, x_):
        assert self.vrep_interface is not None, "vrep_interface is not yet set"
        assert self.vrep_x_name is not None, "x_name is not set"
        return self.vrep_interface.set_object_pose(self.vrep_x_name, x_)

    def get_xd_pose(self):
        assert self.vrep_interface is not None, "vrep_interface is not yet set"
        assert self.vrep_xd_name is not None, "xd_name is not set"
        return self.vrep_interface.get_object_pose(self.vrep_xd_name)

    def _check_sim_running(self):
        if self.vrep_interface is None:
            raise RuntimeError("vrep_interface is not yet set")
        assert self.vrep_interface.is_simulation_running(), "WARNING: Trying to set joints when simulation is not running"

    ##################################################
    # Overloading function for Robot Driver Interface
    ##################################################
    def get_joint_positions(self):
        assert self.vrep_interface is not None, "vrep_interface is not yet set"
        assert self.vrep_robot_joint_names is not None
        return self.vrep_interface.get_joint_positions(self.vrep_robot_joint_names)

    def send_joint_positions(self, qs, force=False, update_pose=False):
        assert self.vrep_robot_joint_names is not None
        if not force:
            self._check_sim_running()
        ret = self.vrep_interface.set_joint_target_positions(self.vrep_robot_joint_names, qs)

        # if self.vrep_x_name is not None and update_pose:
        #     x_ = self.robot_model.fkm(qs)
        #     ret = ret and super().set_object_pose(self.vrep_x_name, x_)
        return ret
