import dqrobotics as dql
import numpy as np


class TrajTranslationEllipse:

    def __init__(self, duration, sampling_time, normal: dql.DQ, center: dql.DQ, radius_a: float,
                 radius_b: float, start_vector: dql.DQ):
        self.duration = duration
        self.sampling_time = sampling_time
        assert dql.is_pure_quaternion(normal), "Normal must be a pure quaternion"
        self.normal = normal.normalize()
        assert dql.is_pure_quaternion(center), "Center must be a pure quaternion"
        self.center = center
        assert radius_a > 0 and radius_b > 0, "Radius must be positive"
        self.radius_a = radius_a
        self.radius_b = radius_b
        assert dql.is_pure_quaternion(start_vector), "Start vector must be a pure quaternion"
        start_vector = start_vector.normalize()

        self.angular_v_ = (2*np.pi) / duration

        # get the UV vector
        self.traj_v_vec_ = dql.cross(self.normal, start_vector).normalize() * self.radius_a
        self.traj_u_vec_ = dql.cross(self.traj_v_vec_, self.normal).normalize() * self.radius_b
        # circle setpoint = center + u_vec * cos(theta) + v_vec * sin(theta)

    def get_setpoint(self, time):
        u_dot = self.angular_v_ * self.traj_u_vec_ * -np.sin(time * self.angular_v_)
        v_dot = self.angular_v_ * self.traj_v_vec_ * np.cos(time * self.angular_v_)

        u = self.traj_u_vec_ * np.cos(time * self.angular_v_)
        v = self.traj_v_vec_ * np.sin(time * self.angular_v_)
        return self.center + u + v, u_dot + v_dot
