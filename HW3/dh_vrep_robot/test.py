import numpy as np
import dqrobotics as dql
from dqrobotics.robot_modeling import DQ_Kinematics

src_Jprim = np.array([
    [0, 0, 0, 0, -0, 0],
    [1, 3.02121e-16, 3.02121e-16, 1.55374e-16, 3.40891e-17, 1.55374e-16],
    [-2.5783e-17, 4.37114e-08, 4.37114e-08, 9.92536e-18, 4.37114e-08, 9.92536e-18],
    [6.33348e-18, -1, -1, -3.94506e-18, -1, -3.94506e-18],
    [0, 0, 0, 0, 0, 0],
    [8.66844e-19, -1.50804e-08, -2.60083e-08, -2.06315e-18, 0.255, -4.0955e-18],
    [0.605, 1.64727e-16, 1.64727e-16, 9.40014e-17, 2.56782e-18, 9.40014e-17],
    [-5.55112e-17, 7.02032e-17, 8.24696e-17, -9.27307e-17, 4.51484e-17, -9.27307e-17],
])

# src_line_dq:  - 1j - 0k + E*(0.605i), tgt_line_dq:  - 1j - 0k + E*(0.605i + 0j - 0.275k)
src_line_dq = dql.DQ([0, 0, -1, 0, 0, 0.605, 0, 0])
tgt_line_dq = dql.DQ([0, 0, -1, 0, 0, 0.605, 0, -0.275])

print(DQ_Kinematics.line_to_line_distance_jacobian(src_Jprim,
                                                   src_line_dq, tgt_line_dq))
