import numpy as np
import cv2
import os
import json


def load_para(json_path):
    assert os.path.exists(json_path), json_path + " does not exist."
    with open(json_path, 'r') as json_file:
        para = json.load(json_file)
    return para


# 加载参数
RT = load_para('./RT.json')
R = np.array(RT['R'])
T = np.array(RT['T'])

para = load_para("camera_intr_opt.json")
mtx = np.array(para["mtx"])
dist = np.array(para["dist"])


def pixel_to_world_with_z(u, v, z_world, mtx, R, T):
    """带Z轴坐标的像素到世界坐标转换"""
    # 去畸变
    undistorted = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32), mtx, dist, None, mtx)
    u_undist, v_undist = undistorted[0][0]

    # 检查去畸变结果
    if not np.isfinite([u_undist, v_undist]).all():
        raise ValueError(f"去畸变结果无效: {u_undist}, {v_undist}")

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]

    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]
    tx, ty, tz = T

    # 线性方程组 AX = B
    A = np.array([
        [fx * r11 - (u_undist - cx) * r31, fx * r12 - (u_undist - cx) * r32],
        [fy * r21 - (v_undist - cy) * r31, fy * r22 - (v_undist - cy) * r32]
    ])
    B = np.array([
        (u_undist - cx) * (r33 * z_world + tz) - fx * (r13 * z_world + tx),
        (v_undist - cy) * (r33 * z_world + tz) - fy * (r23 * z_world + ty)
    ])

    # 确保 A 可解
    if np.linalg.det(A) == 0:
        raise ValueError("A 矩阵是奇异的，无法求解")

    X, Y = np.linalg.solve(A, B)

    return np.array([float(X), float(Y), float(z_world)])


def distance_between_pixels_with_z(u1, v1, z1, u2, v2, z2, mtx, R, T):
    world_p1 = pixel_to_world_with_z(u1, v1, z1, mtx, R, T)
    world_p2 = pixel_to_world_with_z(u2, v2, z2, mtx, R, T)
    return np.linalg.norm(world_p1 - world_p2)


# 测试示例
u1, v1, z1 = 819.17, 1635, -25
u2, v2, z2 = 1837, 1635, -25

distance = distance_between_pixels_with_z(u1, v1, z1, u2, v2, z2, mtx, R, T)
print(f"三维空间距离: {distance:.2f} mm")