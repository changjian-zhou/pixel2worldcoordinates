import numpy as np
import os
import json


def load_para(json_path):
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    para = json.load(json_file)
    return para


RT = load_para('./RT.json')
R, T = RT['R'], RT['T']

para = load_para("camera_intr_opt.json")
mtx, dist = np.array(para["mtx"]), np.array(para["dist"])


def pixel_to_world(u, v, z_world, mtx, R, T):
    # 计算相机坐标 (归一化)
    pixel_coords = np.array([u, v, 1], dtype=np.float32).reshape(3, 1)
    Z_C = (z_world + (np.linalg.inv(R) @ T)[2]) / \
          (np.linalg.inv(R) @ np.linalg.inv(mtx) @ pixel_coords)[2]
    camera_coords = Z_C * np.linalg.inv(mtx) @ pixel_coords

    # 计算世界坐标
    world_coords = np.linalg.inv(R) @ (camera_coords - T)
    X, Y, Z = world_coords.flatten()

    return np.array([X, Y, Z])


def distance_between_pixels(u1, v1, z1, u2, v2, z2, mtx, R, T):
    world_p1 = pixel_to_world(u1, v1, z1, mtx, R, T)
    world_p2 = pixel_to_world(u2, v2, z2, mtx, R, T)

    # 计算两点之间的欧几里得距离
    distance = np.linalg.norm(world_p1 - world_p2)
    return distance


# 示例：两个像素点
u1, v1, z1 = 819.17, 1635, -25
u2, v2, z2 = 1837, 1635, -25

distance = distance_between_pixels(u1, v1, z1, u2, v2, z2, mtx, R, T)
print(f"像素坐标 ({u1}, {v1}, {z1}) 和 ({u2}, {v2}, {z2}) 之间的实际距离: {distance:.2f} mm")
