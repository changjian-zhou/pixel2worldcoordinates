import cv2
import numpy as np
import math
import os
import json


def load_para(json_path):
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    para = json.load(json_file)
    return para


para = load_para("camera_intr_opt.json")

mtx, dist = np.array(para["mtx"]), np.array(para["dist"])

# 3D 真实世界坐标系的点（单位 mm, 确保单位一致）
square_size = 24.375  # 24.375mm 每格
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size  # 计算棋盘格点的 3D 坐标
obj_points = objp  # 存储3D点

# 读取棋盘格照片
img = cv2.imread("./test.jpg")
if img is None:
    print("无法读取图片，请检查文件路径！")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测棋盘格角点
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

if ret:
    img_points = np.array(corners)

    # 在图片上绘制棋盘格角点
    cv2.drawChessboardCorners(img, (9, 6), corners, ret)

    # 解算位姿（求解旋转向量和平移向量）
    _, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)

    # 计算距离（单位 cm）
    distance = math.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2) / 10.0

    # 旋转向量 -> 旋转矩阵
    rvec_matrix = cv2.Rodrigues(rvec)[0]

    # 组合投影矩阵
    proj_matrix = np.hstack((rvec_matrix, tvec))

    # 计算欧拉角
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]

    # 在图像上绘制距离 & 角度信息
    cv2.putText(img, "dist: %.2fcm, yaw: %.2f, pitch: %.2f, roll: %.2f" % (distance, yaw, pitch, roll),
                (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Chessboard Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    RT_dict = {"R": rvec_matrix.tolist(), "T": tvec.tolist()}
    RT = json.dumps(RT_dict, indent=4)
    with open('RT.json', 'w') as json_file:
        json_file.write(RT)
else:
    print("未检测到棋盘格，请检查图片内容！")
