import cv2
import numpy as np
import math
import os
import json

'''
需要修改的参数：
1. 棋盘格每个格子的大小
2. 棋盘格原点的机床坐标系坐标
3. 棋盘格中的每一个点的机床坐标系是否标注正确（加减看好）
4. 读取棋盘格照片
'''

def load_para(json_path):
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    para = json.load(json_file)
    return para


para = load_para("camera_intr_opt.json")

mtx, dist = np.array(para["mtx"]), np.array(para["dist"])

# 棋盘格参数
square_size = 24.375  # 每格大小，单位 mm
pattern_size = (9, 6)  # 棋盘格角点数目（列数, 行数）

# 棋盘格原点的机床坐标系坐标
origin_x = -387.884
origin_y = -340.868

# 生成新的 objp
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)

# 遍历棋盘格中的每一个点
for i in range(pattern_size[1]):      # y方向（行）
    for j in range(pattern_size[0]):  # x方向（列）
        index = i * pattern_size[0] + j
        objp[index, 0] = origin_x - j * square_size  # x坐标
        objp[index, 1] = origin_y + i * square_size  # y坐标
        objp[index, 2] = 0                           # z坐标

obj_points = objp

# 读取棋盘格照片
img = cv2.imread("/home/knd/铣床实验照片/2025_04_25/2025_04_25_13h_35m_29s.jpeg")
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

    print(rvec)
    print(tvec)

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
                (10,  img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 保存处理后的图像
    cv2.imwrite("chessboard_result.jpg", img)

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
