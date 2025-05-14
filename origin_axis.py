import os
import cv2
import numpy as np
import json


def load_para(json_path):
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    para = json.load(json_file)
    return para


para = load_para("camera_intr_opt.json")

mtx, dist = np.array(para["mtx"]), np.array(para["dist"])

rvec, tvec = np.asarray([[-0.06637969], [-0.06594987], [-3.13641332]]), np.asarray(
    [[116.18857133], [ 55.05285571], [558.18590367]])

# 世界坐标原点
point_3d = np.array([[0, 0, 0]], dtype=np.float32)

# 重投影
point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, mtx, dist)
u, v = point_2d[0][0]
print(f"世界坐标 (0, 0, 0) 对应的图像坐标为: ({u:.2f}, {v:.2f})")

# 可视化
img1 = cv2.imread('/home/knd/铣床实验照片/2025_04_25/2025_04_25_13h_35m_29s.jpeg')
cv2.circle(img1, (int(u), int(v)), 10, (0, 0, 255), -1)
cv2.imwrite('origin_projection.jpg', img1)

# 重投影检测
# 坐标轴绘制：
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    print(tuple(imgpts[0].ravel()))
    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[0].ravel())), (255,0,0), 15)
    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[1].ravel())), (0,255,0), 15)
    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[2].ravel())), (0,0,255), 15)
    return img


# 投影坐标轴
axis = np.float32([[80, 0, 0], [0, 80, 0], [0, 0, -80]]).reshape(-1, 3)
imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
img2 = draw(img1, point_2d, imgpts)
cv2.imwrite('1_corners_2.jpg', img2)