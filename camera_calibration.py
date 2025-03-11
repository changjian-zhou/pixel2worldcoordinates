import cv2
import numpy as np
import glob
import json

# 参数设置
CHECKERBOARD = (9, 6)  # 内角点数量
square_size = 24.375  # 单位：mm（需实际测量）
images = glob.glob('calib_images/*.jpg')

# 生成三维点
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 存储标定数据
obj_points = []
img_points = []

# 处理所有图片
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        # 亚像素优化
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners_sub)

# 标定相机
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

# 计算重投影误差
mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print(f"平均重投影误差: {mean_error / len(obj_points):.4f} 像素")


def save_para(filename, matrix, dist):
    camera_para_dict = {"mtx": matrix, "dist": dist}
    temp = json.dumps(camera_para_dict, indent=4)
    with open(filename, 'w') as json_file:
        json_file.write(temp)


# 保存参数
save_para("camera_intr_opt.json", mtx.tolist(), dist.tolist())
