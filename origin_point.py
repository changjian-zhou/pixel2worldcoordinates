import cv2

# 读取棋盘格照片
img = cv2.imread("/home/knd/铣床实验照片/2025_04_25/2025_04_25_13h_35m_29s.jpeg")
if img is None:
    print("无法读取图片，请检查文件路径！")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测棋盘格角点
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

x_origin, y_origin = corners[0][0]
print('相机世界坐标系原点的横坐标像素值：', x_origin)
print('相机世界坐标系原点的纵坐标像素值：', y_origin)
