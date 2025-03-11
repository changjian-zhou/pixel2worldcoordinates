import cv2

# 读取棋盘格照片
img = cv2.imread("./test.jpg")

# 指定像素点坐标 (x, y)
pixel_point = (1086, 1279)
pixel_point1 = (1339, 1285)

# 在该像素点位置画两个半径为 2 的红色圆圈
cv2.circle(img, pixel_point, radius=2, color=(0, 0, 255), thickness=-1)
cv2.circle(img, pixel_point1, radius=2, color=(0, 0, 255), thickness=-1)

# 显示图像
cv2.imshow("Image with Circle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
