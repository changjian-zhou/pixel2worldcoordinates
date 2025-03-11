import cv2


# 鼠标点击回调函数
def get_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        print(f"坐标: ({x}, {y}) 像素值: {image[y, x]}")  # 注意 OpenCV 的索引是 (y, x)


# 读取图像
image = cv2.imread("test.jpg")

# 创建窗口并绑定鼠标事件
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", get_pixel)

while True:
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
        break

cv2.destroyAllWindows()
