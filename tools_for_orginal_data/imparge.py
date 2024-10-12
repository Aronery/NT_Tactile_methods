import cv2
import numpy as np

# 假设 image 是一个 4x4 的图像，这里我们创建一个示例图像

#image = cv2.imread("H:/python_project/NT-SR3/dataset/celebahq_4_40/lr_4/152.png")

image = cv2.imread("H:/python_project/NT-SRGAN and NT-SRCNN/utility/shou_1.png")
image = image.astype(np.uint8)  # 确保图像数据类型正确
# 使用 OpenCV 的 resize 函数进行双线性插值
interpolated_image = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR)

# 显示原始图像和插值后的图像
#cv2.imshow('Original Image', image)
cv2.imshow('Interpolated Image', interpolated_image)
cv2.imwrite('path_to_save_image.jpg', interpolated_image)
# 等待按键后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()