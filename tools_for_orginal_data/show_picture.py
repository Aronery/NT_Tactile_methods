




root="H:/python_project/tools for orginal data/dataset/picture_40x40/train/LR__8_rot_2.png"

import cv2

# 读取图像
image = cv2.imread(root)

# 设置新的图像尺寸
new_width = 40
new_height = 40

# 使用双线性插值缩放图像
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# 显示原图像和缩放后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image - Bilinear Interpolation', resized_image)
cv2.imwrite("pic.png", resized_image)
# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()