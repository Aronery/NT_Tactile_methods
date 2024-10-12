import cv2
import numpy as np

# 读取图片
image = cv2.imread("C:/Users/18142/Desktop/raw_data/picture/28.jpeg")
# Image Height: 480
# Image Width: 640
# Number of Channels: 3
#height, width, channels = image.shape
# print("Image Height:", height)
# print("Image Width:", width)
# print("Number of Channels:", channels)
# new_height = 10
# new_width = 10
target_resolution = (40, 40)
#downsampled_image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_AREA)

# 保存下采样后的图像
#cv2.imwrite('downsampled_image_6666.png', downsampled_image)

def downsampled_image(root_image):
    """
    :param root_image:
    下采样HR图片,以及转为灰度图
    """
    image = cv2.imread(root_image)
    downsampled_image = cv2.resize(image, target_resolution, interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(downsampled_image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("downsampled_image_40x40_"+str(name)+".png", gray_image)
    return gray_image

if __name__ == "__main__":
    root="C:/User/18142/Desktop/raw_data/picture/28.jpeg"
    gray_image=downsampled_image(root)
    cv2.imwrite("downsampled_image",gray_image)
    #pass