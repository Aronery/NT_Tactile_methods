from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 图片路径
lr="H:/python_project/NT-SR3/experiments/sr_ffhq_240516_173805/results/3667/110000_1_lr.png"
hr="H:/python_project/NT-SR3/experiments/sr_ffhq_240516_173805/results/3667/110000_1_hr.png"
sr="H:/python_project/NT-SR3/experiments/sr_ffhq_240516_173805/results/3667/110000_1_sr.png"


image_paths = [lr, hr, sr]

# 文本标识
labels = ['4x4', 'ground true', 'S R']

# 加载字体
font = ImageFont.truetype('arial.ttf', 16)

# 读取图片并添加标签
images_with_labels = []
for i, image_path in enumerate(image_paths):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    label = labels[i]
    text_width, text_height = draw.textsize(label, font)

    # 计算文本位置，这里放在图片上方居中
    text_x = (img.width - text_width) // 2
    text_y = -(text_height // 2)

    # 添加文本
    draw.text((text_x, text_y), label, (255, 0, 0), font=font)

    # 保存带有标签的图片，如果你需要的话
    # img.save(f'{label}.jpg')

    images_with_labels.append(img)

# 确定组合图片的尺寸
max_width = max(img.width for img in images_with_labels)
total_height = sum(img.height for img in images_with_labels)

# 组合图片
new_im = Image.new('RGB', (max_width, total_height))

x_offset = 0
y_offset = 0
for img in images_with_labels:
    new_im.paste(img, (0, y_offset))
    y_offset += img.height

# 显示图片
plt.imshow(new_im)
plt.axis('off')  # 不显示坐标轴
plt.show()

# 保存组合后的图片
new_im.save('combined_images.jpg')