import matplotlib.pyplot as plt

# 给定的数据
epochs = [334, 667, 1000, 1334, 1667, 2000, 2334, 2667, 3000, 3334, 3667]
psnrs = [60.357, 189.92, 197.02, 225.44, 285.28, 369.37, 385.14, 361.71, 314.87, 396.23, 367.52]

# 将PSNR值从科学计数法转换为浮点数
psnrs = [float(f"{psnr:.2f}") for psnr in psnrs]

# 绘制折线图
plt.figure(figsize=(10, 5))  # 设置图表大小
plt.plot(epochs, psnrs, marker='o')  # 绘制折线图，使用圆点标记数据点

# 添加标题和轴标签
plt.title('SR3模型训练时，PSNR变化图')
plt.xlabel('Epochs')
plt.ylabel('PSNR*0.1  /dB')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()