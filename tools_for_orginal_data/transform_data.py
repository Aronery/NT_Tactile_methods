import numpy as np

# # 原始数据
# data = [0.112897751, 0.14431681, 0.656797946, -0.159469304, 0.021129921,
#         -0.018489427, 0.326147512, -0.410474868, 0.034586897, 0.005395367,
#         0.386022328, -0.159029512, 0.184995421, 0.491107296, 1.813853573, -0.478933726]
#
# # 指数变换
# transformed_data = [np.exp(x) for x in data]
#
# # 将数据映射到0到255之间的范围
# min_val = np.min(transformed_data)
# max_val = np.max(transformed_data)
# mapped_data = [(x - min_val) / (max_val - min_val) * 255 for x in transformed_data]
#
# # 打印映射后的数据
# for i, value in enumerate(mapped_data):
#     print("原始数据:", data[i], "\t映射后:", value)



import numpy as  np
import math
# 原始数据
# data = [0.112897751, 0.14431681, 0.656797946, -0.159469304, 0.021129921,
#         -0.018489427, 0.326147512, -0.410474868, 0.034586897, 0.005395367,
#         0.386022328, -0.159029512, 0.184995421, 0.491107296, 1.813853573, -0.478933726]
#
# # 将数据移动到非负范围
# min_val = min(data)
# shifted_data = [x - min_val for x in data]
#
# # 应用幂律变换
# gamma = 1/math.e  # 幂律参数，可以调整
# transformed_data = [x ** gamma for x in shifted_data]
#
# # 将数据映射到0到255之间的范围
# min_val = np.min(transformed_data)
# max_val = np.max(transformed_data)
# mapped_data = [(x - min_val) / (max_val - min_val) * 255 for x in transformed_data]
#
# # 打印映射后的数据
# print(mapped_data)
import numpy as np

# 原始数据
data = np.array([[-0.010561337129199998, 0.1773841084041, 0.10835137658065999, -0.1558491678725],
                 [-0.041919381513, 1.5295558340151998, 0.5309871680089999, -2.0701220178637993],
                 [-0.018874013214499996, 0.08173584718139999, 0.08141037459624999, -0.11824232890669999],
                 [-0.016767680097800004, -0.0040747637888, -0.05258671653459999, -0.044512991630000005]])

# 将数据移动到非负范围
min_val = np.min(data)
shifted_data = data - min_val

# 应用指数变换
base = 2  # 指数的基数，可以调整
transformed_data = np.exp(shifted_data * base)

# 将数据映射到0到255之间的范围
min_val = np.min(transformed_data)
max_val = np.max(transformed_data)
mapped_data = (transformed_data - min_val) / (max_val - min_val) * 255

# 打印映射后的数据
print("映射后的数据：\n", mapped_data.astype(int))


