import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置随机种子以确保可重复性
np.random.seed(0)

# 生成固定的原始数据（仅生成一次，保持不变）
data_base = np.random.uniform(1, 10, (20, 2))
data_base[:, 1:] = 0.5 * data_base[:, 0:1] + np.random.uniform(-2, 2, (20, 1))

# 创建一个图形窗口
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# 设置坐标轴范围
size = 10
ax.set_xlim(-12.5, 12.5)
ax.set_ylim(-12.5, 12.5)
ax.set_title('Result of PCA')
ax.plot([-size, size], [0, 0], c='black')  # x轴
ax.plot([0, 0], [-size, size], c='black')  # y轴

# 初始化用于绘制的对象
scatter_orig, = ax.plot([], [], 'o', label='origin data')
scatter_recon, = ax.plot([], [], 'o', label='restructured data')
scatter_outlier, = ax.plot([], [], 'o', label='outsider')
line_eigen1, = ax.plot([], [], label='eigen vector 1')
line_eigen2, = ax.plot([], [], label='eigen vector 2')
lines_connect = []  # 用于存储原始数据和重构数据的连线

# 动画更新函数
def update(i):
    # 清空之前的绘图对象
    scatter_orig.set_data([], [])
    scatter_recon.set_data([], [])
    scatter_outlier.set_data([], [])
    line_eigen1.set_data([], [])
    line_eigen2.set_data([], [])
    for line in lines_connect:
        line.remove()
    lines_connect.clear()

    # 使用固定的原始数据
    data = data_base.copy()

    # 加入异常值
    angle = 2 * i / 360 * np.pi
    data_outsider = [np.sin(angle) * 5 * angle / (2 * np.pi), np.cos(angle) * 5 * angle / (2 * np.pi)]
    data = np.vstack((data, data_outsider))

    # 去中心化
    data_normal = data - data.mean(0)
    X = data_normal[:, 0]
    Y = data_normal[:, 1]

    # 协方差矩阵
    C = np.cov(data_normal.T)

    # 计算特征值和特征向量
    vals, vecs = np.linalg.eig(C)

    # 重新排序，从大到小
    vecs = vecs[:, np.argsort(-vals)]
    vals = vals[np.argsort(-vals)]

    # 数据在主成分1上的投影坐标
    zcf1 = np.matmul(data_normal, vecs[:, 0])

    # 只用主成分1重构数据
    data_ = np.matmul(zcf1.reshape(len(data), 1), vecs[:, 0].reshape(1, 2)) + data.mean(0)

    # 绘制散点
    scatter_orig.set_data(data[:, 0], data[:, 1])
    scatter_recon.set_data(data_[:, 0], data_[:, 1])
    scatter_outlier.set_data([data[-1, 0]], [data[-1, 1]])  # 修正：将标量包装为列表

    # 绘制两个主成分的方向
    ev1 = np.array([vecs[:, 0] * -1, vecs[:, 0]]) * size + data.mean(0)
    line_eigen1.set_data(ev1[:, 0], ev1[:, 1])

    ev2 = np.array([vecs[:, 1] * -1, vecs[:, 1]]) * size + data.mean(0)
    line_eigen2.set_data(ev2[:, 0], ev2[:, 1])

    # 绘制原始数据和重构数据的连线
    for j in range(len(data)):
        line, = ax.plot([data[j, 0], data_[j, 0]], [data[j, 1], data_[j, 1]], c='black', linewidth=0.5)
        lines_connect.append(line)

    return [scatter_orig, scatter_recon, scatter_outlier, line_eigen1, line_eigen2] + lines_connect

# 创建动画
ani = FuncAnimation(fig, update, frames=range(0, 7200, 10), interval=50, blit=True)

# 添加图例
ax.legend()

# 显示动画
plt.show()
