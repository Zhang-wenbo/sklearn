import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression   # 线性回归模型
from sklearn.preprocessing import PolynomialFeatures    # 构建多项式特征
from sklearn.model_selection import train_test_split    # 划分训练集和测试集
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

'''
1. 生成数据
2. 划分训练集和测试集（验证集）
3. 定义模型（线性回归模型）
4. 训练模型
5. 预测结果，计算误差
'''

plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False

# 1. 生成数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)

print(X.shape)
print(y.shape)

# 画出散点图（3个子图）
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(X, y, c='y')
ax[1].scatter(X, y, c='y')
ax[2].scatter(X, y, c='y')
# plt.show()

# 2. 划分训练集和测试集（验证集）
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义模型（线性回归模型）
model = LinearRegression()

# 一、欠拟合（直线）
x_train1 = trainX
x_test1 = testX

# 4. 训练模型
model.fit(x_train1, trainY)

# 打印查看模型参数
print(model.coef_)
print(model.intercept_)

# 5. 预测结果，计算误差
y_pred1 = model.predict(x_test1)
test_loss1 = mean_squared_error(testY, y_pred1)
train_loss1 = mean_squared_error(trainY, model.predict(x_train1))

# 画出拟合曲线，并写出训练误差和测试误差
ax[0].plot(X, model.predict(X), 'r')
ax[0].text(-3, 1, f"测试误差：{test_loss1:.4f}")
ax[0].text(-3, 1.3, f"训练误差：{train_loss1:.4f}")

# 二、恰好拟合（5次多项式）
poly5 = PolynomialFeatures(degree=5)
x_train2 = poly5.fit_transform(trainX)
x_test2 = poly5.fit_transform(testX)
print(x_train2.shape)
print(x_test2.shape)

# 4. 训练模型
model.fit(x_train2, trainY)
#
# # 打印查看模型参数
# print(model.coef_)
# print(model.intercept_)
#
# 5. 预测结果，计算误差
y_pred2 = model.predict(x_test2)
test_loss2 = mean_squared_error(testY, y_pred2)
train_loss2 = mean_squared_error(trainY, model.predict(x_train2))

# 画出拟合曲线，并写出训练误差和测试误差
ax[1].plot(X, model.predict(poly5.fit_transform(X)), 'r')
ax[1].text(-3, 1, f"测试误差：{test_loss2:.4f}")
ax[1].text(-3, 1.3, f"训练误差：{train_loss2:.4f}")

# 三、过拟合（20次多项式）
poly20 = PolynomialFeatures(degree=20)
x_train3 = poly20.fit_transform(trainX)
x_test3 = poly20.fit_transform(testX)
print(x_train3.shape)
print(x_test3.shape)

# 4. 训练模型
model.fit(x_train3, trainY)

# 5. 预测结果，计算误差
y_pred3 = model.predict(x_test3)
test_loss3 = mean_squared_error(testY, y_pred3)
train_loss3 = mean_squared_error(trainY, model.predict(x_train3))

# 画出拟合曲线，并写出训练误差和测试误差
ax[2].plot(X, model.predict(poly20.fit_transform(X)), 'r')
ax[2].text(-3, 1, f"测试误差：{test_loss3:.4f}")
ax[2].text(-3, 1.3, f"训练误差：{train_loss3:.4f}")

plt.show()