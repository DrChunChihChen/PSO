import numpy as np
import random
import matplotlib.pyplot as plt
import math

# import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Data = pd.read_csv('kc_house_data_1.csv')

Data.head(5).T


# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


X = Data.drop('price',axis =1).values
y = Data['price'].values

#splitting Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#standardization scaler - fit&transform on train, fit only on test
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))


# having 19 nueron is based on the number of available featurs

model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=40)

model.summary()

loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(12,8))

y_pred = model.predict(X_test)


class PSO():
    def __init__(self, pN, dim, max_iter):
        #定义所需变量
        self.w = 0.8
        self.c1 = 2#学习因子
        self.c2 = 2

        self.r1 = 0.6#超参数
        self.r2 = 0.3

        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数

        #定义各个矩阵大小
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度矩阵
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置矩阵
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1  # 全局最佳适应值

        self.init_Population()

    #目标函数，根据使用场景进行设置
    def function(self, x):
        function_inputs = np.array([x])

        output = model.predict (function_inputs)
        fitness = 1.0 / np.abs (output - self.fit)
        print ('狀態1', fitness)
        print ('狀態組合', function_inputs)
        return fitness

    #初始化粒子群
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 1)
                self.V[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])
                if (temp < self.p_fit[i]):  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.fit):  # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                #粒子群算法公式
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            print(self.fit)  # 输出最优值
            print(self.gbest)
        return fitness

if __name__ == '__main__':
    my_pso = PSO(pN=30, dim=4, max_iter=100)
    fitness = my_pso.iterator()
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, 100)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='b', linewidth=3)
    plt.show()