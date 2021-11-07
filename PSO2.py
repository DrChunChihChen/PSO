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

model.add(Dense(19,activation='relu'))
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





import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high):
        # 初始化
        self.dimension = dimension  # 變數個數
        self.time = time  # 迭代的代數
        self.size = size  # 種群大小
        self.bound = []  # 變數的約束范圍
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每個粒子最優的位置
        self.g_best = np.zeros((1, self.dimension))[0]  # 全域最優的位置

        # 初始化第0代初始全域最優解
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]  # 儲存最優的個體
            fit = self.fitness(self.p_best[i])
            # 做出修改
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, x):
        """
        個體適應值計算
        """
        function_inputs = np.array ([x])

        output = model.predict (function_inputs)
        fitness = 1.0 / np.abs (output - 1)
        # print ('狀態1', fitness)
        # print ('狀態組合', function_inputs)
        # x1 = x[0]
        # x2 = x[1]
        # x3 = x[2]
        # x4 = x[3]
        # x5 = x[4]
        # y = math.floor((x2 * np.exp(x1) + x3 * np.sin(x2) + x4 + x5) * 100) / 100
        # print(y)
        return fitness

    def update(self, size):
        c1 = 2.0  # 學習因子
        c2 = 2.0
        w = 0.8  # 自身權重因子
        for i in range(size):
            # 更新速度(核心公式)
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) > self.fitness(self.g_best):
                self.g_best = self.x[i]

    def pso(self):
        best = []
        self.final_best = np.array([1, 2, 3, 4])
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            print('當前最佳位置：{}'.format(self.final_best))
            temp = self.fitness(self.final_best)
            print('當前的最佳適應度：{}'.format(temp))
            best.append(temp)
        t = [i for i in range(self.time)]
        plt.figure()
        plt.plot(t, best, color='red', marker='.', ms=15)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次數")  # X軸標簽
        plt.ylabel(u"適應度")  # Y軸標簽
        plt.title(u"迭代程序")  # 標題
        plt.show()


if __name__ == '__main__':
    time = 50
    size = 100
    dimension = 4
    v_low = -1
    v_high = 1
    low = [-1, -1, -1, -1]
    up = [2, 2, 2, 2]
    pso = PSO(dimension, time, size, low, up, v_low, v_high)
    pso.pso()
