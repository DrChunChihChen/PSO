import random
import numpy as np
from sklearn import svm
from sklearn import model_selection
import matplotlib.pyplot as plt

#設定PSO關鍵引數
particle_num = 30  #粒子數量，個數要適量，不能太大和太小
particle_dim = 2    #每個粒子的維度，維度要與優化的引數個數相同
iter_num = 400  # 最大迭代次數
c1 = 1.5  # 參考自己歷史最優的權重
c2 = 1  # 參考全域性最優的的權重
w_max = 1  # 慣性因子，表示粒子之前運動方向在本次方向上的慣性
w_min = 0.5
min_value = 0.0001  # 引數最小值
best_fit = []
no_change = 0 #全域性最優未改變次數，達到50次則終止程式

#讀取資料並進行預處理
def load_data(data_file):
    data = []   #儲存特徵
    label = []  #儲存標籤
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        label.append(float(lines[0]))#提取標籤
        #提取特徵
        index = 0
        tmp = [] #儲存每個標籤對應的特徵
        #資料預處理,對於缺少的特徵以0補上
        for i in range(1, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:     #特徵存在
                tmp.append(float(li[1]))
            else:       #特徵不存在以零補上，並將當前的新增上
                while (int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:    #處理特徵在最後缺少的情況
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.array(data), np.array(label).T

#PSO
#1.粒子及其運動方向初始化
def inti_origin():
    particle_loc = []   #記錄所有粒子移動過的位置，為一個二維陣列，
    particle_dir = []   #記錄所有粒子要移動的方向，及其歷史方向，為二維陣列
    for i in range(particle_num):
        tempA = [] #儲存每個粒子的引數
        tempB = []  #儲存每個例子的移動方向
        for j in range(particle_dim):
            a = random.random()*(20-min_value) #隨機引數
            b = random.random()  # 隨機移動方向
            tempA.append(a)
            tempB.append(b)
        particle_loc.append(tempA)
        particle_dir.append(tempB)
    return particle_loc,particle_dir
#2.計算SVM的適應度函式值,並尋找區域性和全域性最優適應度，及其對應的引數
def fitness(particle_loc,trainX, trainY,pbest_fit,gbest_fit,pbest_loc,gbest_loc):
    #particle_loc粒子所在位置集合
    #pbest_fit 區域性最優適應度
    #gbest_fit 全域性最優適應度
    #pbest_loc 區域性最優引數
    #gbest_loc 全域性最優引數
    fit_value = []
    #對每個粒子計算其適應度，以3-fold拆分資料，RBF為核函式，求出3個適應度值，取平局值為最終結果
    for i in range(particle_num):
        rbf_svm = svm.SVC(kernel='rbf', C=particle_loc[i][0], gamma=particle_loc[i][1])
        cv_scores = model_selection.cross_val_score(rbf_svm, trainX, trainY, cv=3, scoring='accuracy')
        fit_value.append(cv_scores.mean())
    k = 0  # 記錄迴圈下標
    for i in range(len(fit_value)):
        if(pbest_fit[k] < fit_value[k]):  #新的區域性最優出現
            pbest_fit[k] = fit_value[k]
            pbest_loc[k] = particle_loc[k]
            if(fit_value[k] > gbest_fit): #新的全域性最優出現
                gbest_fit = fit_value[k]
                gbest_loc = particle_loc[k]
    print('最優適應度為{}，對應引數是{}'.format(gbest_fit,gbest_loc))
    best_fit.append(gbest_fit)
    return pbest_fit,gbest_fit,pbest_loc,gbest_loc
#3.更新粒子的位置
def updata(particle_loc,particle_dir,pbest_loc,gbest_loc,k):
    #k為當前迭代次數，動態更新w時使用
    w = w_max - (w_max-w_min)*((k+1)/iter_num)    #w採用新的更新方式
    for i in range(particle_num):   # 速度更新
        #新的速度由三部分組成
        a1 = [w * particle_dir[i][0],w * particle_dir[i][1]]
        a2 = [c1 * random.random() * (pbest_loc[i][0] - particle_loc[i][0]),c1 * random.random() * (pbest_loc[i][1] - particle_loc[i][1])] #參考區域性
        a3 = [c2 * random.random() * (gbest_loc[0] - particle_loc[i][0]),c2 * random.random() * (gbest_loc[1] - particle_loc[i][1])] #參考全域性
        new_dir = [a1[0]+a2[0]+a3[0],a1[1]+a2[1]+a3[1]]
        particle_dir[i] = new_dir #更新速度
        particle_loc[i] = [particle_dir[i][0] + new_dir[0],particle_dir[i][1] + new_dir[1]] #更新位置

        #由於SVM的要求是引數不能小於零，並且應該保持在一定範圍保證其效能，所以要對更新後的引數進行判斷
        particle_loc[i][0] = max(particle_loc[i][0], min_value)
        particle_loc[i][1] = max(particle_loc[i][1], min_value)

    return particle_dir,particle_loc
#畫適應度折線圖
def plot(best_fit):
    X = []
    Y = []
    for i in range(len(best_fit)):
        X.append(i + 1)
        Y.append(best_fit[i])
    plt.plot(X, Y)
    plt.xlabel('迭代次數', size=15)
    plt.ylabel('適應度', size=15)
    plt.show()
#畫散點圖
def plot_point(pbest_loc):
    X = []
    Y = []
    for i in range(len(pbest_loc)):
        X.append(pbest_loc[i][0])
        Y.append(pbest_loc[i][1])
    plt.figure(figsize=(20, 8), dpi=100) #建立畫布
    plt.scatter(X,Y)
    plt.show()
if __name__ == '__main__':
    #讀取資料
    trainX, trainY = load_data('rbf_data.txt')
    #初始化
    particle_loc, particle_dir = inti_origin()
    all_fitness = [] #記錄所有計算過的適應度的值
    pbest_fit = [] #記錄每個粒子各自的最優適應度
    pbest_loc = []  #記錄每個粒子各自的最優位置
    for i in range(particle_num):
        pbest_fit.append(0.0)
        pbest_loc.append([0.0,0.0])
    gbest_fit = 0.0 #全域性最優適應度
    gbest_loc = [0.0,0.0] #全域性最優位置
    for i in range(iter_num):
        print('迭代次數：',i+1,end=",")
        old_gbest_fit = gbest_fit
        pbest_fit,gbest_fit,pbest_loc,gbest_loc = fitness(particle_loc, trainX, trainY,pbest_fit,gbest_fit,pbest_loc,gbest_loc)
        #如果過連續50代沒有產生新的最優值則停止迴圈
        if(old_gbest_fit == gbest_fit):
            no_change += 1
        else:
            no_change = 0
        if(no_change == 50):
            break
        particle_dir, particle_loc = updata(particle_loc,particle_dir,pbest_loc,gbest_loc,i)
    plot_point(pbest_loc)
    plot(best_fit)