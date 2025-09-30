
import collections
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import pickle

def plotRes(X,Y,title_name,x_label,y_label):
    # plt.figure(figsize=(12, 8))
    
    # 为每条轨迹分配不同颜色和标签
    colors = plt.cm.tab10.colors  # 使用10种区分度高的颜色
    plt.subplot(2,2,1)
    plt.plot(
        X, 
        Y["AiSq10D"]["result"], 
        marker='x', 
        markersize=2,
        linestyle='solid',
        linewidth=1,
        color='#e97200',
        label='T1'
    )
    plt.plot(
        X, 
        Y["AiSq5DP"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label='T2'
    )
    plt.plot(
        X, 
        Y["AiSq10DP"]["result"], 
        marker='1', 
        markersize=2,
        linestyle='dashdot',
        linewidth=1,
        color='#143fe2',
        label='T3'
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("T1-T3")
    plt.legend(loc='best')

    plt.subplot(2,2,2)
    plt.plot(
        X, 
        Y["StSt5R"]["result"], 
        marker='x', 
        markersize=2,
        linestyle='solid',
        linewidth=1,
        color='#e97200',
        label='T4'
    )
    plt.plot(
        X, 
        Y["StSt10R"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label='T5'
    )
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("T4-T5")
    plt.legend(loc='best')
    #e01c64
    #143fe2
    plt.subplot(2,2,3)
    plt.plot(
        X, 
        Y["StSt5M"]["result"], 
        marker='x', 
        markersize=2,
        linestyle='solid',
        linewidth=1,
        color='#e97200',
        label='T6'
    )
    plt.plot(
        X, 
        Y["StSt10M"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label='T7'
    )
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("T6-T7")
    plt.legend(loc='best')

    plt.subplot(2,2,3)
    plt.plot(
        X, 
        Y["UnCh5M"]["result"], 
        marker='x', 
        markersize=2,
        linestyle='solid',
        linewidth=1,
        color='#e97200',
        label='T8'
    )
    plt.plot(
        X, 
        Y["UnCh10M"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label='T9'
    )
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("T8-T9")
    plt.legend(loc='best')

    plt.show()
    

def loadData(path):
    with open(path, 'r') as f:
        routes = f.readlines()
    return routes


def findSourceDestination(routes):
    X = []
    for i in range(len(routes)):
        route = np.array(eval(routes[i]))
        X.append(route[0])
        X.append(route[-1])
    X = np.array(X)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    res = kmeans.cluster_centers_
    return res


def positionOfLine(A, B, C):
    Ax, Ay, Bx, By, X, Y = A[0], A[1], B[0], B[1], C[0], C[1]
    position = np.sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
    if position >= 0:
        return 1
    else:
        return -1


def calAngle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b

    t1 = np.dot(ba, bc)
    t2 = np.linalg.norm(ba)
    t3 = np.linalg.norm(bc)

    if np.isnan(t1) or np.isnan(t2) or (t2 * t3) == 0:
        return 0
    if np.isnan(t3):
        return 180

    else:
        cosine_angle = t1 / (t2 * t3)
        if cosine_angle > 1:
            cosine_angle = 1
        if cosine_angle < -1:
            cosine_angle = -1
        angle = np.arccos(cosine_angle)
        p = positionOfLine(a, b, c)
        if p == 1:
            return angle * 180 / np.pi
        else:
            return 360 - angle * 180 / np.pi


def extractTrajFearure(route, res, k):
    # route：轨迹
    # res=[S,D], S为出发点，D为目的地
    # k：扇区划分参数

    if k <= 0:
        return "error: k should be more than 0"
    SrcP,MP,DesP=res[0], sum(res) / 2, res[1]

    sector_angle_range=180/k
    sector_pt=[0 for i in range(2*k)]
    feature_dist=[0 for i in range(2*k)]#特征向量,0-180和180-360分别分为k个扇区
    feature_angle=[0 for i in range(2*k)]
    
    for i in range(1, len(route) - 1):#注意，这里仅计算轨迹点，出发点和终点不计算
        angles = calAngle(SrcP, MP, route[i])
        angles_bin = angles // sector_angle_range 
        idx = int(angles_bin)
        sector_pt[idx] += 1
        feature_dist[idx]+= np.sqrt(np.sum((route[i] - MP) ** 2))
        feature_angle[idx]+=(angles-idx*sector_angle_range)

    for i in range(len(sector_pt)):
        if sector_pt[i]>=2:
            feature_dist[i]=feature_dist[i]/sector_pt[i]
            feature_angle[i]=feature_angle[i]/sector_pt[i]
        else:
            if sector_pt[i]==0:
                if i>0 and i<len(sector_pt)-1:
                    if sector_pt[i-1] > 0 and sector_pt[i+1] > 0:
                        feature_dist[i]=(feature_dist[i-1]+feature_dist[i+1]/sector_pt[i+1])/2
                        feature_angle[i]=(feature_angle[i-1]+feature_angle[i+1]/sector_pt[i+1])/2#sector_angle_range/2
    
    feature = feature_dist + feature_angle
    return feature

results={
    "T1":{"neighbors":24,"bins":10,"result":[]},
    "T2":{"neighbors":5,"bins":17,"result":[]},
    "T3":{"neighbors":9,"bins":9,"result":[]},
    "T4":{"neighbors":4,"bins":11,"result":[]},
    "T5":{"neighbors":9,"bins":10,"result":[]},
    "T6":{"neighbors":9,"bins":10,"result":[]},
    "T7":{"neighbors":7,"bins":10,"result":[]},
    "T8":{"neighbors":28,"bins":3,"result":[]},
    "T9":{"neighbors":22,"bins":3,"result":[]},
    }
for f in ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]:
    
    path_inner = "dataset/" + f + "/inners.txt"
    path_outlier = "dataset/" + f + "/outliers.txt"

    outliers = loadData(path_outlier)
    inners = loadData(path_inner)

    label = [1] * len(outliers) + [0] * len(inners)
    data = outliers + inners
    

    SD_points = findSourceDestination(data)

    time_e0 = time.time()

    #多参数循环测试
    for k in range(3,30):  #扇区划分k值
        features = []
        for i in range(len(data)):
            p1_resorted = np.array(eval(data[i]))
            p1_resorted = np.concatenate((np.array([SD_points[0]]), p1_resorted, np.array([SD_points[-1]])), axis=0)
            feature = extractTrajFearure(p1_resorted, SD_points, k)
            features.append(feature)


        # LOF detector
        n_neighbors=results[f]["neighbors"]
        clf = LocalOutlierFactor(n_neighbors)
        OutlierScore = -clf.fit_predict(features)
        roc = roc_auc_score(label, OutlierScore)
            
        results[f]["result"].append(roc)

# 保存对象到文件
with open("k_effect_result.pickle", "wb") as file:
    pickle.dump(results, file)
    # print(f,":", results[f]["result"])
    # plotRes(range(3,30),results,"",r"$\psi$","AUC")



    
