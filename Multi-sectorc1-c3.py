import collections
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import math
from sklearn.preprocessing import StandardScaler
import random  # 新增：用于随机翻转标签

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

def min_max_normalize(lst):
    """对列表进行最大最小归一化"""
    if not lst:
        return []

    min_val = min(lst)
    max_val = max(lst)

    if max_val == min_val:
        return [0.5] * len(lst)

    normalized = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized

def extractTrajFearure(route, res, k):
    if k <= 0:
        return "error: k should be more than 0"
    SrcP,MP,DesP=res[0], sum(res) / 2, res[1]

    sector_angle_range=180/k
    sector_pt=[0 for i in range(2*k)]
    feature_dist=[0 for i in range(2*k)]
    feature_angle=[0 for i in range(2*k)]

    for i in range(1, len(route) - 1):
        angles = calAngle(SrcP, MP, route[i])
        angles_bin = angles // sector_angle_range
        idx = int(angles_bin)
        sector_pt[idx] += 1
        feature_dist[idx]+= np.sqrt(np.sum((route[i] - MP) ** 2))
        feature_angle[idx]+=(angles-idx*sector_angle_range)

    feature_dist1=[0 for i in range(2*k)]
    feature_dist2=[0 for i in range(2*k)]

    for i in range(len(sector_pt)):
        if sector_pt[i]>=2:
            feature_angle[i]=feature_angle[i]/sector_pt[i]
            feature_dist[i]=feature_dist[i]/sector_pt[i]

        else:
            if sector_pt[i]==0:
                if i>0 and i<len(sector_pt)-1:
                    if sector_pt[i-1] > 0 and sector_pt[i+1] > 0:
                        feature_dist[i]=(feature_dist[i-1]+feature_dist[i+1]/sector_pt[i+1])/2
                        feature_angle[i]=(feature_angle[i-1]+feature_angle[i+1]/sector_pt[i+1])/2

        feature_dist1[i]=feature_dist[i]*math.cos(math.radians(feature_angle[i]))
        feature_dist2[i]=feature_dist[i]*math.cos(math.radians(sector_angle_range-feature_angle[i]))

    return feature_dist1 , feature_dist2

# ===================== 核心修改：标签随机翻转函数 =====================
def flip_labels(labels, flip_ratio=0.05):
    """
    随机翻转指定比例的标签
    labels: 原始标签列表 1=异常, 0=正常
    flip_ratio: 翻转比例，默认5%
    返回：翻转后的标签列表
    """
    flipped_labels = labels.copy()
    n_total = len(flipped_labels)
    n_flip = int(math.ceil(n_total * flip_ratio))  # 向上取整保证至少翻转1个
    
    # 随机选择要翻转的索引
    flip_indices = random.sample(range(n_total), n_flip)
    
    # 执行翻转：0↔1
    for idx in flip_indices:
        flipped_labels[idx] = 1 - flipped_labels[idx]
    
    return flipped_labels
# ====================================================================

results=[]
flip_results = []  # 保存翻转5%标签后的结果

# 数据集列表不变
for f in ["C1","C2","C3"]:
    best_auc={'auc':0,'bins':0,'n_neighbors':0}
    best_flip_auc={'auc':0,'bins':0,'n_neighbors':0}  # 翻转后的最优AUC
    path_inner = "dataset/" + f + "/inners.txt"
    path_outlier = "dataset/" + f + "/outliers.txt"

    outliers = loadData(path_outlier)
    inners = loadData(path_inner)

    # 原始真实标签
    label = [1] * len(outliers) + [0] * len(inners)
    data = outliers + inners

    # ===================== 修改：仅对 C1/C2/C3 翻转5%标签 =====================
    if f in ["C1", "C2", "C3"]:
        np.random.seed(42)  # 固定随机种子，结果可复现
        random.seed(42)
        flipped_label = flip_labels(label, flip_ratio=0.05)
        print(f"\n========== {f} 标签信息 ==========")
        print(f"总样本数: {len(label)}")
        print(f"翻转标签数: {sum(1 for a,b in zip(label,flipped_label) if a!=b)}")
        print(f"原始标签异常数: {sum(label)}")
        print(f"翻转后标签异常数: {sum(flipped_label)}")
    else:
        flipped_label = label  # 非成都数据集不翻转
    # ========================================================================

    SD_points = findSourceDestination(data)

    time_e0 = time.time()

    for k in range(3,40):
        features1 = []
        features2 = []
        for i in range(len(data)):
            p1_resorted = np.array(eval(data[i]))
            p1_resorted = np.concatenate((np.array([SD_points[0]]), p1_resorted, np.array([SD_points[-1]])), axis=0)
            feature1,feature2 = extractTrajFearure(p1_resorted, SD_points, k)
            features1.append(feature1)
            features2.append(feature2)

        scaler = StandardScaler()
        features1 = scaler.fit_transform(features1)
        features2 = scaler.fit_transform(features2)
        features = np.concatenate((features1, features1), axis=1)

        for n_neighbors in range(3,60):
            clf = LocalOutlierFactor(n_neighbors)
            OutlierScore = -clf.fit_predict(features)
        
            # 计算原始标签AUC
            roc = roc_auc_score(label, OutlierScore)
            if best_auc['auc']<roc:
                best_auc['auc']=roc
                best_auc['bins']=k
                best_auc['n_neighbors']=n_neighbors
            
            # ===================== 计算翻转标签AUC =====================
            roc_flip = roc_auc_score(flipped_label, OutlierScore)
            if best_flip_auc['auc'] < roc_flip:
                best_flip_auc['auc'] = roc_flip
                best_flip_auc['bins'] = k
                best_flip_auc['n_neighbors'] = n_neighbors
            # ==========================================================

    results.append(best_auc)
    flip_results.append(best_flip_auc)
    
    # 输出：原始结果 + 翻转5%标签结果
    print(f"\n===== {f} 最终结果 =====")
    print(f"原始标签最优AUC: {best_auc}")
    if f in ["C1", "C2", "C3"]:
        print(f"5%标签翻转后最优AUC: {best_flip_auc}")
    print("="*50)