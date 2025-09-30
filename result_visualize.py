import pickle
import matplotlib.pyplot as plt
import ast  # 安全解析字符串格式的坐标
import json

def plot_K_Effect_Res(X,Y,title_name,x_label,y_label):
    # plt.figure(figsize=(12, 8))
    font_size=15
    # 为每条轨迹分配不同颜色和标签
    # colors = plt.cm.tab10.colors  # 使用10种区分度高的颜色
    plt.subplot(2,2,1)
    plt.plot(
        X, 
        Y["AiSq10D"]["result"], 
        marker='x', 
        markersize=2,
        linestyle='solid',
        linewidth=1,
        color='#e97200',
        label='T1,k=24'
    )
    plt.plot(
        X, 
        Y["AiSq5DP"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label='T2,k=5'
    )
    plt.plot(
        X, 
        Y["AiSq10DP"]["result"], 
        marker='1', 
        markersize=2,
        linestyle='dashdot',
        linewidth=1,
        color='#143fe2',
        label='T3,k=9'
    )
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.title("T1-T3",fontsize=font_size)
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
        label='T4,k=4'
    )
    plt.plot(
        X, 
        Y["StSt10R"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label='T5,k=9'
    )
    
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.title("T4-T5",fontsize=font_size)
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
        label='T6,k=9'
    )
    plt.plot(
        X, 
        Y["StSt10M"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label='T7,k=7'
    )
    
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.title("T6-T7",fontsize=font_size)
    plt.legend(loc='best')

    plt.subplot(2,2,4)
    plt.plot(
        X, 
        Y["UnCh5M"]["result"], 
        marker='x', 
        markersize=2,
        linestyle='solid',
        linewidth=1,
        color='#e97200',
        label='T8,k=28'
    )
    plt.plot(
        X, 
        Y["UnCh10M"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label='T9,k=22'
    )
    
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.title("T8-T9",fontsize=font_size)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

def plot_ngh_Effect_Res(X,Y,title_name,x_label,y_label):
    # plt.figure(figsize=(12, 8))
    font_size=15
    # 为每条轨迹分配不同颜色和标签
    # colors = plt.cm.tab10.colors  # 使用10种区分度高的颜色
    plt.subplot(2,2,1)
    plt.plot(
        X, 
        Y["AiSq10D"]["result"], 
        marker='x', 
        markersize=2,
        linestyle='solid',
        linewidth=1,
        color='#e97200',
        label=r'T1,$\psi$='+str(Y["AiSq10D"]["bins"])
    )
    plt.plot(
        X, 
        Y["AiSq5DP"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label=r'T2,$\psi$='+str(Y["AiSq5DP"]["bins"])
    )
    plt.plot(
        X, 
        Y["AiSq10DP"]["result"], 
        marker='1', 
        markersize=2,
        linestyle='dashdot',
        linewidth=1,
        color='#143fe2',
        label=r'T3,$\psi$='+str(Y["AiSq10DP"]["bins"])
    )
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.title("T1-T3",fontsize=font_size)
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
        label=r'T4,$\psi$='+str(Y["StSt5R"]["bins"])
    )
    plt.plot(
        X, 
        Y["StSt10R"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label=r'T5,$\psi$='+str(Y["StSt10R"]["bins"])
    )
    
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.title("T4-T5",fontsize=font_size)
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
        label=r'T6,$\psi$='+str(Y["StSt5M"]["bins"])
    )
    plt.plot(
        X, 
        Y["StSt10M"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label=r'T7,$\psi$='+str(Y["StSt10M"]["bins"])
    )
    
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.title("T6-T7",fontsize=font_size)
    plt.legend(loc='best')

    plt.subplot(2,2,4)
    plt.plot(
        X, 
        Y["UnCh5M"]["result"], 
        marker='x', 
        markersize=2,
        linestyle='solid',
        linewidth=1,
        color='#e97200',
        label=r'T8,$\psi$='+str(Y["UnCh5M"]["bins"])
    )
    plt.plot(
        X, 
        Y["UnCh10M"]["result"], 
        marker='D', 
        markersize=2,
        linestyle='dashed',
        linewidth=1,
        color='#e01c64',
        label=r'T9,$\psi$='+str(Y["UnCh10M"]["bins"])
    )
    
    plt.xlabel(x_label,fontsize=font_size)
    plt.ylabel(y_label,fontsize=font_size)
    plt.title("T8-T9",fontsize=font_size)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 从文件中加载对象
    with open("ngh_effect_result.pickle", "rb") as file:
        results = pickle.load(file)
        plot_ngh_Effect_Res(range(3,30),results,"","k","AUC")