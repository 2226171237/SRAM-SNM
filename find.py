#-*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
import argparse
import csv
import copy

parser=argparse.ArgumentParser(description='input file path')
parser.add_argument('--file_path',default='./Vout_P1_N1.csv',help='data file path')  # 文件保存路径
parser.add_argument('--factor',default=100,help='expand factot times of origon data') # 数据插值后数据量是原来的多少倍
args=parser.parse_args()


def read_data():
    VoutX=[]
    VoutY=[]
    with open(args.file_path,'r') as f:
        csv_f=csv.reader(f)
        next(csv_f) # 去掉标题字段
        for one_data in csv_f:
            VoutX.append(float(one_data[0]))
            VoutY.append(float(one_data[1]))
    return np.array(VoutX),np.array(VoutY)

if __name__=='__main__':

    VoutX,VoutY=read_data()
    fig,ax=plt.subplots(1,1)


    # 曲线拟合插值
    x1=np.linspace(np.min(VoutX),np.max(VoutX),len(VoutY)*int(args.factor))
    f=interp1d(VoutX,VoutY,kind='cubic')
    y1=f(x1)
    x2=copy.deepcopy(y1)
    y2=copy.deepcopy(x1)
    ax.plot(x1,y1,'r')
    ax.plot(x2,y2,'b')

    # 求两个曲线交点坐标索引
    cross_index = np.argwhere(np.diff(np.sign(y1 - y2)) != 0).reshape(-1) + 0
    cross_index = cross_index[len(cross_index)//2]
    ax.plot(x1[cross_index], y1[cross_index], 'go')

    if y1[cross_index-1]<y2[cross_index+1]:
        x1,y1,x2,y2=x2,y2,x1,y1

    # 45度直线扫描
    b=np.diff(y1[:cross_index])[::-1][:len(y1)-cross_index]
    b=np.cumsum(-b)
    max_len=0  #最大对角长度
    mask_index1=-1
    mask_index2=-1
    for bi in b:  # 45度直线的偏置 f(x)-(x+b)=0 求解 ，45度直线扫描
        cross_index1=np.argwhere(np.diff(np.sign(y1-x1-bi))!=0).reshape(-1)
        cross_index2=np.argwhere(np.diff(np.sign(y2-x2-bi))!=0).reshape(-1)
        line_len=(x1[cross_index1]-x2[cross_index2])**2+\
                 (y1[cross_index1]-y2[cross_index2])**2
        if line_len>max_len:
            mask_index1=cross_index1
            mask_index2=cross_index2
            max_len=line_len
    x,y=[x2[mask_index2],x1[mask_index1]],[y2[mask_index2],y1[mask_index1]] # 最终找到的点
    edge=np.sqrt(max_len/2) # 找到的最大边
    ## 画方形
    ax.text(x[0] + edge+0.05, y[0] + edge+0.05, 'SNM= %.3f' % np.sqrt(max_len), fontsize=15)
    ax.add_patch(patches.Rectangle((x[0],y[0]),width=edge,height=edge,fill=False))
    ax.plot([x[0],x[0]+edge], [y[0],y[0]+edge], 'k')

    # 关于y=x 对称
    ax.add_patch(
        patches.Rectangle((y[0], x[0]), width=edge, height=edge, fill=False))
    ax.plot([y[0],y[0]+edge], [x[0],x[0]+edge], 'k')

    ax.axis('equal')
    ax.grid()
    plt.ylabel('V(v)')
    plt.xlabel('V(v)')
    plt.show()
