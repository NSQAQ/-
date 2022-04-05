import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据处理
data=pd.read_csv("./ex2data1.txt",names=["score_1","score_2","y"])
data_y0=data[data.y==0]
data_y1=data[data.y>0]
# 查看散点图
plt.plot(data_y0.score_1,data_y0.score_2,"ro",data_y1.score_1,data_y1.score_2,"b*")
plt.show()