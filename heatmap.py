from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


#模拟联系数据
data=[[0.1,0.5,0.7,0.9,0.4],[0.1,0.6,0.4,0.3,0.8],[0.5,0.1,0.3,0.2,0.9],[0.1,0.8,0.7,0.5,0.2],[0.1,0.5,0.4,0.8,0.9]]
data=pd.DataFrame(data)


#绘制热图
plot=sns.heatmap(data)


plt.show()