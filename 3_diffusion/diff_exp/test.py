import os
import pandas as pd
import numpy as np

data_pth = '../../dataset/magic/magic04.data'
data = pd.read_csv(data_pth, header=None).iloc[:, 1:]
print(" === data ===", data)

# 데이터 불러오기
# df_data = pd.read_csv(data_pth)
# df_data.columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
# train_col = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
# df_data['class'] = df_data['class'].replace({'g':0, 'h':1})
# data = df_data