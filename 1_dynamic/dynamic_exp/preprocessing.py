import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import poisson


def missing_value_generator(X, missing_rate, seed):
    row_num = X.shape[0]
    column_num = X.shape[1]
    missing_value_average_each_row = column_num * (missing_rate/100)

    np.random.seed(seed)
    # rvs 함수 : 무작위로 표본을 만들때 생성
    poisson_dist = poisson.rvs(mu = missing_value_average_each_row, size = row_num, random_state = seed)
    # numpy.clip(array, min, max) : array 내의 element들에 대해서 min 값보다 작은값들을 min으로 바꿔주고 max값보다 큰 값들을 max값으로 바꿔주는 함수
    poisson_dist = np.clip(poisson_dist, 0, X.shape[1]-1)
    
    column_idx = np.arange(column_num)
    X_missing = X.copy().astype(float)
    for i in range(row_num):
        missing_idx = np.random.choice(column_idx, poisson_dist[i], replace=False)
        for j in missing_idx:
            X_missing[i,j] = np.nan
    
    
    return X_missing


def preprocessing(x,y, missing_rate, seed):
    
    x = missing_value_generator(x, missing_rate, seed)

    # StandardScaler():각 열의 feature 값의 평균을 0으로 잡고, 표준편차를 1로 간주하여 정규화시키는 방법
    # 각 데이터가 평균에서 몇 표준편차만큼 떨어져있는지를 기준으로 삼고, 데이터의 특징을 모르는 경우 사용하는 정규화 방법
    scaler_x = StandardScaler()
    x = scaler_x.fit_transform(x)
    #x_tst = scaler_x.transform(x_tst)
    
    # np.unique() : 배열(리스트, numpy array 등) 자료만 input으로, 1차원 shape으로 변환하고 정렬을 진행한 결과 반환
    if len(np.unique(y)) > 2:
        # One-Hot Encoding : n개의 범주형 데이터를 n개의 비트(0,1) 벡터로 표현, 서로 다른 범주 데이터는 독립적인 관계라는 것을 나타낼 수 있음
        enc = OneHotEncoder(sparse=False)
        y = enc.fit_transform(y.reshape(-1,1))
        #y_tst = enc.fit_transform(y_tst.reshape(-1,1))
    
    else:
        y = y.reshape(-1,1)
        #y_tst = y_tst.reshape(-1,1)
    
    return x,y