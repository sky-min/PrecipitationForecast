"""
# 기온과 습도로 강수량 예측하기

## 데이터 준비

### csv불러오기
"""

import pandas as pd
df = pd.read_csv('./weather.csv')

"""### 기온 습도 값 가져오기"""

x = df[['temp', 'rh']]
x_arr = x.to_numpy()
print(x_arr)

"""### 강수량 가져오기"""

y = df['rain']
y_arr = y.to_numpy()
print(y_arr)

"""##  모델 만들기

### 훈련 데이터와 테스트 데이터 나누기
"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_arr,
    y_arr,
    train_size=0.7,
    test_size=0.3
)

"""### 모델 학습시키기"""

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

"""## 모델테스트

### 테스트 데이터 입력
"""

y_predict = lr.predict(x_test)
print(y_predict)

"""### 그래프 그리기"""

import matplotlib.pyplot as plt
plt.scatter(y_test, y_predict, alpha=0.3)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Precipitation forecast")
plt.show()

"""### 모델 점수
1점 만점 정확도를 나타냄
"""

print(lr.score(x_train, y_train))