import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostClassifier, Pool, CatBoostRegressor

data = pd.read_csv('import_data/TB_SSC_DELIVERY/TB_SSC_DELIVERY.csv',encoding='utf8')

#매출금액합 분포 확인
print(data['매출금액합'].mean())
print(data['매출금액합'].std())
print(data['매출금액합'].min())
print(data['매출금액합'].max())
quantiles = data['매출금액합'].quantile([0.25, 0.5, 0.75])
print(quantiles[0.25])
print(quantiles[0.5])
print(quantiles[0.75])
print(data[data['매출금액합'] == (data['매출금액합'].min())])
print(data[data['매출금액합'] == (data['매출금액합'].max())])

#데이터 탐색 및 시각화
# 해당 열의 고유한 범주 목록을 출력
print(data['기준년월'].unique())
print(data['평일휴일구분'].unique())
print(data['요일구분'].unique())
print(data['시간대구분'].unique())
print(data['성별'].unique())
print(data['직업'].unique())
print(data['연령대'].unique())
print(data['대분류명'].unique())
print(data['중분류명'].unique())
print(data['소분류명'].unique())
print(data['매출건수'].unique())

# 데이터 변수명 영어로 변경
data.loc[(data['기준년월'] == 202201),'기준년월'] = '01'
data.loc[(data['기준년월'] == 202202),'기준년월'] = '02'
data.loc[(data['기준년월'] == 202203),'기준년월'] = '03'
data.loc[(data['기준년월'] == 202204),'기준년월'] = '04'
data.loc[(data['기준년월'] == 202205),'기준년월'] = '05'
data.loc[(data['기준년월'] == 202206),'기준년월'] = '06'
data.loc[(data['기준년월'] == 202207),'기준년월'] = '07'
data.loc[(data['기준년월'] == 202208),'기준년월'] = '08'
data.loc[(data['기준년월'] == 202209),'기준년월'] = '09'
data.loc[(data['기준년월'] == 202210),'기준년월'] = '10'
data.loc[(data['기준년월'] == 202211),'기준년월'] = '11'
data.loc[(data['기준년월'] == 202212),'기준년월'] = '12'

data.loc[(data['평일휴일구분'] == '평일'),'평일휴일구분'] = 'weekday'
data.loc[(data['평일휴일구분'] == '휴일'),'평일휴일구분'] = 'weekend'

data.loc[(data['요일구분'] == '월'),'요일구분'] = 'MON'
data.loc[(data['요일구분'] == '화'),'요일구분'] = 'TUES'
data.loc[(data['요일구분'] == '수'),'요일구분'] = 'WED'
data.loc[(data['요일구분'] == '목'),'요일구분'] = 'THUR'
data.loc[(data['요일구분'] == '금'),'요일구분'] = 'FRI'
data.loc[(data['요일구분'] == '토'),'요일구분'] = 'SAT'
data.loc[(data['요일구분'] == '일'),'요일구분'] = 'SUN'

data.loc[(data['시간대구분'] == 'A.04-10시'),'시간대구분'] = '04-10'
data.loc[(data['시간대구분'] == 'B.10-16시'),'시간대구분'] = '10-16'
data.loc[(data['시간대구분'] == 'C.16-22시'),'시간대구분'] = '16-22'
data.loc[(data['시간대구분'] == 'D.22-04시'),'시간대구분'] = '22-04'

data.loc[(data['직업'] == 'A.전문직'),'직업'] = 'Specialized Job'
data.loc[(data['직업'] == 'B.회사원(대기업)'),'직업'] = 'Office Worker(Conglomerate)'
data.loc[(data['직업'] == 'C.회사원(일반)'),'직업'] = 'Office Worker(General)'
data.loc[(data['직업'] == 'D.공무원'),'직업'] = 'Civil Servant'
data.loc[(data['직업'] == 'E.교육인'),'직업'] = 'Educational'
data.loc[(data['직업'] == 'F.자영업자'),'직업'] = 'Self-Employed'
data.loc[(data['직업'] == 'J.기타'),'직업'] = 'Etc'

data.loc[(data['중분류명'] == '간편음식'),'중분류명'] = 'Simple Food'
data.loc[(data['중분류명'] == '고기'),'중분류명'] = 'Meat'
data.loc[(data['중분류명'] == '디저트'),'중분류명'] = 'Dessert'
data.loc[(data['중분류명'] == '양식/아시안'),'중분류명'] = 'Western/Asian Food'
data.loc[(data['중분류명'] == '음식배달'),'중분류명'] = 'Food Delivery'
data.loc[(data['중분류명'] == '중식'),'중분류명'] = 'Chinese Food'
data.loc[(data['중분류명'] == '피자/치킨'),'중분류명'] = 'Pizza/Chicken'
data.loc[(data['중분류명'] == '한식'),'중분류명'] = 'Korean Food'
data.loc[(data['중분류명'] == '일식/회'),'중분류명'] = 'Japanese Food/Sashimi'
data.loc[(data['중분류명'] == '주점'),'중분류명'] = 'Pub'

# 막대 그래프를 이용한 시각화
sns.barplot(data = data, x = '기준년월', y = '매출건수')
plt.xlabel('Month')
plt.ylabel('Sales count')

sns.barplot(data = data, x = '평일휴일구분', y = '매출건수')
plt.xlabel('Weekday&Weekend')
plt.ylabel('Sales count')

sns.barplot(data = data, x = '요일구분', y = '매출건수')
plt.xlabel('Day')
plt.ylabel('Sales count')

sns.barplot(data = data, x = '시간대구분', y = '매출건수')
plt.xlabel('Time')
plt.ylabel('Sales count')

sns.barplot(data = data, x = '성별', y = '매출건수')
plt.xlabel('Sex')
plt.ylabel('Sales count')

sns.barplot(data = data, x = '직업', y = '매출금액합')
plt.xticks(rotation = 90)
plt.xlabel('Job')
plt.ylabel('Sales count')

sns.barplot(data = data, x = '직업', y = '매출금액합')
plt.xticks(rotation = 90)
plt.xlabel('Sectors')
plt.ylabel('Sales count')

#데이터 전처리
def preprocessing():
    data = pd.read_csv('import_data/TB_SSC_DELIVERY/TB_SSC_DELIVERY.csv',encoding='utf8')
    data.drop(data[(data['중분류명']=='음식배달')].index, inplace=True)
    data["log매출"] = np.log(data["매출금액합"])
    data = data.iloc[:,[0,1,3,4,5,6,8,10,12]]
    return data


#모델 1: 선형 회귀
data = preprocessing()
# one-hot encoding: 순서적 관계가 없는 변수에 대해서 적용하는 인코딩 방식_더미 변수 드롭 ㅇ
data = pd.get_dummies(data, columns = ['기준년월'], drop_first = True)
data = pd.get_dummies(data, columns = ['평일휴일구분'], drop_first = True)
data = pd.get_dummies(data, columns = ['시간대구분'], drop_first = True)
data = pd.get_dummies(data, columns = ['성별'], drop_first = True)
data = pd.get_dummies(data, columns = ['직업'], drop_first = True)
data = pd.get_dummies(data, columns = ['연령대'], drop_first = True)
data = pd.get_dummies(data, columns = ['중분류명'], drop_first = True)
data = data.astype(int)

X = data.drop('log매출', axis = 1)
y = data['log매출']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())


#모델 2, 3: 릿지, 라쏘
df = preprocessing()
df=pd.get_dummies(df, columns = ['기준년월', '평일휴일구분', '시간대구분', '성별', '직업', '연령대', '중분류명'], drop_first=True)
print(df.columns)

x=df[[ '기준년월_202202', '기준년월_202203','기준년월_202204', '기준년월_202205', '기준년월_202206', '기준년월_202207', \
    '기준년월_202208', '기준년월_202209', '기준년월_202210', '기준년월_202211', '기준년월_202212', \
    '평일휴일구분_휴일', '시간대구분_B.10-16시', '시간대구분_C.16-22시', \
    '시간대구분_D.22-04시', '성별_M', '직업_B.회사원(대기업)', '직업_C.회사원(일반)', '직업_D.공무원', \
    '직업_E.교육인', '직업_F.자영업자', '직업_J.기타', '연령대_30', '연령대_40', '연령대_50', \
    '연령대_60', '중분류명_고기', '중분류명_디저트', '중분류명_양식/아시안', \
    '중분류명_일식/회', '중분류명_주점', '중분류명_중식', '중분류명_피자/치킨', '중분류명_한식']]
y=np.log(df["매출금액합"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=1.0) 
lasso.fit(x_train, y_train)
lasso_predictions = lasso.predict(x_test)
print(lasso_predictions)

mse = mean_squared_error(y_test, lasso_predictions)
r2 = r2_score(y_test, lasso_predictions)
coefficients = lasso.coef_
print("평균 제곱 오차 (MSE):", mse)
print("R-squared (결정 계수):", r2)
for i, coef in enumerate(coefficients):
    print(f"변수{i}: {coef}")

ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
ridge_predictions = ridge.predict(x_test)

mse = mean_squared_error(y_test, ridge_predictions) 
r2 = r2_score(y_test, ridge_predictions)
coefficients = lasso.coef_
print("평균 제곱 오차 (MSE):", mse)
print("R-squared (결정 계수):", r2)
for i, coef in enumerate(coefficients):
    print(f"변수{i}: {coef}")


#모델 4: 신경망
df = preprocessing()
encoder = LabelEncoder()
categorical_columns = ['기준년월', '평일휴일구분', '시간대구분', '성별', '직업', '연령대', '중분류명']
for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

X = df[['기준년월', '평일휴일구분', '시간대구분', '성별', '직업', '연령대', '중분류명']]
y = df["log매출"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"평균 제곱 오차 (MSE): {mse}")
print(f"R-squared: {r2}")


#모델 5: 랜덤포레스트
data = preprocessing()
data = pd.get_dummies(data, columns = ['기준년월'])
data = pd.get_dummies(data, columns = ['평일휴일구분'])
data = pd.get_dummies(data, columns = ['시간대구분'])
data = pd.get_dummies(data, columns = ['성별'])
data = pd.get_dummies(data, columns = ['직업'])
data = pd.get_dummies(data, columns = ['연령대'])
data = pd.get_dummies(data, columns = ['중분류명'])
data = data.astype(int)

X = data.drop('log매출', axis = 1)
y = data['log매출']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=10000, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"mean squared error: {mse}")
print(r2)
importance = rf.feature_importances_
print(importance)


#모델 6: CatBoost
data = preprocessing()
y = data["log매출"]
X = data.drop("log매출", axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(iterations=100, learning_rate=0.05)

model.fit(X_train, y_train, cat_features=['기준년월','평일휴일구분','시간대구분','성별','직업','연령대','중분류명'])
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse, r2)


#최종 catboost 모델 최적화
df_feature = pd.DataFrame({"feature" : ['weekend', 'month','time','sex','jobs','ages','food types'], \
    "feature_importance":feature_importance})

sns.barplot(data = df_feature, y = "feature", x = "feature_importance" \
    , order = df_feature.sort_values("feature_importance",ascending = False).feature )

#가능한 모든 조합 생성
attributes = {
    '기준년월' : ['202201', '202202', '202203', '202204', '202205', '202206', '202207', '202208', '202209', '202210', '202211', '202212'],
    '평일휴일구분' : ['평일', '휴일'],
    '성별' : ['F', 'M'],
    '연령대' : ['20', '30', '40', '50' ,'60'],
    '직업' : ['C.회사원(일반)', 'J.기타', 'F.자영업자', 'B.회사원(대기업)', 'D.공무원', 'E.교육인', 'A.전문직'],
    '중분류명': ['한식', '고기', '디저트', '간편음식', '일식/회', '피자/치킨', '양식/아시안', '중식', '음식배달', '주점'],
    '시간대구분' : ['C.16-22시', 'B.10-16시', 'D.22-04시', 'A.04-10시']
}

from itertools import product
combinations = list(product(*attributes.values()))
ex = pd.DataFrame(combinations, columns=attributes.keys())
ex_pred = np.e**(model.predict(ex))
ex['예측값'] = ex_pred
print(ex)
print(ex[ex['예측값'] ==ex['예측값'].max()]) #feature 3개 추가 제거 결정

#성별, 평일휴일구분, 기준년월 제외
y = data["log매출"]
X = data.drop(["log매출", '기준년월', '평일휴일구분', '성별'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostRegressor(iterations=100, learning_rate=0.05)

model.fit(X_train, y_train, cat_features=['시간대구분','직업','연령대','중분류명'])
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)

feature_importance = model.get_feature_importance()
print(feature_importance)

df_feature = pd.DataFrame({"feature" : ['time','jobs','ages','food types'], \
    "feature_importance":feature_importance})

sns.barplot(data = df_feature, y = "feature", x = "feature_importance" \
    , order = df_feature.sort_values("feature_importance",ascending = False).feature )

attributes2 = {
    '연령대' : ['20', '30', '40', '50' ,'60'],
    '직업' : ['C.회사원(일반)', 'J.기타', 'F.자영업자', 'B.회사원(대기업)', 'D.공무원', 'E.교육인', 'A.전문직'],
    '중분류명': ['한식', '고기', '디저트', '간편음식', '일식/회', '피자/치킨', '양식/아시안', '중식', '음식배달', '주점'],
    '시간대구분' : ['C.16-22시', 'B.10-16시', 'D.22-04시', 'A.04-10시']
}

combinations2 = list(product(*attributes2.values()))
ex2 = pd.DataFrame(combinations2, columns=attributes2.keys())
ex_pred2 = np.e**(model.predict(ex2))
ex2['예측값'] = ex_pred2
print(ex2)
print(ex2[ex2['예측값'] == ex2['예측값'].max()])
print(ex2[ex2['예측값'] == ex2['예측값'].min()])


#하이퍼파라미터 튜닝
parameters ={'iterations':[500,1000,2000], \
    'depth' : [5], \
    'learning_rate' : [0.1,0.07,0.05], \
    'random_seed' :[42], \
}

catboost_grid = GridSearchCV(estimator=model,param_grid=parameters,cv=3)

catboost_grid.fit(X_train,y_train)
print(catboost_grid.best_params_) #iteration 1000, learning_rate=0.1

#최종 모델 생성
model = CatBoostRegressor(iterations=1000, learning_rate=0.1)

model.fit(X_train, y_train, cat_features=['시간대구분','직업','연령대','중분류명'])
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(r2)

feature_importance = model.get_feature_importance()
print(feature_importance)

df_feature = pd.DataFrame({"feature" : ['time','jobs','ages','food types'], \
    "feature_importance":feature_importance})

sns.barplot(data = df_feature, y = "feature", x = "feature_importance" \
    , order = df_feature.sort_values("feature_importance",ascending = False).feature )

ex3 = pd.DataFrame(combinations2, columns=attributes2.keys())
ex_pred3 = np.e**(model.predict(ex3))
ex3['예측값'] = ex_pred3
print(ex3)
print(ex3[ex3['예측값'] == ex3['예측값'].max()])
print(ex3[ex3['예측값'] == ex3['예측값'].min()])

ex3_sorted = ex3.sort_values(by='예측값', ascending = False)
print(ex3_sorted.head(n=10)) #매출금액합 best10
print(ex3_sorted.tail(n=10)) #worst10