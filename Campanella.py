import pandas as pd
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import os

# -----
MaxN = 1148
# -----
# Read file
data = pd.read_excel("D:\Python\Hello\CPT 34 to 68 - 1cm intervals.xlsx", sheet_name="CPT-68", usecols=[0, 1],
                     engine='openpyxl')
filename = os.path.basename("D:\Python\Hello\CPT 34 to 68 - 1cm intervals.xlsx")
X = data["D"]
Y = data["C"]

# Teta
teta = 0.5
# ========

n = 0
m = n + 1
MySum = 0
Predicted = list()
Predicted.append(1)
for Lag in X:
    MySum = X[m] - X[n] + MySum
    Predicted.append(math.exp(-2 * MySum / teta))
    n = n + 1
    m = n + 1
    if m == len(X):
        break
del Predicted[40:]


LagAverage = MySum / m
print("LagAverage/n")
print(LagAverage)

X = X.values.reshape(-1, 1)
Y = Y.values.reshape(-1, 1)
model = LinearRegression().fit(X, Y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

for myvalue in X:
    detrendY = Y - model.coef_ * X - model.intercept_


print("====")
b = np.std(detrendY)

print(b)
print("====")

for myvalue in detrendY:
    detrendYNormalized = detrendY / b

StanddetrendYNormalized = np.std(detrendYNormalized)

print("===")
RAverage = list()

n = 0
m = 0

StandVector = list()
FLUCList = list()
Taw = list()
for Window_Var in range(1, MaxN):
    df = pd.DataFrame(detrendYNormalized)
    df = df.rolling(window=Window_Var).mean()
    StandVector.append(np.std(df))
    Taw.append(LagAverage * Window_Var)
    FLUCList.append(pow(np.std(df) / StanddetrendYNormalized, 2) * LagAverage * Window_Var)


print("^^^")
print(Taw)

fig = plt.figure()
fig.text(.5, 0.9, filename)
plt.plot(Taw, FLUCList)


print("****")

print(np.max(FLUCList))
plt.show()
