import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your data
features = pd.read_csv("C:\\Users\\alich\\Downloads\\hsieh_lab\\real-time predictor of network bursts\\Output\\420_cn_I15918_0001.csv")

X = features[:-1]
y = features.sum(axis=1).shift(-1).dropna()

X = X.iloc[:-1]

model = LinearRegression()
model.fit(X, y)

print("R²:", model.score(X, y))