import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

data = pd.read_csv(r'Cotação Histórica BTCUSD.csv')
#print(data.tail())
data2 = pd.read_csv(r'Cotação Histórica ETHUSD.csv')

data["Date"] = pd.to_datetime(data["Date"])
data = data.dropna()

#fig, ax = plt.subplots()
#data.plot(x="Date", y="Close", ax=ax)
#data2.plot(x="Date", y="Close", ax=ax)
#plt.title("Historical BTC and ETH price (in USD)")
#plt.show()

tree = DecisionTreeRegressor()
scaler = StandardScaler()
X = data.drop(["Date", "Close"], axis=1)
y = data["Close"]
#print(X)

X_scaled = scaler.fit_transform(X)
#print(X_scaled)

new_data = pd.read_csv(r'Cotação Histórica BTCUSD 2.csv')
new_data = new_data.dropna()
X_new = new_data.drop(['Date', 'Close'], axis=1)
y_new = new_data['Close']

X_new_scaled = scaler.fit_transform(X_new)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(r2_score(y_test, y_pred))
y_pred_new = tree.predict(X_new_scaled)
print(r2_score(y_new, y_pred_new))

forest = RandomForestRegressor()
forest.fit(X_train, y_train)
y_pred2 = forest.predict(X_test)
print(r2_score(y_test, y_pred2))
y_pred_new2 = forest.predict(X_new_scaled)
print(r2_score(y_new, y_pred_new2))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.25)
tree.fit(X_train2, y_train2)
y_pred3 = tree.predict(X_test2)
print(r2_score(y_test2, y_pred3))
y_pred_new3 = tree.predict(X_new)
print(r2_score(y_new, y_pred_new3))

forest.fit(X_train2, y_train2)
y_pred4 = forest.predict(X_test2)
print(r2_score(y_test2, y_pred4))
y_pred_new4 = forest.predict(X_new)
print(r2_score(y_new, y_pred_new4))


bst = xgb.XGBRegressor()
bst.fit(X_train2, y_train2)
bst_pred = bst.predict(X_test2)
print(r2_score(y_test2, bst_pred))

bst_trial = bst.predict(X_new)
print(r2_score(y_new, bst_trial))

print(y_new.tail())
df = pd.DataFrame(bst_trial, index=y_new)
print(df)

#bst.fit(X_train, y_train)
#bst_pred2 = bst.predict(X_test)
#print(r2_score(y_test, bst_pred2))

#bst_trial2 = bst.predict(X_new_scaled)
#print(r2_score(y_new, bst_trial2))


fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(y_test, color='k', lw=3)
#ax.plot(y_pred, color='r', lw=2)
#ax.plot(y_pred2, color='b', lw=1)
#ax.plot(y_pred3, color='m', lw=0)
#ax.plot(y_pred4, color='aqua', lw=4)
#ax.plot(pred_new, color='c', lw=5)
#ax.plot(pred_new2, color='g', lw=6)
ax.plot(y_new, color='g')
ax.plot(bst_trial, color='m')
ax.legend()
#plt.show()

TRIAL = pd.DataFrame({'Open': 35723.18,'High': 37181.09, 'Low':33705.22, 'Adj Close': 34084.70, 'Volume': 47158423552}, index=['Teste'])
print(TRIAL)
TRIAL_BY_FIRE = bst.predict(TRIAL)
print(TRIAL_BY_FIRE)
