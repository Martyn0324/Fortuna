import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

#Loading data and checking it out

data = pd.read_csv(r'Cotação Histórica BTCUSD.csv')
print(data.tail())

data["Date"] = pd.to_datetime(data["Date"])
data = data.dropna()

#Plotting the data, just for fun and to see how it works

fig, ax = plt.subplots()
data.plot(x="Date", y="Close", ax=ax)
plt.title("Historical BTC price (in USD)")
plt.show()

#Loading models. DecisionTree and Random Forests because I like them

tree = DecisionTreeRegressor()
forest = RandomForestRegressor()
scaler = StandardScaler()

#Now, the samples and features

X = data.drop(["Date", "Close"], axis=1)
y = data["Close"]
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Loading new data, to make sure the models won't overfit

new_data = pd.read_csv(r'Cotação Histórica BTCUSD 2.csv')
new_data = new_data.dropna()
X_new = new_data.drop(['Date', 'Close'], axis=1)
y_new = new_data['Close']

#Scaling the data, just to make sure

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_new_scaled = scaler.transform(X_new)

#Now, the magic begins

tree.fit(X_train_scaled, y_train)
y_pred = tree.predict(X_test_scaled)
print(r2_score(y_test, y_pred))
y_pred_new = tree.predict(X_new_scaled)
print("R² score for scaled data: " + str(r2_score(y_new, y_pred_new)))

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(r2_score(y_test, y_pred))
y_pred_new = tree.predict(X_new_scaled)
print("R² score for unscaled data: " + str(r2_score(y_new, y_pred_new)))

forest = RandomForestRegressor()
forest.fit(X_train_scaled, y_train)
y_pred2 = forest.predict(X_test_scaled)
print(r2_score(y_test, y_pred2))
y_pred_new2 = forest.predict(X_new_scaled)
print("R² score for scaled data: " + str(r2_score(y_new, y_pred_new2)))

forest.fit(X_train, y_train)
y_pred2 = forest.predict(X_test)
print(r2_score(y_test, y_pred2))
y_pred_new2 = forest.predict(X_new)
print("R² score for unscaled data: " + str(r2_score(y_new, y_pred_new2)))

#Scaling is irrelevant ---> Tree models


Time for hyperparameter tuning

params = {'n_estimators': np.arange(100,1000, 50)}
CV_forest = RandomizedSearchCV(estimator=forest, param_distributions=params, n_iter=10, cv=10)

CV_forest.fit(X_train, y_train)
best_forest_pred = CV_forest.best_estimator_.predict(X_test)
print("Best score predicted: " + str(r2_score(y_test, best_forest_pred)))
print(CV_forest.best_estimator_)
print(CV_forest.best_params_)
print(CV_forest.best_score_)

#Best hyperparameter n_estimators = 400

forest = RandomForestRegressor(n_estimators=400)
forest.fit(X_train, y_train)
y_pred_tuned = forest.predict(X_test)
print("Tuned Random Forest R²: " + str(r2_score(y_test, y_pred_tuned)), "MSE: " + str(mean_squared_error(y_test, y_pred_tuned)))


#Now, the mighty one, XGBooster

bst = xgb.XGBRegressor()
bst.fit(X_train, y_train)
bst_pred = bst.predict(X_test)
print("Untuned XGB R² score:" + str(r2_score(y_test, bst_pred)))

bst_trial = bst.predict(X_new)
print(r2_score(y_new, bst_trial))

#Let's tune it

params = {'n_estimators': np.arange(310,345, 5), 'learning_rate': np.arange(0.3, 0.5, 0.01), 'max_depth': np.arange(3, 10, 1)}
CV_xgb = RandomizedSearchCV(bst, param_distributions=params, n_iter = 10, cv=10)
CV_xgb.fit(X_train, y_train)
best_xgb_pred = CV_xgb.best_estimator_.predict(X_test)
print("Best score predicted: " + str(r2_score(y_test, best_xgb_pred)))
print(CV_xgb.best_estimator_)
print(CV_xgb.best_params_)

#best score = 0.9994 with dart
#best score = 0.9994 with gbtree

bst_tuned = xgb.XGBRegressor(n_estimators=345, learning_rate=0.32, max_depth=9)
bst_tuned.fit(X_train, y_train)
bst_tuned_pred = bst_tuned.predict(X_test)
print('Tuned forest MSE: ' + str(mean_squared_error(y_test, y_pred_tuned)), 'Tuned XGB MSE: ' + str(mean_squared_error(y_test, bst_tuned_pred)))


#Plotting the real data and predicted data in order to visually check accuracy

fig, ax = plt.subplots(figsize=(15,5))
ax.plot(y_test, color='k', lw=3)
ax.plot(y_pred, color='r', lw=2)
ax.plot(y_pred2, color='b', lw=1)
ax.plot(pred_new, color='c', lw=5)
ax.plot(pred_new2, color='g', lw=6)
ax.plot(y_new, color='g')
ax.plot(bst_trial, color='m')
ax.legend()
plt.show()

#Insert here today's data in order to predict today's close price. Since the tuned Random Forest had a better performance, we'll be using it

TRIAL = pd.DataFrame({'Open': 35723.18,'High': 37181.09, 'Low':33705.22, 'Adj Close': 34084.70, 'Volume': 47158423552}, index=['Teste'])
print(TRIAL)
TRIAL_BY_FIRE = forest.predict(TRIAL)
print(TRIAL_BY_FIRE)
