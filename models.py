from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

#Separating data for validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#First, the Decision Tree. Using Regressor, since the objective here is to predict prices.

tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(r2_score(y_test, y_pred))
y_pred_new = tree.predict(X_new)
print("R² score for Decision Tree: {}".format(r2_score(y_new, y_pred_new)))

#Now, the RandomForest

forest = RandomForestRegressor()
forest.fit(X_train, y_train)
y_pred2 = forest.predict(X_test)
print(r2_score(y_test, y_pred2))
y_pred_new2 = forest.predict(X_new)
print("R² score for untuned Random Forest: {}".format(r2_score(y_new, y_pred_new2)))

#Now, time for hyperparameter tuning for the Random Forest. No hyperparameter tuning for Decision Tree since I don't know how to do that.

params = {'n_estimators': [100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]}
CV_forest = RandomizedSearchCV(estimator=forest, param_distributions=params, n_iter=10, cv=10)

#More parameters need to be tested. But let's just leave it like this for simplification

CV_forest.fit(X_train, y_train)
best_forest_pred = CV_forest.best_estimator_.predict(X_test)
print("Best score predicted: " + str(r2_score(y_test, best_forest_pred)))
print(CV_forest.best_estimator_)
print(CV_forest.best_params_)
print(CV_forest.best_score_)

#Best parameter n_estimators = 400

forest = RandomForestRegressor(n_estimators=400)
forest.fit(X_train, y_train)
y_pred_tuned = forest.predict(X_test)
print("R² score for tuned Random Forest: " + str(r2_score(y_test, y_pred_tuned)), "\nMSE for tuned Random Forest: " + str(mean_squared_error(y_test, y_pred_tuned)))

#Time for the XGBoost

bst = xgb.XGBRegressor()
bst.fit(X_train, y_train)
bst_pred = bst.predict(X_test)
print("Untuned model score:" + str(r2_score(y_test, bst_pred)))

#Hyperparameter tuning for XGBoost

params = {'n_estimators': [310, 315, 320, 325, 330, 335, 340, 345], 'learning_rate': [0.32], 'max_depth': [9]}
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

#Interestingly enough, the Random Forest Regressor has a better performance.
