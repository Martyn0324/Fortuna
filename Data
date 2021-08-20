#Loading the data

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'Cotação Histórica BTCUSD.csv')

data["Date"] = pd.to_datetime(data["Date"])
data = data.dropna()
X = data.drop(["Date", "Close"], axis=1)
y = data["Close"]

#Plotting the data, just for fun

fig, ax = plt.subplots()
data.plot(x="Date", y="Close", ax=ax)
plt.title("Historical BTC price (in USD)")
plt.show()

#Loading new data for validation purposes

new_data = pd.read_csv(r'Cotação Histórica BTCUSD 2.csv')
new_data = new_data.dropna()
X_new = new_data.drop(['Date', 'Close'], axis=1)
y_new = new_data['Close']

#Scaling the data - This is for educational purposes, since this part is irrelevant for the selected models.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.transform(y)

X_new_scaled = scaler.transform(X_new)
y_new_scaled = scaler.transform(y_new)
