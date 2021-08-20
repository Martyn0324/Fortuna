import models
import pandas as pd

#Remember: No commas. Also: Adj Close = The price right now.

TRIAL = pd.DataFrame({'Open': 5675,'High': 5700, 'Low':5605, 'Adj Close': 5688, 'Volume': 39541620736}, index=['Test'])

Trial_by_fire = forest.predict(TRIAL)
print(Trial_by_fire)
