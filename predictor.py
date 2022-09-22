import pandas as pd
import matplotlib.pyplot as plt
db = pd.read_csv('train-stocks.csv')
print(db.head())

db = db [['Date','Open','Close']]
db['Date'] = pd.to_datetime(db['Date'].apply(lambda x: x.split()[0]))
db.set_index('Date',drop=True,inplace=True)
print(db.head())

fg, ax =plt.subplots(1,2,figsize=(15,7))
ax[0].plot(db ['Open'],label='Open',color='green')
ax[0].set_xlabel('Date',size=15)
ax[0].set_ylabel('Price',size=15)
ax[0].legend()
ax[1].plot(db ['Close'],label='Close',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Price',size=15)
ax[1].legend()
plt.show()