import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
db = pd.read_csv('train-stocks.csv')
test = pd.read_csv('test-stocks.csv')

db = db [['Date','Open','Close']]
test = test [['Date','Open','Close']]

db['Date'] = pd.to_datetime(db['Date'].apply(lambda x: x.split()[0]))
test['Date'] = pd.to_datetime(test['Date'].apply(lambda x: x.split()[0]))



db.set_index('Date',drop=True,inplace=True)
test.set_index('Date',drop=True,inplace=True)
from sklearn.preprocessing import MinMaxScaler
Ms = MinMaxScaler()
db[db.columns] = Ms.fit_transform(db)
training_size = round(len(db ) * 0.95)
train_data = db [:training_size]
test_data  = db [training_size:]

print(db.head())
print(test.head())


'''
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
'''


#test[test.columns] = Ms.fit_transform(test)
#print(db.head())


def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0
    for stop_idx in range(1,len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1
    return (np.array(sequences),np.array(labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

#print(test_label)

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1))
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

model.fit(train_seq, train_label, epochs=80,validation_data=(test_seq, test_label), verbose=1)
test_predicted = model.predict(test_seq)
test_inverse_predicted = Ms.inverse_transform(test_predicted)
# Merging actual and predicted data for better visualization
db_slic = pd.concat([db.iloc[-61:].copy(),pd.DataFrame(test_inverse_predicted,columns=['open_predicted','close_predicted'],index=db .iloc[-61:].index)], axis=1)
db_slic[['Open','Close']] = Ms.inverse_transform(db_slic[['Open','Close']])
db_slic.to_csv(r'pred_results.csv')
print(db_slic.head())
