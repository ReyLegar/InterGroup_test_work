from datetime import datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


df = pd.read_csv('data.csv')
df = df.replace('\s+', '', regex=True)

df = df.sort_values(by=['Day', 'Month', 'Year'], ignore_index=True)

#добавление shift чтобы регрессия основывалась на прошлых значениях цены актива
df['lag1'] = df['Close'].shift(periods=1)
df['lag2'] = df['Close'].shift(periods=2)
df['lag3'] = df['Close'].shift(periods=3)
df['lag4'] = df['Close'].shift(periods=4)
df['lag5'] = df['Close'].shift(periods=5)

df['lag1'].fillna(df['lag1'].mean(), inplace=True)
df['lag2'].fillna(df['lag2'].mean(), inplace=True)
df['lag3'].fillna(df['lag3'].mean(), inplace=True)
df['lag4'].fillna(df['lag4'].mean(), inplace=True)
df['lag5'].fillna(df['lag5'].mean(), inplace=True)

X = df.drop('Close', axis=1)
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #разделение данных

model = Sequential()

model.add(LSTM(11, input_shape=[None, 1], activation="selu", return_sequences=False))


model.add(Dense(1))

model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, batch_size = 30, epochs=10, validation_data=(X_test,y_test))

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)

last = df.iloc[-1]
data = {'Year': int(last['Year']), 'Month': int(last['Month']), 'Day': int(last['Day']), 'Hour': int(last['Hour']), 'Minute': int(last['Minute']), 'Close': last['Close'], 'lag1': last['lag1'], 'lag2': last['lag2'], 'lag3': last['lag3'], 'lag4': last['lag4'], 'lag5': last['lag5']}
date_time = datetime(int(last['Year']), int(last['Month']), int(last['Day']), int(last['Hour']), int(last['Minute']))
new_df = pd.DataFrame(data, index=[0])
for i in range(1, 61):
    date_time_new = date_time + timedelta(minutes=i)
    date_object = datetime.strptime(str(date_time_new), "%Y-%m-%d %H:%M:%S")

    pred = model.predict(new_df.loc[[i - 1]], verbose=0)

    data = {
        'Year': int(date_object.year), 
            'Month': int(date_object.month), 
            'Day': int(date_object.day), 
            'Hour': int(date_object.hour), 
            'Minute': int(date_object.minute), 
            'Close': pred[0][0], 
            'lag1': new_df.loc[i-1, 'Close'], 
            'lag2': new_df.loc[i-1, 'lag1'], 
            'lag3': new_df.loc[i-1, 'lag2'], 
            'lag4': new_df.loc[i-1, 'lag3'],
            'lag5': new_df.loc[i-1, 'lag4']
            }
    
    new_df = new_df.append(data, ignore_index=True)

close = new_df['Close'].to_list()

minute_list = [i + 1 for i in range(len(close))]

plt.plot(minute_list, close)
plt.show()