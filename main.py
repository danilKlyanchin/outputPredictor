import pandas as pd
from prophet import Prophet
import json


df = pd.read_csv('dz_data.csv')
df['дата'] = df['дата'].astype(str)

data = list(map(lambda x: '.'.join(x.split(',')), df['выход'].values))
data = list(map(float, data))
df['выход'] = data

df = df[::-1]
df.index = pd.RangeIndex(len(df))

normal_dates = []
for date in df['дата'].values:
    d,m,y = date.strip().split('.')
    normal_dates.append(f'{y}-{m}-{d}')
df['дата'] = normal_dates

cur_df = df[['дата', 'выход']]
cur_df = cur_df.rename(columns={'дата': 'ds', 'выход': 'y'})

predict_df = pd.read_csv('predict.csv')

# Prophet
model = Prophet()
model.fit(cur_df)

future = model.make_future_dataframe(periods=len(predict_df))
forecast = model.predict(future)

# My predict
predict_out = forecast['yhat'][-len(predict_df):].values


data1 = list(map(lambda x: round(x, 3), predict_out))
data2 = [0] * len(predict_df)

with open('forecast_value.json', 'w') as file:
    json.dump(data1, file)

with open('forecast_class.json', 'w') as file:
    json.dump(data2, file)
