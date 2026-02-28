import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\Happy Home\OneDrive\Desktop\Inhouse projects\minor_project\archive\USA_GA_Albany-Dougherty.County.AP.722160_TMY3_BASE.csv"
)


df['Date/Time'] = df['Date/Time'].str.strip()

is_24 = df['Date/Time'].str.contains('24:00:00')
df.loc[is_24, 'Date/Time'] = df.loc[is_24, 'Date/Time'].str.replace(
    '24:00:00', '00:00:00', regex=False
)

df['Date/Time'] = df['Date/Time'].str.replace(r'\s+', ' ', regex=True)

df['datetime'] = pd.to_datetime(
    df['Date/Time'],
    format='%m/%d %H:%M:%S',
    errors='raise'
)

df.loc[is_24, 'datetime'] += pd.Timedelta(days=1)

df.set_index('datetime', inplace=True)

ts = df['Electricity:Facility [kW](Hourly)']

plt.figure()
plt.plot(ts)
plt.title("Raw Time Series")
plt.xlabel("Time")
plt.ylabel("Electricity KW")
plt.show()

trend = ts.rolling(window=24*30).mean()  

plt.figure()
plt.plot(ts, alpha=0.4, label="Original")
plt.plot(trend, label="Trend")
plt.legend()
plt.title("Trend using Moving Average")
plt.show()

daily_seasonality = ts.groupby(ts.index.hour).mean() #training and val plot separate

plt.figure()
plt.plot(daily_seasonality)
plt.title("Daily Seasonality (Hourly Pattern)")
plt.xlabel("Hour of Day")
plt.ylabel("Electricity KW")
plt.show()

monthly_seasonality = ts.groupby(ts.index.month).mean()

plt.figure()
plt.plot(monthly_seasonality)
plt.title("Yearly Seasonality (Monthly Pattern)")
plt.xlabel("Month")
plt.ylabel("Electricity KW")
plt.show()

detrended = ts - trend

plt.figure()
plt.plot(detrended)
plt.title("Randomness (After Removing Trend)")
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts.dropna(), model='additive', period=24)

decomposition.plot()
plt.show()
