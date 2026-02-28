import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(
    r"C:\Users\Happy Home\OneDrive\Desktop\Inhouse projects\minor_project\archive\USA_GA_Albany-Dougherty.County.AP.722160_TMY3_BASE.csv"
)

ts = df.select_dtypes(include=[np.number]).iloc[:, 0]

split = int(0.8 * len(ts))
train, test = ts[:split], ts[split:]

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape-50

ar_model = ARIMA(train, order=(5, 0, 0))
ar_fit = ar_model.fit()

ar_pred = ar_fit.forecast(steps=len(test))

ar_mae, ar_rmse, ar_mape = metrics(test, ar_pred)

print("AR Model Metrics")
print("MAE :", ar_mae)
print("RMSE:", ar_rmse)
print("MAPE:", ar_mape)
