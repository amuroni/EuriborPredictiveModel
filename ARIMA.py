from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import acf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
from Main import df

# testing P, D and Q values in the ARIMA model
model = ARIMA(df.Euribor, order=(1, 1, 1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# P value for ar1 and ma1 is less than 0.05 so highly significant, that's good.

# residuals analysis
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# near zero means, very uniform variance, great.

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()

# P, D and Q values chosen are a good fit for this model
# now we can validate the model prediction capabilities

# setting train and test series
y = 0.75
x = int(len(df.Euribor) * y)
train = df.Euribor[:x]  # x% of my time series
test = df.Euribor[x:]  # remaining % of my time series

# building the model for train/test
model = ARIMA(train, order=(1, 2, 0))  # changing parameters values to get better accuracy metrics
fitted = model.fit(disp=-1)
forecast_series = int(len(df.Euribor)*(1-y)+1)

# forecast with 95% conf value (P 0.05)
fc, se, conf = fitted.forecast(forecast_series, alpha=0.05)

# all series are Pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# plot to visualize results
bottom, top = ylim()
plt.figure(figsize=(12, 5), dpi=120)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.ylim(bottom=-20, top=20)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
print(fitted.summary())

# Accuracy metrics


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE, very important and between 1 and 0
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]   # corr, very important and between 1 and 0
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax , very important and between 1 and 0
    acf1 = acf(fc-test)[1]                      # ACF1
    print({'mape': mape, 'me': me, 'mae': mae,
            'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
            'corr': corr, 'minmax': minmax})


forecast_accuracy(fc, test.values)
