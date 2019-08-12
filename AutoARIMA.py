import pmdarima as pm
import matplotlib.pyplot as plt
from Main import df

# auto modeling to get the more precise model possible
model = pm.auto_arima(df.Euribor, start_p=1, start_q=1,
                      test='adf',        # use adftest to find optimal 'd'
                      max_p=4, max_q=4,  # maximum p and q
                      m=1,               # frequency of series
                      d=None,            # let model determine 'd'
                      seasonal=False,    # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(model.summary())

# then analyse visually the results by plotting diagnostics
model.plot_diagnostics()
plt.show()
