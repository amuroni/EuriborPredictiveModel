from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs
from Main import df

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
fig, axes = plt.subplots(4, 2)

# Original Series
axes[0, 0].plot(df.Euribor)
axes[0, 0].set_title('Original Series')
plot_acf(df.Euribor, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.Euribor.diff())
axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.Euribor.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.Euribor.diff().diff())
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.Euribor.diff().diff().dropna(), ax=axes[2, 1])

# 3rd Differencing
axes[3, 0].plot(df.Euribor.diff().diff().diff())
axes[3, 0].set_title('3rd Order Differencing')
plot_acf(df.Euribor.diff().diff().diff().dropna(), ax=axes[3, 1])

plt.show()

# Final tests to better choose a D parameter.
y = df.Euribor

# Adf Test
print("ADF test result %f" % ndiffs(y, test='adf'))  # result 1

# KPSS test
print("KPSS test result %f" % ndiffs(y, test='kpss'))  # result 1

# PP test:
print("PP test result %f" % ndiffs(y, test='pp'))  # result 1

# The correct D parameter for my Euribor Series is therefore = 1
# Thus the Q parameter is also 1, given the 2nd order differencing


