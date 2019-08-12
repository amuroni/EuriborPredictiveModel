from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
from Main import df

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

fig, axes = plt.subplots(3, 2)

# 1st differencing
axes[0, 0].plot(df.Euribor.diff())
axes[0, 0].set_title('1st Differencing')
axes[0, 1].set(ylim=(0, 1.5))
plot_pacf(df.Euribor.diff().dropna(), ax=axes[0, 1])

# 2nd differencing
axes[1, 0].plot(df.Euribor.diff().diff())
axes[1, 0].set_title('2nd Differencing')
axes[1, 1].set(ylim=(0, 1.5))
plot_pacf(df.Euribor.diff().diff().dropna(), ax=axes[1, 1])

# 3rd differencing
axes[2, 0].plot(df.Euribor.diff().diff().diff())
axes[2, 0].set_title('3rd Differencing')
axes[2, 1].set(ylim=(0, 2))
plot_pacf(df.Euribor.diff().diff().diff().dropna(), ax=axes[2, 1])

plt.show()

# 2nd differencing, with value P = 1, could be considered the most conservative choice
# e.g. only one point above the blue region
