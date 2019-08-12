import requests
from lxml import html
import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
# import numpy as np
from statsmodels.tsa.stattools import adfuller

# scrape Euribor data from ECB website
url = "http://sdw.ecb.europa.eu/quickview.do?SERIES_KEY=143.FM.M.U2.EUR.RT.MM.EURIBOR6MD_.HSTA"
page = requests.get(url)
tree = html.fromstring(page.content)
values = tree.xpath("//td")

# separate dates from Euribor rates and eliminate third column values (not useful)
dates_string = []
rates = []
c = 0
for i in range(61, len(values)):
    data = values[i].text
    c = c + 1
    if c == 3:
        c = 0
    elif c == 1:
        dates_string.append(data)
    else:
        rates.append(data)

# reformat dates for indexing
dates_index = []
for date in dates_string:
    x = parser.parse(date).date()
    dates_index.append(str(x.year) + "-" + str(x.month))

# invert list values (1994->2019)
dates_index.reverse()
rates.reverse()

# convert date str values to date
pd.to_datetime(dates_index, format='%Y-%m')
# convert Euribor rates into numeric values
pd.to_numeric(rates)

# create pandas dataframe with reformatted dates for index and Euribor monthly rates
# is indexing better with default integer or with the ISO format date? TBD

df = pd.DataFrame(data={"Dates": dates_index[257:], "Euribor": rates[257:]})  # reduced series size
df.Euribor = pd.to_numeric(df.Euribor)
ax = df.plot(kind="bar", x="Dates", y="Euribor")
ax.set_xticklabels([t if not i % 12 else "" for i, t in enumerate(ax.get_xticklabels())])  # one tick each year
plt.xticks(rotation=45)
plt.tight_layout()
plt.autoscale()
plt.show()

# testing the order of differencing (D)
result = adfuller(df.Euribor.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
