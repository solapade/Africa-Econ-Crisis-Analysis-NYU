import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import re



data = pd.read_csv("africa-economic-banking-and-systemic-crisis-data/african_crises.csv").dropna()
data = data.drop(labels = ["case","cc3"],axis=1)
crisisdata = (data.loc[data['systemic_crisis'] == 1
 |  (data['currency_crises'] == 1)
  | (data['inflation_crises'] == 1 )
   | (data['banking_crisis']=='crisis')])
# print(type(data),type(crisisdata))
# crisisdata = crisisdata.drop(labels = ["case","cc3"],axis=1)
noncrisis =data[~data.index.isin(crisisdata.index)]
# Select the ones you want
crisisData = crisisdata[['exch_usd','inflation_annual_cpi']]
crisisExc = crisisdata['exch_usd']
crisisInf = crisisdata['inflation_annual_cpi']
crisisData['_logarithm_base10'] = np.log10(crisisData['inflation_annual_cpi'])
crisisData['__logarithm_base10'] = -1 * (np.log10(-1 * crisisData['inflation_annual_cpi']))
crisisData['Inflation_logarithm_base10'] = crisisData['_logarithm_base10'].astype(str) + crisisData['__logarithm_base10'].astype(str)
crisisData['Inflation_logarithm_base10'] = crisisData['_logarithm_base10'].astype(str) + crisisData['__logarithm_base10'].astype(str)
crisisData = crisisData.replace(to_replace ='nan', value = '', regex = True)
crisisData = crisisData.drop(labels = ["_logarithm_base10","__logarithm_base10"],axis=1)
crisisData['Inflation_logarithm_base10'] = pd.to_numeric(crisisData['Inflation_logarithm_base10'],errors='coerce')


noncrisis = noncrisis[['exch_usd','inflation_annual_cpi']]
noncrisisExch = noncrisis['exch_usd']
noncrisisInfl = noncrisis['inflation_annual_cpi']
noncrisis['_logarithm_base10'] = np.log10(noncrisis['inflation_annual_cpi'])
noncrisis['__logarithm_base10'] = -1 * (np.log10(-1 * noncrisis['inflation_annual_cpi']))
noncrisis['Inflation_logarithm_base10'] = noncrisis['_logarithm_base10'].astype(str) + noncrisis['__logarithm_base10'].astype(str)
noncrisis['Inflation_logarithm_base10'] = noncrisis['_logarithm_base10'].astype(str) + noncrisis['__logarithm_base10'].astype(str)
noncrisis = noncrisis.replace(to_replace ='nan', value = '', regex = True)
noncrisis = noncrisis.drop(labels = ["_logarithm_base10","__logarithm_base10"],axis=1)
noncrisis['Inflation_logarithm_base10'] = pd.to_numeric(noncrisis['Inflation_logarithm_base10'],errors='coerce')

# noncrisis['logarithm_base10'] = noncrisis['__logarithm_base10'].astype(str) + noncrisis['_logarithm_base10'].astype(str)

# re.sub('nan', '', str)


# crisisData.to_csv('crisisData.csv')
# noncrisis.to_csv('noncrisisData.csv')

# crisisdata1inf = crisisdata1inf.transpose()
# plot0 = crisisdata1inf.plot()
#plot0.title("crises values")
# plot0.set_xlabel("Index")
# plot0.set_ylabel("Inflation")

# crisisPlot = crisisInf.plot(title="Crises values")
# crisisPlot.set_xlabel("Index")
# crisisPlot.set_ylabel("Inflation")
# plt.show()

# logcrisisInf = crisisData['Inflation_logarithm_base10']
# logcrisisPlot = logcrisisInf.plot(title="Log Crises values")
# logcrisisPlot.set_xlabel("Index")
# logcrisisPlot.set_ylabel("Inflation")
# plt.show()

logNoncrisisInf = noncrisis['Inflation_logarithm_base10']
logNoncrisisPlot = logNoncrisisInf.plot(title="Log NonCrises values")
logNoncrisisPlot.set_xlabel("Index")
logNoncrisisPlot.set_ylabel("Inflation")
plt.show()

# noncrisisPlot = noncrisisInfl.plot(title="NonCrises values")
# noncrisisPlot.set_xlabel("Index")
# noncrisisPlot.set_ylabel("Inflation")
# plt.show()


# crisisPlot = crisisExc.plot(title="Crises values")
# crisisPlot.set_xlabel("Index")
# crisisPlot.set_ylabel("Exchange to USD")
# plt.show()


# noncrisisPlot = noncrisisExch.plot(title="NonCrises values")
# noncrisisPlot.set_xlabel("Index")
# noncrisisPlot.set_ylabel("Exchange to USD")
# plt.show()


# crisisdata.to_csv('crisisdata.csv')
# non# crisis.to_csv('crisis.csv')
# crisisdata1inf.to_csv('crisisdata1inf.csv')



# print((data.loc[data['systemic_crisis'] == 1
#  |  (data['currency_crises'] == 1)
#   | (data['inflation_crises'] == 1 )
#    | (data['banking_crisis']=='crisis')]))

# print(crisisdata['exch_usd'].max())
# print(crisisdata['exch_usd'].min())
# print(crisisdata['inflation_annual_cpi'].max())
# print(crisisdata['inflation_annual_cpi'].min())

# print(noncrisis['exch_usd'].max())
# print(noncrisis['exch_usd'].min())
# print(noncrisis['inflation_annual_cpi'].max())
# print(noncrisis['inflation_annual_cpi'].min())
# 
# mine.plot()
# data.plot()
# scaler = MinMaxScaler()
# print(fit(X)[data["exch_usd"]])
# print(scaler.fit(data["exch_usd"]))
# print(scaler.data_max_)
# data.head()

# data.info()
# data.to_csv('test.csv')
# sns.heatmap(data.corr(),cbar=True,annot=True,cmap='Blues')


# data = data.groupby("country")
# sns.pairplot(data)
# correlation = data.corr()
# data.plot()
# data.plot.scatter(x="year", y="inflation_annual_cpi")

# plt.show() 

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
