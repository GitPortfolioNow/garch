import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from arch import arch_model

df = pd.read_csv(r'C:\Users\simon\Documents\Derivatives\volatility targetting\Garch Analysis\nasdaq_2009.csv')
df['Price_Lag'] = df['Price'].shift(1)
df.dropna(inplace=True)
df['return'] = (df['Price']/df["Price_Lag"]-1)*100
df['realized variance'] = df['return'].rolling(window=200).var()
return_array = df['return'].to_numpy()
std = np.std(return_array)  # Storing the standard deviation of the array 
mean = np.mean(return_array)  # Storing the mean of the array 
  
plt.boxplot(return_array) 
plt.title('Array with Outliers') 
plt.show() 
print(mean)
print(std)

WinsorizedArray = winsorize(return_array,(0.04,0.04)) 
  
plt.boxplot(WinsorizedArray) 
plt.title('Winsorized array') 
plt.show()
WinsorizedMean = np.mean(WinsorizedArray)
WinsorizedStd = np.std(WinsorizedArray)
print(WinsorizedMean)
print(WinsorizedStd)
print(df)

definedGM = arch_model(WinsorizedArray[-200:], p = 1, q = 1, 
                      mean = 'constant', vol = 'GARCH', dist = 'skewt')
fittedGM = definedGM.fit()
gmForecast = fittedGM.forecast(horizon = 1)
print(gmForecast.variance[-1:])

#simulation
for index,row in df.iterrows():
    if index == 199:
        df.at[index,'shares'] = 0
        df.at[index,'cash'] = 1000000
        df.at[index,'equity'] = df.at[index,'shares'] + df.at[index,'cash']
    if index > 199:
        return_array = df.iloc[index-50: index,3].to_numpy()
        WinsorizedArray = winsorize(return_array,(0.04,0.04))
        definedGM = arch_model(WinsorizedArray[-46:], p = 1, q = 1, 
                      mean = 'constant', vol = 'GARCH', dist = 'skewt')
        fittedGM = definedGM.fit()
        gmForecast = fittedGM.forecast(horizon = 1)

        df.at[index,'forecasted variance'] = float(gmForecast.variance[-1:]['h.1'])

        df.at[index,'leverage'] = 1.5*df.at[index,'realized variance']/df.at[index,'forecasted variance'] #target the long term variance
        #df.at[index,'leverage'] = 6.3492/df.at[index,'forecasted variance'] #target annualized 40% volatility

        if df.at[index,'leverage'] > 2:
             df.at[index,'leverage'] = 2
        # if df.at[index,'leverage'] > 15:
        #    df.at[index,'leverage'] = 15
        
        #df.at[index,'leverage'] = 1.5/df.at[index,'forecasted variance']
        df.at[index,'equity'] = df.at[index-1,'shares']*df.at[index,'Price'] + df.at[index-1,'cash']
        df.at[index,'shares'] = df.at[index,'leverage']*df.at[index,'equity'] / df.at[index,'Price']
        df.at[index,'cash'] = df.at[index-1,'cash'] - (df.at[index,'shares'] - df.at[index-1,'shares'])*df.at[index,'Price']

df.to_csv('df.csv')
print("done")
