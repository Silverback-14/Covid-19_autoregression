# -*- coding: utf-8 -*-
"""




"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
data=pd.read_csv("D://daily_covid_cases.csv")


#1

#a
fig, ax = plt.subplots()
ax.plot(data["Date"],data["new_cases"])
loc = plticker.MultipleLocator(base=60.0)
ax.xaxis.set_major_locator(loc)
plt.xticks(rotation=60)
plt.xlabel("day")
plt.ylabel("number of covid-19 cases")
plt.title("Autocorrelation plot for number of covid cases per day in India")
plt.show()


#b
l=sm.tsa.acf(data["new_cases"],nlags=1)
print("Pearson correlation coefficient with 1 day lag is ",l[1])
data["lag_1"]= data.new_cases.shift(1)

#c
correlation = data["lag_1"].corr(data["new_cases"])
print("The correlation between the  two data frames : ",correlation)
plt.scatter(data["new_cases"],data["lag_1"])
plt.xlabel("Given time sequence")
plt.ylabel("1 day lagged time sequence")
plt.title("Scatter plot  between actual data and lagged data")
plt.show()

#d
lagval = range(1,7)
corrcoef = []
for i in lagval:
    corrcoef.append(data["new_cases"].autocorr(lag=i))
plt.plot(lagval, corrcoef)
plt.xlabel("number days lagged")
plt.ylabel("correlation coefficient")
plt.title("Line plot between correlation coeficients and number of days lagged ")
plt.show()

#e
data = data[['Date', 'new_cases']].set_index(['Date'])
plot_acf(data,lags=50)
plt.xlabel("number days lagged")
plt.ylabel("correlation coefficient")
plt.show()

# 2
test_size = 0.35 
X = data.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

plt.plot(train,label="train")
plt.plot(test,label="test")
plt.xlabel("number of days")
plt.ylabel("number of cases")
plt.title("plot od train and test data")
plt.legend()
plt.show()

window = 5
model = AR(train, lags=5)
model_fit = model.fit()
coef = model_fit.params
print(model_fit.summary())

history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
rmse = (math.sqrt(mean_squared_error(test, predictions))/np.mean(test))*100
mape=mean_absolute_percentage_error(test, predictions)
print("Mean absolute percentage error : ",mape)
print('Test RMSE: ' ,rmse)

plt.plot(test,label="actual")
plt.plot(predictions,label="predicted")
plt.legend()
plt.xlabel("actual data")
plt.ylabel("predicted data")
plt.title("Line plot ")
plt.show()
plt.scatter(test,predictions)
plt.xlabel("actual data")
plt.ylabel("predicted data")
plt.title("scatter plot ")
plt.show()

#3

test_size = 0.35 
X = data.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
rm=[]
mp=[]
l=[1,5,10,15,25]
for i in l:
    window = i
    model = AR(train, lags=i)
    model_fit = model.fit()
    coef = model_fit.params
    
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
    	length = len(history)
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	obs = test[t]
    	predictions.append(yhat)
    	history.append(obs)
    rmspe = (math.sqrt(mean_squared_error(test, predictions))/np.mean(test))*100
    rm.append(rmspe)
    mape = np.mean(np.abs((test - predictions)/test))*100
    mp.append(mape)


dict = {'Lag values ': l, 'RMSE': rm, 'MAPE': mp} 
df = pd.DataFrame(dict)
print(df)


plt.bar(l,rm)
plt.xlabel("Lag values")
plt.ylabel("Rmse for corresponding lag")
plt.title("Bar graph between RMSE and Lag values ")
plt.xticks(l)
plt.show()

plt.bar(l,mp)
plt.xlabel("Lag values")
plt.ylabel("Mean absolute percentage error for corresponding lag")
plt.title("Bar graph between MAPE and Lag values ")
plt.xticks(l)
plt.show()

#4
test_size = 0.35 
X = data.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

window= 1
while window< len(data):
  corr = pearsonr(train[window:].ravel(), train[:len(train)-window].ravel())
  if(abs(corr[0]) <= 2/math.sqrt(len(train[window:]))):
    print('optimal number of lags :',window-1)
    break
  window+=1

window=window-1

model = AR(train, lags=window)

model_fit = model.fit()
coef = model_fit.params 
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() 
for t in range(len(test)):
  length = len(history)
  Lag = [history[i] for i in range(length-window,length)] 
  yhat = coef[0] 
  for d in range(window):
    yhat += coef[d+1] * Lag[window-d-1] 
  obs = test[t]
  predictions.append(yhat) 
  history.append(obs) 
rmspe = (math.sqrt(mean_squared_error(test, predictions))/np.mean(test))*100
mape = np.mean(np.abs((test - predictions)/test))*100
print('RMSE :',rmspe)
print('MAPE:',mape)
