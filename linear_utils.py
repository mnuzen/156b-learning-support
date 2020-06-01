# !pip install statsmodels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import LinearRegression

from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.formula.api as smf
from scipy.stats import norm

"""Initialize cumulative cases"""

def linear(PREDS):
    url = 'https://raw.githubusercontent.com/mnuzen/156b-learning-support/master/data/us/covid/deaths.csv'
    df = pd.read_csv(url, header=0)
    cum_cases = df.iloc[:, 4:]
    countyFIPS = df.iloc[:, 0].to_numpy()
    cum_cases = cum_cases.to_numpy()

    all_zeros = [0 for i in range(14)]
    PREDS = []
    reg = LinearRegression()
    for i in range(len(cum_cases)):
        test = cum_cases[i, -14:]
        train = cum_cases[i, :-14]
        if train[-1] == 0: # no training
            PREDS.append(all_zeros)
        else:
            j = 0
            while train[j] == 0:
                j+=1
            j-=1
            y = train[j:]
            x = np.array(range(len(y))).reshape((len(y), 1))
            x_pred = np.array(range(len(y), len(y)+14)).reshape((14,1))
            reg.fit(x, y)
            pred = reg.predict(x_pred)
            PREDS.append(pred)

    PREDS = np.array(PREDS)
    #print(PREDS.shape)
    DAILY_PRED = np.zeros((3195,14))
    DAILY_PRED[:, 0] = np.subtract(PREDS[:,0], cum_cases[:, -15])
    for i in range(1, len(DAILY_PRED[0])):
        DAILY_PRED[:, i] = np.subtract(PREDS[:,i], PREDS[:, i-1])

    FINAL_PRED = []
    dates = np.loadtxt('dates.txt', dtype=np.str)
    # assume normal distribution
    for county in range(len(DAILY_PRED)):
        for date in range(len(DAILY_PRED[0])):
            mean = DAILY_PRED[county, date]
            std = max(1, mean)**(1/2)
            heading = dates[date] + '-' + str(countyFIPS[county])
            quantiles = np.linspace(norm.ppf(0.1, mean, std), norm.ppf(0.9, mean, std), 9)
            quantiles = quantiles.clip(0).tolist()
            quantiles.insert(0, heading)
            FINAL_PRED.append(quantiles)

    FINAL_PRED = np.array(FINAL_PRED)
    #print(FINAL_PRED.shape)
    #print(FINAL_PRED[9960])

    # predict the last two weeks
    all_zeros = [0 for i in range(14)]
    PREDS = []
    reg = LinearRegression()
    for i in range(len(cum_cases)):
        if cum_cases[i, -1] == 0: # no training
            PREDS.append(all_zeros)
        else:
            j = 0
            y = cum_cases[i]
            while y[j] == 0:
                j+=1
            j-=1
            y = y[j:]
            x = np.array(range(len(y))).reshape((len(y), 1))
            x_pred = np.array(range(len(y), len(y)+14)).reshape((14,1))
            reg.fit(x, y)
            pred = reg.predict(x_pred)
            PREDS.append(pred)

    PREDS = np.array(PREDS)
    print(PREDS.shape)
    DAILY_PRED = np.zeros((3195,14))
    DAILY_PRED[:, 0] = np.subtract(PREDS[:,0], cum_cases[:, -1])
    for i in range(1, len(DAILY_PRED[0])):
        DAILY_PRED[:, i] = np.subtract(PREDS[:,i], PREDS[:, i-1])

    FINAL_PRED = []
    dates = np.loadtxt('dates_final.txt', dtype=np.str)
    # assume normal distribution
    for county in range(len(DAILY_PRED)):
        for date in range(len(DAILY_PRED[0])):
            mean = DAILY_PRED[county, date]
            std = max(1, mean)**(1/2)
            heading = dates[date] + '-' + str(countyFIPS[county])
            quantiles = np.linspace(norm.ppf(0.1, mean, std), norm.ppf(0.9, mean, std), 9)
            quantiles = quantiles.clip(0).tolist()
            quantiles.insert(0, heading)
            FINAL_PRED.append(quantiles)

    return FINAL_PRED
