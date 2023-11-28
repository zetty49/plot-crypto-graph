import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta

# HMA-Kahlman Trend Module
def hma(x, length):
    half_length = int(length/2)
    wma1 = 2 * pd.Series(x).rolling(window=half_length).mean()
    wma2 = pd.Series(x).rolling(window=length).mean()
    hma = pd.Series(wma1 - wma2).rolling(window=int(np.sqrt(length))).mean()
    return hma.values[-1]

def hma3(x, length):
    p = int(length / 2)
    wma1 = pd.Series(x).rolling(window=int(p/3)).mean() * 3
    wma2 = pd.Series(x).rolling(window=int(p/2)).mean()
    wma3 = pd.Series(x).rolling(window=p).mean()
    hma3 = pd.Series(wma1 - wma2 - wma3).rolling(window=int(np.sqrt(length))).mean()
    return hma3.values[-1]

def kahlman(x, g):
    kf = 0.0
    velo = 0.0
    for i in range(len(x)):
        dk = x[i] - kf
        smooth = kf + dk * np.sqrt(g * 2)
        velo += g * dk
        kf = smooth + velo
    return kf

def get_hma_kahlman_trend(price_data, lookback_window=22, use_kahlman=True, gain=0.7):
    a = kahlman(hma(price_data, lookback_window), gain) if use_kahlman else hma(price_data, lookback_window)
    b = kahlman(hma3(price_data, lookback_window), gain) if use_kahlman else hma3(price_data, lookback_window)
    c = 'lime' if b > a else 'red'
    crossdn = a > b and a[-2] < b[-2]
    crossup = b > a and b[-2] < a[-2]
    return a, b, c, crossdn, crossup

# Trendlines Module
def trendline(input_function, delay, only_up):
    Ax, Bx, By, slope = 0, 0, 0.0, 0.0
    Ay = pd.Series(input_function)
    for i in range(1, len(Ay)):
        if not np.isnan(Ay[i]) and Ay[i] != Ay[i-1]:
            Ax, By = i-delay, Ay[i-1]
            Bx = i-1-delay
            slope = (Ay[i] - By) / (i - Bx)
            break
    Axbis = Ax
    Aybis = Ay[0] + (Axbis-delay) * slope
    if slope < 0 and only_up:
        return np.nan, np.nan, np.nan
    elif slope > 0 and not only_up:
        return np.nan, np.nan, np.nan
    else:
        line_color = 'lime' if slope > 0 else 'red'
        return Ax, Axbis, By, Aybis, slope, line_color

def pivot(price_data, lookback_window):
    high_point = ta.pivothigh(price_data, lookback_window, lookback_window)
    low_point = ta.pivotlow(price_data, lookback_window, lookback_window)
    time = np.arange(len(price_data))
    slope_high, intercept_high = trendline(high_point, time[-lookback_window:])
    slope_low, intercept_low = trendline(low_point, time[-lookback_window:])
    
    fig, ax = plt.subplots()
    ax.plot(price_data)
    ax.scatter(time[high_point == price_data], price_data[high_point == price_data], color='green')
    ax.scatter(time[low_point == price_data], price_data[low_point == price_data], color='red')
    ax.plot(time[-lookback_window:], slope_high * time[-lookback_window:] + intercept_high, color='green')
    ax.plot(time[-lookback_window:], slope_low * time[-lookback_window:] + intercept_low, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    plt.show()
    
    return [high_point, low_point, slope_high, slope_low]