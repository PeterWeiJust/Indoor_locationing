# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:44:07 2020

@author: weixijia
"""

import pandas as pd
import numpy as np
import glob

Edinburgh = pd.read_csv('edin.csv',dtype=np.float64, header=0)
Bucharest = pd.read_csv('bucha.csv',dtype=np.float64)

Bucharest_WiFi = Bucharest.iloc[:,3::]
Edinburgh_WiFi = Edinburgh.iloc[:,18::]

time_bucha=(Bucharest.iloc[-1]['t']-Bucharest.iloc[0]['t'])/1000/60 # format in minutes
time_edin=(Edinburgh.iloc[-1][0]-Edinburgh.iloc[0][0])/1000/60 # format in minutes

AP_bucha=Bucharest_WiFi.shape[1]
AP_edin=Edinburgh_WiFi.shape[1]

sum_RSS_bucha=Bucharest_WiFi.sum(axis=1) # ap0 starts from column 3
sum_RSS_edin=Edinburgh_WiFi.sum(axis=1) # ap0 starts from column 18

diversity_bucha=sum_RSS_bucha.unique().shape[0]/sum_RSS_bucha.shape[0] #diversity upon all the samples, calculate the percentage pf how many unique sum RSS values over all WiFi samples
diversity_edin=sum_RSS_edin.unique().shape[0]/sum_RSS_edin.shape[0]

RichInfo_bucha=(Bucharest_WiFi != 0).sum(axis=1)/AP_bucha # For the number of non-zeros in each row divided by total num of ap for each sample
RichInfo_edin=(Edinburgh_WiFi != 0).sum(axis=1)/AP_edin

RSSperAP_bucha=(Bucharest_WiFi != 0).sum(axis=0)/Bucharest_WiFi.shape[0] # get each AP RSS values among all samples (check how may non-zero values of each AP among each column)
RSSperAP_edin=(Edinburgh_WiFi != 0).sum(axis=0)/Edinburgh_WiFi.shape[0]

print('Bucharest Rich info: '+str(RSSperAP_bucha.mean() *100)+' %')
print('Edinburgh Rich info: '+str(RSSperAP_edin.mean() *100)+' %') 