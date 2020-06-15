# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:11:42 2020

@author: mwei_archor
"""
import xml.etree.ElementTree as ET
import math
import numpy as numpy
import numpy as np
import pandas as pd
import os

def data_generator(time_step, sensor_data, rawdata, timerfilepath):
    df = pd.read_csv(rawdata)
    df.drop(columns=['rt'],inplace=True)
    point = pd.read_csv(timerfilepath,index_col=False)
    point.drop_duplicates(['Time'],inplace=True)
    point=point.reset_index(drop=True)
    
    df['lat']=numpy.NaN
    df['lng']=numpy.NaN
    
    sequcnce=0
    s=0.0
    sl=0.0
    l_s=0
    for i in range (len(df)):
        if sequcnce >len(point)-1:
            break
        if df['st'][i]==point['Time'][sequcnce]:
            df['lat'][i]=point['lat'][sequcnce]
            df['lng'][i]=point['Lng'][sequcnce]
            diff=(point['lat'][sequcnce] - s)/(i-l_s)
            difflng=(point['Lng'][sequcnce] - sl)/(i-l_s)
            counter=1
            sum=s
            suml=sl
            for j in range (l_s+1,i):
                if counter%time_step==0:
                    sum=sum+diff*time_step
                    suml=suml+difflng*time_step
                df['lat'][j]=sum
                df['lng'][j]=suml
                counter=counter+1
            
            s=point['lat'][sequcnce]
            sl=point['Lng'][sequcnce]
            sequcnce=sequcnce+1
            l_s=i
    
    df=df.drop(df[df.st < point['Time'][0]].index)
    df=df.drop(df[df.st > point['Time'][len(point)-1]].index)
    
    df.to_csv(sensor_data)
    
timestep=100

for i in range(1,2):
    for j in range(1,9):       
        sensor_data="Scenario_"+str(i)+"/"+str(j)+"_timestep100.csv"
        rawdata="converteddata/Scenario_"+str(i)+"_"+str(j)+"_converted.csv"
        timerfilepath="Scenario_"+str(i)+"/"+str(j)+"/"+str(i)+".csv"
        data_generator(timestep,sensor_data,rawdata,timerfilepath)

for i in range(1,2):
    for j in range(1,9):
        df=pd.read_csv("Scenario_"+str(i)+"/"+str(j)+"_timestep100.csv")
        df=df.drop(df.columns[0],axis=1)
        dfunique=df.loc[(df.index%100==0)&(df.index!=len(df)-1)]
        dfunique.to_csv("Scenario_"+str(i)+"/"+str(j)+"_timestep100_unique.csv",index=0)
        
for i in range(1,2):
    for j in range(1,9):
        dfwifi=pd.read_csv("Timed Data/Scenario_"+str(i)+"/scenario"+str(i)+"-route"+str(j)+".csv")
        dfwifi['match']=-1
        dfwifi.to_csv("Timed Data/Scenario_"+str(i)+"/scenario"+str(i)+"-route"+str(j)+".csv",index=0)
        
        
        