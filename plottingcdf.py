# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:20:45 2020

@author: mwei_archor
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import plotting_functions as pf
import pandas as pd
from data_functions import normalisation,overlap_data,read_overlap_data,downsample_data,DownsampleDataset,WifinozerosDataset,SensorDataset
from keras.models import Sequential,Model,load_model

sensor=SensorDataset()
downsample=DownsampleDataset()

SensorTest=sensor.testx
locationtest=sensor.testy

DownsampleSensorTest=downsample.sensortest
Downsamplelocationtest=downsample.labeltest
DownsampleWifiTest=downsample.wifitest

sensor_baseline=load_model("model/sensor_baseline_model.h5")
sensor_downsample=load_model("romaniamodel/sensor_downsample_model.h5")
wifi_all=load_model("romaniamodel/wifi_DNN_model.h5")
mmloc_multi=load_model("romaniamodel/mmloc_multi.h5")
mmloc_multi_residual=load_model("romaniamodel/mmloc_multi_residual.h5")

#sensorbaselineloc = sensor_baseline.predict(SensorTest, batch_size=100)
sensordownsampleloc = sensor_downsample.predict(DownsampleSensorTest, batch_size=100)
wifiloc = wifi_all.predict(DownsampleWifiTest, batch_size=100)
mmloc_multiloc = mmloc_multi.predict([DownsampleSensorTest,DownsampleWifiTest], batch_size=100)
mmloc_multi_residualloc = mmloc_multi_residual.predict([DownsampleSensorTest,DownsampleWifiTest], batch_size=100)

bin_edgeses=[]
cdfs=[]
'''
names=["sensor_baseline","sensor_downsample","wifi","mmloc","mmloc_residual"]
colors=['green','blue','black','yellow','red']
'''
names=["sensor_downsample","wifi","mmloc_residual","mmloc"]
colors=['green','blue','black','red']
'''
bin_edge,cdf=pf.cdfdiff(target=locationtest, predict=sensorbaselineloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
'''
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=sensordownsampleloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=wifiloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=mmloc_multiloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=mmloc_multi_residualloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)


fig = plt.figure()
for i in range(len(cdfs)): 
    plt.plot(bin_edgeses[i][0:-1], cdfs[i],linestyle='--', label=names[i],color=colors[i-1])
    
plt.xlim(xmin = 0)
plt.ylim((0,1))
plt.xlabel("metres")
plt.ylabel("CDF")
plt.legend(names,loc='upper right')
plt.grid(True)
plt.title('Scenario1 CDF')
fig.savefig("CDF.pdf")

