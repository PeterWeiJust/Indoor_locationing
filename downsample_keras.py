# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:48:28 2020

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
import plotting_functions as pf
import pandas as pd
import wandb
from keras import metrics
from data_functions import normalisation,overlap_data,read_overlap_data,downsample_data,DownsampleDataset
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from wandb.keras import WandbCallback

np.random.seed(7)
# Hyper-parameters
timestep=100
input_size = 3
hidden_size = 193
num_layers = 1
output_dim = 2
LR = 0.001
epoch=100

wandb.init(entity="sensor_downsample",project="sensor_downsample_edinburgh",sync_tensorboard=True,
           config={"epochs": epoch,"batch_size": 100,    
                   }
           )

train_sensor=DownsampleDataset()
SensorTrain=train_sensor.sensortrain
locationtrain=train_sensor.labeltrain
SensorVal=train_sensor.sensorval
locationval=train_sensor.labelval
SensorTest=train_sensor.sensortest
locationtest=train_sensor.labeltest

model_name = "sensor_downsample_model_romania"
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
sensorlstm=LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),units=128)(sensorinput)
sensoroutput=Dense(2)(sensorlstm)
model=Model(inputs=[sensorinput],outputs=[sensoroutput])

model.compile(optimizer=RMSprop(LR),
                 loss='mse',metrics=["acc"])

model.fit(SensorTrain, locationtrain,
                       validation_data=(SensorVal,locationval),
                       epochs=epoch, batch_size=100, verbose=1,callbacks=[tensorboard,WandbCallback()]
                       #shuffle=False,
                       )

model.save("romaniamodel/sensor_downsample_model.h5")
model.save(os.path.join(wandb.run.dir, "wanbd_sensor_downsample.h5"))
fig1=plt.figure()
locPrediction = model.predict(SensorTest, batch_size=100)
aveLocPrediction = pf.get_ave_prediction(locPrediction, 100)
data=pf.normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
plt.legend(['target','prediction'],loc='upper right')
plt.xlabel("x-latitude")
plt.ylabel("y-longitude")
plt.title('sensor_downsample_model prediction')
fig1.savefig("predictionpng/sensor_downsample_locprediction.png")
wandb.log({"chart": wandb.Image("predictionpng/sensor_downsample_model_locprediction.png")})
#draw cdf picture
plt.close()
fig=plt.figure()
bin_edge,cdf=pf.cdfdiff(target=locationtest,predict=locPrediction)
plt.plot(bin_edge[0:-1],cdf,linestyle='--',label="sensor_downsample",color='r')
plt.xlim(xmin = 0)
plt.ylim((0,1))
plt.xlabel("metres")
plt.ylabel("CDF")
plt.legend("sensor_downsample",loc='upper right')
plt.grid(True)
plt.title('sensor_downsample CDF')
fig.savefig("sensor_downsample_CDF.pdf")