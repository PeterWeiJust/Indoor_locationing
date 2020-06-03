# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:04:54 2020

@author: mwei_archor
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import math
import json
import plotting_functions as pf
import pandas as pd
import wandb
from data_functions import normalisation,overlap_data,read_overlap_data,downsample_data,DownsampleDataset,SensorDataset
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
batch_size=100
input_size = 3
hidden_size = 193
num_layers = 1
output_dim = 2
lstm_hidden_units = 128
LR = 0.001 
epoch=2

model_name = "sensorBaseline_bucharest"

wandb.init(entity="mmloc",project=model_name,sync_tensorboard=True,
           config={"epochs": epoch,"batch_size": batch_size,"hidden_size":hidden_size,
                   "learning_rate":LR,"sensor_input_size":input_size,
                   "output_dim":output_dim,"lstm_hidden_units":lstm_hidden_units
                   }
           )

train_sensor=SensorDataset()
SensorTrain=train_sensor.trainx
locationtrain=train_sensor.trainy
SensorVal=train_sensor.valx
locationval=train_sensor.valy
SensorTest=train_sensor.testx
locationtest=train_sensor.testy

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
sensorlstm=LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),units=lstm_hidden_units)(sensorinput)
sensoroutput=Dense(output_dim)(sensorlstm)
model=Model(inputs=[sensorinput],outputs=[sensoroutput])

model.compile(optimizer=RMSprop(LR),
                 loss='mse',metrics=['acc'])

model.fit(SensorTrain, locationtrain,
                       validation_data=(SensorVal,locationval),
                       epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard,WandbCallback()]
                       #shuffle=False,
                       )

model.save("romaniamodel/"+str(model_name)+".h5")
model.save(os.path.join(wandb.run.dir, "wanbd_"+str(model_name)+".h5"))
fig1=plt.figure()
locPrediction = model.predict(SensorTest, batch_size=batch_size)
aveLocPrediction = pf.get_ave_prediction(locPrediction, batch_size)
data=pf.normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
plt.legend(['target','prediction'],loc='upper right')
plt.xlabel("x-latitude")
plt.ylabel("y-longitude")
plt.title(str(model_name)+" Prediction")
fig1.savefig("romaniapredictionpng/"+str(model_name)+"_locprediction.png")
wandb.log({"chart": wandb.Image("romaniapredictionpng/"+str(model_name)+"_locprediction.png")})
#draw cdf picture

fig=plt.figure()
bin_edge,cdf=pf.cdfdiff(target=locationtest,predict=locPrediction)
plt.plot(bin_edge[0:-1],cdf,linestyle='--',label=str(model_name),color='r')
plt.xlim(xmin = 0)
plt.ylim((0,1))
plt.xlabel("metres")
plt.ylabel("CDF")
plt.legend(str(model_name),loc='upper right')
plt.grid(True)
plt.title((str(model_name)+' CDF'))
fig.savefig("romaniacdf/"+str(model_name)+"_CDF.pdf")
wandb.log({"chart": wandb.Image("romaniacdf/"+str(model_name)+"_CDF.pdf")})