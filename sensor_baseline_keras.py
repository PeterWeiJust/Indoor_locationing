# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:04:54 2020

@author: mwei_archor
"""

import numpy as np
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
input_size = 3
hidden_size = 193
num_layers = 1
output_dim = 2
LR = 0.005
epoch=50

wandb.init(entity="sensor_baseline",project="sensor_baseline_edinburgh",sync_tensorboard=True,
           config={"epochs": num_epochs,"batch_size": batch_size,    
                   }
           )

train_sensor=SensorDataset()
SensorTrain=train_sensor.trainx
locationtrain=train_sensor.trainy
SensorVal=train_sensor.valx
locationval=train_sensor.valy
SensorTest=train_sensor.testx
locationtest=train_sensor.testy

model_name = "sensor_baseline_model_romania"
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
sensorlstm=LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),units=128)(sensorinput)
sensoroutput=Dense(2)(sensorlstm)
model=Model(inputs=[sensorinput],outputs=[sensoroutput])

model.compile(optimizer=RMSprop(LR),
                 loss='mse',metrics=['acc'])

model.fit(SensorTrain, locationtrain,
                       validation_data=(SensorVal,locationval),
                       epochs=epoch, batch_size=100, verbose=1,callbacks=[tensorboard,WandbCallback()]
                       #shuffle=False,
                       )

model.save("model/sensor_baseline_model.h5")
mmloc.save(os.path.join(wandb.run.dir, "wanbd_sensor_baseline.h5"))
fig=plt.figure()
locPrediction = model.predict(SensorTest, batch_size=100)
aveLocPrediction = pf.get_ave_prediction(locPrediction, 100)
data=pf.normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
plt.legend(['target','prediction'],loc='upper right')
plt.xlabel("x-latitude")
plt.ylabel("y-longitude")
plt.title('sensor_baseline_model prediction')
fig.savefig("predictionpng/sensor_baseline_locprediction.png")
wandb.log({"chart": wandb.Image("predictionpng/sensor_baseline_model_locprediction.png")})