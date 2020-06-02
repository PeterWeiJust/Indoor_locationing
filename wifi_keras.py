# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:01:12 2020

@author: mwei_archor
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import tensorflow as tf
import plotting_functions as pf
import pandas as pd
import wandb
from keras import metrics
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop,SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from data_functions import DownsampleDataset,WifiDataset
from keras.utils import np_utils
from wandb.keras import WandbCallback

np.random.seed(7)

# Hyper-parameters
wifi_input_size = 193
batch_size = 100
hidden_size = 128
output_dim = 2
learning_rate = 0.001
epoch=100

wandb.init(entity="wifi_DNN",project="wifi_DNN",sync_tensorboard=True,
           config={"epochs": epoch,"batch_size": batch_size,    
                   }
           )

training=WifiDataset()
WifiTrain=training.trainx
locationlabel=training.trainy

WifiVal=training.valx
locationval=training.valy

WifiTest=training.testx
locationtest=training.testy

location=np.concatenate([locationlabel,locationval])
mins=min(np.unique(location,axis=0))
locationlabel=np_utils.to_categorical(locationlabel-mins)
locationval=np_utils.to_categorical(locationval-mins)

output_dim=locationlabel.shape[1]

model_name = "wifi_DNN_model_romania"
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
model = Sequential()
model.add(Dense(hidden_size,activation='relu',input_dim=wifi_input_size))
model.add(Dropout(0.5))
model.add(Dense(hidden_size,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(hidden_size,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(activation='softmax',units=output_dim))
model.compile(optimizer=RMSprop(learning_rate),
                 loss='mse',metrics=["acc"])

model.fit(WifiTrain, locationlabel,
                       validation_data=(WifiVal,locationval),
                       epochs=epoch, batch_size=100, verbose=1,callbacks=[tensorboard]
                       #shuffle=False,
                       )

model.save("romaniamodel/wifi_DNN_model.h5")
model.save(os.path.join(wandb.run.dir, "wanbd_wifi_DNN.h5"))
fig=plt.figure()
locPrediction = model.predict(WifiTest, batch_size=100)
locpredlabel=np.argmax(locPrediction,axis=1)+mins

#aveLocPrediction = pf.get_ave_prediction(locPrediction, 100)
#data=pf.normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
loclatlng=pf.convert_data_to_utm(locpredlabel)
data=pf.normalized_test_data(np.hstack((locationtest,loclatlng)))
plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
plt.legend(['target','prediction'],loc='upper right')
plt.xlabel("x-latitude")
plt.ylabel("y-longitude")
plt.title('wifi_DNN_model prediction')
fig.savefig("romaniapredictionpng/wifi_locprediction.png")
wandb.log({"chart": wandb.Image("romaniapredictionpng/wifi_locprediction.png")})