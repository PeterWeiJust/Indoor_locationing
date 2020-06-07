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

model_name = "wifi_DNN_model_romania"
wandb.init(entity="mmloc",project=model_name,sync_tensorboard=True,
           config={"epochs": epoch,"batch_size": batch_size,"hidden_size":hidden_size,
                   "learning_rate":learning_rate,
                   "output_dim":output_dim,
                   }
           )

training=WifiDataset()
WifiTrain=training.trainx
locationlabel=training.trainy

WifiVal=training.valx
locationval=training.valy

WifiTest=training.testx
locationtest=training.testy


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
                       epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard,WandbCallback()]
                       #shuffle=False,
                       )

model.save("romaniamodel/"+str(model_name)+".h5")
model.save(os.path.join(wandb.run.dir, "wanbd_"+str(model_name)+".h5"))
fig1=plt.figure()
locPrediction = model.predict(WifiTest, batch_size=batch_size)

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