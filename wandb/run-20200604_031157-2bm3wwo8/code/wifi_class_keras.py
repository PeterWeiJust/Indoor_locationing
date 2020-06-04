# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:37:47 2020

@author: weixijia
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
from sklearn.svm import SVC
from keras.optimizers import Adam, RMSprop,SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from data_functions import DownsampleDataset,WifiDataset,WifiClusterDataset
from keras.utils import np_utils
from wandb.keras import WandbCallback

np.random.seed(7)

# Hyper-parameters
# wifi_input_size = 193
# batch_size = 100
# hidden_size = 128
# output_dim = 400
# learning_rate = 0.001
# epoch=5

model_name = "wifiClass_bucharest"


wandb.init(entity="mmloc",project=model_name,sync_tensorboard=True)
# Set and save hyperparameters         
wandb.config.gamma = "auto"
wandb.config.C = 1
wandb.config.seed = 0



training=WifiClusterDataset()
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


# tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
# model = Sequential()
# model.add(Dense(hidden_size,activation='relu',input_dim=wifi_input_size))
# model.add(Dropout(0.5))
# model.add(Dense(hidden_size,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(hidden_size,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(activation='softmax',units=output_dim))
# model.compile(optimizer=RMSprop(learning_rate),
#                  loss='mse',metrics=["acc"])

# model.fit(WifiTrain, locationlabel,
#                        validation_data=(WifiVal,locationval),
#                        epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard,WandbCallback()]
#                        #shuffle=False,
#                        )

# model.save("romaniamodel/"+str(model_name)+".h5")
# model.save(os.path.join(wandb.run.dir, "wanbd_"+str(model_name)+".h5"))

# Fit model
model = SVC(kernel='rbf', random_state=wandb.config.seed, gamma=wandb.config.gamma, C=wandb.config.C)
model.fit(WifiTrain, locationlabel)

# Save metrics
wandb.log({"Train Accuracy": model.score(WifiTrain, locationlabel), 
           "Test Accuracy": model.score(WifiVal,locationval)})

fig1=plt.figure()
locPrediction = model.predict(WifiTest)
locpredlabel=np.argmax(locPrediction,axis=1)+mins

#aveLocPrediction = pf.get_ave_prediction(locPrediction, 100)
#data=pf.normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
loclatlng=pf.convert_data_to_utm(locpredlabel)
data=pf.normalized_test_data(np.hstack((locationtest,loclatlng)))
plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
plt.legend(['target','prediction'],loc='upper right')
plt.xlabel("x-latitude")
plt.ylabel("y-longitude")
plt.title(str(model_name)+" Prediction")
fig1.savefig("romaniapredictionpng/"+str(model_name)+"_locprediction.png")
wandb.log({"chart": wandb.Image("romaniapredictionpng/"+str(model_name)+"_locprediction.png")})
#draw cdf picture
fig=plt.figure()
bin_edge,cdf=pf.cdfdiff(target=locationtest,predict=loclatlng)
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
