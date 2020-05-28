import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import plotting_functions as pf
import pandas as pd
from data_functions import normalisation,overlap_data,read_overlap_data,downsample_data,DownsampleDataset
from keras.models import Model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input,ReLU,Multiply,Add
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard

# Hyper-parameters
sensor_input_size = 3
wifi_input_size = 193
hidden_size = 128
batch_size = 100
output_dim = 2
num_epochs = 300
learning_rate = 0.001

#load downsample dataset
train_sensor=DownsampleDataset()

SensorTrain=train_sensor.sensortrain
locationtrain=train_sensor.labeltrain
WifiTrain=train_sensor.wifitrain

SensorVal=train_sensor.sensorval
locationval=train_sensor.labelval
WifiVal=train_sensor.wifival

SensorTest=train_sensor.sensortest
WifiTest=train_sensor.wifitest
locationtest=train_sensor.labeltest

#construct mmloc model
sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
sensoroutput=LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),units=hidden_size)(sensorinput)

wifiinput=Input(shape=(wifi_input_size,))
wifi=Dense(hidden_size)(wifiinput)
wifi=ReLU()(wifi)
wifi=Dense(hidden_size)(wifi)
wifi=ReLU()(wifi)
wifioutput=Dense(hidden_size)(wifi)

#merge style: multiply
merge=Multiply()([sensoroutput,wifioutput])
hidden=Dense(hidden_size,activation='relu')(merge)
'''
#merge style: Multimodal Residual Networks
#hidden layer1
MRNsensorhidden=Dense(hidden_size,activation='tanh')(sensoroutput)
MRNwifihidden=Dense(hidden_size,activation='tanh')(wifioutput)
merge=Multiply()([MRNsensorhidden,MRNwifihidden])
MRNsensorhidden=Dense(hidden_size)(sensoroutput)
MRNHidden=Add()([MRNsensorhidden,merge])
#hidden layer2
MRNsensorhidden=Dense(hidden_size,activation='tanh')(MRNHidden)
MRNwifihidden=Dense(hidden_size,activation='tanh')(wifioutput)
merge=Multiply()([MRNsensorhidden,MRNwifihidden])
MRNsensorhidden=Dense(hidden_size)(MRNHidden)
MRNHidden=Add()([MRNsensorhidden,merge])
#hidden layer3
MRNsensorhidden=Dense(hidden_size,activation='tanh')(MRNHidden)
MRNwifihidden=Dense(hidden_size,activation='tanh')(wifioutput)
merge=Multiply()([MRNsensorhidden,MRNwifihidden])
MRNsensorhidden=Dense(hidden_size)(MRNHidden)
MRNHidden=Add()([MRNsensorhidden,merge])
'''
output=Dense(output_dim)(hidden)
mmloc=Model(inputs=[sensorinput,wifiinput],outputs=[output])

mmloc.compile(optimizer=RMSprop(learning_rate),
                 loss='mse',metrics=['acc'])

model_name = "mmloc_multi_model_romania"
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

mmloc.fit([SensorTrain,WifiTrain], locationtrain,
                       validation_data=([SensorVal,WifiVal],locationval),
                       epochs=num_epochs, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                       #shuffle=False,
                       )

#save model
mmloc.save("romaniamodel/mmloc_multi.h5")
fig=plt.figure()
locPrediction = mmloc.predict([SensorTest,WifiTest], batch_size=100)
aveLocPrediction = pf.get_ave_prediction(locPrediction, 100)
data=pf.normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
plt.legend(['target','prediction'],loc='upper right')
plt.xlabel("x-latitude")
plt.ylabel("y-longitude")
plt.title('mmloc_multi_prediction')
fig.savefig("romaniapredictionpng/mmloc_multi_locprediction.png")
