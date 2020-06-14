#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:26:25 2020

@author: weixijia
"""

import xml.etree.ElementTree as ET
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import tensorflow as tf

timestep=100

class SensorDataset(torch.utils.data.Dataset):
    def __init__(self,mode="train",transform=torch.from_numpy):
        self.mode=mode
        self.transform=transform
        self.trainx,self.trainy=read_data(1,24)
        self.valx,self.valy=read_data(25,31)
        self.testx,self.testy=read_data(32,32)
        self.length=len(self.trainx)+len(self.testx)
    
    
    def __getitem__(self, index):
        if self.mode=="train":
            data=self.trainx[index]
            label=self.trainy[index]
        elif self.mode=="val":
            data=self.valx[index]
            label=self.valy[index]
        else:
            data=self.testx[index]
            label=self.testy[index]
            
        if self.transform is not None:
            data=self.transform(data)
            label=self.transform(label)
            
        return data,label
        
    def __len__(self):
        if self.mode=="train":
            return len(self.trainx)
        elif self.mode=="val":
            return len(self.valx)
        else:
            return len(self.testx)

class WifiDataset(torch.utils.data.Dataset):
    def __init__(self,mode="train",transform=torch.from_numpy):
        self.mode=mode
        self.transform=transform
        self.trainx,self.trainy=read_wifi_data(1,24)
        self.valx,self.valy=read_wifi_data(25,31)
        self.testx,self.testy=read_wifi_data(32,32)
        self.length=len(self.trainx)+len(self.valx)+len(self.testx)
    
    
    def __getitem__(self, index):
        if self.mode=="train":
            data=self.trainx[index]
            label=self.trainy[index]
        elif self.mode=="val":
            data=self.valx[index]
            label=self.valy[index]
        else:
            data=self.testx[index]
            label=self.testy[index]
        
        if self.transform is not None:
            data=self.transform(data)
            label=self.transform(label)
            
        return data,label
        
    def __len__(self):
        if self.mode=="train":
            return len(self.trainx)
        else:
            return len(self.testx)

class WifiClusterDataset(torch.utils.data.Dataset):
    def __init__(self,mode="train",transform=torch.from_numpy):
        self.mode=mode
        self.transform=transform
        self.trainx,self.trainy=read_wifi_data_classification(1,6)
        self.valx,self.valy=read_wifi_data_classification(7,7)
        self.testx,self.testy=read_wifi_data_classification(8,8)
        self.length=len(self.trainx)+len(self.valx)+len(self.testx)
    
    
    def __getitem__(self, index):
        if self.mode=="train":
            data=self.trainx[index]
            label=self.trainy[index]
        elif self.mode=="val":
            data=self.valx[index]
            label=self.valy[index]
        else:
            data=self.testx[index]
            label=self.testy[index]
        
        if self.transform is not None:
            data=self.transform(data)
            label=self.transform(label)
            
        return data,label
        
    def __len__(self):
        if self.mode=="train":
            return len(self.trainx)
        elif self.mode=="val":
            return len(self.valx)
        else:
            return len(self.testx)
        
class DownsampleDataset(torch.utils.data.Dataset):
    def __init__(self,tw=1000,slide=100,mode="train",transform=torch.from_numpy):
        self.mode=mode
        self.transform=transform
        
        self.sensortrain,self.labeltrain,self.wifitrain=downsample_data(1,1,tw,slide)
        for i in range(2,25):
            sensortrain,labeltrain,wifitrain=downsample_data(i, i, tw, slide)
            self.sensortrain=np.concatenate((self.sensortrain, sensortrain),axis=0)
            self.labeltrain=np.concatenate((self.labeltrain, labeltrain),axis=0)
            self.wifitrain=np.concatenate((self.wifitrain, wifitrain),axis=0)
        
        self.sensorval,self.labelval,self.wifival=downsample_data(25,25,tw,slide)
        for i in range(26,32):
            sensorval,labelval,wifival=downsample_data(i, i, tw, slide)
            self.sensorval=np.concatenate((self.sensorval, sensorval),axis=0)
            self.labelval=np.concatenate((self.labelval, labelval),axis=0)
            self.wifival=np.concatenate((self.wifival, wifival),axis=0)

        self.sensortest,self.labeltest,self.wifitest=downsample_data(32,32,tw,slide)
        self.length=len(self.sensortrain)+len(self.sensortest)
    
    
    def __getitem__(self, index):
        if self.mode=="train":
            sensor=self.sensortrain[index]
            wifi=self.wifitrain[index]
            label=self.labeltrain[index]
        elif self.mode=="val":
            sensor=self.sensorval[index]
            wifi=self.labelval[index]
            label=self.wifival[index]
        
        else:
            sensor=self.sensortest[index]
            wifi=self.labeltest[index]
            label=self.wifitest[index]
            
        if self.transform is not None:
            sensor=self.transform(sensor)
            wifi=self.transform(wifi)
            label=self.transform(label)
            
        return sensor,label,wifi
        
    def __len__(self):
        if self.mode=="train":
            return len(self.sensortrain) 
        elif self.mode=="val":
            return len(self.sensorval)
        else:
            return len(self.sensortest)


def read_wifi_data_classification(file_start,file_end):
    if file_start==file_end:     
        path='sensordata/sensor_wifi_1_'+str(file_start)+'_timestep100.csv'
        dataset=pd.read_csv(path,usecols=[i for i in range(16,59)]) 
        dataset = dataset.dropna()
        Y=pd.read_csv(path,usecols=[209]) 
        Y=np.array(Y)
        wifidata=np.array(dataset)
    else:
        wifi = []
        res_label=[]
        for file_num in range (file_start-1,file_end):
            file_num=file_num+1
            path='sensordata/sensor_wifi_1_'+str(file_num)+'_timestep100.csv'
            data=pd.read_csv(path,usecols=[i for i in range(16,59)])
            data_label=pd.read_csv(path,usecols=[209])
            res_label.append(data_label)
            wifi.append(data)
        dataset_label=pd.concat(res_label, axis=0)
        Y=np.array(dataset_label)#convert df to array
        wifidata=pd.concat(wifi,axis=0)
        wifidata=np.array(wifidata)
    return wifidata,Y


def read_wifi_data(file_start,file_end):
    if file_start==file_end:     
        path='sensordata/sensor_wifi_1_'+str(file_start)+'_timestep100.csv'
        dataset=pd.read_csv(path,usecols=[i for i in range(16,59)]) 
        dataset = dataset.dropna()
        Y=pd.read_csv(path,usecols=[i for i in range(14,16)]) 
        Y=np.array(Y)#convert df to array
        Y=normalisation(Y)#get normalised value
        
        wifidata=np.array(dataset)
    else:
        wifi = []
        res_label=[]
        for file_num in range (file_start-1,file_end):
            file_num=file_num+1
            path='sensordata/sensor_wifi_1_'+str(file_num)+'_timestep100.csv'
            data=pd.read_csv(path,usecols=[i for i in range(16,59)])
            data_label=pd.read_csv(path,usecols=[i for i in range(14,16)])
            res_label.append(data_label)
            wifi.append(data)
        dataset_label=pd.concat(res_label, axis=0)
        Y=np.array(dataset_label)#convert df to array
        Y=normalisation(Y)#get normalised value
        wifidata=pd.concat(wifi,axis=0)
    return wifidata,Y

def read_wifi_data_nozeros(file_start,file_end):
    sensor,label=read_wifi_data(file_start,file_end)
    dataset = np.hstack((sensor, label))
    dataset=pd.DataFrame(dataset)
    dataset['rss_total']= dataset.iloc[:, 0:102].sum(axis=1)
    dataset = dataset.drop(dataset[dataset.rss_total == 0].index)
    X=dataset.iloc[:,0:102]
    X=np.array(X)#convert df to array
    Y=dataset.iloc[:,102:104]
    Y=np.array(Y)#convert df to array
    dataset=np.array(dataset.iloc[:,0:104])
    return X,Y

def read_data(file_start,file_end):
    if file_start==file_end:
        label_num=count_label_num(file_start)
        path='timestep100/1_'+str(file_start)+'timestep100.csv'
        label_path='sensordata/sensor_wifi_1_'+str(file_start)+'_timestep100.csv'
        dataset = pd.read_csv(path,usecols = [12,13,14,15,16])
        dataset_label = pd.read_csv(label_path,usecols = [11,12,13,14,15])
        X=dataset.iloc[0:label_num*timestep,0:3]
        X=np.array(X)#convert df to array
        X=normalisation(X)
        X=X.reshape((X.shape[0]//timestep,timestep,3))#reshape for lstm
        Y=dataset_label.iloc[:,3:5]
        Y=np.array(Y)#convert df to array
        Y=normalisation(Y)#get normalised value
    else:
        res =[]
        res_label =[]
        for file_num in range (file_start-1,file_end):
            file_num=file_num+1
            label_num=count_label_num(file_num)
            path='timestep100/1_'+str(file_num)+'timestep100.csv'
            label_path='sensordata/sensor_wifi_1_'+str(file_num)+'_timestep100.csv'
            data = pd.read_csv(path,usecols = [12,13,14,15,16])
            data_label = pd.read_csv(label_path,usecols = [11,12,13,14,15])
            data = data.iloc[0:label_num*timestep,0:5]
            data_label = data_label.iloc[:,0:5]
            res.append(data)
            res_label.append(data_label)
        dataset=pd.concat(res, axis=0)
        dataset_label=pd.concat(res_label, axis=0)
        X=dataset.iloc[:,0:3]
        X=np.array(X)#convert df to array
        X=normalisation(X)
        X=X.reshape((X.shape[0]//timestep,timestep,3))#reshape for lstm
        Y=dataset_label.iloc[:,3:5]
        Y=np.array(Y)#convert df to array
        Y=normalisation(Y)#get normalised value
    return X,Y

def read_overlap_data(file_start,file_end):
    if file_start==file_end:
        label_num=count_label_num(file_start)
        path='timestep100/1_'+str(file_start)+'timestep100.csv'
        label_path='sensordata/sensor_wifi_1_'+str(file_start)+'_timestep100.csv'
        dataset = pd.read_csv(path,usecols = [12,13,14,15,16])
        dataset = dataset.dropna()
        wifidata = pd.read_csv(label_path,usecols=[i for i in range(16,59)])
        wifidata = wifidata.dropna()
        dataset_label = pd.read_csv(label_path,usecols = [11,12,13,14,15])
        X=dataset.iloc[0:label_num*timestep,0:3]
        X=np.array(X)#convert df to array
        #X=X.reshape((X.shape[0]//timestep,timestep,3))#reshape for lstm
        #X=normalisation(X)
        Y=dataset_label.iloc[:,3:5]
        Y=np.array(Y)#convert df to array
        Y=normalisation(Y)#get normalised value
        wifidata=np.array(wifidata)
    else:
        res =[]
        res_label =[]
        wifi = []
        for file_num in range (file_start-1,file_end):
            file_num=file_num+1
            label_num=count_label_num(file_num)
            path='timestep100/1_'+str(file_num)+'timestep100.csv'
            label_path='sensordata/sensor_wifi_1_'+str(file_num)+'_timestep100.csv'
            data = pd.read_csv(path,usecols = [12,13,14,15,16])
            wifidata = pd.read_csv(label_path,usecols=[i for i in range(16,59)])
            data_label = pd.read_csv(label_path,usecols = [11,12,13,14,15])
            data = data.iloc[0:label_num*timestep,0:5]
            data_label = data_label.iloc[:,0:5]
            res.append(data)
            res_label.append(data_label)
            wifi.append(wifidata)
        dataset=pd.concat(res, axis=0)
        dataset_label=pd.concat(res_label, axis=0)
        X=dataset.iloc[:,0:3]
        X=np.array(X)#convert df to array
        #X=normalisation(X)
        #X=X.reshape((X.shape[0]//timestep,timestep,3))#reshape for lstm
        Y=dataset_label.iloc[:,3:5]
        Y=np.array(Y)#convert df to array
        Y=normalisation(Y)#get normalised value
        wifidata=pd.concat(wifi,axis=0)
    return X,Y,dataset,wifidata

def overlap_data(file_start, file_end, tw,slide):
    X,Y,input_data,wifidata=read_overlap_data(file_start, file_end)
    input_data=np.array(input_data)
    sensor=input_data[:,0:3]
    location=input_data[:,3:5]
    sensor=normalisation(sensor)
    location=normalisation(location)
    input_data=np.concatenate((sensor, location), axis=1)
    inout_seq = []
    label_seq = []
    wifi_seq = []
    wifidata=np.array(wifidata)
    L = len(input_data)
    total_samples=(L-tw)//slide+1
    input_data=np.array(input_data)
    for i in range (total_samples):
        train_seq = input_data[i*slide:i*slide+tw,0:3]
        #train_seq=train_seq.reshape((1,tw,3))#reshape for lstm
        train_label = input_data[i*slide,3:5]
        inout_seq.append((train_seq))
        label_seq.append((train_label))
        wifi_seq.append((wifidata[i]))
    inout_seq=np.array(inout_seq)
    wifi_seq=np.array(wifi_seq)
    return inout_seq,label_seq,wifi_seq #return array of sensor,label and wifi

def downsample_data(file_start, file_end, tw,slide):
    X,Y,input_data,wifidata=read_overlap_data(file_start, file_end)
    input_data=np.array(input_data)
    sensor=input_data[:,0:3]
    location=input_data[:,3:5]
    sensor=normalisation(sensor)
    location=normalisation(location)
    input_data=np.concatenate((sensor, location), axis=1)
    inout_seq = []
    label_seq = []
    wifi_seq = []
    wifidata=np.array(wifidata)
    L = len(input_data)
    total_samples=(L-tw)//slide+1
    print(total_samples)
    input_data=np.array(input_data)
    for i in range (total_samples):
        train_seq = input_data[i*slide:i*slide+tw,0:3]# 1000*3
        train_seq = train_seq[[99,199,299,399,499,599,699,799,899,999],:]
        #train_seq = train_seq[[0,100,200,300,400,500,600,700,800,900],:]
        #train_seq=train_seq.reshape((1,tw,3))#reshape for lstm
        train_label = input_data[i*slide,3:5]
        inout_seq.append((train_seq))
        label_seq.append((train_label))
        wifi_seq.append((wifidata[i]))
    inout_seq=np.array(inout_seq)
    label_seq=np.array(label_seq)
    wifi_seq=np.array(wifi_seq)
    return inout_seq,label_seq,wifi_seq #return array of sensor,label and wifi

def normalisation(X):#Scale data to the range of -1:1
    max_range=1
    min_range=0
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    normalized_X = X_std * (max_range- min_range) + min_range
    return normalized_X

def count_label_num(file_num):
    countlabel_path='sensordata/sensor_wifi_1_'+str(file_num)+'_timestep100.csv'
    dataset = pd.read_csv(countlabel_path,usecols = [11,12,13,14,15])
    label_num=dataset.shape[0]
    return label_num

