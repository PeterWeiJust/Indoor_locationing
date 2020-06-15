# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:14:29 2020

@author: mwei_archor
"""

import xml.etree.ElementTree as ET
import utm
import math
import numpy as numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def raw_file_converter(rawdata):
    tree=ET.parse(rawdata)
    root=tree.getroot()
    read_limit = len(root)
    
    posx=[]
    posy=[]
    times=[]
    for i in range(read_limit):
        posx.append(float(root[i].attrib['lat']))
        posy.append(float(root[i].attrib['long']))
        times.append(root[i].attrib['time'])
    return posx,posy,times


poxs=[]
poys=[]
scenarios=['1']
routefileindex=['1','2','3','4','5','6','7','8']

gminx=999999999999
gminy=999999999999
for i in range(len(scenarios)):
    for j in range(len(routefileindex)):
        lat=[]
        Lng=[]
        lat1,Lng1,Time=raw_file_converter("Scenario_"+scenarios[i]+"/"+routefileindex[j]+"/"+"ground_truth_"+scenarios[i]+".xml")
        for k in range(len(lat1)):
            z=utm.from_latlon(lat1[k],Lng1[k])
            gminx=min(gminx,z[0])
            gminy=min(gminy,z[1])

for i in range(len(scenarios)):
    for j in range(len(routefileindex)):
        lat=[]
        Lng=[]
        xs=[]
        ys=[]
        lat1,Lng1,Time=raw_file_converter("Scenario_"+scenarios[i]+"/"+routefileindex[j]+"/"+"ground_truth_"+scenarios[i]+".xml")
        for k in range(len(lat1)):
            z=utm.from_latlon(lat1[k],Lng1[k])
            xs.append(z[0])
            ys.append(z[1])
        lat=[xs[i]-gminx+100 for i in range(len(xs))]
        Lng=[ys[i]-gminy+100 for i in range(len(ys))]
        data={'Time':Time,'lat1':lat1,'Lng1':Lng1,'x':xs,'y':ys,'lat':lat,'Lng':Lng}
        df=pd.DataFrame(data=data)
        df.to_csv("Scenario_"+scenarios[i]+"/"+routefileindex[j]+"/"+"ground_truth_"+scenarios[i]+".csv",index=False)
