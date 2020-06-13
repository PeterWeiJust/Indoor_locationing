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
    tree = ET.parse(rawdata)
    root = tree.getroot()
    read_limit = len(root)

    posx = []
    posy = []
    times = []
    for i in range(read_limit):
        posx.append(float(root[i].attrib['x_pos']))
        posy.append(float(root[i].attrib['y_pos']))
        times.append(root[i].attrib['time'])
    return posx, posy, times


poxs = []
poys = []
scenarios = ['1', '2', '3', '4', '5']
routefileindex = ['1', '2', '3', '4', '5', '6', '7', '8']
'''
pox,poy,times=raw_file_converter("G:/Indoor_Dataset_fixed_last/Timed Data/Scenario_5/11-15/ground_truth_5.xml")
plt.plot(pox,poy)
plt.show()

'''

for i in range(len(scenarios)):
    for j in range(len(routefileindex)):
        lat = []
        Lng = []
        xs = []
        ys = []
        lat1, Lng1, Time = raw_file_converter(
            "./Timed Data/Scenario_" + scenarios[i] + "/" + routefileindex[j] + "/" + "ground_truth_" + scenarios[
                i] + ".xml")
        for k in range(len(lat1)):
            z = utm.from_latlon(lat1[k], Lng1[k])
            xs.append(z[0])
            ys.append(z[1])
        minx = min(xs)
        miny = min(ys)
        lat = [xs[i] - minx + 100 for i in range(len(xs))]
        Lng = [ys[i] - miny + 100 for i in range(len(ys))]
        data = {'Time': Time, 'lat1': lat1, 'Lng1': Lng1, 'x': xs, 'y': ys, 'lat': lat, 'Lng': Lng}
        df = pd.DataFrame(data=data)
        df.to_csv("./Timed Data/Scenario_" + scenarios[i] + "/" + routefileindex[j] + "/" + "ground_truth_" + scenarios[
            i] + ".csv", index=False)

'''
for i in routefileindex:
    for j in routeindex:
        
        x,y,time=raw_file_converter("F:/pyproject/multimodal_locationing/Timed Data/Routes_"+str(i)+"/"+j+"Route.xml")
        fig = plt.figure()
        plt.title('Routes_'+i+'/'+j+'Route')
        plt.plot(x,y,c='r')
        plt.scatter(x,y,s=10)
        plt.axis([min(x)-0.000005,max(x)+0.000005,min(y)-0.00005,max(y)+0.00005])
        
        fig.savefig('G:/Routes_'+i+'/'+j+'Route.png')
'''
