#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def convert_data_to_utm(dd):
    rowlist=[]
    collist=[]
    for i in dd:
        rowlist.append((i-1)%20*2+91)
        collist.append((i-1)//20*2+91)
    return np.transpose(np.vstack((rowlist,collist)))

def normalized_test_data(dd):
    min_c1 = 99
    max_c1 = 130
    min_c2 = 99
    max_c2 = 130

    d1 = dd[:, 0]
    d2 = dd[:, 1]
    d3 = dd[:, 2]
    d4 = dd[:, 3]

    inverse_to_utm_x = lambda x: (min_c1 + x * (max_c1 - min_c1))
    inverse_to_utm_y = lambda x: (min_c2 + x * (max_c2 - min_c2))
    
    id1 = inverse_to_utm_x(d1)
    id2 = inverse_to_utm_y(d2)
    
    return np.transpose(np.vstack((np.vstack((id1, id2)), np.vstack((d3, d4)))))    

    
def normalized_data_to_utm(dd):
    min_c1 = 99
    max_c1 = 130
    min_c2 = 99
    max_c2 = 130

    d1 = dd[:, 0]
    d2 = dd[:, 1]
    d3 = dd[:, 2]
    d4 = dd[:, 3]

    # inverse_to_utm_x = lambda x: (min_c1 + (x + 1) * (max_c1 - min_c1) / 2)
    # inverse_to_utm_y = lambda x: (min_c2 + (x + 1) * (max_c2 - min_c2) / 2)

    inverse_to_utm_x = lambda x: (min_c1 + x * (max_c1 - min_c1))
    inverse_to_utm_y = lambda x: (min_c2 + x * (max_c2 - min_c2))
    
    id1 = inverse_to_utm_x(d1)
    id2 = inverse_to_utm_y(d2)
    id3 = inverse_to_utm_x(d3)
    id4 = inverse_to_utm_y(d4)
    
    return np.transpose(np.vstack((np.vstack((id1, id2)), np.vstack((id3, id4)))))


def cal_error_in_meters(data):
    data = normalized_data_to_utm(data)
    errors = [np.sqrt(np.square(item[0] - item[2]) + np.square(item[1] - item[3])) for item in data]
    return errors


def cdfpic(data):
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)
    
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    
    hist=hist.astype(float)/len(data)

    cdf = np.cumsum(hist)
    
    return bin_edges,cdf

def cdfdiff(target, predict):
    target_and_predict = np.hstack((target, predict))
    error_in_meters = cal_error_in_meters(target_and_predict)    
    return cdfpic(error_in_meters)

def get_ave_prediction(locPrediction, n):
    weights = np.ones(n)
    weights /= weights.sum()
    x = np.asarray(locPrediction[:,0])
    y = np.asarray(locPrediction[:,1])  
    avelatPrediction = np.convolve(x, weights, mode='full')[:len(x)]
    avelngPrediction = np.convolve(y, weights, mode='full')[:len(y)]
    avelatPrediction[:n] = avelatPrediction[n]
    avelngPrediction[:n] = avelngPrediction[n]
    avelatPrediction=avelatPrediction.reshape(-1,1)
    avelngPrediction=avelngPrediction.reshape(-1,1)
    aveLocPrediction=np.column_stack((avelatPrediction,avelngPrediction))
    return aveLocPrediction

# error line plot into file "/graph_output/errors_visualization{*}.png"
def visualization(Y_test, Y_pre, suffix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel("x-longitude")
    ax.set_ylabel("y-latitude")

    Y_test = np.array(Y_test)
    for target, pred, i in zip(Y_test, Y_pre, range(np.shape(Y_test)[0])):
        plt.plot([pred[0], target[0]], [pred[1], target[1]], color='r',
                 linewidth=0.5, label='error line' if i == 0 else "")
        plt.scatter(pred[0], pred[1], label='prediction' if i == 0 else "", color='b', marker='.')
        plt.scatter(target[0], target[1], label='target' if i == 0 else "", color='c', marker='.')

    ax.set_title("Errors of {}".format(suffix))
    ax.legend()

    # save error line fig
    fig.savefig("errors_visualization_" + str(suffix) + ".png")  # regression [200,200,200]
