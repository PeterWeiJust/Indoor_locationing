#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 23:53:01 2019

@author: chenxingji
"""
import utm
import os
import re
import xml.dom.minidom
import collections
import numpy as np
import pickle
import h5py
from collections import Counter

# define some constant variables
north_west = (55.945139, -3.18781)
south_east = (55.944600, -3.186537)
num_grid_y = 30  # latitude
num_grid_x = 40  # longitude
max_lat = abs(north_west[0] - south_east[0])  # 0.0006 # 0.0005393
max_lng = abs(north_west[1] - south_east[1])  # 0.002  # 0.001280
delta_lat = max_lat / num_grid_y  # 3e-06
delta_lng = max_lng / num_grid_x  # 1e-05




# 本脚本处理xml数据，写入precis_sample_102(*表示wifi维度).h5或out_in_overal_102.txt和out_in_overal_102.csv文件中

# *****************************************************************************************************
# 1. Read the distinct access point id from file("wifi_filename") into dictionary

Scenario=1
wifi_filename = "wifi_id_"+str(Scenario)+".txt"
NUM_COLUMNS = len(open(wifi_filename).readlines(  ))
def read_ap_to_dict(filename):
    ap_dict = collections.OrderedDict()
    with open(filename) as file:
        for line in file:
            elements = re.split(r'[\s]', line.strip())
            ap_dict[elements[0]] = (elements[1], elements[2])
    return ap_dict


WIFI_DICT = read_ap_to_dict(wifi_filename)

# *****************************************************************************************************
# 2. Pre-processing the 8 xml files, generate standard input data
# instantiate a WifiFile object for each background file collected


class WifiFile(object):
    # Class variable
    world_ap_dict = WIFI_DICT
    file_rank = 0

    def __init__(self, file_name):
        # Member variables
        self.wr_dict = collections.OrderedDict()
        self.loc_dict = collections.OrderedDict()
        self.fn = file_name

        # Transfer the data from raw file into internal data structure
        self.first_parse_file(file_name)
        self.sample_num = len(self.loc_dict)
        self.f_inputs = np.zeros((self.sample_num, NUM_COLUMNS))
        self.f_outputs = np.zeros((self.sample_num, 2))

        self.generate_instances()
        # self.f_outputs[:, 2:] = self.latlng_to_utm(self.f_outputs[:, :2], axis_x=488278.2671953467,axis_y=6199937)

        # Save standard input and output into files
        self.save_overall_csv()

    def first_parse_file(self, file_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.documentElement

        wr_list = root.getElementsByTagName('wr')
        #loc_list = root.getElementsByTagName('loc')

        print("# wifi record:", wr_list.length)
        #print("# loc record:", loc_list.length)

        # location(user input)
        i = 0
        #for loc, wr in zip(loc_list, wr_list):
        for  wr in zip(wr_list):
            i = i+1
            # try:
            #     lat = float(loc.getAttribute("lat"))
            #     lng = float(loc.getAttribute("lng"))
            # except ValueError:
            #     print('invalid input: %s,%s'.format(lat, lng))
            # self.loc_dict[i] = (lat, lng)

            # ap_list是一个ap的列表，一个ap_list表示一个<wr>，代表一个time step记录下来的一个ap的列表
            ap_list = list()
            for record, j in zip(wr.childNodes, range(len(wr.childNodes))):  # for each AP
                if j % 2:
                    ap = wr.childNodes[j].getAttribute("b")
                    s = wr.childNodes[j].getAttribute("s")
                    if ap not in self.world_ap_dict.keys():
                        # pass
                        # self.world_wifi[ap] = 1
                        print("{} not in world ap dict".format(ap))
                    else:
                        ap_list.append((ap, s))
            self.wr_dict[i] = ap_list

    def generate_instances(self):
        if len(self.loc_dict) == len(self.wr_dict):
            i = 0
            for out, inp in zip(self.loc_dict.values(), self.wr_dict.values()):
                self.f_inputs[i, :] = self.func(inp)
                self.f_outputs[i, :2] = out
                i = i + 1

    def func(self, ori_input):
        wr = WifiFile.formalize_wr(ori_input)
        return WifiFile.normalize_wifi_inputs(wr)

    # Normalise each APs strength to [0,1]
    @staticmethod
    def normalize_wifi_inputs(wr_inputs):

        zero_index = np.where(wr_inputs == 0)
        wr_inputs[zero_index] = -100

        max = -40
        min = -100
        wr_inputs = (wr_inputs - min) / (max - min)

        return wr_inputs

    @staticmethod
    def formalize_wr(wr):
        element = np.zeros(NUM_COLUMNS)     # standard input need same number of input ap
        for ap in wr:
            ap_id = ap[0]
            ap_val = ap[1]
            # find out the index（column index in element） of this ap_id
            ap_index = int(WifiFile.world_ap_dict[ap_id][1]) - 1
            element[ap_index] = ap_val
        return element

    # @staticmethod
    # def latlng_to_utm(outputs, axis_x, axis_y):
    #     for i in range(len(outputs)):
    #         x = utm.from_latlon(outputs[:, 0], outputs[:, 1])[0] - float(axis_x)
    #         y = utm.from_latlon(outputs[:, 0], outputs[:, 1])[1] - float(axis_y)
    #     return np.vstack((x, y))

    def save_overall_csv(self):
        txt_filename = "sample_" + str(NUM_COLUMNS) + ".csv"
        write_text = np.hstack((self.f_outputs, self.f_inputs))
        with open(txt_filename, "ab") as f:  # 以append的形式附加
            np.savetxt(f, write_text, delimiter=",", newline='\n')


# =============================================================================
# Iterate over all the background file in the directory "background"
def iterate(path):
    dirs = os.listdir(path)
    for dir in dirs:
        if dir.endswith("xml"):
            fi_d = os.path.join(path, dir)
            WifiFile(fi_d)
        else:
            pass
            # using "continue" here is the same as using "pass"
# =============================================================================


file = str(Scenario)+"_sample_" + str(NUM_COLUMNS) + ".csv"
if os.path.isfile(file):
    os.remove(file)

iterate("./")
# WifiFile("foreground(corridor1)_2019-06-01 18-19-08.xml")
