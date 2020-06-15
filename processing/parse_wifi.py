#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import re
import numpy as np
import pandas as pd
import os
import xml.dom.minidom



def read_ap_to_dict(filename):
    ap_dict = collections.OrderedDict()
    with open(filename) as file:
        for line in file:
            elements = re.split(r'[\s]', line.strip())
            ap_dict[elements[0]] = (elements[1], elements[2])
    return ap_dict


wifi_filename = "wifi_id.txt"
WIFI_DICT = read_ap_to_dict(wifi_filename)


# -----------------------------------------------------------------------------------------------


# Normalise each APs strength to [0,1]
def normalize_wifi_inputs(wr_inputs):
    wr_inputs = np.array(wr_inputs).astype(np.int)
    zero_index = np.where(wr_inputs == 0)
    wr_inputs[zero_index] = -100

    wifi_max = -40
    wifi_min = -100
    wr_inputs = (wr_inputs - wifi_min) / (wifi_max - wifi_min)

    return wr_inputs.tolist()


def formalize_wr(wr):
    ap_num = len(WIFI_DICT)  # standard input need same number of input ap
    # element = np.zeros(ap_num)
    element = [0 for i in range(ap_num)]
    for ap in wr:
        ap_id = ap[0]
        ap_val = ap[1]
        # find out the index（column index in element） of this ap_id
        ap_index = int(WIFI_DICT[ap_id][1]) - 1
        element[ap_index] = ap_val
    return element


def parse_wifi(dir, f_id):
    # 读route文件中的label
    # index = (f_id.split("/")[-1]).split("_")[1]
    # label_file = dir + "/" + index + "Route_label.csv"
    # df_label = pd.read_csv(label_file, index_col=0)
    # xy_array = df_label.values
    # t_start = mt.date_to_timestamp(xy_array[0, 0])
    # t_end = mt.date_to_timestamp(xy_array[-1, 0])
    # print(t_start + "  -  " + t_end + "\n")

    # 读scenariu文件中的wifi
    dom = xml.dom.minidom.parse(f_id)
    root = dom.documentElement
    a_t = root.getElementsByTagName('a')[0].getAttribute("t")
    m_t = root.getElementsByTagName('m')[0].getAttribute("t")
    g_t = root.getElementsByTagName('g')[0].getAttribute("t")
    t0 = min(a_t, m_t, g_t)
    wr_list = root.getElementsByTagName('wr')
    list_t_wifi = []
    for item, i in zip(wr_list, range(len(wr_list))):  # for each time step
        t = int(item.getAttribute("t"))
        delta_t = t - int(t0)
        # ap_list是一个ap的列表，一个ap_list表示一个wifi reading,即一个wifi sample inpput
        ap_list = list()
        for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
            if j % 2:
                ap = item.childNodes[j].getAttribute("b")
                s = item.childNodes[j].getAttribute("s")
                if ap not in WIFI_DICT.keys():
                    pass
                else:
                    ap_list.append((ap, s))
        wifi_vector = formalize_wr(ap_list)
        wifi_vector = normalize_wifi_inputs(wifi_vector)
        wifi_vector.insert(0, delta_t)
        wifi_vector.insert(0, t)
        list_t_wifi.append(wifi_vector)

    df = pd.DataFrame(list_t_wifi)
    df.columns = ['t', 'delta_t'] + ['ap' + str(i) for i in range(len(wifi_vector)-2)]
    out_file = "Timed Data/Scenario_1" + "/scenario1-" + f_id[22] + "route.csv"
    if os.path.isfile(out_file):
        os.remove(out_file)
    df.to_csv(out_file)


# Iterate over all the sensor wifi file in the directory "Timed_Data"
def iterate(path):
    dirs = os.listdir(path)
    for dir in dirs:
        if dir != ".DS_Store":
            f_id = os.path.join(path, dir)
            if os.path.isdir(f_id):
                iterate(f_id)
            else:
                if f_id.endswith(".xml") and bool(1 - dir.startswith("ground_truth_")):
                    print("processing: " + f_id + "...")
                    parse_wifi(path, f_id)
        else:
            pass
            # using "continue" here is the same as using "pass"


iterate("Timed Data")
