import os
import xml.dom.minidom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_functions as dfunction


# 由于label文件中存在时间倒序的情况，此文件对Route文件进行重新排序
def read_route(file_name):
    dom = xml.dom.minidom.parse(file_name)
    root = dom.documentElement

    position = root.getElementsByTagName('position')
    position_list = []
    for pos, i in zip(position, range(len(position))):
        x = pos.getAttribute("x_pos")
        y = pos.getAttribute("y_pos")
        time = pos.getAttribute("time")
        position_list.append([time, x, y])
    ll = file_name.split("/")
    # dir_name = str(ll[-3] + "/" + ll[-2] + "/" + ll[-1])[:-4]
    df = pd.DataFrame(position_list, columns=['time', 'x', 'y'])
    df1 = df.sort_values(by='time')

    xy_numpy = np.array(df[['y', 'x']], dtype=float)
    xy_numpy_nor = np.transpose(np.vstack((dfunction.normalisation(xy_numpy[:, 0]), dfunction.normalisation(xy_numpy[:, 1]))))

    # plt.xlabel("normalised x_pos")
    # plt.ylabel("normalised y_pos")

    # plt.plot(xy_numpy_nor[:,0], xy_numpy_nor[:, 1])
    # plt.show()

    fig, ax = plt.subplots()
    # 打开交互模式
    plt.ion()
    for i in range(xy_numpy_nor.shape[0]):
        # 清除原有图像
        # plt.cla()
        data = xy_numpy_nor[:i+1, :] # 每迭代一次，将i放入y1中画出来
        ax.cla()  # 清除键
        ax.plot(data[:, 0], data[:, 1], label=i)
        # plt.pause(0.1)
        ax.scatter(data[:-1, 0], data[:-1, 1], c='g', marker='o')
        ax.plot(data[-1, 0], data[-1, 1], c='r', marker='^')
        # ax.legend()
        plt.title(i)
        plt.pause(0.5)

        # 关闭交互模式
    plt.ioff()

    # 显示图形
    plt.show()

    # dir_name = dir_name + '_label.csv'
    # if os.path.isfile(dir_name):
    #     os.remove(dir_name)
    # df.to_csv(dir_name)


def traverse_all(path):
    dirs = os.listdir(path)
    for dir in dirs:
        if dir != ".DS_Store":
            dir = os.path.join(path, dir)
            files = os.listdir(dir)
            for file in files:
                if file.endswith("Route.xml"):
                    f_id = os.path.join(dir, file)
                    print("processing ", f_id)
                    read_route(f_id)


# traverse_all("./Timed Data")
read_route('./Timed Data/Scenario_1/1/ground_truth_1.xml')