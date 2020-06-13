import os
import xml.dom.minidom
import collections


# 1. scan all the wifi signal in background file, write distinct access point into a file
#    it can iterate over all the background files in a specified directory
class WifiRecord(object):
    world_wifi = {}
    world_ordered_wifi = {}

    def __init__(self, path):
        self.traverse_all(path)
        self.generate_ordered_wifi()

    def scan_wifi(self, file_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.documentElement

        wr_list = root.getElementsByTagName('wr')
        for item, i in zip(wr_list, range(len(wr_list))):  # for each time step
            for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
                if j % 2:
                    ap = item.childNodes[j].getAttribute("b")
                    if ap not in self.world_wifi.keys():
                        self.world_wifi[ap] = 1
                    else:
                        self.world_wifi[ap] = self.world_wifi[ap] + 1

    def traverse_all(self, path):
        dirs = os.listdir(path)
        for dd in dirs:
            if dd.endswith("xml"):
                fi_d = os.path.join(path, dd)
                print("processing ", fi_d)
                self.scan_wifi(fi_d)

    def generate_ordered_wifi(self):
        self.world_ordered_wifi = collections.OrderedDict(
            sorted(self.world_wifi.items(), key=lambda t: t[1], reverse=True))


def write_ap_to_file(wr, filename):
    file = open(filename, 'w')
    for wifi_id, times, rank in zip(wr.world_ordered_wifi.keys(), wr.world_ordered_wifi.values(),
                                    range(len(wr.world_ordered_wifi))):
        file.write('{}\t{}\t{}\n'.format(str(wifi_id), str(times), str(rank+1)))
    file.close()

Scenario=1
wifi_filename = "wifi_id_"+str(Scenario)+".txt"
wr1 = WifiRecord("./")
write_ap_to_file(wr1, wifi_filename)

