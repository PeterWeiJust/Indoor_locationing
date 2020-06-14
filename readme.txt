This is the branch for data processing。
1.wifi_processing:
Bucharest wifi data is saved in Timed Data folder with each scenario and each round of data.
Simple run wifi_feature.py first to go through all the xml files in Timed Data folder to get all the appeared AP (shown as mac address) + appeared frequency + order in the xml file.
Hence, we generate a txt list named as wifi_id. The total number of AP is decided by all the xml files in one scenario folder.

Then run parse_wifi.py which firstly read the wifi_id.txt to convert the raw log xml file to a csv file which format as time + AP0, AP1, ... APn. (n, number of AP)
Please temparely rename the generated csv as scenario1-route1,scenario1-route2...scenario1-route8 in each sub folder to make sure the later sensor & wifi combination code could work normally. (we will optimise this inconvient issues tomorrow 06.16 )