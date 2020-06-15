This is the branch for data processing.
The processing part mainly two steps: wifi and sensor processing.
For wifi:
Bucharest wifi data is saved in Timed Data folder with each scenario and each round of data.
Simple run wifi_feature.py first to go through all the xml files in Timed Data folder to get all the appeared AP (shown as mac address) + appeared frequency + order in the xml file.
Hence, we generate a txt list named as wifi_id. The total number of AP is decided by all the xml files in one scenario folder.

Then run parse_wifi.py which firstly read the wifi_id.txt to convert the raw log xml file to a csv file which format as time + AP0, AP1, ... APn. (n, number of AP)
Please temparely rename the generated csv as scenario1-route1,scenario1-route2...scenario1-route8 in each sub folder to make sure the later sensor & wifi combination code could work normally. (we will optimise this inconvient issues tomorrow 06.16 )


For sensor:
1. Run routeprocess.py, which will extract lat/lng from ground_truth xml and generate the label according to record time.
2. Run generate_ground_truth.cpp, which will generate linear interploated sensor data.
3. Run datagenerate.py, in this process we will generate labels in all routes and get the timestep100 files. The data will be placed under Scenario_1 folder.
4. And then, do wifi process until all scenario-route wifi are generated. The final step is to run Combine_sensor_wifi.cpp and we can get sensor_wifi data file. We combine them according to their record time, choose the time closest wifi RSS record for each sensor record. If no wifi signal found, we fill the RSS with 0. 


