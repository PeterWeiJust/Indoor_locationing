The alpha-keras branch contains the dataset and keras model in all Bucharest scenarios. All sensor and wifi data are placed under timestep100 and sensordata folders, and in every .py file, we use 'wandb' module to make it run online and plot pictures automatically. We can run downsample_keras.py/mmloc_keras.py/mmloc_residual_keras.py/sensor_baseline_keras.py/wifi_regression_keras.py directly without parameters.
Downsample_keras: It the model of downsampling sensor, uses sensor data to predict lat/lng.

mmloc_keras: The multimodal models using both sensor and wifi data to predict lat/lng. But the fusion function of sensor and wifi modules are just simply multiplication.

mmloc_residual_keras.py: Uses multimodal residual network(MRN) to replace multimodal models, also used to predict lat/lng.

sensor_baseline_keras.py: Sensor data without using downsample to predict lat/lng.

wifi_regression_keras.py: Wifi data used to predict lat/lng.

plottingcdf.py: Run this function to draw the cdf  of all models after they are trained.



 