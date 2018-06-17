# Bachelorthesis

The purpose of this software is to enable a comparison of different statistical models for time series prediction.

It is built around a GUI which lets the user import time series (at the moment only in the form of csvs provided by the [EEA weather token service]( http://discomap.eea.europa.eu/map/fme/AirQualityExport.htm).
The GUI has options to
- [] select which files should be imported
- [] navigate and visualize the imported dataframe
- [] display statistical plots (decomposition, acf, pacf, ...)
- [] plot predictions and raw data
- [] present information on the accuracy of a prediction model
- [] compare different prediction models

To start the program simply run 
```
python3 core.py
```
