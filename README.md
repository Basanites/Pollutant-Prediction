# Atmospheric Pollutant Prediction

The purpose of this software is to compare different models used for predictions on the specific use case of air pollutant data.
The used data is sourced from the [EEA E1a and E2a download service](http://discomap.eea.europa.eu/map/fme/AirQualityExport.htm) and downloaded CSVs need to be placed in ./res.

The program is based on python3.6.
To install the requirements run
```
pip install -r requirements.txt
```

For a regular analysis process run
```
python converter.py
python parameter_estimation.py
python evaluation.py
python queries.py
```
