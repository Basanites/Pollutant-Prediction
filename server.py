from flask import Flask, request, jsonify
from model import Model
from util.openaq import get_station_coordinates

app = Flask(__name__)


@app.route('/predict')
def predict():
    station = request.args.get('station')
    pollutant = request.args.get('pollutant')
    forecast_steps = request.args.get('forecast_steps')
    type = request.args.get('type')
    coordinates = get_station_coordinates(station)


    return jsonify({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': coordinates
        },
        'properties': {
            'station': station,
            'pollutant': pollutant,
            'forecast_steps': forecast_steps,
            'type': type,
            'uri': '/predict?{}&{}&{}'.format(station, forecast_steps, type)
        }
    })

@app.route('/pollutants')
def get_pollutants():
    station = request.args['station']
    coordinates = get_station_coordinates(station)
    pollutants = str(model.get_stations_pollutant(station))

    return jsonify({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': coordinates
        },
        'properties': {
            'station': station,
            'pollutants': pollutants
        }
    })

if __name__ == '__main__':
    model = Model()
    model.import_csvs(['./res/DE_1_6085_2013_timeseries.csv'])
    print('imported')
    app.run(debug=True, port=5000)
