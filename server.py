from flask import Flask, request, jsonify
from model import Model

app = Flask(__name__)


@app.route('/predict')
def predict():
    station = request.args.get('station')
    interval = request.args.get('interval')
    type = request.args.get('type')
    coordinates = request.args.get('coordinates')

    return jsonify({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [1, 1]
        },
        'properties': {
            'station': station,
            'interval': interval,
            'model': type,
            'coordinates': coordinates,
            'uri': '/predict?{}&{}&{}'.format(station, interval, type)
        }
    })

@app.route('/pollutants')
def get_pollutants():
    station = request.args['station']

    return jsonify({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [1, 1]
        },
        'properties': {
            'station': station,
            'pollutants': str(model.get_stations_pollutant(station))
        }
    })

if __name__ == '__main__':
    model = Model()
    model.import_csvs(['./res/DE_1_6085_2013_timeseries.csv'])
    print('imported')
    app.run(debug=True, port=5000)
