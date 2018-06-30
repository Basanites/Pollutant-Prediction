from flask import Flask, request, jsonify
from model import Model
from util.openaq import get_station_coordinates

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp


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
            'uri': '{0}?station={1}&pollutant={2}&forecast_steps={3}&type={4}'.format(request.base_url, station,
                                                                                      pollutant, forecast_steps, type)
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


def _build_properties():
    pass


def _build_geoJSON(point_coordinates, type='feature'):
    if type.lower() == 'feature':
        return _build_geoJSON_feature(point_coordinates)
    return {}


def _build_geoJSON_feature(point_coordinates):
    pass


if __name__ == '__main__':
    model = Model()
    model.import_csvs(['./res/DE_1_6085_2013_timeseries.csv'])
    print('imported')
    app.run(debug=True, port=5000)
