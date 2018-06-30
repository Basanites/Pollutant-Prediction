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


@app.route('/forecast')
def forecast():
    station = request.args.get('station')
    pollutant = request.args.get('pollutant')
    forecast_steps = request.args.get('forecast_steps')
    type = request.args.get('type')

    try:
        coordinates = get_station_coordinates(station)
    except IndexError:
        return not_found()
    base_uri = request.base_url
    properties = {'station': station,
                  'pollutant': pollutant,
                  'type': type,
                  'forecast_steps': forecast_steps}

    return jsonify(_build_JSON_response(coordinates, properties, base_uri))


@app.route('/pollutants')
def get_pollutants():
    station = request.args['station']
    coordinates = get_station_coordinates(station)
    pollutants = str(model.get_stations_pollutant(station))
    base_uri = request.base_url
    properties = {'station': station,
                  'pollutants': pollutants}

    return jsonify(_build_JSON_response(coordinates, properties, base_uri))

def _build_JSON_response(coordinates, properties_dict, base_uri, **kwargs):
    response = _build_geoJSON(coordinates, properties_dict, base_uri, **kwargs)
    response['status'] = 200
    return response


def _build_geoJSON(coordinates, properties_dict, base_uri, **kwargs):
    type = kwargs.get('type')
    if type == 'point' or not type:
        return _build_point_geoJSON(coordinates, properties_dict, base_uri)


def _build_point_geoJSON(coordinates, properties_dict, base_uri):
    return {'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': coordinates
            },
            'properties': _build_properties_JSON(properties_dict, base_uri)
            }


def _build_properties_JSON(properties_dict, base_uri):
    query_uri = base_uri + '?'
    out = {}

    for k, v in properties_dict.items():
        if v:
            out[k] = v
            query_uri += '&{}={}'.format(k, v)

    out['uri'] = query_uri.replace('&', '', 1)

    return out


if __name__ == '__main__':
    model = Model()
    model.import_csvs(['./res/DE_1_6085_2013_timeseries.csv'])
    print('imported')
    app.run(debug=True, port=5000)
