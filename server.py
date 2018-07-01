from flask import Flask, request, jsonify
from model import Model
from util.openaq import get_station_coordinates

app = Flask(__name__)

parameters = ['station', 'pollutant', 'forecast_steps', 'type']


@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Not Found: ' + request.url,
    }

    response = jsonify(message)
    response.status_code = 404

    return response


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad request: {}'.format(request.url),
    }

    response = jsonify(message)
    response.status_code = 400

    return response


@app.route('/forecast')
def forecast():
    try:
        station = request.args['station']
        pollutant = request.args['pollutant']
        forecast_steps = request.args.get('forecast_steps') or 24
        type = request.args.get('type') or 'forest'
        coordinates = get_station_coordinates(station)
    except KeyError:
        return bad_request()
    except IndexError:
        return not_found()

    forecast = model.forecast_series(station, pollutant)
    base_uri = request.base_url
    properties = {'station': station,
                  'pollutant': pollutant,
                  'type': type,
                  'forecast_steps': forecast_steps,
                  'forecast': forecast}

    return _build_JSON_response(coordinates, properties, base_uri)


@app.route('/pollutants')
def get_pollutants():
    station = request.args['station']
    coordinates = get_station_coordinates(station)
    pollutants = str(model.get_stations_pollutant(station))
    base_uri = request.base_url
    properties = {'station': station,
                  'pollutants': pollutants}

    return _build_JSON_response(coordinates, properties, base_uri)


def _build_JSON_response(coordinates, properties_dict, base_uri, **kwargs):
    response = _build_geoJSON_dict(coordinates, properties_dict, base_uri, **kwargs)
    response['status'] = 200
    response = jsonify(response)
    response.status_code = 200
    return response


def _build_geoJSON_dict(coordinates, properties_dict, base_uri, **kwargs):
    type = kwargs.get('type')
    if type == 'point' or not type:
        return _build_point_geoJSON_dict(coordinates, properties_dict, base_uri)


def _build_point_geoJSON_dict(coordinates, properties_dict, base_uri):
    return {'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': coordinates
            },
            'properties': _build_properties_JSON_dict(properties_dict, base_uri)
            }


def _build_properties_JSON_dict(properties_dict, base_uri):
    query_uri = base_uri + '?'
    out = {}

    for k, v in properties_dict.items():
        if v is not None:
            out[k] = v
            if v in parameters:
                query_uri += '&{}={}'.format(k, v)

    out['uri'] = query_uri.replace('&', '', 1)

    return out


if __name__ == '__main__':
    model = Model()
    model.import_csvs(['./res/DE_1_6085_2013_timeseries.csv'])
    print('imported')
    app.run(debug=True, port=5000)
