from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict')
def predict():
    station = request.args.get('station')
    interval = request.args.get('interval')
    model = request.args.get('model')

    return jsonify({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [1, 1]
        },
        'properties': {
            'station': station,
            'interval': interval,
            'model': model,
            'uri': '/predict?{}&{}&{}'.format(station, interval, model)
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
