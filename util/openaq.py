import requests

def get_station_coordinates(station):
    payload = {'location': station}
    info = requests.get('https://api.openaq.org/v1/locations', payload).json()
    coordinates = info['results'][0]['coordinates']

    return [coordinates['latitude'], coordinates['longitude']]