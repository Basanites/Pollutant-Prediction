import glob
import os

import pandas as pd


def write_to_file(folder, name, content):
    with open(f'./{folder}/{name}', 'w') as file:
        file.write(content)


if __name__ == '__main__':
    eval_folder = 'eval'
    tex_folder = 'tex'
    if not os.path.exists(f'./{tex_folder}'):
        os.makedirs(f'./{tex_folder}')

    frame = pd.DataFrame()
    for csv in glob.glob(f'{eval_folder}/*.csv'):
        station, rate = csv.replace('.csv', '').replace(f'{eval_folder}/', '').split('-')
        current_frame = pd.read_csv(csv)
        current_frame['station'] = station
        current_frame['rate'] = rate

        frame = pd.concat([frame, current_frame])

    frame = frame.reset_index()

    measures = ['median_absolute_error', 'mean_squared_error', 'mean_absolute_error']
    times = ['fit_time', 'prediction_time']
    norm_measures = [f'norm_{measure}' for measure in measures]

    # Evaluate averages depending on station, rate and pollutant. Ordered by nmse mean.
    for station in frame.station.unique():
        station_df = frame[frame.station == station]
        for rate in station_df.rate.unique():
            rate_df = station_df[station_df.rate == rate]
            for pollutant in rate_df.pollutant.unique():
                pollutant_df = rate_df[rate_df.pollutant == pollutant]

                best_avg = pollutant_df[['model', *norm_measures, *times]].groupby(
                    'model').agg(
                    ['mean', 'median', 'min', 'max']).sort_values(
                    [('norm_mean_absolute_error', 'mean')])

                tex = best_avg.to_latex()
                write_to_file(tex_folder, f'model_avg-{station}-{rate}-{pollutant}.tex', tex)

    # Evaluate distance averages depending on rate and pollutant. Ordered by nmse mean.
    for rate in frame.rate.unique():
        rate_df = frame[frame.rate == rate]
        for pollutant in rate_df.pollutant.unique():
            pollutant_df = rate_df[rate_df.pollutant == pollutant]
            for distance in pollutant_df.distance.unique():
                distance_df = pollutant_df[pollutant_df.distance == distance]

                best_avg = distance_df[['distance', *norm_measures, *times]].groupby(
                    'distance').agg(
                    ['mean', 'median', 'min', 'max']).sort_values(
                    [('norm_mean_absolute_error', 'mean')])

                tex = best_avg.to_latex()
                write_to_file(tex_folder, f'distance_avg-{rate}-{pollutant}-{distance}.tex', tex)
