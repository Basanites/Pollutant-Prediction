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

    for timebased in [True, False]:
        for differenced in [True, False]:
            if timebased:
                options = [False]
            else:
                options = [True, False]

            for artificial in options:
                # count of first places by pollutant and distance for model on mse
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant']].sort_values(
                    by='mean_squared_error').groupby(
                    ['distance', 'station', 'pollutant'], as_index=False).first().groupby(
                    ['pollutant', 'distance', 'model'], as_index=False).count().rename(
                    index=str, columns={'mean_squared_error': 'best_mse_count'})[
                    ['pollutant', 'model', 'distance', 'best_mse_count']].sort_values(
                    by=['pollutant', 'distance', 'best_mse_count'], ascending=[True, True, False])

                # count of first places by pollutant for model over mse mean by distance
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant']].groupby(
                    ['model', 'station', 'pollutant'], as_index=False).mean().sort_values(
                    by='mean_squared_error').groupby(
                    ['station', 'pollutant'], as_index=False).first().sort_values(by='mean_squared_error').groupby(
                    ['model', 'pollutant'], as_index=False).count()[
                    ['pollutant', 'model', 'mean_squared_error']].sort_values(
                    by=['pollutant', 'mean_squared_error'], ascending=[True, False]).rename(
                    index=str, columns={'mean_squared_error': 'best_mse_by_distance_avg'})

                # count of first places by pollutant for model over mse mean by distance. independent of artificial
                frame[(~(frame.direct == timebased)) & (frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant']].groupby(
                    ['model', 'station', 'pollutant'], as_index=False).mean().sort_values(
                    by='mean_squared_error').groupby(
                    ['station', 'pollutant'], as_index=False).first().sort_values(by='mean_squared_error').groupby(
                    ['model', 'pollutant'], as_index=False).count()[
                    ['pollutant', 'model', 'mean_squared_error']].sort_values(
                    by=['pollutant', 'mean_squared_error'], ascending=[True, False]).rename(
                    index=str, columns={'mean_squared_error': 'best_mse_by_distance_avg'})

                # Evaluate model averages depending on distance, rate and pollutant. Ordered by nmae mean.
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced)][
                    ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures, *times]].groupby(
                    ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                    ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                    ['rate', 'pollutant', 'model', 'distance', ('norm_mean_absolute_error', 'mean')])

                # Evaluate model averages depending on distance, rate and pollutant. Best model for pollutant and
                # distance combo based on nmae mean.
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced)][
                    ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures, *times]].groupby(
                    ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                    ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                    [('norm_mean_absolute_error', 'mean'), 'rate', 'pollutant', 'model', 'distance']).groupby(
                    ['rate', 'pollutant', 'distance']).first()

                # Evaluate model averages depending on distance, rate and pollutant. Best model per pollutant mae mean.
                # Count amount of best per pollutant
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced)][
                    ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures, *times]].groupby(
                    ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                    ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                    ['rate', 'pollutant', 'distance', ('norm_mean_absolute_error', 'mean')]).groupby(
                    ['rate', 'pollutant', 'distance']).first().groupby(
                    ['pollutant', 'model']).count()[['rate']].rename(
                    index=str, columns={'rate': 'best_count'}).reset_index().sort_values(
                    ['pollutant', 'best_count'], ascending=[True, False])

                # Evaluate model averages depending on distance, rate and pollutant. Best model per pollutant mae mean.
                # Count amount of best per distance
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced)][
                    ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures, *times]].groupby(
                    ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                    ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                    ['rate', 'pollutant', 'distance', ('norm_mean_absolute_error', 'mean')]).groupby(
                    ['rate', 'pollutant', 'distance']).first().groupby(
                    ['distance', 'model']).count()[['rate']].rename(
                    index=str, columns={'rate': 'best_count'}).reset_index().sort_values(
                    ['distance', 'best_count'], ascending=[True, False])

                # General statistics for models based on rate
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced)][
                    ['model', 'station', 'rate', *norm_measures, *times]].groupby(
                    ['model', 'rate'], as_index=False).agg(
                    ['mean', 'median', 'min', 'max']).reset_index().sort_values(['rate', 'model'])

                # General statistics for models based on distance and rate
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced)][
                    ['model', 'station', 'rate', 'distance', *norm_measures, *times]].groupby(
                    ['model', 'rate', 'distance'], as_index=False).agg(
                    ['mean', 'median', 'min', 'max']).reset_index().sort_values(['rate', 'model', 'distance'])
