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
                # count of best mse for model on pollutant and distance
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant', 'rate']].sort_values(
                    by='mean_squared_error').groupby(
                    ['rate', 'distance', 'station', 'pollutant'], as_index=False).first().groupby(
                    ['rate', 'pollutant', 'distance', 'model'], as_index=False).count().rename(
                    index=str, columns={'mean_squared_error': 'best_mse_count'})[
                    ['rate', 'pollutant', 'model', 'distance', 'best_mse_count']].sort_values(
                    by=['rate', 'pollutant', 'distance', 'best_mse_count'], ascending=[True, True, True, False])

                # count best mse for model on distance
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant', 'rate']].sort_values(
                    by='mean_squared_error').groupby(
                    ['rate', 'distance', 'station', 'pollutant'], as_index=False).first().groupby(
                    ['rate', 'pollutant', 'distance', 'model'], as_index=False).count().rename(
                    index=str, columns={'mean_squared_error': 'best_mse_count'})[
                    ['rate', 'pollutant', 'model', 'distance', 'best_mse_count']].sort_values(
                    by=['rate', 'pollutant', 'distance', 'best_mse_count'],
                    ascending=[True, True, True, False]).groupby(
                    ['rate', 'model', 'distance'], as_index=False).sum().sort_values(
                    ['rate', 'distance', 'best_mse_count'], ascending=[True, True, False])

                # count of first places by pollutant for model over mse mean by distance
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant', 'rate']].groupby(
                    ['rate', 'model', 'station', 'pollutant'], as_index=False).mean().sort_values(
                    by='mean_squared_error').groupby(
                    ['rate', 'station', 'pollutant'], as_index=False).first().sort_values(
                    by='mean_squared_error').groupby(
                    ['rate', 'model', 'pollutant'], as_index=False).count()[
                    ['rate', 'pollutant', 'model', 'mean_squared_error']].sort_values(
                    by=['rate', 'pollutant', 'mean_squared_error'], ascending=[True, True, False]).rename(
                    index=str, columns={'mean_squared_error': 'best_mse_by_distance_avg'}).sort_values(
                    ['rate', 'pollutant', 'best_mse_by_distance_avg'], ascending=[True, True, False])

                # split on rate to preserve internal sense of size
                for rate in ['D', 'H']:
                    # General statistics for models based on rate
                    # Informative nature
                    frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', *norm_measures]].groupby(
                        ['model', ], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        [('norm_median_absolute_error', 'mean')])

                    # General statistics for models based on distance and rate
                    # Informative nature
                    frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', 'distance', *norm_measures]].groupby(
                        ['model', 'distance'], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        ['distance', ('norm_median_absolute_error', 'mean')])

                    # General statistics for models based on pollutant and rate
                    # Informative nature
                    frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', 'pollutant', *norm_measures]].groupby(
                        ['model', 'pollutant'], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        ['pollutant', ('norm_median_absolute_error', 'mean')])

                    # General statistics for models prediction times
                    # Informative nature
                    frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', 'pollutant', *times]].groupby(
                        ['model'], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        [('prediction_time', 'mean')])

                    # General statistics for models fit times
                    # Informative nature
                    frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', 'pollutant', *times]].groupby(
                        ['model'], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        [('fit_time', 'mean')])

                ###########################################################################################
                # model averages depending on distance, rate and pollutant. Ordered by nmae mean.
                # Very big, only for informative purposes.
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures]].groupby(
                    ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                    ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                    ['rate', 'pollutant', 'model', 'distance', ('norm_mean_absolute_error', 'mean')])

                # model averages depending on distance, rate and pollutant. Best model for pollutant and
                # distance combo based on nmae mean.
                # Very big, only for informative purposes.
                frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures]].groupby(
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
                ####################################################################################################
