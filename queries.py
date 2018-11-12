import glob
import os

import pandas as pd


def write_to_file(folder, name, content):
    with open(f'./{folder}/{name}', 'w') as file:
        file.write(content)


def _export_dataframe(df, name):
    df.to_excel(f'./{excel_folder}/{name}.xlsx')
    write_to_file(tex_folder, f'{name}.tex', df.to_latex())


def generate_name(timebased_, differenced_=None, artificial_=None, rate_=None):
    out = f'timebased={timebased_}'
    if differenced_ is not None:
        out += f'-differenced={differenced_}'
    if artificial_ is not None:
        out += f'-artificial={artificial_}'
    if rate_ is not None:
        out += f'-rate={rate}'
    return out


if __name__ == '__main__':
    eval_folder = 'eval'
    tex_folder = 'tex'
    excel_folder = 'excel'
    if not os.path.exists(f'./{tex_folder}'):
        os.makedirs(f'./{tex_folder}')
    if not os.path.exists(f'./{excel_folder}'):
        os.makedirs(f'./{excel_folder}')

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

        # statistical information about prediction times in general
        current_name = generate_name(timebased)
        general_prediction_times = frame[~(frame.direct == timebased)][
            ['model', 'station', 'pollutant', *times]].groupby(
            ['model'], as_index=False).agg(
            ['mean', 'median', 'min', 'max']).reset_index().sort_values(
            [('prediction_time', 'mean')])
        _export_dataframe(general_prediction_times, 'general_prediction_times-' + current_name)

        # statistical information about fit times in general
        current_name = generate_name(timebased)
        general_fit_times = frame[~(frame.direct == timebased)][
            ['model', 'station', 'pollutant', *times]].groupby(
            ['model'], as_index=False).agg(
            ['mean', 'median', 'min', 'max']).reset_index().sort_values(
            [('fit_time', 'mean')])
        _export_dataframe(general_fit_times, 'general_fit_times-' + current_name)

        for differenced in [True, False]:
            if timebased:
                options = [False]
            else:
                options = [True, False]

            for artificial in options:
                current_name = generate_name(timebased, differenced, artificial)

                # count of best mse for model on pollutant and distance
                best_by_dist_and_poll = frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant', 'rate']].sort_values(
                    by='mean_squared_error').groupby(
                    ['rate', 'distance', 'station', 'pollutant'], as_index=False).first().groupby(
                    ['rate', 'pollutant', 'distance', 'model'], as_index=False).count().rename(
                    index=str, columns={'mean_squared_error': 'best_mse_count'})[
                    ['rate', 'pollutant', 'model', 'distance', 'best_mse_count']].sort_values(
                    by=['rate', 'pollutant', 'distance', 'best_mse_count'], ascending=[True, True, True, False])
                _export_dataframe(best_by_dist_and_poll, 'best_by_dist_and_poll-' + current_name)

                # count best mse for model on distance
                best_by_dist = frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
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
                _export_dataframe(best_by_dist, 'best_by_dist-' + current_name)

                # # count of first places by pollutant for model over mse mean by distance
                # best_by_poll = frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                #         frame.differenced == differenced)][
                #     ['mean_squared_error', 'model', 'distance', 'station', 'pollutant', 'rate']].groupby(
                #     ['rate', 'model', 'station', 'pollutant'], as_index=False).mean().sort_values(
                #     by='mean_squared_error').groupby(
                #     ['rate', 'station', 'pollutant'], as_index=False).first().sort_values(
                #     by='mean_squared_error').groupby(
                #     ['rate', 'model', 'pollutant'], as_index=False).count()[
                #     ['rate', 'pollutant', 'model', 'mean_squared_error']].sort_values(
                #     by=['rate', 'pollutant', 'mean_squared_error'], ascending=[True, True, False]).rename(
                #     index=str, columns={'mean_squared_error': 'best_mse_by_distance_avg'}).sort_values(
                #     ['rate', 'pollutant', 'best_mse_by_distance_avg'], ascending=[True, True, False])
                # _export_dataframe(best_by_poll, 'best_by_poll-' + current_name)

                best_by_poll = frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                        frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant', 'rate']].sort_values(
                    by='mean_squared_error').groupby(
                    ['rate', 'pollutant', 'station', 'distance'], as_index=False).first().groupby(
                    ['rate', 'distance', 'pollutant', 'model'], as_index=False).count().rename(
                    index=str, columns={'mean_squared_error': 'best_mse_count'})[
                    ['rate', 'pollutant', 'model', 'distance', 'best_mse_count']].sort_values(
                    by=['rate', 'pollutant', 'distance', 'best_mse_count'],
                    ascending=[True, True, True, False]).groupby(
                    ['rate', 'model', 'pollutant'], as_index=False).sum().sort_values(
                    ['rate', 'pollutant', 'best_mse_count'], ascending=[True, True, False])[
                    ['rate', 'pollutant', 'model', 'best_mse_count']]
                _export_dataframe(best_by_poll, 'best_by_poll-' + current_name)

                # count of first places for nmae split by artificial
                artificial_comparison = frame[(~(frame.direct == timebased)) & (frame.differenced == differenced)][
                    ['mean_squared_error', 'model', 'distance', 'station', 'pollutant', 'rate',
                     'artificial']].sort_values(
                    by='mean_squared_error').groupby(
                    ['station', 'rate', 'pollutant', 'distance', 'model'], as_index=False).first().groupby(
                    ['model', 'artificial', 'rate'], as_index=False).count()[
                    ['rate', 'model', 'artificial', 'mean_squared_error']].rename(
                    index=str, columns={'mean_squared_error': 'best_mse_count'}).sort_values(
                    ['rate', 'model', 'best_mse_count'], ascending=[True, True, False])
                _export_dataframe(artificial_comparison, 'best_by_artificial-' + generate_name(timebased, differenced))

                # count of first places for nmae split by differenced
                differenced_comparison = frame[(~(frame.direct == timebased)) & (frame.artificial == artificial)][
                    ['norm_mean_absolute_error', 'model', 'distance', 'station', 'pollutant', 'rate',
                     'differenced']].sort_values(
                    by='norm_mean_absolute_error').groupby(
                    ['station', 'rate', 'pollutant', 'distance', 'model'], as_index=False).first().groupby(
                    ['model', 'differenced', 'rate'], as_index=False).count()[
                    ['rate', 'model', 'differenced', 'norm_mean_absolute_error']].rename(
                    index=str, columns={'norm_mean_absolute_error': 'count_best_nmae'}).sort_values(
                    ['rate', 'model', 'count_best_nmae'], ascending=[True, True, False])
                _export_dataframe(differenced_comparison,
                                  'best_by_differenced-' + generate_name(timebased, artificial_=artificial))

                # split on rate to preserve internal sense of size
                for rate in ['D', 'H']:
                    current_name = generate_name(timebased, differenced, artificial, rate)

                    # General statistics for models based on rate
                    # Informative nature
                    general_norm_measures = frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', *norm_measures]].groupby(
                        ['model', ], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        [('norm_mean_absolute_error', 'mean')])
                    _export_dataframe(general_norm_measures, 'general_norm_measures-' + current_name)

                    # General statistics for models based on distance and rate
                    # Informative nature
                    distance_norm_measures = frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', 'distance', *norm_measures]].groupby(
                        ['model', 'distance'], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        ['distance', ('norm_mean_absolute_error', 'mean')])
                    _export_dataframe(distance_norm_measures, 'distance_norm_measures-' + current_name)

                    # General statistics for models based on pollutant and rate
                    # Informative nature
                    pollutant_norm_measures = frame[
                        (~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                                frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', 'pollutant', *norm_measures]].groupby(
                        ['model', 'pollutant'], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        ['pollutant', ('norm_median_absolute_error', 'mean')])
                    _export_dataframe(pollutant_norm_measures, 'pollutant_norm_measures-' + current_name)

                    # General statistics for models prediction times
                    # Informative nature
                    general_prediction_times = frame[
                        (~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                                frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', 'pollutant', *times]].groupby(
                        ['model'], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        [('prediction_time', 'mean')])
                    _export_dataframe(general_prediction_times, 'general_prediction_times-' + current_name)

                    # General statistics for models fit times
                    # Informative nature
                    general_fit_times = frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                            frame.differenced == differenced) & (frame.rate == rate)][
                        ['model', 'station', 'pollutant', *times]].groupby(
                        ['model'], as_index=False).agg(
                        ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                        [('fit_time', 'mean')])
                    _export_dataframe(general_fit_times, 'general_fit_times-' + current_name)

                # ###########################################################################################
                # # model averages depending on distance, rate and pollutant. Ordered by nmae mean.
                # # Very big, only for informative purposes.
                # frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                #         frame.differenced == differenced)][
                #     ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures]].groupby(
                #     ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                #     ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                #     ['rate', 'pollutant', 'model', 'distance', ('norm_mean_absolute_error', 'mean')])
                #
                # # model averages depending on distance, rate and pollutant. Best model for pollutant and
                # # distance combo based on nmae mean.
                # # Very big, only for informative purposes.
                # frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                #         frame.differenced == differenced)][
                #     ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures]].groupby(
                #     ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                #     ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                #     [('norm_mean_absolute_error', 'mean'), 'rate', 'pollutant', 'model', 'distance']).groupby(
                #     ['rate', 'pollutant', 'distance']).first()
                #
                # # Evaluate model averages depending on distance, rate and pollutant. Best model per pollutant mae mean.
                # # Count amount of best per pollutant
                # frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                #         frame.differenced == differenced)][
                #     ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures, *times]].groupby(
                #     ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                #     ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                #     ['rate', 'pollutant', 'distance', ('norm_mean_absolute_error', 'mean')]).groupby(
                #     ['rate', 'pollutant', 'distance']).first().groupby(
                #     ['pollutant', 'model']).count()[['rate']].rename(
                #     index=str, columns={'rate': 'best_count'}).reset_index().sort_values(
                #     ['pollutant', 'best_count'], ascending=[True, False])
                #
                # # Evaluate model averages depending on distance, rate and pollutant. Best model per pollutant mae mean.
                # # Count amount of best per distance
                # frame[(~(frame.direct == timebased)) & (frame.artificial == artificial) & (
                #         frame.differenced == differenced)][
                #     ['model', 'distance', 'station', 'pollutant', 'rate', *norm_measures, *times]].groupby(
                #     ['pollutant', 'model', 'rate', 'distance'], as_index=False).agg(
                #     ['mean', 'median', 'min', 'max']).reset_index().sort_values(
                #     ['rate', 'pollutant', 'distance', ('norm_mean_absolute_error', 'mean')]).groupby(
                #     ['rate', 'pollutant', 'distance']).first().groupby(
                #     ['distance', 'model']).count()[['rate']].rename(
                #     index=str, columns={'rate': 'best_count'}).reset_index().sort_values(
                #     ['distance', 'best_count'], ascending=[True, False])
                # ####################################################################################################
