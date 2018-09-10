import math

import pandas as pd

from parameter_estimation import resample_dataframe, get_info, scale_series, rescale_series, scale_dataframe, rescale_dataframe, \
    difference_series, dedifference_series, difference_dataframe, dediffference_dataframe


def get_daily():
    df = pd.DataFrame({
        'Timestamp': ['2013-01-11 23:00:00', '2013-01-12 23:00:00', '2013-01-14 23:00:00'],
        'PM10': [12, 19, 43],
        'PM2.5': [8, 13, 32]
    })

    df = df.set_index('Timestamp')
    df.index = pd.to_datetime(df.index)
    return df


def get_hourly():
    df = pd.DataFrame({
        'Timestamp': ['2013-01-01 00:00:00', '2013-01-01 01:00:00', '2013-01-01 03:00:00'],
        'NO': [36.848, 42.785, 16.865],
        'PM10': [200.165, 205.237, 27.727],
        'SO2': [27.458, 10.514, 4.948]
    })
    df = df.set_index('Timestamp')
    df.index = pd.to_datetime(df.index)
    return df


class TestHelpers:
    def test_daily(self):
        daily = get_daily()

        assert len(daily) is 3
        assert len(daily.columns) is 2
        assert '2013-01-12 23:00:00' in daily.index
        assert '2013-01-13 23:00:00' not in daily.index

    def test_hourly(self):
        hourly = get_hourly()

        assert len(hourly) is 3
        assert len(hourly.columns) is 3
        assert '2013-01-01 01:00:00' in hourly.index
        assert '2013-01-01 02:00:00' not in hourly.index


class TestResampleDataframe:
    def test_daily_resampling(self):
        daily = get_daily()
        resampled = resample_dataframe(daily, rate='D')

        assert len(resampled) is 4
        assert len(resampled.columns) is 2
        assert '2013-01-12' in resampled.index
        assert '2013-01-13' in resampled.index
        assert '2013-01-12 23:00:00' not in resampled.index

    def test_hourly_resampling(self):
        hourly = get_hourly()
        resampled = resample_dataframe(hourly, rate='H')

        assert len(resampled) is 4
        assert len(resampled.columns) is 3
        assert '2013-01-01 01:00:00' in resampled.index
        assert '2013-01-01 02:00:00' in resampled.index


class TestGetInfo:
    def test_daily(self):
        string = 'station-ignored-day.csv'
        station, rate = get_info(string, '')

        assert station == 'station'
        assert rate == 'D'

    def test_hourly(self):
        string = 'station-ignored-hour.csv'
        station, rate = get_info(string, '')

        assert station == 'station'
        assert rate == 'H'


class TestScalingSeries:
    def test_scale(self):
        series = get_daily()['PM10'].astype(float)
        scaled, scaler = scale_series(series)

        assert math.isclose(scaled.max(), 1, rel_tol=0.001)
        assert math.isclose(scaled.min(), -1, rel_tol=0.001)
        assert series.max() is not 1
        assert series.min() is not -1
        assert len(series) is len(scaled)
        assert series.index is scaled.index

    def test_rescales(self):
        series = get_daily()['PM10'].astype(float)
        scaled, scaler = scale_series(series)
        rescaled = rescale_series(scaled, scaler)

        assert rescaled.iloc[0] == series.iloc[0]
        assert len(series) is len(rescaled)
        assert series.index is rescaled.index


class TestScalingDataframe:
    def test_scale(self):
        df = get_daily()
        scaled, scaler = scale_dataframe(df.astype(float))

        assert df.max() is not 1
        assert df.min() is not -1
        assert len(df) is len(scaled)
        assert df.index is scaled.index
        assert df.columns is scaled.columns
        for column in scaled.columns:
            assert math.isclose(scaled[column].max(), 1, rel_tol=0.001)
            assert math.isclose(scaled[column].min(), -1, rel_tol=0.001)

    def test_rescale(self):
        df = get_daily()
        scaled, scaler = scale_dataframe(df.astype(float))
        rescaled = rescale_dataframe(scaled, scaler)

        assert rescaled.iloc[0].equals(df.iloc[0].astype(float))
        assert len(df) is len(rescaled)
        assert df.index is rescaled.index


class TestSeriesDifferencing:
    def test_differencing(self):
        series = get_daily()['PM10']
        differenced = difference_series(series)

        assert differenced.index.equals(series.index[1:])
        assert differenced.iloc[0] == series.iloc[1] - series.iloc[0]

    def test_dedifferencing(self):
        series = get_daily()['PM10']
        differenced = difference_series(series)
        dedifferenced = dedifference_series(differenced, series.iloc[0])

        assert dedifferenced.index is differenced.index
        assert dedifferenced.index.equals(series.index[1:])
        assert dedifferenced.iloc[0].astype(float) == series.iloc[1].astype(float)


class TestDataframeDifferencing:
    def test_differencing(self):
        df = get_daily()
        series = df['PM10']
        differenced = difference_dataframe(df)

        assert differenced.index.equals(df.index[1:])
        assert differenced.iloc[0].astype(float).equals(df.iloc[1].subtract(df.iloc[0]).astype(float))
        assert differenced.iloc[0]['PM10'] == (series.iloc[1] - series.iloc[0])

    def test_dedifferencing(self):
        df = get_daily()
        differenced = difference_dataframe(df)
        dedifferenced = dediffference_dataframe(differenced, df.iloc[0])

        assert dedifferenced.index is differenced.index
        assert dedifferenced.index.equals(df.index[1:])
        assert dedifferenced.iloc[0].astype(float).equals(df.iloc[1].astype(float))
