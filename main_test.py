import pandas as pd

from main import resample_dataframe


def get_daily():
    df = pd.DataFrame({
        'Timestamp': ['2013-01-11 23:00:00', '2013-01-12 23:00:00', '2013-01-14 23:00:00'],
        'PM10': [12, 19, 43],
        'PM2.5': [8, 13, 32]
    })

    df = df.set_index('Timestamp')
    return df


def get_hourly():
    df = pd.DataFrame({
        'Timestamp': ['2013-01-01 00:00:00', '2013-01-01 01:00:00', '2013-01-01 03:00:00'],
        'NO': [36.848, 42.785, 16.865],
        'PM10': [200.165, 205.237, 27.727],
        'SO2': [27.458, 10.514, 4.948]
    })
    df = df.set_index('Timestamp')
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
