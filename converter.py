import glob
import os
import model
import pandas as pd

datadir = './res'
finaldir = './post'

pre = glob.glob(datadir + '/*')

if not os.path.exists(finaldir):
    os.makedirs(finaldir)
    post = list()
else:
    post = glob.glob(finaldir + '/*')

model = model.Model()

for csv in pre:
    model.import_csv(csv)
    df = model.df
    try:
        columns = list(df)
        station = df.iloc[0].AirQualityStationEoICode

        filename = f'{station}-{columns[-1]}.csv'
        filelocation = f'{finaldir}/{filename}'

        if filelocation in post:
            df2 = pd.read_csv(filelocation, index_col=0, parse_dates=[0], infer_datetime_format=True)
            df = pd.concat([df, df2])
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index()
        else:
            post.append(filelocation)
        df.to_csv(filelocation)
    except IndexError:
        pass
