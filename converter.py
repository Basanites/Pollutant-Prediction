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
    post = glob.glob(datadir + '/*')

model = model.Model()

for csv in pre:
    print(f'reading{csv}')
    model.import_csv(csv)
    df = model.df
    try:
        columns = list(df)
        station = df.iloc[0].AirQualityStationEoICode

        filename = f'{station}-{columns[-1]}.pkl'.replace(' ', '_')
        filelocation = f'{finaldir}/{filename}'

        if filelocation in post:
            df2 = pd.read_pickle(filelocation)
            df = df2.concatenate(df)
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index()

        df.to_pickle(filelocation)
    except IndexError:
        pass
