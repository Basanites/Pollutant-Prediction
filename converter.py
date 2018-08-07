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

        filename = f'{station}-{columns[-1]}.pkl'
        filelocation = f'{finaldir}/{filename}'

        if filelocation in post:
            df2 = pd.read_pickle(filelocation)
            df = pd.concat([df, df2])
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index()
        else:
            post.append(filelocation)
        df.to_pickle(filelocation)
    except IndexError:
        pass
