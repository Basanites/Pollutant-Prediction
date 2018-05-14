import matplotlib.pyplot as plt
import numpy
import scipy
import pandas as pd
import glob


df = pd.DataFrame()

print('Loading CSVs\n')
for file in glob.glob('res/*.csv'):
    read = pd.read_csv(file,
                 encoding="utf-16", parse_dates=[13, 14],
                 infer_datetime_format=True,
                 index_col=[4, 8, 13])
    df = pd.concat([df, read[~read.index.get_level_values(0).str.contains('Bulk')]])
print('\nFinished Loading')
print('Sorting')
df = df.sort_index()
print('Finished sorting')
