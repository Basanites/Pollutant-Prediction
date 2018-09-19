import glob

import pandas as pd

if __name__ == '__main__':
    eval_folder = 'eval'

    frame = pd.DataFrame()
    for csv in glob.glob(f'{eval_folder}/*.csv'):
        station, rate = csv.replace('.csv', '').replace(f'{eval_folder}/', '').split('-')
        current_frame = pd.read_csv(csv)
        current_frame['station'] = station
        current_frame['rate'] = rate

        frame = pd.concat([frame, current_frame])

    frame = frame.reset_index().drop(columns=['index'])
