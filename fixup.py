import glob
import os
import re

rates = {"DESN082": "H",
         "DEHH033": "H",
         "DEBE010": "D",
         "DEHH070": "H",
         "DESN059": "D",
         "DEHH049": "H",
         "DEBE051": "H",
         "DEBE068": "D",
         "DEHH079": "H",
         "DEHH050": "H",
         "DEHH059": "H"}

if __name__ == '__main__':
    for csv in glob.glob('./results/*.csv'):
        split = csv.split('-')
        station, rate = re.sub('\./.*/', '', split[0]), split[1]
        if rate not in ['H, D']:
            os.rename(csv, re.sub(station, f'{station}-{rates[station]}', csv))
