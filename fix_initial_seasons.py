import glob
import re

for csv in glob.glob('results/*direct=False.csv'):
    replacement = ''

    with open(csv, 'r') as myfile:
        data = myfile.read()

        malformed = re.search('(")(?s).*(")', data).group(0)
        fixed = re.sub(' +', ' ', malformed.replace('"', '').replace('\n', ' '))

        replacement = data.replace(malformed, fixed)

    with open(csv, 'w') as myfile:
        myfile.write(replacement)