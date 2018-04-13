import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

import numpy as np
import pandas as pd


df = pd.read_csv('res/DE_NO2.csv', encoding='latin-1')

print(df.head(0))