import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

 #Load Datasets

#clicks_df=pd.read_csv('H:/Studies/Last semester/project/recsys-challenge-2015/yoochoose-clicks.dat',
#                      names=['session_id','timestamp','item_id','category'],
#                      dtype={'category': str})
buys_df=pd.read_csv('H:/Studies/Last semester/project/recsys-challenge-2015/yoochoose-buys.dat',
                      names=['session_id', 'timestamp', 'item_id', 'price', 'quantity'])
#chossing random 10000 sessions
#df = clicks_df.loc[np.random.choice(clicks_df.index, size=10000)]
