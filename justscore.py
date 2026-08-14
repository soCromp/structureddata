from hps import calculate_fast_proxy
import pandas as pd
import sys

dataname = sys.argv[1]
synthsamplespath = sys.argv[2]

realsamples = pd.read_csv(f'data/processed/{dataname}/train.csv')
synthsamples = pd.read_csv(synthsamplespath)

calculate_fast_proxy(realsamples, synthsamples, dataname)
