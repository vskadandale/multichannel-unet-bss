import seaborn as sns
import numpy as np
import pandas as pd
import os
from settings import *
import matplotlib.pyplot as plt

path = os.path.join(DUMPS_FOLDER, 'results', TYPE)
df = pd.DataFrame(columns=['source', 'model', 'values'])
dict = {}
for fname in os.listdir(path):
    if not fname.startswith('.'):
        fpath = os.path.join(path, fname)
        for source in SOURCES_SUBSET:
            dict['source'] = source
            dict['model'] = fname[:-4]
            dict['values'] = pd.read_csv(fpath)['SDR_'+source][:50]
            df=df.append(pd.DataFrame(dict))


g = sns.catplot(x="values", y="model",
                row="source",
                data=df,
                sharex=False,
                sharey=True,
                orient="h", height=2, aspect=3, palette="Set3",
                kind="boxen")

plt.savefig(os.path.join(DUMPS_FOLDER, 'results', TYPE, TYPE+'_boxenplots.png'))
plt.show()
4+5
