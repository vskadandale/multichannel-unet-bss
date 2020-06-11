import os
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import median
import matplotlib.pyplot as plt
from settings import *

#### ENERGY DISTRIBUTION PLOTS ####
"""
vocals = np.load(os.path.join(ENERGY_PROFILE_FOLDER, 'vocals_energy_profile.npy'), allow_pickle=True).item()
other = np.load(os.path.join(ENERGY_PROFILE_FOLDER, 'other_energy_profile.npy'), allow_pickle=True).item()
bass = np.load(os.path.join(ENERGY_PROFILE_FOLDER, 'bass_energy_profile.npy'), allow_pickle=True).item()
drums = np.load(os.path.join(ENERGY_PROFILE_FOLDER, 'drums_energy_profile.npy'), allow_pickle=True).item()
v = pd.DataFrame.from_dict(vocals, orient='index')
b = pd.DataFrame.from_dict(bass, orient='index')
d = pd.DataFrame.from_dict(drums, orient='index')
o = pd.DataFrame.from_dict(other, orient='index')
df = pd.concat([v, d, b, o],axis=1)
df.columns = ['vocals', 'drums', 'bass', 'other']

#sns.boxplot(data=[df['vocals'], df['drums'], df['bass'], df['other']])
#sns.violinplot(data=[df['vocals'], df['drums'], df['bass'], df['other']])
splot = sns.violinplot(data=[df['vocals'], df['drums'], df['bass'], df['other']])
#splot = sns.swarmplot(data=[df['vocals'], df['drums'], df['bass'], df['other']],
#                   color="white", edgecolor="gray")
splot.set(xlabel='Source Type', ylabel='Energy', title='Energy distribution across the sources')
#splot.set(yscale="log")


f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)



df1=pd.DataFrame(columns=['vocals', 'drums', 'bass', 'rest'])
v = pd.read_csv(os.path.join(ENERGY_PROFILE_FOLDER, 'vocals_energy_profile.csv'), names=['id','vocals'])
d = pd.read_csv(os.path.join(ENERGY_PROFILE_FOLDER, 'drums_energy_profile.csv'), names=['id','drums'])
b = pd.read_csv(os.path.join(ENERGY_PROFILE_FOLDER, 'bass_energy_profile.csv'), names=['id','bass'])
o = pd.read_csv(os.path.join(ENERGY_PROFILE_FOLDER, 'other_energy_profile.csv'), names=['id','rest'])
a = pd.read_csv(os.path.join(ENERGY_PROFILE_FOLDER, 'accompaniment_energy_profile.csv'), names=['id','acc'])
df1['vocals']=v['vocals']
df1['drums']=d['drums']
df1['bass']=b['bass']
df1['rest']=o['rest']

#splot1 = sns.violinplot(data=df1, orient="h", ax=axes[1], linewidth=2)
splot1 = sns.boxplot(data=df1, orient="h", ax=axes[1], linewidth=2)
splot1.set(ylabel='Source Type', xlabel='Energy', title='Energy distribution in 4-sources setting')

df2=pd.DataFrame(columns=['vocals', 'acc'])
df2['vocals'] = v['vocals']
df2['acc'] = a['acc']
#splot2 = sns.violinplot(data=df2, orient="h", ax=axes[0], linewidth=2)
splot2 = sns.boxplot(data=df2, orient="h", ax=axes[0], linewidth=2)
splot2.set(ylabel='Source Type', xlabel='Energy', title='Energy distribution in 2-sources setting')
"""

f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
df = pd.read_csv(os.path.join('trackwise_energy_profile.csv'))

FONT_SIZE = 16

plt.rc('font', size=14)          # controls default text sizes
df1=pd.DataFrame(columns=['vocals', 'acc'])
df1['vocals'] = df['Vocals']
df1['acc'] = df['Accompaniment']
splot1 = sns.violinplot(data=df1, orient="h", ax=axes[0], linewidth=2)
splot1.set(ylabel='Source Type', xlabel='Energy', title='Energy distribution in 2-sources setting')
splot1.set_ylabel('Source Type', fontsize=FONT_SIZE)
splot1.set_xlabel('Energy', fontsize=FONT_SIZE)
splot1.tick_params(axis='x', labelsize=FONT_SIZE)
splot1.tick_params(axis='y', labelsize=FONT_SIZE)

df2=pd.DataFrame(columns=['vocals', 'drums', 'bass', 'rest'])
df2['vocals']=df['Vocals']
df2['drums']=df['Drums']
df2['bass']=df['Bass']
df2['rest']=df['Other']

splot2 = sns.violinplot(data=df2, orient="h", ax=axes[1], linewidth=2)
splot2.set(ylabel='Source Type', xlabel='Energy', title='Energy distribution in 4-sources setting')
splot2.tick_params(axis='x', labelsize=FONT_SIZE)
splot2.tick_params(axis='y', labelsize=FONT_SIZE)
splot2.set_ylabel('Source Type', fontsize=FONT_SIZE)
splot2.set_xlabel('Energy', fontsize=FONT_SIZE)
splot2.set_xticks(np.arange(0, 3000, step=500))



plt.show()
f.savefig(os.path.join(ENERGY_PROFILE_FOLDER, 'energy_profile.png'), bbox_inches='tight', dpi=600)
