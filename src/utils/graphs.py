import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

engine = create_engine('mysql+pymysql://root:lu_iz@localhost:3306/crowd_detection')

query = "SELECT * FROM crowd_records"
df = pd.read_sql(query, engine)

crowd_grp = df.groupby(['crowd_id'])
medians = crowd_grp['size'].mean().apply(lambda x: int(x))
times = crowd_grp['rec_time'].min()
people_df = pd.concat([medians, times], axis='columns', sort=False)

people_df.set_index('rec_time', inplace=True)


def plot_graph(date, smp_rate, title):
    time_grp = people_df.loc[date]
    time_grp = time_grp.resample(smp_rate).sum()

    date_format = mpl_dates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.plot(time_grp.index, time_grp['size'], '-')

    # formatting graph
    
    plt.grid(True)

    # labels
    plt.xlabel(f'tempo ({smp_rate[:-1]} min)')
    plt.ylabel('pessoas aglomeradas')
    plt.title(title)

    plt.show()


interval = (people_df.index >= '2022-12-27-10:10') & (people_df.index <= '2022-12-27-10:40')
plot_graph(interval, '5T', '2º intervalo (manhã) - 27/12')

def plot_graph_rec(df, datetime, title):
    df.set_index('rec_time', inplace=True)
    time_df = df.loc[datetime]
    time_df = time_df.resample('1S').sum()
    
    plt.plot(time_df.index, time_df['size'], '-')

    # formatting graph
    date_format = mpl_dates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.grid(True)
    plt.ylim(0, 10)

    # labels
    plt.xlabel(f'tempo (1S)')
    plt.ylabel('pessoas aglomeradas')
    plt.title(title)

    plt.show()

'''
mask = '2022-12-28'
plot_graph_rec(df, mask, 'Dia 28/12')
'''