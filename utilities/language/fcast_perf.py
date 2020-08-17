"""
$ python fcast_perf_.py ts_name cutoff_date n_features to_table
fcast performance of the ens fcast
ts_name as usual, mandatory
cutoff_date as usual, optional default last
n_features: number of features (forecasts) to use in the ensemble, optional default 25
to_table = 1 write to table, 0 not. optional default 0
The optionals are hierarchical:
2 args means ts_name and cutoff_date
3 means ts_name, cutoff_date and n_features
4 means all
"""

import os
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'


import sys
import pandas as pd
import numpy as np
import json
from functools import reduce

from capacity_planning.forecast.utilities.language import ens_processing as ep
from capacity_planning.forecast.utilities.language import fcast_processing as fp
from capacity_planning.forecast.utilities.language import fcast_perf as perf
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.data import hql_exec as hql


# pd.set_option('display.max_rows', 100)
# pd.set_option('precision', 4)
# pd.set_option('display.width', 320)
# pd.set_option('display.max_columns', 35)
#

def main(argv):
    # ###########################
    # parameters
    # ###########################
    time_scale = 'W'  # forecasting time scale reset for daily ticket data
    init_date = pd.to_datetime('2016-01-01')
    upr = 12
    lwr = 9
    # ###########################
    # ###########################
    # ###########################
    print(argv)
    if len(argv[1:]) == 1:
        ts_name = argv[-1]
        cutoff_date = pd.to_datetime('today')
        to_table = False
        n_features = 25
    elif len(argv[1:]) == 2:
        ts_name, cutoff_date = argv[1:]
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
            to_table = False
            n_features = 25
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
            sys.exit()
    elif len(argv[1:]) == 3:
        ts_name, cutoff_date, n_features = argv[1:]
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
            to_table = False
            n_features = int(n_features)
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
            sys.exit()
    elif len(argv[1:]) == 4:
        ts_name, cutoff_date, n_features, to_table = argv[1:]
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
            n_features = int(n_features)
            to_table = bool(int(to_table))
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
            sys.exit()
    else:
        s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
        sys.exit()

    # set dates
    sun_date = cutoff_date - pd.to_timedelta(6, unit='D')           # cutoff date in week starting Sunday
    upr_date = sun_date + pd.to_timedelta(upr, unit=time_scale)                                            # horizon for the perf testing at cutoff_date
    lwr_date = sun_date + pd.to_timedelta(lwr, unit=time_scale)      # lwr date for perf testing window

    a_df, d_ff, cfg_dict = fp.cross_validation(ts_name, cutoff_date, upr, lwr, n_features, init_date=init_date, time_scale=time_scale)
    p_ut.save_df(d_ff, '~/my_tmp/d_ff')
    f_list = list()
    for lg, flf_ in d_ff.groupby('language'):
        flf_.drop('language', axis=1, inplace=True)
        flf_['y'] = flf_['y'].astype(float)
        flf_['yhat'] = flf_['yhat'].astype(float)
        a_perf = flf_[['dim_cfg', 'a_err']].drop_duplicates()

        flf = flf_[flf_['dim_cfg'].isin(cfg_dict[lg][0])].copy()
        flf.dropna(axis=1, inplace=True, how='all')  # drop all-null cols (fcast cfgs for other languages)
        p_flf = pd.pivot_table(flf[['ds', 'dim_cfg', 'yhat']].copy(), index=['ds'], columns=['dim_cfg'], values=['yhat']).reset_index()
        cols = [str(c[1]) if c[0] == 'yhat' else c[0] for c in p_flf.columns]
        p_flf.columns = cols
        p_flf = p_flf.merge(flf[['ds', 'y', 'y_shifted', 'adj_y_shifted']].drop_duplicates(), on='ds', how='left')

        s_ut.my_print('\n\n+++++++++++++++++++++++++ starting aggregation for ' + lg + ' ++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        d_list = ep.lang_perf(lg, p_flf, a_perf, cutoff_date, upr, lwr)
        pl = pd.DataFrame(d_list)
        pl['language'] = lg
        pl['n_forecasts'] = n_features
        f_list.append(pl)

    if len(f_list) > 0:
        pf = pd.concat(f_list, axis=0)
        pf['ts_name'] = ts_name
        pf['upr'] = upr_date
        pf['lwr'] = lwr_date
        pf['cutoff'] = cutoff_date
        print(pf)
        print('overall: ' + str(pf['avg_err'].mean()))
        p_ut.save_df(pf, '~/my_tmp/perf/fcast_perf_' + ts_name + '_' + str(cutoff_date.date()))
        if to_table is True:
            tab_cols = ['language', 'err', 'df', 'lwr', 'upr', 'train_cutoff', 'ens', 'n_features', 'n_forecasts']
            partition = {'cutoff': str(cutoff_date.date()), 'ts_name': ts_name}
            ret = hql.to_tble(pf, tab_cols, 'sup.cx_language_forecast_performance', partition)
            if ret != 0:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: forecast performance detail failed for ' + ts_name + ' and cutoff date ' + str(cutoff_date.date()))
                sys.exit()
        print('DONE')
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no data for forecast performance detail failed for ' + ts_name + ' and cutoff date ' + str(cutoff_date.date()))
        sys.exit()


if __name__ == '__main__':
    main(sys.argv)
