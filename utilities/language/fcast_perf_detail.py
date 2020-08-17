"""
$ python fcast_perf.py ts_name cutoff_date to_table
fcast performance of the ens fcast
"""

import os
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'


import sys
import pandas as pd
import numpy as np
import json

from capacity_planning.forecast.utilities.language import ens_processing as ep
from capacity_planning.forecast.utilities.language import fcast_processing as fp
from capacity_planning.forecast.utilities.language import fcast_perf as perf
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.data import hql_exec as hql


pd.set_option('display.max_rows', 100)
pd.set_option('precision', 4)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 35)


def main(argv):
    # ###########################
    # parameters
    # ###########################
    time_scale = 'W'  # forecasting time scale reset for daily ticket data
    init_date = pd.to_datetime('2016-01-01')
    upr = 12
    lwr = 8
    evals = 50
    by_lang = False
    # ###########################
    # ###########################
    # ###########################
    print(argv)
    if len(argv[1:]) == 1:
        ts_name = argv[-1]
        cutoff_date = pd.to_datetime('today')
        to_table = False
    elif len(argv[1:]) == 2:
        ts_name, cutoff_date = argv[1:]
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
            to_table = False
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
            sys.exit()
    elif len(argv[1:]) == 3:
        ts_name, cutoff_date, to_table = argv[1:]
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
            to_table = bool(int(to_table))
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
            sys.exit()
    else:
        s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
        sys.exit()
    ts_cfg, cols = dp.ts_setup(ts_name, cutoff_date, init_date, time_scale)

    # actuals
    actuals_df = dp.ts_actuals(ts_name, ts_cfg, cols)
    actuals_df.rename(columns={ts_cfg['ycol']: 'y'}, inplace=True)

    # forecasts
    f_df = fp.get_lang_fcast(ts_cfg, cutoff_date)
    fcast_date = cutoff_date + pd.to_timedelta(upr, unit=time_scale)

    perf_list = list()
    for xens in ['XGBRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'GradientBoostingRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'lasso']:
        fcast_df = ep.make_fcast(ts_name, f_df, actuals_df, cutoff_date, fcast_date, xens, evals, by_lang, (lwr, upr), lwr=lwr, upr=upr)
        perf_df = perf.fcast_perf(fcast_df, actuals_df, cutoff_date, lwr, upr, time_scale, xens)
        if perf_df is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: forecast performance detail failed for ' + ts_name +
                          ' ,cutoff date ' + str(cutoff_date.date()) + ' and ensemble: ' + str(xens))
        else:
            perf_df['ts_name'] = ts_name
            perf_list.append(perf_df)
    if len(perf_list) > 0:
        pf = pd.concat(perf_list, axis=0)
        p_ut.save_df(pf, '~/my_tmp/perf/fcast_perf_detail_' + ts_name + '_' + str(cutoff_date.date()))
        if to_table is True:
            tab_cols = ['language', 'y', 'yhat', 'err', 'lwr', 'upr', 'ens']
            partition = {'cutoff': str(cutoff_date.date()), 'ts_name': ts_name}
            ret = hql.to_tble(pf, tab_cols, 'sup.cx_language_forecast_performance_detail', partition)
            if ret != 0:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: forecast performance detail failed for ' + ts_name + ' and cutoff date ' + str(cutoff_date.date()))
                sys.exit()
        print('DONE')
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no data for forecast performance detail failed for ' + ts_name + ' and cutoff date ' + str(cutoff_date.date()))
        sys.exit()


if __name__ == '__main__':
    main(sys.argv)
