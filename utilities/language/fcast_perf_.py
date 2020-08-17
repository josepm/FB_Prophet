"""
$ python fcast_perf_.py ts_name cutoff_date to_table
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

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.forecast.utilities.language import fcast_processing as fp
from capacity_planning.data import hql_exec as hql
from capacity_planning.forecast.utilities.language import ens_processing as ep

USE_CACHE = True if platform.system() == 'Darwin' else False
USE_CACHE = True
RENEW = not USE_CACHE


# pd.set_option('display.max_rows', 100)
# pd.set_option('precision', 4)
# pd.set_option('display.width', 320)
# pd.set_option('display.max_columns', 35)


def main(argv):
    # ###########################
    # parameters
    # ###########################
    time_scale = 'W'  # forecasting time scale reset for daily ticket data
    init_date = pd.to_datetime('2016-01-01')
    upr = 12
    lwr = 8
    # ###########################
    # ###########################
    # ###########################
    print(argv)
    if len(argv) == 2:
        ts_name = argv[-2:]
        cutoff_date = pd.to_datetime('today')
        to_table = False
    elif len(argv) == 3:
        ts_name, cutoff_date = argv[-2:]
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
            to_table = False
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
            sys.exit()
    elif len(argv) == 4:
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

    actuals_df = dp.ts_actuals(ts_name, ts_cfg, cols)
    actuals_df.rename(columns={ts_cfg['ycol']: 'y'}, inplace=True)
    fcast_df = fp.get_ens_fcast(ts_name, ts_cfg, cutoff_date)
    ens = fcast_df.loc[fcast_df.index[0], 'ens']
    f_df = ep.fcast_filter(fcast_df, actuals_df, ts_name, cutoff_date + pd.to_timedelta(upr, unit=time_scale), cutoff_date, time_scale)
    pf = fcast_perf(f_df, actuals_df, cutoff_date, lwr, upr, time_scale, ens)
    if pf is None:
        return
    else:
        pf['ts_name'] = ts_name
        p_ut.save_df(pf, '~/my_tmp/perf/fcast_perf_' + ts_name + '_' + str(cutoff_date.date()))
        if to_table is True:
            tab_cols = ['language', 'y', 'yhat', 'err', 'lwr', 'upr', 'ens']
            partition = {'cutoff': str(cutoff_date.date()), 'ts_name': ts_name}
            ret = hql.to_tble(pf, tab_cols, 'sup.cx_language_forecast_performance', partition)
            if ret != 0:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: forecast performance failed for ' + ts_name + ' and cutoff date ' + str(cutoff_date.date()))
                sys.exit()
    print('DONE')


def fcast_perf(f_df, a_df, cutoff_date, lwr, upr, time_scale, ens):
    # computes fcast error between lwr and upr horizons
    # reset for performance
    upr_horizon = cutoff_date + pd.to_timedelta(upr, unit=time_scale)                                           # training fcast date
    lwr_horizon = cutoff_date + pd.to_timedelta(lwr, unit=time_scale)

    if a_df['ds'].min() <= lwr_horizon and a_df['ds'].max() >= upr_horizon:
        ef = a_df[(a_df['ds'] >= lwr_horizon) & (a_df['ds'] <= upr_horizon)].merge(f_df, on=['ds', 'language'], how='left')
        ff = ef[ef['language'] != 'All'].copy()
        if ff['language'].nunique() > 1:
            all_f = ff.groupby('ds').agg({'y': np.sum, 'yhat': np.sum, 'loss': np.mean}).reset_index()
            all_f['language'] = 'All'
            ef = pd.concat([all_f, ff], axis=0)
        pf = ef.groupby('language').agg({'y': np.sum, 'yhat': np.sum, 'loss': np.mean}).reset_index()
        pf['err'] = np.abs(pf['yhat'] / pf['y'] - 1)
        pf['lwr'] = lwr_horizon
        pf['upr'] = upr_horizon
        pf['ens'] = ens
        pf['cutoff'] = cutoff_date
        return pf
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no actuals to compute fcast errors:: actuals max ds: ' +
                      str(a_df['ds'].max().date()) + ' upr horizon: ' + str(upr_horizon.date()))
        return None


def perf_smry(perf_df, cutoff_date, time_scale, ts_name, upr, lwr):
    # print smry and save
    upr_horizon = cutoff_date + pd.to_timedelta(upr, unit=time_scale)
    lwr_horizon = cutoff_date + pd.to_timedelta(lwr, unit=time_scale)
    if perf_df is not None:
        perf_df.sort_values(by='language', inplace=True)
        perf_df.reset_index(inplace=True, drop=True)
        perf_df['ts_name'] = ts_name
        perf_df['cutoff'] = cutoff_date
        s_ut.my_print('###########################  cutoff: ' + str(cutoff_date.date()) + ' ts_name: ' + str(ts_name) +
                      ' performance between ' + str(lwr_horizon.date()) + ' (included) and ' + str(upr_horizon.date()) +
                      ' (included)  ##########################################')
        perf_df.sort_values(by=['language', 'err'], inplace=True)
        print(perf_df.head(10))
        p_ut.save_df(perf_df, '~/my_tmp/fbp/lang_perf_' + ts_name + '_' + str(cutoff_date.date()))
    else:
        s_ut.my_print('WARNING: no actuals to compute fcast errors for the  between ' +
                      str(lwr_horizon.date()) + ' (included) and ' + str(upr_horizon.date()) + ' (included) ' +
                      ' for cutoff: ' + str(cutoff_date.date()) + ' and ts_name: ' + str(ts_name))


if __name__ == '__main__':
    main(sys.argv)
