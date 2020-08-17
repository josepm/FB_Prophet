"""
$ python err_fcast ts_name
forecast errors for a time series
"""

import os
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'


import sys
import pandas as pd
import json
import numpy as np
import copy

from capacity_planning.forecast.utilities.language import time_series as ts
from capacity_planning.forecast import lang_forecast as lf
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut


pd.set_option('display.max_rows', 100)
pd.set_option('precision', 4)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 35)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def fcast_perf(f_df, a_df, lwr_horizon, upr_horizon):
    # computes fcast error between lwr and upr horizons
    adf = a_df[(a_df['ds'] >= lwr_horizon) & (a_df['ds'] <= upr_horizon)].copy()
    if len(adf) > 0:
        y_langs = [l for l in f_df['language'].unique() if 'not-' not in l]
        y_df = f_df[f_df['language'].isin(y_langs)].copy()
        y_df.rename(columns={'yhat': 'yhat_d'}, inplace=True)
        adf = adf[adf['language'].isin(y_langs)].copy()

        n_langs = [l for l in f_df['language'].unique() if 'not-' in l]
        f_df_all = f_df[f_df['language'] == 'All'].copy()
        b_df = f_df[f_df['language'].isin(n_langs)].copy()
        n_df = f_df_all.merge(b_df, on=['ds', 'ens'], how='left')
        n_df['language'] = n_df['language_y'].apply(lambda x: x.replace('not-', ''))
        n_df['yhat_i'] = n_df['yhat_x'] - n_df['yhat_y']
        n_df.drop(['language_x', 'language_y', 'yhat_x', 'yhat_y'], axis=1, inplace=True)
        df_f = y_df.merge(n_df, on=['ds', 'language', 'ens'], how='left')
        df_f['yhat_i'].fillna(value=df_f['yhat_d'], inplace=True)
        ef = adf.merge(df_f, on=['ds', 'language'], how='left')
        if len(ef) == 0:
            s_ut.my_print('No data to compute forecast errors')
            return None
        pf = ef.groupby(['language', 'ts_name', 'ens']).agg({'y': np.sum, 'yhat_d': np.sum, 'yhat_i': np.sum}).reset_index()
        pf['err_d'] = np.abs(pf['yhat_d'] / pf['y'] - 1)
        pf['err_i'] = np.abs(pf['yhat_i'] / pf['y'] - 1)
        print(pf)

        return pf
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no actuals to compute fcast errors:: actuals max ds: ' +
                      str(a_df['ds'].max().date()) + ' upr horizon: ' + str(upr_horizon.date()))
        return None


def set_actuals(ts_name, cutoff_date, time_scale, init_date):
    # collect actuals and prepare them (outliers)
    data_cfg = '~/my_repos/capacity_planning/forecast/config/data_cfg.json'
    with open(os.path.expanduser(data_cfg), 'r') as fp:
        d_cfg = json.load(fp)
    ts_cfg = d_cfg[ts_name]
    ts_cfg['cutoff_date'] = cutoff_date
    ts_cfg['name'] = ts_name
    ts_cfg['time_scale'] = time_scale
    ts_cfg['init_date'] = init_date
    actuals_df, _ = lf.get_actuals(ts_name, ts_cfg)
    if actuals_df is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no actuals for ' + str(ts_name) + ' and cutoff date ' + str(cutoff_date.date()))
        return None
    ts_cfg_ = copy.deepcopy(ts_cfg)
    ts_cfg_['outlier_coef'] = max(ts_cfg['outlier_coef'])  # worst case

    ts_obj = ts.TimeSeries(ts_name, actuals_df, cutoff_date, cutoff_date, init_date, ts_cfg_, time_scale=time_scale)
    af_list = list()
    for lx, fx in ts_obj.df_dict.items():
        if fx['ds'].max() < cutoff_date - pd.to_timedelta((1 + cutoff_date.weekday()) % 7, unit='D'):   # move cutoff to week starting Sunday
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no actuals for training for language ' +
                          str(lx) + ' max ds: ' + str(fx['ds'].max().date()) + ' min ds: ' + str(fx['ds'].min().date()))
            break
        fx['language'] = lx
        af_list.append(fx)
    f_all = pd.concat(af_list, axis=0) if len(af_list) > 0 else None
    return f_all


def get_fcasts(ts_list, froot):
    fname = 'ens_fcast_'
    f_list = list()
    for f in os.listdir(froot):
        for ts in ts_list:
            if fname + ts in f:
                fv = f.split('_')       # ['ens', 'fcast', <ts_name_components>, date.par]
                ts_name_ = '_'.join(fv[2:-1])
                dt_str = fv[-1].split('.')[0]
                dt = pd.to_datetime(dt_str)
                if ts_name_ == ts:
                    fpath = froot + f.split('.')[0]
                    s_ut.my_print('@@@@ loading fcast file: ' + str(fpath))
                    df = p_ut.read_df(fpath)
                    df['cutoff'] = pd.to_datetime(dt)
                    df['ts_name'] = ts
                    f_list.append(df)
    return pd.concat(f_list, axis=0) if len(f_list) > 0 else None


def get_actuals(ts_list, init_date, time_scale):
    a_list = list()
    for ts in ts_list:
        f_act = set_actuals(ts, init_date, time_scale, init_date)
        if f_act is not None:
            f_act['ts_name'] = ts
            a_list.append(f_act)
    return pd.concat(a_list, axis=0) if len(a_list) > 0 else None


def merge_neg(df, b_name, p_name, n_name, ycol):
    all_df = df[df['ts_name'] == b_name].copy()
    all_df.drop('ts_name', axis=1, inplace=True)
    p_df = df[df['ts_name'] == p_name].copy()
    p_df.drop('ts_name', axis=1, inplace=True)
    n_df = df[df['ts_name'] == n_name].copy()
    n_df.drop('ts_name', axis=1, inplace=True)
    np_df = all_df.merge(n_df, on=['ds', 'language'], how='inner')
    np_df[ycol] = np_df[ycol + '_x'] - np_df[ycol + '_y']
    np_df.drop([ycol + '_x', ycol + '_y'], axis=1, inplace=True)
    fout = p_df.merge(np_df, on=['ds', 'language'], how='left')
    ts_languages = [lg for lg in p_df['language'].unique() if 'not-' not in lg]
    fout = fout[fout['language'].isin(ts_languages)].copy()
    fout.rename(columns={ycol + '_x': ycol + '_direct', ycol + '_y': ycol + '_indirect'}, inplace=True)
    return fout


def main(argv):
    print(argv)
    if len(argv) == 2:
        p_name = argv[1]  # at least 3 days after last Saturday with actual data
        if 'not-' in p_name:
            print('invalid series name: ' + str(argv))
            sys.exit()
    else:
        print('invalid args: ' + str(argv))
        sys.exit()

    # ###########################
    # parameters
    # ###########################
    lwr = 9
    upr = 12
    time_scale = 'W'  # forecasting time scale reset for daily ticket data
    init_date = pd.to_datetime('2016-01-01')
    froot = os.path.expanduser('~/my_tmp/fbp/')
    # max_int = 2 if time_scale == 'W' else (5 if time_scale == 'D' else None)
    # if max_int is None:
    #     s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: unsupported time scale: ' + str(time_scale))
    #     sys.exit()
    # ###########################
    # ###########################

    # set complementary TS
    vname = p_name.split('_')  # eg ts, bu
    if len(vname) == 2:
        b_name = vname[0]
        bu = vname[1]
        n_name = b_name + '_not-' + bu
        ts_list = [p_name, n_name, b_name]
    elif len(vname) == 1:
        ts_list = [p_name]
        b_name, n_name = None, None
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: invalid ts name: ' + p_name)
        sys.exit()

    # forecasts
    fcast_df = get_fcasts(ts_list, froot)
    p_ut.save_df(fcast_df, '~/my_tmp/fcast_df')
    if fcast_df is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no fcast data for ' + p_name)
        sys.exit()
    else:
        # actuals
        a_df = get_actuals(ts_list, init_date, time_scale)
        if a_df is not None:
            ts_adf = a_df[a_df['ts_name'] == p_name].copy()
            p_ut.save_df(ts_adf, '~/my_tmp/ts_adf')
            p_list = list()
            for cu, f_cu in fcast_df.groupby('cutoff'):
                fcast_date = f_cu['ds'].max()
                cu -= pd.to_timedelta((1 + cu.weekday()) % 7, unit='D')          # move cutoff to week starting Sunday

                # performance bounds
                upr_horizon = min(fcast_date, cu + pd.to_timedelta(upr, unit=time_scale))
                lwr_horizon = cu + pd.to_timedelta(lwr, unit=time_scale)
                f_cu.drop(['cutoff', 'ts_name'], axis=1, inplace=True)
                m_fcast = merge_neg(f_cu, b_name, p_name, n_name, 'yhat')
                m_fcast.drop_duplicates(inplace=True)
                perf_df = fcast_perf(m_fcast, ts_adf, lwr_horizon, upr_horizon)
                perf_df['ts_name'] = p_name
                perf_df['cutoff'] = cu
                p_list.append(perf_df)
            if len(p_list) > 0:
                perf_out = pd.concat(p_list, axis=0)
                fname = froot + 'ens_perf_'
                p_ut.save_df(perf_out, fname + p_name)
            else:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: perf_df not available')
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no actuals data for ' + p_name)
            sys.exit()
        print('DONE')


if __name__ == '__main__':
    main(sys.argv)
