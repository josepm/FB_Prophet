"""
computes forecast error at language level given a cutoff date
$ python errors.py ts_name cutoff_date_
- cutoff_date is the date fcasts for ts_name were run, not today. If not a Saturday, reset to the prior Saturday
- if no forecasts for (adjusted) cutoff date, no errors
- if there are forecasts,
  - we look for actuals after the cutoff date, if found try to compute errors for the fcasts issued on this cutoff date
- the fcast error is the MAPE between cutoff_date + lwr_horizon and cutoff_date + upr_horizon, currently set 3 months, i.e. weeks 9, 10, 11 and 12 after the cutoff date
- if the actuals do not cover the range up to upr_horizon, no errors are generated
- outliers in actuals are removed before the MAPE computation
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import sys
import json
import copy


from capacity_planning.forecast.utilities.language import time_series as ts
from capacity_planning.utilities import stats_utils as st_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import time_utils as tm_ut

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
upr_horizon = 12 * 7  # days (included)
lwr_horizon = 9 * 7   # days (included)
time_scale = 'W'



def g_err(ef, ycol):
    y = ef[ycol].values
    yhat = ef['yhat'].values
    e_arr = ['sMAPE', 'MAPE', 'mMAPE', 'wMAPE', 'LAR', 'MASE', 'RMSE']  # , 'YJ', 'YJd', 'BC', 'SR']
    d_err = {etype: st_ut.err_func(y, yhat, etype) for etype in e_arr}
    return pd.DataFrame([d_err])


def prep_actuals(adf, ts_cfg, end_date):
    # remove outliers from actuals for fcast error analysis
    # if time_scale == 'W':
        # max_int = 2
    # elif time_scale == 'D':
        # max_int = 5
    # else:
    #     s_ut.my_print('ERROR: invalid time scale: ' + str(time_scale))
    #     sys.exit()

    if 'language' not in adf.columns:
        adf['language'] = 'NULL'
    ycol = ts_cfg['ycol']
    init_date = adf['ds'].min()
    zz = adf.groupby(['ds', 'language']).agg({ycol: np.sum}).reset_index()
    ts_dict = {'agg_dict': {ycol: 'sum'}, 'outlier_coef': ts_cfg['outlier_coef'], 'check_cols': [ycol]}
    ts_obj = ts.TimeSeries(ts_cfg['name'], zz, end_date, end_date, init_date, ts_dict, time_scale=ts_cfg['time_scale'])
    df_list = list()
    for k, v in ts_obj.df_dict.items():
        v['language'] = k
        v.rename(columns={'y': ycol}, inplace=True)
        df_list.append(v)
    return pd.concat(df_list)


def get_actuals_(ts_name, ts_cfg, e_date):
    # get actuals
    froot = ts_cfg['data_path'].split('/')
    a_dir = os.path.expanduser('/'.join(froot[:-1])) + '/'
    fname = froot[-1]
    actuals_df = None
    for f in os.listdir(a_dir):
        if fname in f:
            s_ut.my_print('actuals file: ' + a_dir + str(f.split('.')[0]))
            actuals_df = p_ut.read_df(a_dir + f.split('.')[0])
            if actuals_df is not None and actuals_df['ds'].max() >= e_date:
                break
    if actuals_df is None:
        s_ut.my_print('ERROR: no actuals for ' + ts_name + ' and horizon ' + str(e_date.date()))
        sys.exit()
    else:
        p_ut.set_week_start(actuals_df, tcol='ds')  # week_starting patch
        return actuals_df


def get_fcast_(ts_name, cutoff_date, e_date):
    # get forecast
    froot = '~/my_tmp/fbp/'
    fname = froot + 'lang_fcast_' + ts_name + '_' + str(cutoff_date.date())
    fcast_df = p_ut.read_df(fname)
    if fcast_df is None:
        s_ut.my_print('ERROR: no forecasts for ' + str(ts_name) + ' and cutoff date ' + str(cutoff_date.date()))
        sys.exit()
    elif fcast_df['ds'].max() <= e_date:
        s_ut.my_print('ERROR: no forecasts for ' + str(ts_name) + ' and cutoff date ' + str(cutoff_date.date()) + ' and horizon ' + str(e_date.date()))
        sys.exit()
    else:
        p_ut.set_week_start(fcast_df, tcol='ds')  # week_starting patch
        return fcast_df


def ix_cfg_err(adf, fdf, ycol):
    z = adf.merge(fdf, on=['ds', 'language'], how='left')
    z = z[(z['yhat'] > 0.0) & (z[ycol] > 0.0)].copy()

    # all languages
    z_all = z.groupby('ds').agg({ycol: np.sum, 'yhat': np.sum}).reset_index()
    z_all['language'] = 'All'
    f_all = pd.concat([z_all, z[['ds', 'language', ycol, 'yhat']]], axis=0)
    ef = f_all.groupby('language').apply(g_err, ycol=ycol).reset_index()
    if 'level_1' in ef.columns:
        ef.drop('level_1', axis=1, inplace=True)
    return ef


def fcast_errors_(a_df, f_df, e_date, s_date, ycol):
    a_df = a_df[(a_df['ds'] >= s_date) & (a_df['ds'] <= e_date)].copy()
    f_df = f_df[(f_df['ds'] >= s_date) & (f_df['ds'] <= e_date)].copy()
    zlist = list()
    for ix, ff in f_df.groupby('ix'):
        ff.reset_index(inplace=True, drop=True)
        zix = ix_cfg_err(a_df, ff, ycol)  # err check: need data past cutoff for actuals
        zix['ix'] = ix
        zlist.append(zix)
    e_lang = pd.concat(zlist, axis=0)
    e_lang.reset_index(inplace=True, drop=True)
    return e_lang


def fcast_errors(argv):
    if len(argv) == 3:
        ts_name, run_date = argv[1:]  # at least 3 days after last Saturday with actual data
    else:
        print('invalid args: ' + str(sys.argv))
        sys.exit()

    # data cfg
    data_cfg = FILE_PATH + '/config/data_cfg.json'
    with open(os.path.expanduser(data_cfg), 'r') as fp:
        d_cfg = json.load(fp)

    ts_cfg = d_cfg[ts_name]
    ts_cfg['name'] = ts_name
    ts_cfg['time_scale'] = time_scale
    cutoff_date = tm_ut.get_last_sat(run_date)                            # set to saturday prior run_date or the run_date if a saturday
    e_date = cutoff_date + pd.to_timedelta(upr_horizon, unit='D')         # error check end date (included)
    s_date = cutoff_date + pd.to_timedelta(lwr_horizon, unit='D')         # error check start date (included)

    # get actuals
    actuals_df = get_actuals_(ts_name, ts_cfg, e_date)
    a_df_dict = dict()
    for w in ts_cfg['outlier_coef']:
        ts_cfg_ = copy.deepcopy(ts_cfg)
        ts_cfg_['outlier_coef'] = w
        a_df_dict[w] = prep_actuals(actuals_df, ts_cfg_, e_date)

    # get fcasts
    f_df = get_fcast_(ts_name, cutoff_date, e_date)

    # get language level errors and save err data
    e_list = list()
    for w, a_df in a_df_dict.items():
        ts_cfg_ = copy.deepcopy(ts_cfg)
        ts_cfg_['outlier_coef'] = w
        e_lang_ = fcast_errors_(a_df, f_df, e_date, s_date, ts_cfg_['ycol'])
        e_list.append(e_lang_)
    e_lang = pd.concat(e_list)
    e_lang['ts_name'] = ts_name
    e_lang['cutoff'] = cutoff_date
    froot = '~/my_tmp/fbp/'
    fname = froot + 'lang_cfgs_'
    p_ut.save_df(e_lang, fname + ts_name + '_' + str(cutoff_date.date()))
    return


if __name__ == '__main__':
    s_ut.my_print(sys.argv)
    fcast_errors(sys.argv)
    print('DONE')
