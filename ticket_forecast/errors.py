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
from functools import reduce


from capacity_planning.forecast.utilities.language import time_series as ts
from capacity_planning.utilities import stats_utils as st_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import time_utils as tm_ut
from capacity_planning.utilities import imputer

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
upr_horizon = 12 * 7  # days (included)
lwr_horizon = 9 * 7   # days (included)
time_scale = 'W'


def fcast_errs(a_df, d_df, cutoff_date, fmonth, periods=4):
    dr = pd.date_range(end=cutoff_date, freq='W', periods=periods)
    fdf = d_df[d_df['ds'].isin(dr)].copy()

    # language
    err_df = get_errs_(fdf, a_df, ['language'], 3.0, fmonth)
    d_all = {'language': ['All'], 'err_start': [str(dr.min().date())], 'err_end': [str(dr.max().date())], 'forecast': [fmonth],
             'err (%)': np.round(100 * (err_df['yhat'] - err_df['y']).sum() / err_df['y'].sum(), 1)}
    l_df = pd.concat([pd.DataFrame(d_all), err_df], axis=0)

    # service tier and language
    ls_df = get_errs_(fdf, a_df, ['language', 'service_tier'], 3.0, fmonth)
    ls_df['key'] = ls_df['language'] + '-' + ls_df['service_tier']

    lcols = ['language', 'forecast', 'err_start', 'err_end', 'err (%)']
    lscols = ['key', 'forecast', 'err_start', 'err_end', 'err (%)']
    return l_df[lcols], ls_df[lscols]


def get_errs_(fdf, adf, cols, o_coef, fstring):

    # actuals
    tdf = adf.groupby(['ds'] + cols).sum(numeric_only=True).reset_index()
    t_af_list = list()
    for k, f in tdf.groupby(cols):
        if isinstance(k, str):
            k = (k, )
        ft, _ = trim_outliers(f, None, o_coef, 'ds', ['y'])   # holidays?
        for i in range(len(cols)):
            ft[cols[i]] = k[i]
        t_af_list.append(ft)

    tdf = pd.concat(t_af_list, axis=0) if len(t_af_list) > 0 else None
    g_fdf = fdf.groupby(['ds'] + cols).sum(numeric_only=True).reset_index()
    df = g_fdf.merge(tdf, on=['ds'] + cols)
    xdf = df.groupby(cols).sum(numeric_only=True).reset_index()
    xdf['err (%)'] = np.round(100 * (xdf['yhat'] - xdf['y']) / xdf['y'], 1)
    xdf['forecast'] = fstring
    xdf['err_start'] = str(fdf['ds'].min().date())
    xdf['err_end'] = str(fdf['ds'].max().date())
    return xdf


def get_errs(cutoff_date_, fdf_obj, tdf_, o_coef=3.0, lwr=4, tcol='ds'):
    # gets errs on actuals
    # we should look at weeks 12 to 16 but we do weeks  9, 10, 11, 12 = cutoff_date to get it sooner
    if fdf_obj.data is None:
        s_ut.my_print('no available fcast data for ' + str(cutoff_date_.date()))
        return None, None
    if tdf_ is None:
        s_ut.my_print('ERROR: no available actuals data for ' + str(cutoff_date_.date()))
        sys.exit()
    s_ut.my_print('computing forecast errors for cutoff date: ' + str(cutoff_date_.date()))
    fdf = fdf_obj.data.copy()
    tdf = tdf_.copy()

    # fcast data
    lwr_week = cutoff_date_ - pd.to_timedelta(lwr, unit='W')   # excluded
    upr_week = cutoff_date_
    s_ut.my_print('errors from (excluded) ' + str(lwr_week.date()) + ' to (included) ' + str(cutoff_date_.date()) + ' *********************')
    p_fdf = fdf[(fdf[tcol] > lwr_week) & (fdf[tcol] <= upr_week)].copy()

    # errors
    err_df, off_df = get_errs_(p_fdf, tdf, ['language', 'service_tier'], o_coef, fdf_obj.forecast,  tcol=tcol)
    if 'index' in err_df.columns:
        err_df.drop('index', inplace=True, axis=1)
    if 'level_0' in err_df.columns:
        err_df.drop('level_0', inplace=True, axis=1)
    lang_df = err_df[err_df['service_tier'] == 'All'].copy()
    lang_df.drop('service_tier', axis=1, inplace=True)
    lang_df.drop_duplicates(inplace=True)
    lang_df.reset_index(drop=True, inplace=True)
    tier_df = err_df[err_df['service_tier'] != 'All'].copy()
    tier_df.drop_duplicates(inplace=True)
    tier_df.reset_index(drop=True, inplace=True)
    return lang_df, tier_df, off_df


def get_fcast_file(cutoff_date_, froot, months=3):
    f_month = 1 + (cutoff_date_.month - months) % 12          # (old) fcast issue month
    yr = cutoff_date_.year if f_month < cutoff_date_.month else cutoff_date_.year - 1
    dm = pd.to_datetime(str(yr) + '-' + str(f_month) + '-01')  # 1st day of issue month
    wd = dm.weekday()
    fcast_sat = dm - pd.to_timedelta(wd + 2, unit='D') if wd < 5 else dm - pd.to_timedelta(wd - 5, unit='D')
    fcast_f = froot + str(fcast_sat.date())
    return fcast_f


def get_fcast(cutoff_date_, froot, months=3):
    # get the fcast issued <months> ago
    f_month = 1 + (cutoff_date_.month - months) % 12   # fcast issue month
    yr = cutoff_date_.year if f_month < cutoff_date_.month else cutoff_date_.year - 1
    dm = pd.to_datetime(str(yr) + '-' + str(f_month) + '-01')  # 1st day of issue month
    wd = dm.weekday()
    fcast_sat = dm - pd.to_timedelta(wd + 2, unit='D') if wd < 5 else dm - pd.to_timedelta(wd - 5, unit='D')
    fcast_f = froot + str(fcast_sat.date())
    try:
        fdf = p_ut.read_df(os.path.expanduser(fcast_f))
    except OSError:
        s_ut.my_print('file not found: ' + froot)
        return None
    if fdf is None:
        return None
    else:
        p_ut.set_week_start(cfg_df, tcol='ds')  # week_starting patch

        fdf.rename(columns={'ticket_count': 'forecasted_count'}, inplace=True)
        s_ut.my_print('getting forecast from ' + fcast_f)
        return fdf


def c_group(a_df):
    def _c_group(_df):
        b_df = _df.groupby('channel').agg({'actual_count': np.sum}).reset_index()
        x = b_df.loc[b_df.index[0],]
        x['channel'] = 'all'
        x['actual_count'] = b_df['actual_count'].sum()
        return pd.concat([b_df, pd.DataFrame(x).transpose()])
    a_out = a_df.groupby(['ds', 'business_unit', 'language']).apply(_c_group).reset_index(level=[0, 1, 2])  # lang/BU and channel actuals
    return a_out


def get_actuals(cutoff_date_):
    fdir = os.path.expanduser('~/my_tmp/cleaned/')  # '~/my_tmp/in_df_data_'
    adf = None
    for f in os.listdir(fdir):
        if str(cutoff_date_.date()) in f and 'tickets_' in f and 'old' not in f:   # 'in_df_data_' in f:   # we do not know the rolling window
            s_ut.my_print('getting actuals from ' + fdir + f)
            adf = p_ut.read_df(fdir + f)
            break
    if adf is None:
        s_ut.my_print('no available actuals data for ' + str(cutoff_date_.date()))
        return None
    adf.reset_index(inplace=True, drop=True)
    p_ut.clean_cols(adf, ["language", "service_tier", "channel", "business_unit"],
                    '~/my_repos/capacity_planning/data/config/col_values.json',
                    check_new=False,
                    do_nan=False,
                    rename=True)
    adf.rename(columns={'ticket_count': 'y', 'ds_week_starting': 'ds'}, inplace=True)
    i_vals = ['nan', 'NULL', None, 'other', np.nan, 'null', 'N/A']
    imp_data = imputer.impute(adf, i_vals=i_vals, ex_cols=['ds'])
    imp_data['y'] = np.round(imp_data['y'].values, 0)
    return imp_data


def trim_outliers(mf_, h_df_, o_coef_, tcol, cols_):
    def _ps_outliers(f_, tc_, c_, ocoef, hdates_, lbl_dict):
        x, _, o_df = p_ut.ts_outliers(f_, tc_, c_, coef=ocoef, verbose=False, replace=True, ignore_dates=hdates_, lbl_dict=lbl_dict)
        return x[[tc_, c_]], o_df

    g_cols = list(set(mf_.columns) - set([tcol] + cols_))
    vals = mf_.loc[mf_.index[0], g_cols]
    if isinstance(vals, pd.core.frame.DataFrame):
        vals.set_index(g_cols, inplace=True)
        vals_d = vals.to_dict(orient='index')
    else:
        vals_d = vals.to_dict()
    if h_df_ is not None:       # hols: always None here
        dates_arr = h_df_.apply(
            lambda x: [x[tcol] + pd.to_timedelta(ix, unit='D') for ix in range(x['lower_window'], x['upper_window'])], axis=1).values  # holiday and window
        h_dates = list(set([dt for dt_list in dates_arr for dt in dt_list]))  # list of hol dates
    else:
        h_dates = list()
    df_list_ = [_ps_outliers(mf_[[tcol, c]].copy(), tcol, c, o_coef_, h_dates, vals_d) for c in cols_]
    df_list = [f[0] for f in df_list_ if f[0] is not None]
    of_list = [f[1] for f in df_list_ if f[1] is not None]
    off = pd.concat(of_list, axis=0) if len(of_list) > 0 else None
    t_df_out = reduce(lambda x, y: x.merge(y, on=tcol, how='inner'), df_list) if len(df_list) > 0 else None
    return t_df_out, off



