"""
$ python biz_ticket_forecast run_date with_bu
run_date: a date. will be reset to the sta prior (if Tues to Fri) or 2 sat prior (if Sun, Mon)
with_bu: if 1, use a forecast by lang and BU otherwise only language. Default = 1
"""
import os
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
from scipy import stats as sp_s
import pandas as pd
import json
import numpy as np

from capacity_planning.forecast.utilities.tickets import tkt_utils as t_ut
from capacity_planning.forecast.utilities.language import regressors as regs
from capacity_planning.forecast.utilities.language import time_series as ts
from capacity_planning.forecast.utilities.language import data_prep as dtp
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import time_utils as tm_ut


pd.set_option('display.max_rows', 100)
pd.set_option('precision', 4)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 35)


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
is_mp = True


def ens_fcast(fdf, adf, cutoff_date, g_cols, b_cols, normalize=True):
    fdf_idx = t_ut.set_cfg_idx(fdf.copy())
    t_start = max(adf['ds'].min(), fdf_idx['ds'].min())
    t_fdf = fdf_idx[(fdf_idx['ds'] <= cutoff_date) & (fdf_idx['ds'] >= t_start)].copy()
    t_adf = adf[(adf['ds'] <= cutoff_date) & (adf['ds'] >= t_start)][['ds', 'ticket_count'] + b_cols + g_cols].copy()
    v_fdf = fdf_idx[(fdf_idx['ds'] > cutoff_date)].copy()

    cols = b_cols + g_cols
    g_adf_dict = {gc: fgc for gc, fgc in t_adf.groupby(cols)}
    g_fdf_dict = {gc: fgc for gc, fgc in v_fdf.groupby(cols)}
    arg_list = [[l, lf, g_adf_dict.get(l, None), g_fdf_dict.get(l, None), 'ticket_count', cols, normalize]
                for l, lf in t_fdf[['ds', 'ticket_count', 'cfg_idx'] + cols].groupby(cols)]
    d_list = s_ut.do_mp(t_ut.lasso_selection, arg_list, is_mp=True, cpus=None, do_sigkill=True)
    f_all = pd.concat([d['res'] for d in d_list], axis=0)
    f_all.dropna(inplace=True)
    f_all = f_all[f_all['y_pred'] > 0]
    f_all['y_pred'] = np.round(f_all['y_pred'].values, 0)
    return f_all


def biz_fcast(fdf, adf, g_cols, b_cols, cutoff_date, time_scale, inc_start, inc_end):
    df_list = list()
    for l, f_df in fdf.groupby(g_cols):
        s_ut.my_print('biz_fcast for ' + str(l))
        if len(l) == 2:
            lang, bu = l
            a_df = adf[(adf['language'] == lang) & (adf['business_unit'] == bu)].copy()
        else:
            lang = l
            a_df = adf[adf['language'] == lang].copy()
        b_fcast = t_ut.tier_split(a_df, f_df, b_cols, cutoff_date, time_scale, inc_start, inc_end=inc_end)
        df_list.append(b_fcast[b_fcast['ticket_count'] > 0])
    return pd.concat(df_list) if len(df_list) > 0 else None


def main(argv):
    print(argv)
    time_scale = 'W'    # reset for daily ticket data
    if len(sys.argv) == 2:
        run_date = sys.argv[1]   # at least 3 days after last Saturday with actual data
        with_bu = True
        s_ut.my_print('WARNING: rerun not set in command line. Assuming no rerun')
    elif len(sys.argv) == 3:
        _, run_date, bu = sys.argv  # at least 3 days after last Saturday with actual data
        with_bu = bool(int(bu))
    elif len(sys.argv) == 1:
        with_bu = True
        run_date = str(pd.to_datetime('today').date())
    else:
        print('invalid args: ' + str(sys.argv))
        sys.exit()

    cutoff_date = tm_ut.get_last_sat(run_date)   # set to last saturday

    if time_scale == 'W':
        upr_horizon, lwr_horizon = 75, None
        fcast_days = 7 * upr_horizon                         # regardless of time_scale
        inc_start, inc_end = 4, 0
    else:
        upr_horizon, lwr_horizon = 75 * 7, None
        fcast_days = upr_horizon
        inc_start, inc_end = 28, 0

    fcast_date = cutoff_date + pd.to_timedelta(fcast_days, unit='D')

    # get actuals
    act_df = p_ut.read_df('~/my_tmp/tix_act_df_' + str(cutoff_date.date()))
    if act_df is None:
        s_ut.my_print('ERROR: No actuals found')
        sys.exit()
    p_ut.set_week_start(act_df, tcol='ds')  # week_starting patch

    # get lang fcast
    froot = '~/my_tmp/fbp_tix_'
    fname = froot + 'lwbu_fcast_' if with_bu is True else froot + 'lnbu_fcast_'
    fcast_df = p_ut.read_df(fname + str(cutoff_date.date()))
    if fcast_df is None:
        s_ut.my_print('ERROR: No fcast found')
        sys.exit()
    p_ut.set_week_start(fcast_df, tcol='ds')  # week_starting patch

    b_cols = ['agent_sector', 'channel']
    g_cols = ['language']
    if with_bu is False:
        b_cols.append('business_unit')
        fcast_df.drop('business_unit', inplace=True, axis=1)  # all None
    else:
        g_cols.append('business_unit')
    s_ut.my_print('------------------------- start biz level forecast from cutoff date ' + str(cutoff_date.date()) +
                  ' to forecast date ' + str(fcast_date.date()) + ' with business columns: ' + str(b_cols) + '  ------------')

    b_fcast = biz_fcast(fcast_df, act_df, g_cols, b_cols, cutoff_date, time_scale, inc_start, inc_end)
    if b_fcast is not None:  # save all the fcats for eahc fcast cfg
        froot = '~/my_tmp/fbp_tix_'
        fname = froot + 'wbu_b_fcast_' if with_bu is True else froot + 'nbu_b_fcast_'
        p_ut.save_df(b_fcast, fname + str(cutoff_date.date()))
    else:
        print('ERROR: no business fcast')
        sys.exit()

    # final fcast (ens_avg)
    ens_df = ens_fcast(b_fcast, act_df, cutoff_date, g_cols, b_cols)
    froot = '~/my_tmp/fbp_tix_'
    fname = froot + 'wbu_e_fcast_' if with_bu is True else froot + 'nbu_e_fcast_'
    p_ut.save_df(ens_df, fname + str(cutoff_date.date()))

    print('++++++++++++++ Error Summary ++++++++++++')
    # check for language error
    fdf = ens_df.groupby(['ds', 'language']).agg({'y_pred': np.sum}).reset_index()
    months = 3
    m_start = pd.to_datetime(str(cutoff_date.year) + '-' + str(cutoff_date.month) + '-01') + pd.DateOffset(months=months+1)
    end_date = tm_ut.last_saturday_month(m_start)                     # max date for err check
    collect_date = cutoff_date - pd.DateOffset(months=months)
    start_date = end_date - pd.to_timedelta(2, unit='W')       # start date for error check
    a_df, _ = t_ut.get_actuals(end_date, collect_date)            # actuals from collect date to end_date
    fa = t_ut.set_act(a_df, ['language'])  # clean TS for each language
    fa = fa[(fa['ds'] > start_date) & (fa['ds'] <= end_date)].copy()
    z = fa.merge(fdf, on=['ds', 'language'], how='left')
    z = z[(z['y_pred'] > 0) & z['ticket_count'] > 0].copy()
    z_lang = z.groupby('language').agg({'ticket_count': np.sum, 'y_pred': np.sum}).reset_index()
    z_all = pd.DataFrame({'language': ['All'], 'ticket_count': [z_lang['ticket_count'].sum()], 'y_pred': [z_lang['y_pred'].sum()]})
    z_lang = pd.concat([z_all, z_lang], axis=0)
    z_lang['err'] = np.abs((z_lang['y_pred'] / z_lang['ticket_count']) - 1)
    print(z_lang)



    # t_ut.err_chk(ens_df, cutoff_date, [['language']], ycol='y_pred', months=3)
    print('DONE')


if __name__ == '__main__':
    s_ut.my_print(sys.argv)
    main(sys.argv)

