"""
python ratio_forecast.py ts_name to_table
ts_name: time series name. Mandatory. Must be a top level series, e.g. ticket_counts, prod_hours not a BU level
to_table: default to 0
The actual cutoff date for the forecast is in the ratio_forecast_cfg file. If missing, it is teh cutoff_date argument
applies ratios to get service tier forecasts
"""
import os
import platform
from functools import reduce
import json

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
import pandas as pd
import numpy as np

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.utilities import adj_forecast as adj
from capacity_planning.forecast.utilities import hts as hts
from capacity_planning.utilities import imputer
from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.data import hql_exec as hql


def main(argv):
    print(argv)
    if len(argv) == 2:
        ts_name = argv[-1]
        to_table = False
    elif len(argv) == 3:
        ts_name, to_table = argv[1:]
        try:
            to_table = bool(int(to_table))
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, to_table): ' + str(argv))
            sys.exit()
    else:
        s_ut.my_print('ERROR: invalid arguments (ts_name, to_table): ' + str(argv))
        sys.exit()

    if any([bu in ts_name for bu in ['Homes', 'Experiences', 'China']]):
        s_ut.my_print('ERROR: time series cannot be a BU time series: ' + str(ts_name))
        sys.exit()

    data_cfg = os.path.expanduser('~/my_repos/capacity_planning/forecast/config/ratio_forecast_cfg.json')
    if os.path.isfile(data_cfg):
        with open(data_cfg, 'r') as fptr:
            rf_dict = json.load(fptr)
    else:
        s_ut.my_print('ERROR: ' + data_cfg + ' file not found')
        sys.exit()

    d_date = rf_dict.get('data_date', None)
    if d_date is None:
        s_ut.my_print('ERROR: data_date cannot be null')
        sys.exit()
    data_date = pd.to_datetime(d_date)                                     # this is the cutoff date we get data from tables

    a_date = rf_dict.get('adjust_date', None)                              # if None, nothing to adjust and adj_date = data_date
    adjust_date = data_date if a_date is None else pd.to_datetime(a_date)  # this is the actual cutoff date

    window = rf_dict.get('ratio_windows', dict())
    if len(window) == 0:  # not set
        window = {'default': {'start': adjust_date - pd.to_timedelta(6, unit='W'), 'end': adjust_date}}
    else:
        for k, v in window.items():
            for kk, vv in v.items():
                v[kk] = pd.to_datetime(vv)

    s_ut.my_print('************************* read table date: ' + str(data_date.date()) + ' ********************************************')
    s_ut.my_print('************************* write table date: ' + str(adjust_date.date()) + ' *******************************************')

    # ###############################
    # ###############################
    time_scale = 'W'
    init_date = pd.to_datetime('2016-01-01')
    # ###############################
    # ###############################

    df_tilde, bottom_ts = hts.main(ts_name, data_date, do_cov=True)                 # coherent forecasts at language level + language level adjustments
    f_df = adj.main(df_tilde, 'language', bottom_ts, ts_name, adjust_date)            # must adjust at language level before service level ratios
    fr_list, fr_cols = list(), list()
    a_list = list()
    ts_list = bottom_ts   # ratios only on bottom_ts then aggregate to top TS
    for ts in ts_list:
        s_ut.my_print('============= starting ' + str(ts))
        ts_cfg, _ = dp.ts_setup(ts, data_date, init_date, time_scale)
        a_df = dp.ts_actuals(ts, ts_cfg, ['language', 'business_unit', 'channel', 'service_tier'], drop_cols=False)
        b_df = filter_actuals(a_df, window)
        fr = ts_ratio(ts, b_df.copy(), f_df[['ds', 'language', ts + '_tilde']].copy(), window, data_date)
        fr_list.append(fr)
        fr_cols.append(fr.columns)
        a_list.append(b_df)
        check_ratios(ts, b_df, fr, True, 'service_tier')
        check_ratios(ts, b_df, fr, False, 'service_tier')
        check_ratios(ts, b_df, fr, True, 'channel')
        check_ratios(ts, b_df, fr, False, 'channel')

    # must adjust together to ensure coherence
    fr = reduce(lambda x, y: x.merge(y, on=['ds', 'language', 'channel', 'service_tier'], how='outer'), fr_list) if len(fr_list) > 0 else None
    fr.fillna(0, inplace=True)
    for k_col in ['channel', 'service_tier']:            # language adj must be done before ratios
        fr = adj.main(fr, k_col, bottom_ts, ts_name, adjust_date)

    # save data
    f_list = list()
    for idx in range(len(ts_list)):
        ts = ts_list[idx]
        fx = fr[fr_cols[idx]].copy()
        fx.rename(columns={ts + '_tilde': 'yhat'}, inplace=True)
        fx['yhat'] = np.round(fx['yhat'].values, 0)  # this makes input totals and output totals to be a bit off
        fx = fx[fx['yhat'] > 0]
        fx['ts_name'] = ts
        fx['cutoff'] = adjust_date
        f_list.append(fx)

    # get the aggregate series
    fall = pd.concat(f_list, axis=0)
    gall = fall.groupby(['ds', 'language', 'channel', 'service_tier']).sum(numeric_only=True).reset_index()
    gall['cutoff'] = adjust_date
    gall['ts_name'] = ts_name

    # align cols (ap gets confused otherwise?)
    tcols = ['ds', 'language', 'channel', 'service_tier', 'yhat', 'ts_name', 'cutoff']
    fall = fall[tcols].copy()
    gall = gall[tcols].copy()

    # final DF to save
    fout = pd.concat([gall, fall], axis=0)
    p_ut.save_df(fout, '~/my_tmp/fbp/ratios_fcast_' + ts_name + '_' + str(adjust_date.date()))
    ts_cfg, _ = dp.ts_setup('ticket_count', data_date, init_date, time_scale)
    a_df = dp.ts_actuals('ticket_count', ts_cfg, ['language', 'business_unit', 'channel', 'service_tier'], drop_cols=False)
    p_ut.save_df(a_df, '~/my_tmp/a_df_ticket_count_' + str(adjust_date.date()))

    # data summary
    s_ut.my_print('**************** Data Summary *******************')
    for c in ['language', 'channel', 'service_tier']:
        s_ut.my_print('unique ' + c + ': ' + str(fout[c].unique()))

    # save to DB
    if to_table is True:
        tcols.remove('cutoff')
        tcols.remove('ts_name')
        tcols.insert(1, 'ds_week_starting')
        tcols.insert(2, 'fcst_date_inv_ending')               # ds_week_ending
        fout = fout[fout['ds'] > adjust_date.date()].copy()   # only save forecasted values
        fout['ds_week_starting'] = fout['ds']
        fout['fcst_date_inv_ending'] = fout['ds'] + pd.to_timedelta(6, unit='D')
        for ts in fout['ts_name'].unique():
            partition = {'cutoff': str(adjust_date.date()), 'ts_name': ts}
            tb_df = fout[fout['ts_name'] == ts].copy()
            ret = hql.to_tble(tb_df, tcols, 'sup.cx_weekly_forecasts', partition)
            if ret != 0:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no forecasts loaded to table for ' + ts_name +
                              ' and cutoff date ' + str(adjust_date.date()))
                sys.exit()
            else:
                s_ut.my_print('>>>>>>>>>>>>>>> SUCCESS: data saved to table <<<<<<<<<<<<<<<<<<<')
    else:
        s_ut.my_print('>>>>>>>>>>>>>>> WARNING: no data saved to table <<<<<<<<<<<<<<<<<<<')


def check_ratios(ts, a_df, f_df, is_dl, col):
    s_ut.my_print('ratio check to ' + str(ts) + ' and DL: ' + str(is_dl))
    b_df = a_df[a_df['channel'] == 'directly'].copy() if is_dl is True else a_df[a_df['channel'] != 'directly'].copy()
    a_r = get_ratios(b_df, 'ticket_count', [col]) if len(b_df) > 0 else None
    g_df = f_df[f_df['channel'] == 'directly'].copy() if is_dl is True else f_df[f_df['channel'] != 'directly'].copy()
    f_r = get_ratios(g_df, ts + '_tilde', [col]) if len(g_df) > 0 else None
    p_ut.save_df(a_df, '~/my_tmp/a_df_' + ts + '_' + str(is_dl) + '_' + col)
    p_ut.save_df(f_df, '~/my_tmp/f_df_' + ts + '_' + str(is_dl) + '_' + col)
    if a_r is not None and f_r is not None:
        raf = a_r.merge(f_r, on=[col, 'language'], how='left')
        raf.columns = [col, 'language', 'a_ratio', 'f_ratio']
        raf['diff'] = np.abs(raf['f_ratio'] - raf['a_ratio'])
        p_ut.save_df(raf, '~/my_tmp/raf_' + ts + '_' + str(is_dl) + '_' + col)
        if raf['diff'].min() > 1.0e-3:
            raf['ts_name'] = ts
            raf['is_dl'] = is_dl
            s_ut.my_print('WARNING: ' + col + ' ratios off <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print(raf[raf['diff'] > 0])
    elif (a_r is None and f_r is not None) or (a_r is not None and f_r is None):
        s_ut.my_print('ERROR: error in ratios for ' + ts)
        sys.exit()
    else:  # nothing to check
        pass


def filter_actuals(a_df, window):
    af_list = list()
    for lg, lf in a_df.groupby('language'):
        w = window.get(lg, window['default'])
        lf = lf[(lf['ds'] >= w['start']) & (lf['ds'] <= w['end'])].copy()
        af_list.append(lf)
    return pd.concat(af_list, axis=0)


def tmp_ratios(cu, window, gcols):  # tmp fix
    wf = pd.read_parquet('~/my_tmp/cleaned/old_tickets_2020-02-29.par')
    _ = p_ut.clean_cols(wf, ["language", "service_tier", "channel", "business_unit"], '~/my_repos/capacity_planning/data/config/col_values.json', check_new=True,
                        do_nan=True, rename=True)
    wf.rename(columns={'ds_week_starting': 'ds'}, inplace=True)
    wf['channel'] = wf.apply(lambda x: 'directly' if x['service_tier'] == 'directly' else x['channel'], axis=1)
    i_vals = ['nan', 'NULL', None, 'other', np.nan, 'null', 'N/A']
    wf['ds'] = wf['ds'].dt.date.astype(str)
    wf = imputer.impute(wf, i_vals=i_vals, ex_cols=['ds'])
    wf['ds'] = pd.to_datetime(wf['ds'])
    wf = wf[(wf['ds'] <= cu) & (wf['ds'] >= cu - pd.to_timedelta(window, unit='W'))].copy()
    wf['channel'] = wf.apply(lambda x: 'directly' if x['service_tier'] == 'directly' else x['channel'], axis=1)  # again in case imputation added directly wrongly
    a_df = wf[wf['channel'] != 'directly'].copy()
    lct_df = a_df.groupby(gcols).sum(numeric_only=True).reset_index()
    l_df = lct_df.groupby(['language']).sum(numeric_only=True).reset_index()
    lct_ratio = lct_df.merge(l_df, on=['language'], how='left')
    lct_ratio['ratio'] = lct_ratio['ticket_count' + '_x'] / lct_ratio['ticket_count' + '_y']
    lct_ratio.drop(['ticket_count' + '_x', 'ticket_count' + '_y'], axis=1, inplace=True)
    return lct_ratio


def get_ratios(df, ycol, gcols):
    l_df = df.groupby(['language']).sum(numeric_only=True).reset_index()
    acols = list(set(['language'] + gcols))
    lg_df = df.groupby(acols).sum(numeric_only=True).reset_index()
    df_ratio = lg_df.merge(l_df, on='language', how='left')
    df_ratio['ratio'] = df_ratio[ycol + '_x'] / df_ratio[ycol + '_y']
    df_ratio.drop([ycol + '_x', ycol + '_y'], axis=1, inplace=True)
    return df_ratio


def set_ratios(l_df, r_df, ycol, scale=1.0):
    df = l_df.merge(r_df, on='language', how='left')
    df[ycol] *= df['ratio'] * scale
    df.drop('ratio', inplace=True, axis=1)
    return df


def n_func(af):
    af['ratio'] /= af['ratio'].sum()
    return af


def ts_ratio(ts, act_df, f_df, wd, cutoff_date):
    # use the last w dates of actuals to set the counts at channel and service tier level for each language
    ycol = 'ticket_count'
    gcols = ['language', 'channel', 'service_tier']

    # drop 'All' language in case it is there
    act_df = act_df[act_df['language'] != 'All'].copy()
    f_df = f_df[f_df['language'] != 'All'].copy()
    f_df.rename(columns={ts + '_tilde': 'yhat'}, inplace=True)

    # directly
    ad_df = act_df[act_df['channel'] == 'directly'].copy()
    dl_scale = ad_df['ticket_count'].sum() / act_df['ticket_count'].sum()
    if len(ad_df) > 0:
        dl_ratio = get_ratios(ad_df, ycol, gcols)
        dl_fdf = set_ratios(f_df, dl_ratio, 'yhat', scale=dl_scale)
    else:
        dl_fdf = None

    # non directly
    lct_df = act_df[act_df['channel'] != 'directly'].copy()
    lct_scale = lct_df['ticket_count'].sum() / act_df['ticket_count'].sum()

    # ###############################################################################################
    # ###############################################################################################
    # ###############################################################################################
    # ######################################## temporal fix #########################################
    if str(cutoff_date.date()) == '2020-02-29':
        rcols = ['language', 'channel', 'service_tier', 'ratio']
        if ts == 'ticket_count_Experiences':
            my_tiers = ['Experiences', 'Claims', 'Safety']
            my_channels = ['phone', 'email', 'messaging', 'directly', 'chat']
        elif ts == 'ticket_count_Homes':
            my_tiers = ['Claims', 'Community Education', 'Payments', 'ProHost + Plus', 'Regulatory Response', 'Resolutions 1', 'Resolutions 2', 'Safety', 'SafetyHub', 'Neighbors']
            my_channels = ['phone', 'email', 'messaging', 'directly', 'chat']
        elif ts == 'ticket_count_China':
            my_tiers = ['Resolutions 1', 'Community Education', 'ProHost + Plus', 'Resolutions 2', 'Claims', 'Regulatory Response', 'Payments', 'Safety', 'SafetyHub']
            my_channels = ['phone', 'email', 'messaging', 'directly', 'chat']
        else:
            print('??????????????')
            sys.exit()
        lct_ratio_new = get_ratios(lct_df, ycol, gcols)
        lct_ratio_tmp = tmp_ratios(cutoff_date, wd, gcols)

        if ts == 'ticket_count_Experiences':
            lct_ratio = lct_ratio_new[rcols].copy()
        elif ts == 'ticket_count_Homes':
            svc_tiers = ['Claims', 'Community Education', 'Neighbors', 'Payments', 'Plus',
                         'ProHost', 'Regulatory Response', 'Resolutions 1', 'Resolutions 2', 'Safety', 'SafetyHub', 'Social Media']
            tmp_tiers = ['Claims', 'Community Education', 'Payments', 'Plus',
                         'Regulatory Response', 'Resolutions 1', 'Resolutions 2', 'Safety', 'SafetyHub']
            new_tiers = ['Neighbors', 'ProHost', 'Social Media']
            lct_ratio = lct_ratio_tmp.merge(lct_ratio_new, on=['language', 'channel', 'service_tier'], how='outer')
            lct_ratio.fillna(0, inplace=True)
            lct_ratio.rename(columns={'ratio_x': 'ratio_tmp', 'ratio_y': 'ratio_new'}, inplace=True)
            lct_ratio['ratio'] = lct_ratio.apply(lambda x: x['ratio_tmp'] if x['service_tier'] in tmp_tiers else x['ratio_new'], axis=1)
            lct_ratio = lct_ratio[lct_ratio['service_tier'].isin(my_tiers)].copy()
            lct_ratio = lct_ratio[rcols].groupby('language').apply(n_func)
        elif ts == 'ticket_count_China':
            svc_tiers = ['Resolutions 1', 'Community Education', 'Plus', 'Resolutions 2', 'Claims', 'Regulatory Response', 'Payments', 'Safety',
                         'Experiences', 'SafetyHub', 'Neighbors', 'Social Media']
            tmp_tiers = ['Resolutions 1', 'Community Education', 'Plus', 'Resolutions 2', 'Claims', 'Regulatory Response', 'Payments', 'Safety', 'SafetyHub']
            new_tiers = ['Experiences', 'Neighbors', 'Social Media']
            lct_ratio = lct_ratio_tmp[rcols].copy()
        else:
            print('??????????????')
            sys.exit()
        # ###################### NEEDS ROLLING PER LORETA'S RULES #######################################
        # ###############################################################################################
        # ###############################################################################################
        lct_ratio = lct_ratio[lct_ratio['service_tier'].isin(my_tiers)].copy()
        lct_ratio = lct_ratio[rcols].groupby('language').apply(n_func)
        # ###############################################################################################
        # ###############################################################################################
        # ###############################################################################################
    else:
        lct_ratio = get_ratios(lct_df, ycol, gcols)

    # language, channel service_tier
    lct_fdf = set_ratios(f_df, lct_ratio, 'yhat', scale=lct_scale)

    # merge directly and non-directly
    fout = lct_fdf if dl_fdf is None else pd.concat([dl_fdf, lct_fdf], axis=0)

    fout = fout[lct_fdf['yhat'] > 0].copy()
    fout.rename(columns={'yhat': ts + '_tilde'}, inplace=True)
    fout.dropna(inplace=True)
    return fout


if __name__ == '__main__':
    s_ut.my_print(sys.argv)
    main(sys.argv)
    print('DONE')
