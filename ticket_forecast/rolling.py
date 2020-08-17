"""
builds rolling average of <window> forecasts from run_date backwards and also generates the xls of the rolling forecast
assumes forecasts are run each Saturday
$ python rolling.py <window> cutoff_date
window: integer
cutoff_date: this should be a Saturday
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import pandas as pd
import numpy as np

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.ticket_forecast import errors as errs
from capacity_planning.forecast.ticket_forecast import to_excel as txl
# from capacity_planning.forecast.ticket_forecast import tix_to_table as t2t
from capacity_planning.utilities import time_utils as tm_ut
from capacity_planning.forecast.utilities.language import time_series as ts

DO_MP = True

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

# year: target year
# count: target forecast
# variability: max variability in tgt_year tgt before triggering adjustment
# adj: factor to reduce error and control variability when the delta is slightly above the variability.
target = {'year': 2020, 'count': 28.6e+06, 'variability': 0.03, 'adjust': 2.0}


def process_w_df(df_, cutoff_date_, tcol, gcols):
    zd = df_.duplicated()
    if df_[zd]['ticket_count'].sum() > 0:
        s_ut.my_print('ERROR: non-zero duplicated counts: ' + str(df_[zd]['ticket_count'].sum()))
        sys.exit()
    df_.drop_duplicates(inplace=True)
    df_ = df_.groupby(gcols + [tcol]).apply(lambda x: x['ticket_count'].max()).reset_index()
    df_.rename(columns={0: 'ticket_count'}, inplace=True)
    df_ = df_[df_['ticket_count'] > 0].copy(0)
    df_[tcol] = pd.to_datetime(df_[tcol])
    fcast_df_ = df_[df_[tcol] > cutoff_date_].copy()
    return fcast_df_


def table_load(dr_, cutoff_date_, m_adj=1.0):
    # load to adjusted data to the table
    # read table files '~/Forecasts/par/' + 'table_output_'
    gcols = ['dim_business_unit', 'dim_language', 'dim_tier', 'dim_channel', 'time_interval']
    t_list, max_ds_ = list(), None
    for d_ in dr_:
        fname_ = os.path.expanduser('~/Forecasts/par/table_output_' + str(d_.date())) + '.par'
        s_ut.my_print('rolling date: ' + str(d_.date()) + ' fname: ' + str(fname_))
        if os.path.exists(fname_):
            fx = p_ut.read_df(fname_)
            p_ut.set_week_start(fx, tcol='fcst_date_inv_endings')  # week_starting patch

            # week_starting patch
            df_cols__ = fx.columns
            if 'ds_week_ending' in df_cols__ and 'ds_week_starting' not in df_cols__:
                fx['ds_week_ending'] = pd.to_datetime(fx['ds_week_ending'])
                fx['ds_week_starting'] = fx['ds_week_ending'] - pd.to_timedelta(6, unit='D')

            fv = process_w_df(fx, cutoff_date_, 'fcst_date_inv_ending', gcols + ['run_date_inv_ending'])
            max_ds_ = fv['fcst_date_inv_ending'].max() if max_ds_ is None else min(max_ds_, fv['fcst_date_inv_ending'].max())
            t_list.append(fv)
    tdf = pd.concat(t_list, axis=0)
    t_fdf = tdf.groupby(gcols + ['fcst_date_inv_ending']).apply(lambda x: x['ticket_count'].mean()).reset_index()
    t_fdf.rename(columns={0: 'ticket_count'}, inplace=True)
    avg_tdf = t_fdf[t_fdf['fcst_date_inv_ending'] <= max_ds_].copy()
    avg_tdf['run_date_inv_ending'] = str(cutoff_date_.date())
    avg_tdf.reset_index(inplace=True)
    avg_tdf.rename(columns={'index': 'fcst_horizon'}, inplace=True)
    avg_tdf['fcst_date_inv_ending'] = avg_tdf['fcst_date_inv_ending'].dt.date.astype(str)
    avg_tdf['ticket_count'] *= m_adj
    print('******* saving data to load to sup.dim_cx_ticket_forecast >>>>>>>>>>>>>>')
    p_ut.save_df(avg_tdf, '~/my_tmp/tab_data_' + str(cutoff_date_.date()))
    print(888888888888888888888888)
    print('---------------- SKIPPING TABLE ---------------------')
    ret =-1
    # ret = t2t.to_table(avg_tdf, str(cutoff_date_.date()), 'sup.dim_cx_ticket_forecast')         # 'josep.dim_ticket_facst_test    # 'sup.dim_cx_ticket_forecast'
    if ret == -1:
        s_ut.my_print('ERROR: table push failed')
    return ret


def adjust_DF(curr_obj, cutoff_date_, target_):
    # adjust wrt to raw (non adjusted) previous data
    # manage fluctuations around target_
    tgt_y = target_['year']
    adj = target_['adjust']
    max_y_err = target_['variability']
    y_start, y_end = str(tgt_y) + '-01-01', str(tgt_y) + '-12-31'

    froot = curr_obj.froot
    dt = pd.to_datetime(str(cutoff_date_.year) + '-' + str(cutoff_date_.month) + '-01')
    last_cu = tm_ut.get_last_sat(dt)
    if last_cu.month == cutoff_date_.month:  # the first of the month is a Sat!
        last_cu -= pd.to_timedelta(7, unit='D')
    fpath = froot + str(last_cu.date())
    s_ut.my_print('last forecast path: ' + fpath)
    last_obj = ts.TicketForecast(fpath)

    curr_df = curr_obj.data.copy()   # current forecast
    last_df = last_obj.data.copy()
    if last_df is None or curr_df is None:
        s_ut.my_print('WARNING: data not found. Cannot adjust')
        return curr_df, 1.0, 0.0

    last_ttl = txl.get_total(last_df, None, y_start, y_end)['All']
    curr_ttl = txl.get_total(curr_df, None, y_start, y_end)['All']
    dev = np.abs(curr_ttl - last_ttl) / last_ttl
    string = '\n BU: All raw last total: ' + str(int(last_ttl)) + ' current raw total: ' + str(int(curr_ttl)) + ' dev: ' + str(np.round(100.0 * dev, 2)) + '%'
    if dev <= max_y_err:
        m = 1.0
    elif max_y_err < dev <= adj * max_y_err:
        if last_ttl < curr_ttl:
            m = (1.0 + (dev / adj)) * (last_ttl / curr_ttl)
        else:
            m = (1.0 - (dev / adj)) * (last_ttl / curr_ttl)
    else:
        p_ut.save_df(curr_df, '~/my_tmp/curr_df')
        p_ut.save_df(last_df, '~/my_tmp/last_df')
        s_ut.my_print('ERROR: deviation is too large')
        sys.exit()

    curr_df['ticket_count'] = curr_df.apply(lambda x: x['ticket_count'] * m if x['ds_week_ending'] > cutoff_date else x['ticket_count'], axis=1)  # adjust fcasts only
    adj_ttl = txl.get_total(curr_df, None, y_start, y_end)['All']  # adjusted current
    new_dev = np.abs(adj_ttl - last_ttl) / last_ttl
    string += '\n BU: All last raw total: ' + str(int(last_ttl)) + \
              ' current adjusted total: ' + str(int(adj_ttl)) + ' adjusted dev: ' + str(np.round(100.0 * new_dev, 2)) + '%'

    # deltas by BU after adjusting (from last month raw)
    s_ut.my_print('BU level changes (2020)')
    last_bu = txl.get_total(last_df, 'business_unit', y_start, y_end)
    curr_bu = txl.get_total(curr_df, 'business_unit', y_start, y_end)
    for k, last_v in last_bu.items():
        curr_v = curr_bu[k]
        bu_dev = np.abs(curr_v - last_v) / last_v
        string += '\n BU: ' + str(k) + ' last raw total: ' + str(int(last_v)) + \
                  ' current adjusted total: ' + str(int(curr_v)) + ' adjusted dev: ' + str(np.round(100.0 * bu_dev, 2)) + '%'

    # adjustment print
    s_ut.my_print('\n' + string + '\n')
    return curr_df, m, new_dev


if __name__ == '__main__':
    # ##########################################
    # ##########################################
    # ##########################################

    print(sys.argv)
    if len(sys.argv) == 3:
        _, window, run_date = sys.argv
    elif len(sys.argv) == 2:
        _, window = sys.argv
        run_date = pd.to_datetime('today')
    else:
        print('???????????')
        sys.exit()
    window = int(window)

    # #########################
    actuals_weeks = 12
    target_year = 2020
    # #########################

    # generate the list of dates:
    # run date can be any weekday
    cutoff_date = tm_ut.get_last_sat(run_date)          # get max{saturdays} such that saturday <= run_date
    dr = pd.date_range(end=cutoff_date, periods=window, freq='7D')
    s_ut.my_print('rolling weeks: ' + str([str(x.date()) for x in dr]))

    # read weekly par files
    df_list, act_df, max_ds = list(), None, None
    for d in dr:
        # fname = os.path.expanduser('~/Forecasts/par/xls_df_' + str(d.date())) + '.par'
        # s_ut.my_print('rolling date: ' + str(d.date()) + ' fname: ' + str(fname))
        for bu in ['Homes', 'Experiences', 'China']:
            fname = os.path.expanduser('~/my_tmp/fbp/ratios_fcast_tickets_' + bu + '_' + str(d.date())) + '.par'
            print(fname)
            if os.path.exists(fname):
                df = p_ut.read_df(fname)
                df['business_unit'] = bu
                df_list.append(df)
    df = pd.concat(df_list, axis=0)
    df.rename(columns={'yhat': 'ticket_count'})
    max_ds = df['ds'].max()
        # p_ut.set_week_start(df, tcol='ds')  # week_starting patch
        # if 'ds_week_ending' in df.columns:
        #     df['ds_week_ending'] = pd.to_datetime(df['ds_week_ending'])
        # fcast_df = process_w_df(df, cutoff_date, 'ds_week_ending', ['dim_business_unit', 'dim_language', 'dim_tier', 'dim_channel'])
        # max_ds = fcast_df['ds_week_ending'].max() if max_ds is None else min(max_ds, fcast_df['ds_week_ending'].max())
        # df_list.append(fcast_df)
    # else:
    #         s_ut.my_print('ERROR: ' + str(fname) + ' is missing')
    #         sys.exit()

    # rolling average of fcasts
    avg_fdf = df.copy()
    # fdf = pd.concat(df_list, axis=0)
    # r_fdf = fdf.groupby(['dim_business_unit', 'dim_language', 'dim_tier', 'dim_channel', 'ds_week_ending']).apply(lambda x: x['ticket_count'].mean()).reset_index()
    # r_fdf.rename(columns={0: 'ticket_count'}, inplace=True)
    # avg_fdf = r_fdf[(r_fdf['ds_week_ending'] <= max_ds) & (r_fdf['ds_week_ending'] > cutoff_date)].copy()

    # save fcast only, raw (not adjusted)
    # fcast_f = '~/Forecasts/rolling/par/raw_r_fcast_' + str(window) + '_' + str(cutoff_date.date())  # only non-ADJUSTED fcast data
    # avg_fdf['adj'] = False
    # avg_fdf['initiative'] = False
    # s_ut.my_print('saving raw fcast data to ' + fcast_f)
    # p_ut.save_df(avg_fdf, fcast_f)   # ONLY forecasted data

    # accuracy for the <months> months old raw forecast
    adf = errs.get_actuals(cutoff_date)                                                                                   # raw actuals up to cutoff_date
    window = 4
    fcast_file = errs.get_fcast_file(cutoff_date, '~/Forecasts/rolling/par/raw_r_fcast_' + str(window) + '_', months=3)   # file path from <months> months old forecast from cutoff
    fdf_obj = ts.TicketForecast(fcast_file)                                                                            # fcast obj from 3 months ago
    fdf = fdf_obj.data
    p_ut.clean_cols(fdf, ["language", "service_tier", "channel", "business_unit"],
                    '~/my_repos/capacity_planning/data/config/col_values.json',
                    check_new=False,
                    do_nan=False,
                    rename=True)
    fdf.rename(columns={'ticket_count': 'forecasted_count'}, inplace=True)
    # fdf['ds_week_ending'] = pd.to_datetime(fdf['ds_week_ending'])
    # fdf['ds'] = fdf['ds_week_ending'] - pd.to_timedelta(6, unit='D')
    if fdf is None or adf is None:
        lang_errs, tier_errs = None, None
    else:
        s_ut.my_print('Error wrt actuals for an old forecast')
        lang_errs, tier_errs, off_df = errs.get_errs(cutoff_date, fdf_obj, adf, tcol='ds')                                                    # errs on filtered actuals for old raw fcast
        lang_errs['adj'] = False
        lang_errs['initiative'] = False
        tier_errs['adj'] = False
        tier_errs['initiative'] = False
        if off_df is not None:
            off_df['adj'] = False
            off_df['initiative'] = False
            p_ut.save_df(off_df, '~/Forecasts/rolling/errors/raw_r_outliers_' + str(cutoff_date.date()))
        p_ut.save_df(lang_errs, '~/Forecasts/rolling/errors/raw_r_language_' + str(cutoff_date.date()))
        p_ut.save_df(tier_errs, '~/Forecasts/rolling/errors/raw_r_tier_' + str(cutoff_date.date()))

    # save raw rolling data, NOT adjusted
    act_df = adf[['business_unit', 'language', 'service_tier', 'channel', 'ds',  'actual_count']].copy()
    act_df.rename(columns={'actual_count': 'ticket_count'}, inplace=True)
    act_df = act_df[act_df['ds'] >= cutoff_date - pd.to_timedelta(actuals_weeks,  unit='W')]
    avg_fdf.rename(columns={'yhat': 'ticket_count'}, inplace=True)
    b_df = pd.concat([act_df, avg_fdf], axis=0)

    b_df['is_actual'] = b_df['ds'].apply(lambda x: 1 if x <= cutoff_date else 0)
    b_df['run_date'] = cutoff_date
    b_df['adj'] = False
    b_df['initiative'] = False
    xls_fr = '~/Forecasts/rolling/par/raw_r_xls_' + str(window) + '_' + str(cutoff_date.date())   # 16 weeks of actuals and forecast. NOT adjusted
    s_ut.my_print('saving raw data (fcast + actuals) to ' + xls_fr)
    p_ut.save_df(b_df, xls_fr)
    curr_obj = ts.TicketForecast(xls_fr)

    # adjust fcast wrt previous rolling fcast
    # adj_fdf, mval, dev = adjust_DF(curr_obj, cutoff_date, target)

    # save rolling data, adjusted
    # adj_fdf['is_actual'] = adj_fdf['ds_week_ending'].apply(lambda x: 1 if x <= cutoff_date else 0)
    # adj_fdf['run_date'] = cutoff_date
    # adj_fdf['ds_week_starting'] = adj_fdf['ds_week_ending'] - pd.to_timedelta(6, unit='D')
    # adj_fdf['adj'] = True
    # adj_fdf['initiative'] = False
    # xls_fa = '~/Forecasts/rolling/par/adj_r_xls_' + str(window) + '_' + str(cutoff_date.date())   # actuals_weeks of actuals and forecast. adjusted
    # s_ut.my_print('saving adjusted data (fcast + actuals) to ' + xls_fa)
    # p_ut.save_df(adj_fdf, xls_fa)
    # curr_adj_obj = ts.TicketForecast(xls_fa)   # current adjusted fcast

    # get the previous adj forecast
    last_rolling = tm_ut.get_last_sat(str(cutoff_date.year) + '-' + str(cutoff_date.month) + '-01')
    if last_rolling.month == cutoff_date.month:  # the first of the month is a Sat!
        last_rolling -= pd.to_timedelta(7, unit='D')
    # froot = curr_adj_obj.froot
    froot = curr_obj.froot
    f_prev = os.path.expanduser(froot + str(last_rolling.date())) + '.par'
    s_ut.my_print('previous adjusted fcast: ' + str(f_prev))
    prev_adj_obj = ts.TicketForecast(f_prev)   # prior adj fcast

    # save fcast only, raw (adjusted)
    fcast_f = '~/Forecasts/rolling/par/adj_r_fcast_' + str(window) + '_' + str(cutoff_date.date())  # only ADJUSTED fcast data
    # adj_fdf['adj'] = True
    # adj_fdf['initiative'] = False
    # s_ut.my_print('saving adj fcast data to ' + fcast_f)
    # p_ut.save_df(adj_fdf[adj_fdf['ds_week_ending'] > cutoff_date], fcast_f)

    # prepare excel output
    # xl_file = '~/Forecasts/rolling/xls/adj_r_fcast_' + str(window) + '_' + str(cutoff_date.date()) + '.xlsx'
    xl_file = '~/my_tmp/xls.xlsx'
    froot = '~/Forecasts/rolling/par/adj_r_fcast_' + str(window) + '_'
    lact_df = errs.c_group(adf)                     # actuals aggregated by dim_tier (i.e. no dim_tier)
    # txl.to_excel_(xl_file, cutoff_date, curr_adj_obj, prev_adj_obj, target_year, lang_errs, tier_errs, lact_df)
    txl.to_excel_(xl_file, cutoff_date, curr_obj, prev_adj_obj, target_year, lang_errs, tier_errs, lact_df)

    # load to table: load adjusted values as these are the published ones
    s_ut.my_print('\n*********** forecast rolling to ' + str(cutoff_date.date()) +
                  ' has month-to-month adjustment of ' + str(mval) + ' and new deviation of ' + str(np.round(100.0 * dev, 2)) + '% ************')
    ret = table_load(dr, cutoff_date, m_adj=mval)
    if ret == -1:
        s_ut.my_print('ERROR: table did not load')
        sys.exit()
    else:
        print('DONE')
