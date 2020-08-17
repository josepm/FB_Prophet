"""
generates excel output
assumes forecasts are run each Saturday
$ python cutoff_date
cutoff_date: date used to load from ens_forecast table
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
import copy

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.ticket_forecast import errors as errs
from capacity_planning.forecast.ticket_forecast import excel_utils as xl_ut
from capacity_planning.utilities import time_utils as tm_ut
from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.forecast.utilities.language import fcast_processing as fp

DO_MP = True

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def get_fcast(ts_name_, ts_cfg_, cutoff_date_, bu_, use_cache=None):
    ts = ts_name_ + '_' + bu_
    d_cfg = copy.deepcopy(ts_cfg_)
    d_cfg['ts_key'] = ts_cfg['ycol'] + '_' + bu_
    d_cfg.pop('business_units')
    t_df = fp.get_ratio_fcast(ts, d_cfg, cutoff_date_, use_cache=use_cache)
    t_df.rename(columns={'yhat': d_cfg['ycol']})
    t_df['business_unit'] = bu_
    return t_df


if __name__ == '__main__':
    # ##########################################
    # ##########################################
    # ##########################################

    print(sys.argv)
    if len(sys.argv) == 2:
        _, cutoff_date = sys.argv
        cutoff_date = pd.to_datetime(cutoff_date)
    else:
        print('invalid args: ' + str(sys.argv))
        sys.exit()

    # #########################
    actuals_weeks = 12
    target_year = 2020
    init_date = pd.to_datetime('2016-01-01')
    time_scale = 'W'
    ts_name = 'ticket_count'
    # #########################

    # get actuals
    ts_cfg, _ = dp.ts_setup(ts_name, cutoff_date, init_date, time_scale)
    adf = dp.ts_actuals(ts_name, ts_cfg, ['language', 'business_unit', 'channel', 'service_tier'], drop_cols=False, use_cache=False)
    start, end = tm_ut.iso_dates(target_year - 1)
    adf = adf[adf['ds'] >= start]   # use for YoY comparison
    adf.rename(columns={'ticket_count': 'y'}, inplace=True)
    p_ut.save_df(adf, '~/my_tmp/a_df')

    # current forecast
    s_ut.my_print('current forecast')
    cf_df = pd.concat([get_fcast(ts_name, ts_cfg, cutoff_date, bu, use_cache=False) for bu in ts_cfg['business_units']], axis=0)
    cf_df = cf_df[cf_df['ds'] > cutoff_date].copy()
    p_ut.save_df(cf_df, '~/my_tmp/cf_df')

    # target year summary
    xl_ut.year_summary(adf, cf_df, target_year, cutoff_date)

    # get 90 day old forecast (last Sat of a month)
    s_ut.my_print('3 months old forecast')
    date = pd.to_datetime(str(cutoff_date.year) + '-' + str(cutoff_date.month) + '-01')
    cu90 = tm_ut.last_saturday_month(date - pd.to_timedelta(3, unit='M'))
    f90_df = pd.concat([get_fcast(ts_name, ts_cfg, cu90, bu, use_cache=False) for bu in ts_cfg['business_units']], axis=0)

    # get prev old forecast (last Sat of a month)
    s_ut.my_print('past forecast forecast')
    date = pd.to_datetime(str(cutoff_date.year) + '-' + str(cutoff_date.month) + '-01')
    prev_cu = tm_ut.last_saturday_month(date)
    pf_df = pd.concat([get_fcast(ts_name, ts_cfg, prev_cu, bu, use_cache=False) for bu in ts_cfg['business_units']], axis=0)
    p_ut.save_df(cf_df, '~/my_tmp/cf_df')

    # get 90 day errors
    m90 = (cu90 + pd.to_timedelta(2, 'W')).month_name()
    l_df, ls_df = errs.fcast_errs(adf, f90_df, cutoff_date, m90, periods=4)

    # ############################### excel tabs ##############################
    # excel writer
    dt = cutoff_date + pd.to_timedelta(2, unit='W')
    mm = dt.month_name()
    yy = dt.year
    fx = os.path.expanduser('~/my_tmp/xlsx/ticket_forecast_' + mm[:3] + '_' + str(yy) + '_test.xlsx')
    s_ut.my_print('xl file: ' + fx)
    xlwriter = pd.ExcelWriter(fx, engine='xlsxwriter')

    # fcast delta tab
    s_ut.my_print('Delta Tabs')
    dr = pd.date_range(end=cutoff_date + pd.to_timedelta(12, unit='W'), freq='W', periods=4)
    xl_ut.fcast_delta(xlwriter, cf_df, pf_df, dr, cutoff_date, prev_cu, 'Forecast Delta (All)')
    for bu in ts_cfg['business_units']:
        c_fcast = cf_df[cf_df['business_unit'] == bu].copy()
        p_fcast = pf_df[pf_df['business_unit'] == bu].copy()
        xl_ut.fcast_delta(xlwriter, c_fcast, p_fcast, dr, cutoff_date, prev_cu, 'Forecast Delta (' + bu + ')')

    # target year totals by language-tier-channel tab
    s_ut.my_print('Totals Tabs')
    for bu in [None] + ts_cfg['business_units']:
        xl_ut.fcast_totals(xlwriter, cf_df, adf, cutoff_date, target_year, 'Totals', bu=bu)

    # language errors tab
    s_ut.my_print('Language Accuracy Tab')
    if l_df is not None:
        l_df.fillna(0, inplace=True)
        l_df.to_excel(xlwriter, 'Language Accuracy', index=False)
        workbook = xlwriter.book
        bold = workbook.add_format({'bold': True})
        worksheet = xlwriter.sheets['Language Accuracy']
        worksheet.set_column('A:ZZ', 20)
        worksheet.set_row(0, None, bold)
        format_key = workbook.add_format({'bg_color': '#E06666', 'font_color': '#660000'})
        worksheet.conditional_format(0, 0, 0, len(l_df.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    else:
        s_ut.my_print('No data for ' + 'language accuracy')

    # tier errors tab
    s_ut.my_print('Tier Accuracy Tab')
    if ls_df is not None:
        ls_df.fillna(0, inplace=True)
        ls_df.to_excel(xlwriter, 'Tier Accuracy', index=False)
        workbook = xlwriter.book
        bold = workbook.add_format({'bold': True})
        worksheet = xlwriter.sheets['Tier Accuracy']
        worksheet.set_column('A:ZZ', 20)
        worksheet.set_row(0, None, bold)
        format_key = workbook.add_format({'bg_color': '#E06666', 'font_color': '#660000'})
        worksheet.conditional_format(0, 0, 0, len(ls_df.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    else:
        s_ut.my_print('No data for ' + 'tier accuracy')

    # channel mix tab???

    # actuals tab???

    # svc tier tabs
    adf.rename(columns={'y': 'yhat'}, inplace=True)
    z = adf.groupby(['ds', 'service_tier']).sum().reset_index()
    zz = z[z['ds'] >= '2020-02-01']   # covid

    cols = ['ds', 'business_unit', 'channel', 'language', 'service_tier', 'yhat']
    df = pd.concat([adf[cols], cf_df[cols]], axis=0)
    p_ut.save_df(adf, '~/my_tmp/adf')
    p_ut.save_df(cf_df, '~/my_tmp/cf_df')
    counts_dict = xl_ut.rs_to_excel(df, 'ds', 'yhat')
    a_dr = [str(x.date()) for x in pd.date_range(start=df['ds'].min(), end=cutoff_date, freq='W')]
    for ky, sht in counts_dict.items():
        tab_name = ky.replace('/', '-')
        s_ut.my_print(tab_name + ' Tab')
        sht.fillna(0, inplace=True)
        sht = sht.applymap(lambda x: np.round(x, 0) if isinstance(x, (float, np.float64, int, np.int64)) else x)
        sht.to_excel(xlwriter, tab_name, index=False)
        xl_ut.set_colors(xlwriter, tab_name, sht, a_dr)
        workbook = xlwriter.book
        bold = workbook.add_format({'bold': True})
        worksheet = xlwriter.sheets[tab_name]
        worksheet.set_column('B:ZZ', 20, bold)
        worksheet.set_column('A:A', 20, bold)
        # p_ut.save_df(sht, '~/my_tmp/sht_' + ky)

    s_ut.my_print('saving xls forecast to ' + fx)
    xlwriter.save()


