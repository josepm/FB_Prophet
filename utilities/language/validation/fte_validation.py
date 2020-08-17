"""
$python cutoff_date ts_name

ts_name: phone-inbound, phone-outbound

"""

import os
import sys
import pandas as pd
import numpy as np
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.forecast.utilities.language import data_processing as dtp


with s_ut.suppress_stdout_stderr():
    import airpy as ap


def to_float(x):
    try:
        return float(x)
    except ValueError:
        try:
            return float(x.replace(',', ''))
        except ValueError:
            print(x)
            return x


def get_cx_type(ts_):
    if ts_ == 'phone-inbound':
        return ['IB FTE', 'Escalation FTE']
    elif ts_ == 'phone-outbound':
        return ['OB FTE']
    else:
        print('invalid ts_name: ' + str(ts_))


if __name__ == '__main__':
    # ############################################
    # parameters
    horizon = 112                # fcast horizon
    # cxfname = '~/my_tmp/cx_data/cx_IB&OB_PHONE_2019-09-28.tsv'
    # cxfname = '~/my_tmp/cx_data/cx_IB&OB_PHONE_2019-07-27.tsv'
    # ############################################

    print(sys.argv)
    cutoff_date, ts_name = sys.argv[1:]

    # CX FTEs
    cxfname = os.path.expanduser('~/my_tmp/cx_data/cx_Capacity_' + cutoff_date + '.tsv')
    try:
        dfx = pd.read_csv(cxfname, sep='\t')
    except FileNotFoundError:
        print(str(cxfname) + ' does not exist')
        sys.exit()

    dfx.columns = [str(c) for c in dfx.columns]
    if 'Week Start' not in dfx.columns:
        print('Week Start column missing. Are date week endings?')
        sys.exit()
    to_drop = [c for c in dfx.columns if 'Unnamed:' in c or 'nan' == c or c == 'Week Start']
    dfx.drop(to_drop, axis=1, inplace=True)
    cx_type = get_cx_type(ts_name)
    dfx = dfx[dfx['Type'].isin(cx_type) & (dfx['Channel'] == 'Phone')].copy()

    dfx.rename(columns={'Type': 'type', 'Channel': 'channel', 'Language': 'language', 'Service': 'sector'}, inplace=True)
    l_cols = ['type', 'sector', 'channel', 'language']
    d_cols = [pd.to_datetime('20' + c.split('/')[2] + '-' + c.split('/')[0] + '-' + c.split('/')[1]) for c in dfx.columns if c not in l_cols]
    init = pd.to_datetime(min(d_cols))
    dr = [str((x + pd.to_timedelta(6, unit='D')).date()) for x in pd.date_range(start=min(d_cols), periods=len(dfx.columns) - len(l_cols), freq='7D')]

    dfx.columns = l_cols + dr
    dfm = pd.melt(dfx, value_vars=dr, id_vars=l_cols, var_name='ds_week_ending', value_name='cx_yhat')
    start = pd.to_datetime(cutoff_date) + pd.to_timedelta(7, unit='D')  # cutoff is a week_ending date. Go to the next week
    dfm['ds_week_ending'] = pd.to_datetime(dfm['ds_week_ending'].values)
    dfm = dfm[(dfm['ds_week_ending'] >= start)].copy()

    _ = p_ut.clean_cols(dfm, ['sector', 'language'], '~/my_repos/capacity_planning/data/config/col_values.json', check_new=False)
    dfm['cx_yhat'] = dfm['cx_yhat'].apply(lambda x: to_float(x))
    p_df = pd.pivot_table(dfm, index=['ds_week_ending', 'language', 'sector'], values='cx_yhat', columns=['type']).reset_index()
    cx_df = p_df[p_df['sector'].isin(['Claims', 'Community Education', 'Experiences', 'PST', 'Payments',
                                     'Regulatory Response', 'Resolutions 1', 'Resolutions 2', 'Safety'])].copy()
    cx_df.fillna(0, inplace=True)
    cx_df['cx_FTEs'] = cx_df[cx_type].sum(axis=1)
    cx_df.drop(cx_type, axis=1, inplace=True)

    # DS FTEs
    qry = 'select * from sup.fct_agent_forecasts where cutoff = \'' + cutoff_date + '\' and ts_name = \'' + ts_name + '\';'
    ds_df = ap.presto.query(qry, use_cache=False)
    ds_df = ds_df[['ds_week_starting', 'language', 'tier', 'agents']].copy()
    ds_df['ds_week_starting'] = pd.to_datetime(ds_df['ds_week_starting'].values)
    ds_df['ds_week_ending'] = ds_df['ds_week_starting'] + pd.to_timedelta(6, unit='D')
    ds_df.drop('ds_week_starting', axis=1, inplace=True)
    ds_df.rename(columns={'agents': 'ds_FTEs', 'tier': 'sector'}, inplace=True)
    ds_df['ds_FTEs'].fillna(0, inplace=True)

    all_df = cx_df.merge(ds_df, on=['ds_week_ending', 'language', 'sector'], how='left')
    all_df['delta'] = 2.0 * np.abs(all_df['cx_FTEs'] - all_df['ds_FTEs']) / (all_df['cx_FTEs'] + all_df['ds_FTEs'])
    all_df['delta'].fillna(0, inplace=True)
    all_df['x'] = np.abs(all_df['cx_FTEs'] - all_df['ds_FTEs'])



