"""
fcast validation: get fcast error from cx data
errors from ds data obtained in ens_cfg

$ python cx_fcast.py cutoff_date ts_name
- input file in ~/my_tmp/cx_data/cx_IB&OB_PHONE_<cutoff_date>.tsv
- input file top rows and dates fixed in xls or similar
- ts_name = phone-inbound-vol, phone-inbound-aht, phone-inbound-vol, phone-outbound-aht
set parameters in code:
    cutoff_date = '2019-09-28'   the date we use as last actuals available
    horizon_date = '2019-11-25'  how far we compare the forecasts
    modify the list of n_cfg and min_cnt values if needed

MUST FIX DATE VALUES AND FORMAT IN TSV FILE

trick to get the top cfg by language: z = df.groupby('language').apply(pd.DataFrame.nsmallest, n=1, columns=['ds_err']).reset_index(drop=True)
"""
import os
import sys
import pandas as pd
import numpy as np
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.forecast.utilities.language import data_processing as dtp


def set_demand(t_df, abn_par, name):
    if name == 'phone-inbound-vol':
        # :::::::::::: adjust abandonments with abandonment retries :::::::::::::: #
        # pABN = a / (A + a): observed abandonment prob
        # A: accepted calls
        # a: observed (non-deduped) abandonments
        # r = number of retries per abandonment
        # b = unique (de-duplicated) abandonments: b * (1 + r) = a
        # actual demand D = A + b = A + a / (1 + r)
        # retries model r = r0 * pABN / (1 - pABN) because retries grow with pABN
        # r0 = 10.0  because at pABN ~ 5%, avg retries are about 0.5, ie r0 = 10
        # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
        pABN = t_df['abandons'] / (t_df['accepted'] + t_df['abandons'])
        retries = abn_par * pABN / (1.0 - pABN)
        t_df['y'] = t_df['accepted'] + t_df['abandons'] / (1 + retries)
        ctype = 'IB Offered'
    elif name == 'phone-outbound-vol':
        # t_df = t_df[t_df['interaction_type'] == 'outbound'].copy()
        t_df.rename(columns={'calls': 'y'}, inplace=True)
        ctype = 'OB Offered'
    elif name == 'phone-inbound-aht':
        # t_df = t_df[t_df['interaction_type'] == 'inbound'].copy()
        t_df = t_df[t_df['calls'] > 0].copy()
        ttl_mins = t_df['agent_mins']      # + t_df['agent_rcv_consult_mins'] + t_df['agent_init_consult_mins']
        t_df['y'] = ttl_mins / t_df['calls']
        ctype = 'IB AHT'
    elif name == 'phone-outbound-aht':
        t_df = t_df[t_df['calls'] > 0].copy()
        ttl_mins = t_df['agent_mins']      # + t_df['agent_rcv_consult_mins'] + t_df['agent_init_consult_mins']
        t_df['y'] = ttl_mins / t_df['calls']
        ctype = 'OB AHT'
    elif name == 'deferred':
        ctype = None
    elif name == 'deferred_hbat':
        ctype = None
    else:
        s_ut.my_print('unknown ts_name: ' + str(name))
        sys.exit(0)
    return t_df.copy(), ctype


def err(l_df, ds_cols, wcol):
    e_dict = dict()
    e_dict['w'] = [l_df[wcol].mean()]
    for c in ds_cols:
        ds = 2.0 * np.abs(l_df[c] - l_df['y']) / (l_df[c] + l_df['y'])
        key = c.replace('yhat', 'err')
        e_dict[key] = [ds.mean()]
    return pd.DataFrame(e_dict)


def func_err(a_df_, ts):
    a_df = a_df_.copy()
    if 'aht' in ts:
        ds = a_df['ds_err'].mean()
    else:
        ds = (a_df['w'] * a_df['ds_err']).sum() / a_df['w'].sum()
    return pd.DataFrame({'ds_err': [ds]})


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
    if ts_ == 'phone-inbound-vol':
        return 'IB Offered'
    elif ts_ == 'phone-outbound-vol':
        return 'OB Offered'
    elif ts_ == 'phone-inbound-aht':
        return 'IB AHT'
    elif ts_ == 'phone-outbound-aht':
        return 'OB AHT'
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

    if ts_name == 'phone-inbound-vol':
        fname = dtp.get_data_file('~/my_tmp/cleaned/phone-vol_cleaned_', cutoff_date)
        interaction_type = 'inbound'
    else:
        fname = dtp.get_data_file('~/my_tmp/cleaned/phone-aht_cleaned_', cutoff_date)
        interaction_type = 'inbound' if 'inbound' in ts_name else 'outbound'

    # actuals
    s_ut.my_print('actuals file: ' + str(fname))
    q_df = pd.read_parquet(fname)

    # week_starting patch
    df_cols_ = q_df.columns
    if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
        q_df['ds_week_ending'] = pd.to_datetime(q_df['ds_week_ending'])
        q_df['ds_week_starting'] = q_df['ds_week_ending'] - pd.to_timedelta(6, unit='D')

    q_df['ds'] = pd.to_datetime(q_df['ds'].values)

    # use week starting
    if ts_name == 'phone-inbound-vol':
        q_df = q_df.groupby([pd.Grouper(key='ds', freq='W-SUN'), 'language']).agg({'offered': np.sum, 'accepted': np.sum, 'abandons': np.sum}).reset_index()
    else:
        q_df = q_df[q_df['interaction_type'] == interaction_type].copy()
        q_df = q_df.groupby([pd.Grouper(key='ds', freq='W-SUN'), 'language']).agg({'calls': np.sum, 'agent_mins': np.sum}).reset_index()

    m_df, ctype = set_demand(q_df, 10, ts_name)
    a_df = m_df.copy()
    horizon_date = min(pd.to_datetime(cutoff_date) + pd.to_timedelta(horizon, unit='D'), a_df['ds'].max())
    a_df['ds_week_ending'] = a_df['ds'] + pd.to_timedelta(6, unit='D')  # switch to week ending so that we do not have incomplete weeks at end
    a_df = a_df[(a_df['ds_week_ending'] <= horizon_date) & (a_df['ds_week_ending'] > cutoff_date)].copy()
    a_df.drop('ds', axis=1, inplace=True)
    w_col = 'y' if 'vol' in ts_name else 'calls'

    # CX forecasts
    cxfname = '~/my_tmp/cx_data/cx_IB&OB_PHONE_' + cutoff_date + '.tsv'
    try:
        dfx = pd.read_csv(cxfname, sep='\t')
    except FileNotFoundError:
        print(str(cxfname) + ' does not exist')
        sys.exit()

    # week_starting patch
    df_cols_ = dfx.columns
    if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
        dfx['ds_week_ending'] = pd.to_datetime(dfx['ds_week_ending'])
        dfx['ds_week_starting'] = dfx['ds_week_ending'] - pd.to_timedelta(6, unit='D')

    dfx.columns = [str(c) for c in dfx.columns]
    if 'Week Start' not in dfx.columns:
        print('Week Start column missing. Are date week endings?')
        sys.exit()
    to_drop = [c for c in dfx.columns if 'Unnamed:' in c or 'nan' == c or c == 'Week Start']
    dfx.drop(to_drop, axis=1, inplace=True)
    dfx = dfx[dfx['Type'].isin(['IB Offered', 'OB Offered', 'OB Volumne', 'IB FTE', 'OB FTE', 'Escalation FTE', 'IB AHT', 'OB AHT']) & (dfx['Channel'] == 'Phone')].copy()

    dfx.rename(columns={'Type': 'type', 'Channel': 'channel', 'Language': 'language', 'Service': 'sector'}, inplace=True)
    l_cols = ['type', 'sector', 'channel', 'language']
    d_cols = [pd.to_datetime('20' + c.split('/')[2] + '-' + c.split('/')[0] + '-' + c.split('/')[1]) for c in dfx.columns if c not in l_cols]
    init = pd.to_datetime(min(d_cols))
    dr = [str((x + pd.to_timedelta(6, unit='D')).date()) for x in pd.date_range(start=min(d_cols), periods=len(dfx.columns) - len(l_cols), freq='7D')]

    dfx.columns = l_cols + dr
    dfm = pd.melt(dfx, value_vars=dr, id_vars=l_cols, var_name='ds_week_ending', value_name='cx_yhat')
    start = pd.to_datetime(cutoff_date) + pd.to_timedelta(7, unit='D')  # cutoff is a week_ending date. Go to the next week
    dfm['ds_week_ending'] = pd.to_datetime(dfm['ds_week_ending'].values)
    dfm = dfm[(dfm['ds_week_ending'] >= start) & (dfm['ds_week_ending'] <= horizon_date)].copy()

    _ = p_ut.clean_cols(dfm, ['sector', 'language'], '~/my_repos/capacity_planning/data/config/col_values.json', check_new=False)
    dfm['cx_yhat'] = dfm['cx_yhat'].apply(lambda x: to_float(x))
    p_df = pd.pivot_table(dfm, index=['ds_week_ending', 'language', 'sector'], values='cx_yhat', columns=['type']).reset_index()
    c_df = p_df[p_df['sector'].isin(['Claims', 'Community Education', 'Experiences', 'PST', 'Payments',
                                     'Regulatory Response', 'Resolutions 1', 'Resolutions 2', 'Safety'])].copy()
    c_df.fillna(0, inplace=True)

    # language level: no language level agg for FTE
    if 'inbound-vol' in ts_name:
        g_df = c_df.groupby(['ds_week_ending', 'language']).agg({'IB Offered': np.sum}).reset_index()
        g_df.rename(columns={'IB Offered': 'cx_yhat'}, inplace=True)
    elif 'outbound-vol' in ts_name:
        g_df = c_df.groupby(['ds_week_ending', 'language']).agg({'OB Offered': np.sum}).reset_index()
        g_df.rename(columns={'OB Offered': 'cx_yhat'}, inplace=True)
    elif 'aht' in ts_name:  # AHT
        c_df['w_ib_aht'] = c_df['IB AHT'] * c_df['IB Offered']
        c_df['w_ob_aht'] = c_df['OB AHT'] * c_df['OB Offered']
        g_df = c_df.groupby(['ds_week_ending', 'language']).agg({'w_ib_aht': np.sum, 'IB Offered': np.sum, 'w_ob_aht': np.sum, 'OB Offered': np.sum}).reset_index()
        g_df['cx_yhat'] = g_df['w_ib_aht'] / g_df['IB Offered'] if 'inbound' in ts_name else g_df['w_ob_aht'] / g_df['OB Offered']
    else:
        print('invalid ts_name: ' + str(ts_name))

    # language level errors weekly
    wf = g_df.merge(a_df, on=['ds_week_ending', 'language'], how='inner')
    wf['cx_err'] = 2.0 * np.abs(wf['cx_yhat'] - wf['y']) / (wf['cx_yhat'] + wf['y'])
    wf['ts_name'] = ts_name
    wf['cutoff_date'] = cutoff_date
    wf['horizon_date'] = str(horizon_date.date())
    p_ut.save_df(wf, '~/my_tmp/cx_fcast_weekly_err_' + ts_name + '_' + cutoff_date)

    # summary by language
    wf['x'] = wf['cx_err'] * wf[w_col]
    lf = wf.groupby('language').agg({'x': np.sum, w_col: np.sum}).reset_index()
    lf['cx_err'] = lf['x'] / lf[w_col]
    all_err = (lf[w_col] * lf['cx_err']).sum() / lf[w_col].sum()
    adf = pd.DataFrame({'language': ['All'], 'cx_err': [all_err]})
    lf = lf[['language', 'cx_err']].copy()
    fdf = pd.concat([lf, adf], axis=0)
    fdf['ts_name'] = ts_name
    fdf['cutoff_date'] = cutoff_date
    fdf['horizon_date'] = str(horizon_date.date())
    p_ut.save_df(fdf, '~/my_tmp/cx_fcast_err_' + ts_name + '_' + cutoff_date)
    print('horizon date: ' + str(horizon_date))
    print(fdf)

    print('DONE')

