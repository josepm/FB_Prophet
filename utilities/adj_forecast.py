"""
$ python adj_forecast.py ts_name cutoff adj_name a_df
may runs before ratio_forecast (language level adjustments) or after ratios (service tier adj)
"""

import os
import platform
import json

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
import pandas as pd
import numpy as np

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut


def fcast_adj(k_col, adjust_date):
    # k_col: key columns to be adjusted
    # forecast adjuster (engineering, ECs, ...)
    # ds is the start date of an adj factor for a given k_col and ts_name
    # at the last adj time all adj factors must be 1, otherwise we overwrite
    # All missing adj factors fill to 1
    # All missing k_col values at a given ds get an adj factor or 1
    if adjust_date is None:
        return list()
    else:
        data_cfg = os.path.expanduser('~/my_repos/capacity_planning/forecast/config/fcast_adjust_' + str(adjust_date.date()) + '.json')
        if os.path.isfile(data_cfg):
            with open(data_cfg, 'r') as fptr:
                adj_dict = json.load(fptr)
        else:
            s_ut.my_print('>>>>>>>>>>>>>>>> WARNING: could not find adjustments file ' + data_cfg + '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            return list()

    # find dicts in adj_dict that contain k_col
    adj_names = list()
    for k, vlist in adj_dict.items():
        for v in vlist:
            if k_col in v.keys():
                adj_names.append(k)
                break

    # check adj cfg
    adj_df_list = list()
    for adj_name in adj_names:   # list of dicts in adj_dict that contain k_col
        adj_list = adj_dict[adj_name]
        adj_df_ = pd.DataFrame(adj_list)
        adj_df_['ds'] = pd.to_datetime(adj_df_['ds'].values)

        # add missing entries a ds
        ts_cols = [c for c in adj_df_.columns if c not in ['ds', k_col]]
        f_list = [adj_df_]
        v_list = adj_df_[k_col].unique()
        for ds, f in adj_df_.groupby('ds'):
            m_vals = list(set(v_list) - set(f[k_col].unique()))
            if len(m_vals) > 0:
                lf = pd.DataFrame({'ds': [ds] * len(m_vals), k_col: m_vals})
                f_list.append(lf)
        adj_df = pd.concat(f_list, axis=0, sort=True)
        adj_df[ts_cols].fillna(1, inplace=True)   # default is not adjust

        # check latest date has all 1's
        b = adj_df['ds'] == adj_df['ds'].max()
        d_max = adj_df[b].copy()
        if d_max[ts_cols].min().min() != 1 or d_max[ts_cols].max().max() != 1:
            s_ut.my_print('WARNING: last adjust values not 1. Resetting')
            d_max[ts_cols] = 1
            adj_df = pd.concat([adj_df[~b].copy(), d_max], axis=0)
        fa = adj_df.reset_index(drop=True)
        fa.fillna(1.0, inplace=True)
        adj_df_list.append(fa)
    return adj_df_list


def merge_adj(df, adj_df, k_col, ts_cols, master_ts):  # keeps coherence at master_ts
    mf = pd.concat([set_factor(ds, kval, k_col, adj_df, ts_cols) for (ds, kval), f in df.groupby(['ds', k_col])], axis=0)
    p_ut.save_df(mf, '~/my_tmp/mf')
    fx = df.merge(mf, on=['ds', k_col], how='left')
    gx = fx.apply(row_adjust, ts_list=ts_cols, axis=1)
    gx[master_ts + '_tilde'] = gx[[c + '_tilde' for c in ts_cols]].sum(axis=1)
    return gx.copy()


def row_adjust(a_row, ts_list):
    for ts in ts_list:
        a_row[ts + '_tilde'] *= a_row[ts]
    return a_row


def set_factor(ds, kval, k, adj_df, ts_cols):
    lf = adj_df[adj_df[k] == kval]
    mf = lf[lf['ds'] <= ds].copy()
    nf = mf[mf['ds'] == mf['ds'].max()]
    if len(nf) == 0:
        dout = {c: [1.0] for c in ts_cols}
        dout['ds'] = [ds]
        dout[k] = [kval]
        return pd.DataFrame(dout)
    else:
        nf['ds'] = ds
        return nf


# def hts_adj(row, top_col, hat_cols):
#     m = row[top_col] / row[hat_cols].sum()
#     for c in hat_cols:
#         row[c] = int(m * row[c])
#     return row

def main(in_df, k_col, ts_list, top_ts, adjust_date):
    if k_col in ts_list:
        s_ut.my_print('ERROR: invalid k_col: ' + str(k_col) + ' cannot one of ' + str(ts_list) + ' has to be a DF column')
        sys.exit()
    adj_df_list = fcast_adj(k_col, adjust_date)
    if len(adj_df_list) > 0:
        fout = in_df.copy()
        for adj_df in adj_df_list:
            fout = merge_adj(fout, adj_df, k_col, ts_list, top_ts)
        p_ut.save_df(fout, '~/my_tmp/fout_' + k_col)
        return fout
    else:
        s_ut.my_print('WARNING: nothing to adjust for ' + k_col + ' <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        return in_df



