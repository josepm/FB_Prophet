"""

"""
import sys
import os

import itertools
import pandas as pd
import numpy as np
import json
import platform

from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import stats_utils as st_ut
# import sklearn.linear_model as l_mdl
from capacity_planning.data import hql_exec as hql
from capacity_planning.forecast.ticket_forecast import excel_utils as xl_ut
from capacity_planning.forecast.utilities.language import data_processing as dp

with s_ut.suppress_stdout_stderr():
    import airpy as ap
USE_CACHE = True if platform.system() == 'Darwin' else False
USE_CACHE = False
RENEW = not USE_CACHE


def get_year_ticket(yyyy, cutoff_date, ts_name):
    start, end = xl_ut.iso_dates(yyyy)
    cutoff_date = pd.to_datetime(cutoff_date)
    ts_cfg, _ = dp.ts_setup(ts_name, cutoff_date, pd.to_datetime('2016-01-01'), 'W')
    a_df = dp.ts_actuals(ts_name, ts_cfg, ['language', 'business_unit', 'channel', 'service_tier'], drop_cols=False)
    a_df['ts_name'] = ts_name
    f_df = ap.hive.query('select * from sup.cx_weekly_forecasts where ts_name = \'' + ts_name + '\' and cutoff =\'' + str(cutoff_date.date()) + '\';')
    f_df.columns = [c.split('.')[1] for c in f_df.columns]
    a_df['ds'] = pd.to_datetime(a_df['ds'].values)
    a_df = a_df[(a_df['ds'] >= start) & (a_df['ds'] <= cutoff_date)].copy()
    f_df['ds'] = pd.to_datetime(f_df['ds'].values)
    f_df = f_df[f_df['ds'] <= end].copy()
    f_df.rename(columns={'yhat': ts_name}, inplace=True)
    return pd.concat([a_df, f_df], axis=0)


def to_days(a_df, start=True, t_col='ds', y_col='y'):
    # approximates a weekly DF to daily
    def _to_days(row, _list, _t_col, _y_col, _cols, _start):
        r_dict = {c: [row[c]] * 7 for c in _cols}
        r_dict[_t_col] = pd.date_range(start=row['ds'], periods=7, freq='D') if _start is True else pd.date_range(end=row['ds'], periods=7, freq='D')
        r_dict[_y_col] = [row[_y_col] / 7] * 7
        f = pd.DataFrame(r_dict)
        _list.append(f)
        return row
    df_list = list()
    cols = [c for c in a_df.columns if c not in [t_col, y_col]]
    _ = a_df.apply(_to_days, _list=df_list, _start=start, _t_col=t_col, _y_col=y_col, _cols=cols, axis=1)
    return pd.concat(df_list, axis=0)


def to_monthly(a_df, t_col, y_col):
    gcols = [c for c in a_df.columns if c not in [t_col, y_col]]
    return a_df.groupby(gcols + [pd.Grouper(key='ds', freq='MS')]).sum().reset_index()


def get_actuals(ts_dict, gcols, use_cache=None):          # actuals with a max ds >= cutoff_date
    cutoff_date = ts_dict['cutoff_date']
    init_date = ts_dict['init_date']
    ts_name = ts_dict['name']
    ycol = ts_dict['ycol']
    s_ut.my_print('getting ' + ts_name + ' actuals from table')
    r_date = hql.get_rmax(ycol, use_cache=USE_CACHE)
    qcols = list(set(['ds', 'language', 'y'] + gcols))
    col_str = ','.join(qcols)
    print('rmax: ' + str(r_date))
    qry = 'select ' + col_str + ' from sup.cx_weekly_actuals where ts_name=\'' + ycol + '\' and run_date=\'' + r_date + '\';'
    try:
        uc = USE_CACHE if use_cache is None else use_cache
        df = hql.from_tble(qry, ['ds'], use_cache=uc, renew=RENEW)
        s_ut.my_print(qry + ' completed. Got ' + str(len(df)) + ' rows')
    except:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: query: ' + qry + ' failed. No data for ts ' + ts_name)
        sys.exit()
    s_ut.my_print('pid: ' + str(os.getpid()) + ' got actuals for ' + ts_name + ' from table ' + 'sup.cx_weekly_actuals')
    df.rename(columns={'y': ycol}, inplace=True)    # unique name needed (may mix with regressors later)
    df = df[(df['ds'] >= init_date)].copy()
    if df['ds'].max() < cutoff_date:
        s_ut.my_print('ERROR: no actuals up to cutoff date for ' + ts_name)
        return None
    else:
        if len(df) > 0:
            return df.groupby(['ds'] + gcols).sum().reset_index()
        else:
            return None


# def tier_split(a_df, f_df, b_cols, cutoff_date, time_scale, inc_start, inc_end=0):
#     def _tier_fcast(row, b_f_, b_list_):
#         b_df = pd.DataFrame(columns=row.index, index=range(len(b_f_)))
#         for ix in row.index:
#             b_df[ix] = row[ix]
#         zx = pd.concat([b_df, b_f_], axis=1)
#         zx['w'] = zx['ticket_count'].values
#         zx['ticket_count'] = np.round(zx['w'] * zx['yhat'], 0)
#         b_list_.append(zx)
#         return row
#
#     t_start = cutoff_date - pd.to_timedelta(inc_start, unit=time_scale)
#     t_end = cutoff_date - pd.to_timedelta(inc_end, unit=time_scale)
#     b_df = a_df[(a_df['ds'] >= t_start) & (a_df['ds'] <= t_end)].copy()
#     bg_df = b_df.groupby(b_cols).agg({'ticket_count': np.sum})
#     bg_df['ticket_count'] /= bg_df['ticket_count'].sum()
#     bg_df.reset_index(inplace=True)
#     b_list = list()
#     _ = f_df.apply(_tier_fcast, b_f_=bg_df, b_list_=b_list, axis=1)
#     tier_f = pd.concat(b_list, axis=0)
#     return tier_f


# def set_cfg_idx(df):
#     cfg_cols = ['growth', 'y_mode', 'w_mode', 'r_mode', 'xform', 'h_mode', 'training', 'do_res', 'changepoint_range']
#     df.fillna('None', inplace=True)
#     df['cfg_str'] = df.apply(lambda x: json.dumps(x[cfg_cols].to_dict()), axis=1)
#     z = df['cfg_str'].drop_duplicates()
#     zf = pd.DataFrame(z)
#     zf.reset_index(inplace=True, drop=True)
#     zf.reset_index(inplace=True)
#     zf.columns = ['cfg_idx', 'cfg_str']
#     return df.merge(zf, on=['cfg_str'], how='left')


# def lasso_selection(gbk, gf, af, ff, ycol, cols, normalize=True):
#     s_ut.my_print('lasso selection for: ' + str(gbk))
#     cfg_list = gf['cfg_idx'].unique()
#     pcols = [c for c in gf.columns if c not in ['cfg_idx', ycol]]
#     pf = pd.pivot_table(gf, index=pcols, columns='cfg_idx', values=ycol).reset_index()
#     pf.dropna(inplace=True)  # ensure all fcast cfgs have the same time range
#     df = pf.merge(af, on='ds', how='inner')
#     pars = len(cfg_list)     # regression parameters: one per fcast_cfg plus 3 extra pars: constant + 2 regularization
#     ndata = len(df)          # data points
#     while ndata < pars:      # more regression unknowns than data
#         s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: not enough data for cfg_list: ' + str(cfg_list))
#         cfg_list = cfg_list[:int(0.75 * ndata)]   # pick the top one by score
#         pars = len(cfg_list)
#     if len(cfg_list) == 0:
#         s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: not enough data for cfg_list: ' + str(cfg_list))
#         return None
#
#     X_train = df[cfg_list].values
#     y_train = df[ycol].values
#     lasso_mdl = l_mdl.LassoLarsIC(criterion='aic', normalize=normalize)
#     lasso_mdl.fit(X_train, y_train)
#     m_pars = lasso_mdl.coef_              # pars excluding intercept
#     new_cfg_list = [int(cfg_list[i]) for i in range(len(cfg_list)) if m_pars[i] != 0.0]
#
#     f_pf = pd.pivot_table(ff, index=pcols, columns='cfg_idx', values=ycol).reset_index()
#     if len(new_cfg_list) > 0:
#         X_test = f_pf[cfg_list].values
#         y_test = lasso_mdl.predict(X_test)
#         f_pred = pd.DataFrame({'ds': f_pf['ds'], 'y_pred': y_test})
#         for ix in range(len(cols)):
#             f_pred[cols[ix]] = gbk[ix]
#         return {'cfg_idx': new_cfg_list, 'alpha': lasso_mdl.alpha_, 'pars': m_pars, 'res': f_pred}
#     else:
#         f_pred = pd.DataFrame({'ds': f_pf['ds'], 'y_pred': [np.nan] * len(f_pf)})
#         for ix in range(len(cols)):
#             f_pred[cols[ix]] = gbk[ix]
#         return {'cfg_idx': new_cfg_list, 'alpha': np.nan, 'pars': list(), 'res': f_pred}
#

# def g_err(ef):
#     y = ef['ticket_count'].values
#     yhat = ef['yhat'].values
#     e_arr = ['sMAPE', 'MAPE', 'mMAPE', 'wMAPE', 'LAR', 'MASE', 'RMSE']  # , 'YJ', 'YJd', 'BC', 'SR']
#     d_err = {etype: st_ut.err_func(y, yhat, etype) for etype in e_arr}
#     return pd.DataFrame([d_err])

