"""

"""

import os
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
import pandas as pd
import numpy as np
from functools import reduce
import itertools

import sklearn.linear_model as l_mdl
# import xgboost as xgb

from capacity_planning.utilities import stats_utils as st_ut
from capacity_planning.utilities import time_utils as t_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.data import hql_exec as hql
from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.forecast.utilities.language import data_processing as dtp


pd.set_option('display.max_rows', 100)
pd.set_option('precision', 4)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_colwidth', 400)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
USE_CACHE = True if platform.system() == 'Darwin' else False
RENEW = not USE_CACHE
USE_CACHE = True


def get_lang_fcast(ts_cfg, cutoff_date, eq=True):
    ts_name = ts_cfg['name']
    ts = ts_cfg['ts_key']
    tble = 'sup.cx_language_forecast'
    s_ut.my_print('getting ' + ts + ' language forecast from table ' + tble + ' and cutoff date: ' + str(cutoff_date.date()))
    if eq is True:
        qry = 'select ds, language, yhat, dim_cfg, cutoff from ' + tble + ' where cutoff = \'' + str(cutoff_date.date()) + '\' and ts_name = \'' + ts + '\';'
    else:
        qry = 'select ds, language, yhat, dim_cfg, cutoff from ' + tble + ' where cutoff <= \'' + str(cutoff_date.date()) + '\' and ts_name = \'' + ts + '\';'
    try:
        fcast_df = hql.from_tble(qry, ['ds', 'cutoff'], use_cache=USE_CACHE, renew=RENEW)
        s_ut.my_print(qry + ' completed. Got ' + str(len(fcast_df)) + ' rows')
    except:
        s_ut.my_print('ERROR: ' + qry + ' failed')
        sys.exit()
    if fcast_df is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no data for eq: ' + str(eq) + ', ts: ' + ts_name + ' and cutoff date ' + str(cutoff_date.date()))
    return fcast_df


def get_ens_fcast(ts_name, ts_cfg, cutoff_date):
    tble = 'sup.cx_ens_forecast'
    s_ut.my_print('getting ' + ts_cfg['ts_key'] + ' ens forecast from table ' + tble + ' and cutoff date: ' + str(cutoff_date.date()))
    qry = 'select * from ' + tble + ' where cutoff = \'' + str(cutoff_date.date()) + '\' and ts_name = \'' + ts_cfg['ts_key'] + '\';'
    try:
        fcast_df = hql.from_tble(qry, ['ds', 'cutoff'], use_cache=USE_CACHE, renew=RENEW)
    except:
        s_ut.my_print('ERROR: ' + qry + ' failed')
        sys.exit()
    if fcast_df is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no data for ' + ts_name + ' and cutoff date ' + str(cutoff_date.date()))
    return fcast_df


def get_ratio_fcast(ts_name, ts_cfg,  cutoff_date, use_cache=None):  # >>>>>>>>>>>>> used in to_excel <<<<<<<<<<<<<<<<<
    tble = 'sup.cx_weekly_forecasts'
    s_ut.my_print('get_ratio_forecast: getting ' + ts_cfg['ts_key'] + ' ratio forecast from ' + tble + ' and cutoff date ' + str(cutoff_date.date()))
    qry = 'select * from ' + tble + ' where cutoff = \'' + str(cutoff_date.date()) + '\' and ts_name = \'' + ts_cfg['ts_key'] + '\';'
    try:
        uc = USE_CACHE if use_cache is None else use_cache
        fcast_df = hql.from_tble(qry, ['ds', 'cutoff'], use_cache=uc, renew=RENEW)
    except:
        s_ut.my_print('ERROR: ' + qry + ' failed')
        sys.exit()
    if fcast_df is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no data for ' + ts_name + ' and cutoff date ' + str(cutoff_date.date()))
    return fcast_df


def cfg_perf(fl, upr, lwr, cu_sun):
    fl.reset_index(inplace=True)
    lg, dim_cfg, ws = fl.loc[fl.index[0], ['language', 'dim_cfg', 'opt_shift']]
    fx = f_ok(fl, cu_sun - pd.to_timedelta(upr, unit='W'), cu_sun + pd.to_timedelta(upr, unit='W'), 'yhat')    # check fcast OK at the real cutoff date
    if fx is None:
        s_ut.my_print('WARNING: invalid cfg ' + str(dim_cfg) + ' config for cutoff ' + str(cu_sun.date()) + ' and language ' + str(lg))
        p_ut.save_df(fl, '~/my_tmp/fl_' + lg + '_' + str(dim_cfg))
        return None
    else:
        # actual performance for this cfg
        fa = fx[(fx['ds'] >= cu_sun + pd.to_timedelta(lwr, unit='W')) & (fx['ds'] <= cu_sun + pd.to_timedelta(upr, unit='W'))].copy()
        a_err = st_ut.wmape(fa[['y', 'yhat']])

        # training performance: use last year's data adjusted
        g_train = fx[(fx['ds'] >= cu_sun + pd.to_timedelta(lwr, unit='W')) & (fx['ds'] <= cu_sun + pd.to_timedelta(upr, unit='W'))].copy()
        t_err = st_ut.wmape(g_train[['adj_y_shifted', 'yhat']], y_col='adj_y_shifted') if len(g_train) > 0 else np.nan    # "training" performance for this fcast cfg
        if np.isnan(t_err):
            s_ut.my_print('WARNING: cannot get training performance of config ' + str(dim_cfg) + ' for cutoff ' + str(cu_sun.date()) + ' and language ' + str(lg))
        else:
            s_ut.my_print('OK: get performance of config ' + str(dim_cfg) + ' for cutoff ' + str(cu_sun.date()) + ' and language ' + str(lg))
        fx['train_err'] = t_err
        fx['a_err'] = a_err
        return fx


def get_shift(fl, cu_sun, upr, lwr):
    # extra shift to max correlation
    lg = fl.loc[fl.index[0], 'language']
    fl.drop('language', inplace=True, axis=1)
    zn = fl.copy()
    zn.set_index('ds', inplace=True)
    zs = zn[['y']].shift(364, freq='D')                                 # shift prior year data to cutoff date
    f = zn.merge(zs, left_index=True, right_index=True, how='left')
    f.rename(columns={'y_y': 'y_shifted_', 'y_x': 'y'}, inplace=True)
    ws_adj = 4                                                          # max adj shift
    lx = [_corr(f.copy(), cu_sun, v, upr) for v in range(-ws_adj, ws_adj + 1)]
    lx.sort(key=lambda x: x[0])
    rho_max, ws_opt, le = lx[-1]
    print('language: ' + lg + ' week shift: ' + str(ws_opt) + ' best corr: ' + str(rho_max) + ' len: ' + str(le))

    f['y_shifted'] = f['y_shifted_'].shift(ws_opt)
    f.drop('y_shifted_', axis=1, inplace=True)
    f.dropna(inplace=True)
    fx = lm_mdl(f, cu_sun, upr, lwr)
    if fx is not None:
        fx['opt_shift'] = ws_opt
        fx['corr'] = rho_max
        p_ut.save_df(fx, '~/my_tmp/fshift_' + lg)
        return fx
    else:
        return None


def _corr(f, cu, v, upr):
    f['y_shifted'] = f['y_shifted_'].shift(v)
    g = f[(f.index > cu - pd.to_timedelta(upr, unit='W')) & (f.index <= cu)].copy()
    g.dropna(inplace=True)
    if len(g) < np.floor(upr / 2):                                       # not enough data
        corr = -1.0
    else:
        corr = g[['y', 'y_shifted']].corr().loc['y', 'y_shifted']
    return corr, v, len(g)


def lm_mdl(f, cu, upr, lwr):
    g_train = f[(f.index > cu - pd.to_timedelta(upr, unit='W')) & (f.index <= cu)].copy()      # no leakage
    g_train.dropna(inplace=True)
    g_test = f[(f.index >= cu + pd.to_timedelta(lwr, unit='W')) & (f.index <= cu + pd.to_timedelta(upr, unit='W'))].copy()
    g_test.dropna(inplace=True)
    if len(g_train) == 0 or len(g_test) == 0:
        return None
    else:
        X = np.reshape(g_train['y_shifted'].values, (-1, 1))                                   # this year's y shifted to this years dates
        y = g_train['y'].values                                                                # last year y on this year time scale
        lm = l_mdl.LinearRegression(fit_intercept=False).fit(X, y)                             # get the scaling factor from last year to this year
        r2 = lm.score(X, y)
        g_test['adj_y_shifted'] = lm.predict(np.reshape(g_test['y_shifted'].values, (-1, 1)))  # scale last year's y's to this year's levels
        f['adj_y_shifted'] = lm.predict(np.reshape(f['y_shifted'].values, (-1, 1)))
        f['R2'] = r2
        return f                                                                         # regression R2 and training error


def basic_perf(ts_name, cutoff_date, upr, lwr, init_date='2016-01-01', time_scale='W'):
    ts_cfg, cols = dp.ts_setup(ts_name, cutoff_date, init_date, time_scale)

    # actuals
    actuals_df = dp.ts_actuals(ts_name, ts_cfg, cols)
    actuals_df.rename(columns={ts_cfg['ycol']: 'y'}, inplace=True)
    actuals_df.drop_duplicates(inplace=True)                          # not sure why there are dups. Table problem?

    # forecasts
    f_df = get_lang_fcast(ts_cfg, cutoff_date, eq=True)
    f_df.drop_duplicates(inplace=True)                                # not sure why there are dups. Table problem?
    f_idx = fcast_idx(f_df)
    f_df = f_df.merge(f_idx, on='dim_cfg', how='left')
    f_df.drop('dim_cfg', axis=1, inplace=True)
    f_df.rename(columns={'index': 'dim_cfg'}, inplace=True)
    df = f_df.merge(actuals_df, on=['ds', 'language'], how='left')
    df.drop('cutoff', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)                                  # not sure why there are dups
    p_ut.save_df(df, '~/my_tmp/df_all')

    # perf
    cu_sun = cutoff_date - pd.to_timedelta(6, unit='D')
    sf = df[['ds', 'language', 'y']].drop_duplicates()
    sf.dropna(inplace=True)
    f_shift = sf.groupby('language').apply(get_shift, cu_sun=cu_sun, upr=upr, lwr=lwr).reset_index()  # find best shift
    nf = df.merge(f_shift, on=['ds', 'language', 'y'], how='left')
    nf.set_index(['language', 'dim_cfg'], inplace=True)     # avoids drop of nuisance cols
    nf.dropna(inplace=True)
    nf.drop_duplicates(inplace=True)
    zperf = nf.groupby(['language', 'dim_cfg']).apply(cfg_perf, upr=upr, lwr=lwr, cu_sun=cu_sun).reset_index(drop=True)  # do not groupby df.index
    zperf.dropna(inplace=True)
    p_ut.save_df(zperf, '~/my_tmp/zperf_' + ts_name + '_' + str(cutoff_date.date()))
    nf.reset_index(inplace=True)
    p_ut.save_df(nf, '~/my_tmp/nf')
    fout = nf.merge(zperf, on=list(nf.columns), how='left')
    return actuals_df, fout


def cfg_select(ts_name, cutoff_date, upr, lwr, cm_min, init_date='2016-01-01', time_scale='W'):          # selects for each language good common cfgs
    actuals, df_all = basic_perf(ts_name, cutoff_date, upr, lwr, init_date=init_date, time_scale=time_scale)
    lang_cfg = {lg: lang_cfg_select(fl, cm_min) for lg, fl in df_all.groupby('language')}
    return lang_cfg, actuals, df_all


def lang_cfg_select(fl, cm_min):
    fx = fl[['dim_cfg', 'train_err', 'a_err']].drop_duplicates()
    my_cfg = list(fx.nsmallest(cm_min, columns=['train_err'])['dim_cfg'])
    best_cfg = list(fx.nsmallest(cm_min, columns=['a_err'])['dim_cfg'])
    return my_cfg, best_cfg


def cross_validation(ts_name, cutoff_date, upr, lwr, features, init_date='2016-01-01', time_scale='W'):
    # find best fcast cfgs for cutoff_date
    if lwr < 1:
        s_ut.my_print('ERROR: invalid lwr bound for performance (must be >= 1): ' + str(lwr))
        sys.exit()
    if upr < lwr:
        s_ut.my_print('ERROR: lwr bound should be at most upr. lwr: ' + str(lwr) + ' upr: ' + str(upr))
        sys.exit()

    cfg_dict, actuals_df, df = cfg_select(ts_name, cutoff_date, upr, lwr, features, init_date=init_date, time_scale=time_scale)
    return actuals_df, df, cfg_dict


def fcast_idx(f_df):
    fg = pd.DataFrame(f_df['dim_cfg'].copy())
    fg.drop_duplicates(inplace=True)
    fg.reset_index(drop=True, inplace=True)
    fg.reset_index(inplace=True)
    p_ut.save_df(fg, '~/my_tmp/f_idx')
    return fg


def f_ok(sf, cu, fd, col):
    # sf: DF with one dim_cfg
    # cu: cutoff date
    # fd: fcast date
    # col: col in DF (yhat)
    gf = sf[~sf[col].isnull()].copy()
    gf, _ = dtp.de_gap(gf.copy(), 1, col, 'ds', 7)               # fills gaps and removes NaNs
    return None if gf['ds'].min() > cu or gf['ds'].max() < fd or len(gf) == 0 or gf[col].isnull().sum() > 0 else gf


