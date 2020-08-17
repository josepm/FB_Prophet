"""
prepare volume regressors
"""
import pandas as pd
import os
import sys
from functools import reduce
import platform

import sklearn.linear_model as l_mdl

# from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import time_utils as t_ut
from capacity_planning.data import hql_exec as hql
from capacity_planning.forecast.utilities.language import data_processing as dp

DEFAULT_LANGUAGES = ['English_APAC', 'English_EMEA', 'English_NA', 'French', 'German', 'Italian', 'Japanese', 'Korean', 'Mandarin', 'Portuguese', 'Russian', 'Spanish']
USE_CACHE = True if platform.system() == 'Darwin' else False
USE_CACHE = True
RENEW = not USE_CACHE

with s_ut.suppress_stdout_stderr():
    import airpy as ap


def ens_fcast(ts_name, regs, cutoff_date, time_scale, fcast_days, init_date, a_df):
    r_list = list()
    for rname in regs:
        r_cfg, _ = dp.ts_setup(rname, cutoff_date, init_date, time_scale)
        if r_cfg is None:
            s_ut.my_print('ERROR: invalid regressor name: ' + rname)
            sys.exit()

        if r_cfg['do_fcast'] is True:
            qry = 'select * from sup.cx_ens_forecast where cutoff = \'' + str(cutoff_date.date()) + '\' and ts_name = \'' + rname + '\';'
            rdf = hql.from_tble(qry, ['ds'], use_cache=USE_CACHE, renew=RENEW)
            if rdf is None:              # no ens fcast file found
                s_ut.my_print('ERROR: no forecast for regressor: ' + rname)
                sys.exit()
            else:
                cols = ['ds', 'language', 'yhat'] if 'language' in rdf.columns else ['ds', 'yhat']
                rdf = rdf[rdf['ds'] > cutoff_date][cols].copy()
                adf = get_actuals(r_cfg, init_date='2016-01-01')
                adf = adf[adf['ds'] <= cutoff_date].copy()
                adf.rename(columns={r_cfg['ycol']: 'yhat'}, inplace=True)
                rdf = pd.concat([adf[cols].copy(), rdf], axis=0)
        else:   # static regressors
            s_ut.my_print(rname + ' is a static regressor')
            try:
                reg_func = getattr(sys.modules[__name__], rname)   # function to set up the static regressor
                args = [cutoff_date, init_date, fcast_days, time_scale]
                rdf = reg_func(*args)
            except AttributeError as e:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' No static regressor with name ' + rname + ': ' + str(e))
                rdf = None

            if rdf is not None:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' found static regressor: ' + str(rname))
            else:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: regressor ' + rname + ' not found <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        if rdf is not None and len(rdf) > 0:
            rdf.rename(columns={'yhat': rname}, inplace=True)
            r_list.append(rdf)
    if len(r_list) > 0:
        r_dict = merge_regressors_(r_list, init_date)
        r_dict = {lx: selector(ts_name, lx, a_df[a_df['language'] == lx].copy(), rl, cutoff_date) for lx, rl in r_dict.items()}
        s_dict = {lx: fl for lx, fl in r_dict.items() if fl is not None}
        return s_dict
    else:
        return dict()


def merge_regressors_(df_list, init_date):       # put all regressors in a single DF merging by ds and language
    init_date = pd.to_datetime(init_date)
    min_ds = max([init_date] + [f['ds'].min() for f in df_list])
    max_ds = min([f['ds'].max() for f in df_list])
    new_df_list = [f[(f['ds'] >= min_ds) & (f['ds'] <= max_ds)] for f in df_list]
    _ = [f.drop('language', axis=1, inplace=True) for f in new_df_list if 'language' in f.columns and f['language'].nunique() == 1 and f['language'].unique()[0] == 'NULL']
    yl_list = [f for f in new_df_list if 'language' in f.columns and len(f) > 0]                 # regressors with language
    l_rdf = reduce(lambda x, y: x.merge(y, on=['ds', 'language'], how='left'), yl_list) if len(yl_list) > 0 else None
    nl_list = [f for f in new_df_list if 'language' not in f.columns and len(f) > 0]                 # regressors without language
    n_rdf = reduce(lambda x, y: x.merge(y, on=['ds'], how='left'), nl_list) if len(nl_list) > 0 else None
    reg_fdf = n_rdf if l_rdf is None else (l_rdf if n_rdf is None else l_rdf.merge(n_rdf, on='ds', how='left'))
    if reg_fdf is not None:
        if 'language' in reg_fdf.columns:
            l_dict = {lx: rl for lx, rl in reg_fdf.groupby('language')}
        else:
            l_dict = dict()
            for lang in DEFAULT_LANGUAGES:
                lf = reg_fdf.copy()
                lf['language'] = lang
                l_dict[lang] = lf
        return l_dict
    else:
        return dict()


def monthly(cutoff_date, init_date, fcast_days, time_scale):
    if time_scale == 'D':
        start = init_date
        freq = 'D'
    elif time_scale == 'W':
        dw = init_date.weekday()
        start = init_date + pd.to_timedelta(6 - dw, unit='D')  # set to week-starting Sunday
        freq = 'W-SUN'
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: freq not supported: ' + str(time_scale))
        sys.exit()

    dr = pd.date_range(start=start, end=cutoff_date + pd.to_timedelta(fcast_days + 7, unit='D'), freq=freq)
    df = pd.DataFrame({'ds': dr})
    for mm in range(1, 13):
        df[str(mm)] = df.apply(lambda x: 1 if x['ds'].month == mm else 0, axis=1)
    return df


def weekly(cutoff_date, init_date, fcast_days, time_scale):
    if time_scale == 'D':
        start = init_date
        freq = 'D'
    elif time_scale == 'W':
        dw = init_date.weekday()
        start = init_date + pd.to_timedelta(6 - dw, unit='D')  # set to week-starting Sunday
        freq = 'W-SUN'
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: freq not supported: ' + str(time_scale))
        sys.exit()

    dr = pd.date_range(start=start, end=cutoff_date + pd.to_timedelta(fcast_days + 7, unit='D'), freq=freq)
    df = pd.DataFrame({'ds': dr})
    for ww in range(1, 53):       # ignore week 53 when present
        df[str(ww)] = df.apply(lambda x: 1 if t_ut.week_from_date(x['ds']) == ww else 0, axis=1)
    return df


def get_actuals(rcfg, init_date='2016-01-01'):
    cutoff_date = rcfg['cutoff_date']
    init_date = rcfg.get('init_date', init_date)
    r_name = rcfg['name']
    int_type = rcfg.get('interaction_type', None)
    s_ut.my_print('getting ' + r_name + ' regressor actuals from table')
    ts_name = rcfg['ycol']
    r_date = hql.get_rmax(ts_name, use_cache=USE_CACHE)
    qry = 'select ds, language, y from sup.cx_weekly_actuals where ts_name=\'' + ts_name + '\' and run_date=\'' + r_date + '\';'
    try:
        rdf = hql.from_tble(qry, ['ds'], use_cache=USE_CACHE, renew=RENEW)
        s_ut.my_print(qry + ' completed. Got ' + str(len(rdf)) + ' rows')
    except:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: query: ' + qry + ' failed. No data for regressor ' + r_name)
        sys.exit()
    rdf.rename(columns={'y': rcfg['ycol']}, inplace=True)
    s_ut.my_print('pid: ' + str(os.getpid()) + ' got ts_name ' + ts_name + ' from table ' + 'sup.cx_weekly_actuals')
    if rdf is None:
        return None
    else:
        rdf = rdf[(rdf['ds'] >= pd.to_datetime(init_date))].copy()
        if rdf['ds'].max() < pd.to_datetime(cutoff_date):
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: ' + r_name + ' max date (' + str(rdf['ds'].max().date()) + ') is smaller than cutoff date (' + str(cutoff_date.date()) + ')')
            return None
        elif len(rdf) > 0:
            if 'interaction_type' in rdf.columns and int_type is not None:
                rdf = rdf[rdf['interaction_type'] == int_type].copy()
            rdf.reset_index(inplace=True, drop=True)
            rdf.rename(columns={'y': rcfg['ycol']}, inplace=True)
            return rdf
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no data for regressor  ' + r_name)
            return None


def selector(ts_name, lang, a_df, r_df, cutoff_date):
    # for each language, include only regressors that correlate well with the time series to forecast
    # use lasso to find out which ones are to be kept
    r_list = [c for c in r_df.columns if c not in ['ds', 'language']]
    ycol = [c for c in a_df.columns if c not in ['ds', 'language']][0]

    if len(a_df) == 0:
        s_ut.my_print('WARNING: empty actuals for ' + lang + ' and ts ' + ts_name)
        return None
    if len(r_df) == 0:
        s_ut.my_print('WARNING: empty regressors for ' + lang + ' and ts ' + ts_name)
        return None

    ds_min = max(a_df['ds'].min(), r_df['ds'].min())
    s_df = r_df[(r_df['ds'] >= ds_min) & (r_df['ds'] <= cutoff_date)].copy()
    b_df = a_df[(a_df['ds'] >= ds_min) & (a_df['ds'] <= cutoff_date)].copy()

    df = b_df.merge(s_df, on=['ds', 'language'], how='left')
    df.dropna(inplace=True)
    X_train = df[r_list].values
    y_train = df[ycol].values
    lasso_mdl = l_mdl.LassoLarsIC(criterion='aic', normalize=True)
    lasso_mdl.fit(X_train, y_train)
    m_pars = lasso_mdl.coef_              # pars excluding intercept
    new_r_list = [r_list[i] for i in range(len(r_list)) if m_pars[i] != 0.0]
    s_ut.my_print('regressor selector::regressors for ' + ts_name + ' and language ' + lang + ': ' + str(new_r_list))
    return r_df[['ds', 'language'] + new_r_list].copy() if len(new_r_list) > 0 else None
