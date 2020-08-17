"""
Get the ensemble configs to use per ts and language by comparing with actuals
$ python ens_cfg.py cutoff_date ts_name
ts_name = phone-inbound-vol, phone-inbound-aht, phone-inbound-vol, phone-outbound-aht
set parameters in code:
    horizon = horizon in days (112) but horizon_date = min(cutoff_date + horizon, actuals.ds..max())
output: in DFs saved as parquet
    ~/my_tmp/ens_cfg/lang_all_<ts_name>_<cutoff_date>_<horizon_date>.par  --> all results for all languages
    ~/my_tmp/ens_cfg/lang_top_<ts_name>_<cutoff_date>_<horizon_date>.par  --> best result per language
    ~/my_tmp/ens_cfg/no_lang_all_<ts_name>_<cutoff_date>_<horizon_date>.par  --> avg for ts (no language)

"""

import os
import sys
import platform
import multiprocessing as mp

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'  # before pandas load
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'  # before pandas load

import pandas as pd
import numpy as np
import json
import itertools
import scipy.stats as sps
import sklearn.linear_model as l_mdl
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import itertools
from capacity_planning.utilities import sys_utils as s_ut


from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import stats_utils as st_ut
from capacity_planning.forecast.interactions_forecast import lang_fcast as lfc
from capacity_planning.forecast.utilities.language import data_processing as dtp
from capacity_planning.data import hql_exec as hql
from capacity_planning.utilities import xforms as xf

err_arr = ['sMAPE', 'MAPE', 'mMAPE', 'wMAPE', 'LAR', 'MASE', 'RMSE', 'YJ', 'YJd', 'BC', 'SR']


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
        # t_df = t_df[t_df['interaction_type'] == 'outbound'].copy()
        t_df = t_df[t_df['calls'] > 0].copy()
        ttl_mins = t_df['agent_mins']      # + t_df['agent_rcv_consult_mins'] + t_df['agent_init_consult_mins']
        t_df['y'] = ttl_mins / t_df['calls']
        ctype = 'OB AHT'
    elif name == 'deferred':
        ctype = None
    elif name == 'deferred_hbat':
        ctype = None
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' unknown ts_name: ' + str(name))
        sys.exit(0)
    return t_df.copy(), ctype


def process_mdl_args(fdf_, a_df_, cutoff_date_, l_dict, reg_mdl):
    return [[
        l_dict[l]['cfg_idx'], l, lf, a_df_[a_df_['language'] == l], cutoff_date_, l_dict[l]['alpha'], reg_mdl, l_dict[l]['normalize']
    ] for l, lf in fdf_[['ds', 'yhat', 'cfg_idx', 'language']].groupby('language')]


def set_reg_mdl(reg_mdl, alpha, normalize):
    if reg_mdl == 'EN':
        return l_mdl.ElasticNet(alpha=alpha, normalize=normalize), 3  # elastic net regression
    elif reg_mdl == 'BR':
        return l_mdl.BayesianRidge(normalize=normalize), 2  # Bayesian Ridge
    elif reg_mdl == 'RG':
        return l_mdl.Ridge(normalize=normalize), 2  # Ridge regression
    elif reg_mdl == 'LS':
        return l_mdl.Lasso(alpha=alpha, normalize=normalize), 2  # Lassoregression
    elif reg_mdl == 'LR':
        return l_mdl.LinearRegression(), 1  # OLS regression
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: invalid regression model: ' + str(reg_mdl))
        return None, None


def process_mdl(cfg_list, lang, lang_f, af, cutoff_date_, alpha, selection, opt, reg_mdl, normalize):
    f = lang_f[lang_f['cfg_idx'].isin(cfg_list)]
    if len(f) == 0:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no data for cfg_list: ' + str(cfg_list) + ' and language ' + str(lang))
        p_ut.save_df(lang_f, '~/my_tmp/lang_f')
        return None
    pf = pd.pivot_table(f[['ds', 'yhat', 'cfg_idx', 'language']], index=['ds', 'language'], columns='cfg_idx', values='yhat').reset_index()  # (level=0).reset_index(drop=True)
    pf.dropna(inplace=True)               # ensure all fcast cfgs have the same time range
    df = pf.merge(af[['ds', 'language', 'y']], on=['ds', 'language'], how='inner')
    df.drop('language', inplace=True, axis=1)
    df_train = df[df['ds'] <= cutoff_date_]

    if selection != 'single':
        r_mdl, rpars = set_reg_mdl(reg_mdl, alpha, normalize)
        if r_mdl is None:
            return None
        pars = len(cfg_list) + rpars
    else:
        pars, rpars, r_mdl = 1, 0, None

    ndata = len(df_train)     # data points
    if ndata < pars:          # more regression unknowns than data
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: not enough data for cfg_list: ' + str(cfg_list))
        return None
    else:
        d_out = dict()
        X_train = df_train[cfg_list].values
        y_train = df_train['y'].values

        df_test = df[df['ds'] > cutoff_date_]
        X_test = df_test[cfg_list].values
        y_test = df_test['y'].values

        # in-sample performance
        if selection != 'single':
            r_mdl.fit(X_train, y_train)

        y_samp = np.reshape(X_train, (-1, 1)) if selection == 'single' else r_mdl.predict(X_train)                                      # in-sample prediction
        for et in err_arr:
            d_out[et + '_train'] = st_ut.err_func(y_train, y_samp, et)

        # off-sample performance
        yhat = np.reshape(X_test, (-1, 1)) if selection == 'single' else r_mdl.predict(X_test)                                           # off-sample prediction
        for et in err_arr:
            d_out[et + '_test'] = st_ut.err_func(y_test, yhat, et)

        # AIC:
        # see https://kourentzes.com/forecasting/2016/06/17/how-to-choose-a-forecast-for-your-time-series/ (note should be MSE = (1/n) sum_i (y_1 - f_i)^2)
        # https://www.stat.berkeley.edu/~binyu/summer08/Hurvich.AICc.pdf  eq 4
        m_pars = [1] if selection == 'single' else r_mdl.coef_   # pars list excluding intercept
        if len(cfg_list) != len(m_pars):
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: cfg_list: ' + str(cfg_list) + ' pars: ' + str(m_pars))
            sys.exit()
        new_cfg_list = [int(cfg_list[i]) for i in range(len(cfg_list)) if m_pars[i] != 0.0]
        new_pars = len(new_cfg_list) + rpars
        mse_ = np.mean((y_samp - y_train) ** 2)
        d_out['aic'] = 2 * new_pars + ndata * np.log(mse_) + 2 * new_pars * (1 + new_pars) / (ndata - new_pars - 1)
        d_out['language'] = lang
        d_out['selection'] = selection
        d_out['opt'] = opt
        d_out['cfg_idx'] = json.dumps(new_cfg_list)
        d_out['alpha'] = alpha
        d_out['normalize'] = normalize
        return d_out


def lasso_selection(lang, lang_f, af, normalize=True):
    cfg_list = lang_f['cfg_idx'].unique()
    f = lang_f[lang_f['cfg_idx'].isin(cfg_list)]
    if len(f) == 0:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no data for cfg_list: ' + str(cfg_list))
        return None
    pf = pd.pivot_table(f[['ds', 'yhat', 'cfg_idx', 'language']], index=['ds', 'language'], columns='cfg_idx', values='yhat').reset_index(level=0).reset_index(drop=True)
    pf.dropna(inplace=True)  # ensure all fcast cfgs have the same time range
    df = pf.merge(af, on='ds', how='inner')
    pars = len(cfg_list)     # regression parameters: one per fcast_cfg plus 3 extra pars: constant + 2 regularization
    ndata = len(df)          # data points
    while ndata < pars:      # more regression unknowns than data
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: not enough data for cfg_list: ' + str(cfg_list))
        cfg_list = cfg_list[:int(0.75 * ndata)]   # pick the top one by score
        pars = len(cfg_list)
    if len(cfg_list) == 0:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: not enough data for cfg_list: ' + str(cfg_list))
        return None

    X_train = df[cfg_list].values
    y_train = df['y'].values
    lasso_mdl = l_mdl.LassoLarsIC(criterion='aic', normalize=normalize)
    lasso_mdl.fit(X_train, y_train)
    m_pars = lasso_mdl.coef_              # pars excluding intercept
    new_cfg_list = [int(cfg_list[i]) for i in range(len(cfg_list)) if m_pars[i] != 0.0]
    return {'cfg_idx': new_cfg_list, 'alpha': lasso_mdl.alpha_, 'language': lang, 'normalize': normalize}


def variable_selection(step, cfg_list, lang, lang_f, af, cutoff_date_, alpha, reg_mdl, normalize, ctr=0):
    # add (step = fwd) or remove (step = bwd) one cfg at a time
    # cfg_list is the initial cfg, which is always set by the Lasso regression (good starting point)
    # for step = fwd, if cfg_list = list(), it will incrementally add one cfg till aic cannot be improved any longer by adds
    # for step = bwd, cfg_list = all_cfg, it will incrementally remove one cfg till aic cannot be improved any longer by removals
    # after the fwd or bwd step is completed, we do a rounds of alternating add/remove to see if we can further improve (lower aic)

    all_cfgs = lang_f['cfg_idx'].unique()  # all available cfgs
    if 'fwd' in step:
        new_cfgs = list(set(all_cfgs) - set(cfg_list))  # possible cfgs to add
    elif 'bwd' in step:
        new_cfgs = [c for c in cfg_list]                # possible cfgs to rm
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: invalid step: ' + str(step))
        sys.exit()
        
    # always refine the initial cfg using lasso (unless recursive call)
    f_list = list()
    init_selection = step if '_alt' in step else 'lasso'
    f = process_mdl(cfg_list, lang, lang_f, af, cutoff_date_, alpha, init_selection, 'lasso', reg_mdl, normalize)
    if f is not None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' variable_selection:: step: ' + str(step) + ' init cfg: ' + str(cfg_list) + ' f: ' + str(f))
        f_list.append(f)
        min_aic = f['aic']
        best_cfg = json.loads(f['cfg_idx'])
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: could not find initial lasso cfg. Using ' + str(cfg_list))
        min_aic = np.inf
        best_cfg = [c for c in cfg_list]

    while True:
        better_cfg = None
        for c in new_cfgs:
            if 'fwd' in step:
                if c in best_cfg:    # cannot add it: already in the best_cfg
                    continue
                else:
                    this_cfg = [x for x in best_cfg] + [c]
            else:     # bwd
                if c not in best_cfg:  # cannot remove it: try another one
                    continue
                else:
                    this_cfg = [x for x in best_cfg if x != c]
            f = process_mdl(this_cfg, lang, lang_f, af, cutoff_date_, alpha, step, 'NULL', reg_mdl, normalize)
            if f is not None:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' variable selection::step: ' + str(step) + ' ctr: ' + str(ctr) + ' cfg: ' + str(this_cfg) + ' f: ' + str(f))
                f_list.append(f)
                if f['aic'] < min_aic:                    # found a better cfg
                    min_aic = f['aic']
                    better_cfg = json.loads(f['cfg_idx'])
        if better_cfg is not None:
            best_cfg = [c for c in better_cfg]
        else:
            break

    # try refine by alternating add/remove features
    if '_alt' not in step:
        while ctr < min(5.0, len(best_cfg)):
            step = 'fwd_alt' if 'bwd' in step else 'bwd_alt'
            f_alt = variable_selection(step, best_cfg, lang, lang_f, af, cutoff_date_, alpha, reg_mdl, normalize, ctr=ctr+1)
            if f_alt is None:
                break
            else:
                f_list += f_alt
                f_alt.sort(key=lambda x: x['aic'])  # list of dicts sorted by increasing aic
                if f_alt[0]['aic'] < min_aic:
                    min_aic = f_alt[0]['aic']
                    best_cfg = json.loads(f_alt[0]['cfg_idx'])
                else:
                    break
    return f_list   # list of dicts


def set_all(e_df, w_df, etype_arr):
    opt = e_df.loc[e_df.index[0], 'opt']
    fx = e_df.merge(w_df, on='language', how='inner')

    # NOTE: must use None in cfg_idx to be able to save as parquet
    d_out = {'ens_score': (fx['ens_score'] * fx['weight']).sum() / fx['weight'].sum(), 'language': 'All', 'cfg_idx': None, 'selection': 'N/A', 'opt': opt, 'aic': np.nan}
    for etype in etype_arr:
        for k in ['_test', '_train']:
            d_out[etype + k] = (fx[etype + k] * fx['weight']).sum() / fx['weight'].sum(),
    return pd.DataFrame(d_out, index=[0])


def topn_fcasts(l, lf, n=None):
    # top n forecasts by language from idx
    if n is None:
        n = len(lf)
    lf['cfg_idx'] = lf['cfg_idx'].astype(int)
    min_idx = lf['cfg_idx'].min()
    t_lf = lf[lf['cfg_idx'] < min_idx + n].copy()
    t_lf['cfg_idx'] = t_lf['cfg_idx'].astype(str)
    t_lf['language'] = l
    return t_lf


def best_score(sdf, etype):
    e_df = sdf.groupby('language').apply(pd.DataFrame.nsmallest, n=1, columns=[etype]).reset_index(drop=True)  # best scores
    e_df['opt'] = e_df['opt'].apply(lambda x: etype if x == 'NULL' else x)
    return e_df


def main(ts_name_, cutoff_date_):
    cfg_cols = ['growth', 'y_mode', 'w_mode', 'r_mode', 'xform', 'h_mode', 'training', 'do_res', 'changepoint_range']
    upr_horizon, lwr_horizon = 112, 84
    lbl = ts_name_ + '_' + cutoff_date_

    if ts_name_ == 'phone-inbound-vol':
        fname = dtp.get_data_file('~/my_tmp/cleaned/phone-vol_cleaned_', cutoff_date_)
        interaction_type = 'inbound'
    else:
        fname = dtp.get_data_file('~/my_tmp/cleaned/phone-aht_cleaned_', cutoff_date_)
        interaction_type = 'inbound' if 'inbound' in ts_name_ else 'outbound'

    # ######################################################################
    # ######################################################################
    # actuals
    s_ut.my_print('pid: ' + str(os.getpid()) + ' actuals file: ' + str(fname))
    q_df = pd.read_parquet(fname)

    # week_starting patch
    df_cols_ = q_df.columns
    if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
        q_df['ds_week_ending'] = pd.to_datetime(q_df['ds_week_ending'])
        q_df['ds_week_starting'] = q_df['ds_week_ending'] - pd.to_timedelta(6, unit='D')

    q_df['ds'] = pd.to_datetime(q_df['ds'].values)
    w_col = 'y' if 'vol' in ts_name_ else 'calls'

    # daily actuals (language level)
    if ts_name_ == 'phone-inbound-vol':
        q_df = q_df.groupby(['ds','language']).agg({'offered': np.sum, 'accepted': np.sum, 'abandons': np.sum}).reset_index()
    else:
        q_df = q_df[q_df['interaction_type'] == interaction_type].copy()
        q_df = q_df.groupby(['ds', 'language']).agg({'calls': np.sum, 'agent_mins': np.sum}).reset_index()
    a_ddf, ctype = set_demand(q_df.copy(), 10, ts_name_)
    w_df = a_ddf[a_ddf['ds'] <= cutoff_date_].groupby('language').agg({w_col: np.sum}).reset_index()
    w_df.columns = ['language', 'weight']
    p_ut.save_df(a_ddf, '~/my_tmp/a_daily_df_' + lbl)

    # weekly level: use week starting
    if ts_name_ == 'phone-inbound-vol':
        m_df = q_df.groupby(pd.Grouper(key='ds', freq='W-SUN')).agg({'offered': np.sum, 'accepted': np.sum, 'abandons': np.sum}).reset_index()
    else:
        m_df = q_df.groupby(pd.Grouper(key='ds', freq='W-SUN')).agg({'calls': np.sum, 'agent_mins': np.sum}).reset_index()
    a_wdf_, ctype = set_demand(m_df, 10, ts_name_)

    a_wdf = a_wdf_.copy()
    horizon_date = min(pd.to_datetime(cutoff_date_) + pd.to_timedelta(upr_horizon, unit='D'), a_wdf['ds'].max())
    a_wdf['ds_week_ending'] = a_wdf['ds'] + pd.to_timedelta(6, unit='D')  # switch to week ending so that we do not have incomplete weeks at end
    a_wdf = a_wdf[(a_wdf['ds_week_ending'] <= horizon_date) & (a_wdf['ds_week_ending'] > cutoff_date_)].copy()
    a_wdf.drop('ds', axis=1, inplace=True)
    a_wdf['ts_name'] = ts_name_
    p_ut.save_df(a_wdf, '~/my_tmp/a_weekly_df_' + lbl)
    # ######################################################################
    # ######################################################################

    # ######################################################################
    # ######################################################################
    # DS forecasts: select the top fcast cfgs for each language, score them based on past performance and forecast them
    sdir = '~/my_tmp/cfg_sel/'
    df_best = p_ut.read_df(sdir + 'cfg_best_' + ts_name_ + '_' + cutoff_date_)  # best ensembles by idx
    p_ut.set_week_start(df_best, tcol='ds')  # week_starting patch

    z = df_best[['language', 'cfg_idx']].copy()
    z.set_index('language', inplace=True)
    dx = z.to_dict()['cfg_idx']
    dx = {k: list(v) for k, v in dx.items()}

    df_idx = p_ut.read_df(sdir + 'cfg_idx_' + ts_name_ + '_' + cutoff_date_)   # map cfg_idx to fcast cfg
    p_ut.set_week_start(df_idx, tcol='ds')  # week_starting patch
    df_idx = df_idx[['language', 'cfg_idx'] + cfg_cols].copy()
    df_idx.drop_duplicates(inplace=True)

    # fix None for fcasts
    for c in cfg_cols:
        df_idx[c] = df_idx[c].apply(lambda x: None if x == 'None' else x)
    df_idx['h_mode'] = df_idx['h_mode'].apply(lambda x: True if x == 1 else False)
    df_idx['do_res'] = df_idx['do_res'].apply(lambda x: True if x == 1 else False)
    cfg_df = pd.concat([lf[lf['cfg_idx'].isin(dx[l])] for l, lf in df_idx.groupby('language')], axis=0)

    # run the fcasts for the selected cfg's
    file_out = lfc.main(ts_name_, cutoff_date_, cfg_cols, to_db=False, df_cfg=cfg_df.copy(), is_mp=True)   # , is_fcast=False)
    if file_out is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no fcast file returned')
        sys.exit()
    s_ut.my_print('pid: ' + str(os.getpid()) + ' +++++++++++++ completed forecasts:: file: ' + str(file_out) + ' +++++++++++++++ ')
    fdf = pd.read_parquet(file_out)

    # week_starting patch
    df_cols_ = fdf.columns
    if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
        fdf['ds_week_ending'] = pd.to_datetime(fdf['ds_week_ending'])
        fdf['ds_week_starting'] = fdf['ds_week_ending'] - pd.to_timedelta(6, unit='D')

    print(fdf.head())

    # make sure fdf and perf_df have the same cfg's
    # fdf_list = list(fdf['cfg_idx'].unique())
    # perf_df = perf_df[perf_df['cfg_idx'].isin(fdf_list)].copy()
    # cfg_df = cfg_df[cfg_df['cfg_idx'].isin(fdf_list)].copy()
    # arr = [int(x * num_cfg_) for x in [1.0, 0.75, 0.5, 0.25, 0.125]]
    # arg_list = [[fdf, a_ddf, w_df, cfg_df, ts_name_, cutoff_date_, horizon_date] for k in arr if k > 1]
    # m_list = s_ut.do_mp(get_models, arg_list, is_mp=True, cpus=len(arr), do_sigkill=True)  # list of dicts
    # m_list = get_models(fdf, a_ddf, w_df, cfg_df, ts_name_, cutoff_date_, horizon_date)
    m_list = get_models(fdf, a_ddf, w_df, ts_name_, cutoff_date_, horizon_date)
    s_ut.my_print('pid: ' + str(os.getpid()) + ' ============== main: get_models complete. appending results ==========')
    if len(m_list) > 0:
        d_out = dict()
        for dv in m_list:
            for k, fname in dv.items():
                if k not in d_out.keys():
                    d_out[k] = list()
                fz = p_ut.read_df(fname)
                p_ut.set_week_start(fz, tcol='ds')  # week_starting patch
                d_out[k].append(fz)
        return {k: pd.concat(d_out[k], axis=0) for k, arr in d_out.items()}
    else:
        return None


def single_cfg(fdf, a_ddf, cutoff_date_):
    s_list = [process_mdl([cfg], l, lf, a_ddf, cutoff_date_, None, 'single', 'NULL', None, None) for l, lf in fdf.groupby('language') for cfg in lf['cfg_idx'].unique()]
    return pd.DataFrame(s_list)


def select_cfg(fdf, a_ddf, cutoff_date_):
    s_list = [process_mdl(lf['cfg_idx'].unique(), l, lf, a_ddf, cutoff_date_, None, 'All', 'NULL', 'LR', True) for l, lf in fdf.groupby('language') for cfg in lf['cfg_idx'].unique()]
    return pd.DataFrame(s_list)


# def correlation_results(fdf, a_ddf, cutoff_date_, perf_df, cfg_df, p_col_, num_cfg_, reg_mdl_):
#     dict_list = corr_selection(perf_df, cfg_df, p_col_, num_cfg_, 1.0)   # select only cfg's with corr <= 0.0
#     corr_df = pd.DataFrame(dict_list)
#     corr_df.set_index('language', inplace=True)
#     dx_ = corr_df.to_dict()['cfg_list']            # dx = {lang: cfg_list, ...}
#     dx = {k: list(v) for k, v in dx_.items() if len(v) > 0}
#     d_arr = {l: list({max(1, int(x * len(dl))) for x in [0.4, 0.2, 0.1, 0.05, 0.02, 0.01]}) for l, dl in dx.items()}
#     c_list = [process_mdl(dx[l][:ln], l, lf, a_ddf, cutoff_date_, None, 'corr', 'NULL', reg_mdl_, None) for l, lf in fdf.groupby('language') for ln in d_arr[l]]
#     return pd.DataFrame(c_list), dx


def lasso_vble_sel_results(fdf, a_ddf, cutoff_date_, normalize, reg_mdl):
    # find an initial cfg with lasso and improve aic with variable selection
    n_lang = fdf['language'].nunique()
    t_fdf = fdf[fdf['ds'] <= cutoff_date_].copy()
    t_adf = a_ddf[a_ddf['ds'] <= cutoff_date_][['ds', 'language', 'y']].copy()

    arg_list = [[l, lf, t_adf[t_adf['language'] == l].copy(), normalize]
                for l, lf in t_fdf[['ds', 'yhat', 'cfg_idx', 'language']].groupby('language')]
    d_list = s_ut.do_mp(lasso_selection, arg_list, is_mp=True, cpus=n_lang, do_sigkill=True)
    lasso_df = pd.DataFrame(d_list)
    lasso_df.set_index('language', inplace=True)
    lasso_dict = lasso_df.to_dict(orient='index')   # {lang: {'alpha': ..., 'cfg_list': [...]}, ..}

    # improve best lasso cfg by variable selection starting from initial Lasso cfg
    arg_list = process_mdl_args(fdf.copy(), a_ddf.copy(), cutoff_date_, lasso_dict, reg_mdl)
    f_arg_list = [['fwd'] + a for a in arg_list]
    f_list_ = s_ut.do_mp(variable_selection, f_arg_list, is_mp=True, cpus=n_lang, do_sigkill=True)  # list of lists
    f_list = [d for dl in f_list_ for d in dl if d is not None]                                      # flatten the list
    b_arg_list = [['bwd'] + a for a in arg_list]
    b_list_ = s_ut.do_mp(variable_selection, b_arg_list, is_mp=True, cpus=n_lang, do_sigkill=True)   # list of lists
    b_list = [d for dl in b_list_ for d in dl if d is not None]                                       # flatten the list
    scores_df = pd.DataFrame(f_list + b_list)
    scores_df.dropna(inplace=True)
    scores_df.reset_index(inplace=True, drop=True)
    return scores_df


def get_models(fdf, a_ddf, w_df, ts_name_, cutoff_date_, horizon_date):
    p = mp.current_process()
    p.daemon = False

    # #########################
    # parameters
    # #########################

    s_ut.my_print('pid: ' + str(os.getpid()) + ' :::::::: get_models ::::::::::::::::')
    
    # get the fcast DF
    lbl = ts_name_ + '_' + cutoff_date_
    # fdf_file = '~/my_tmp/ens_cfg/top_fcasts_' + lbl
    # p_ut.save_df(fdf, fdf_file)
    # p_ut.save_df(a_ddf, '~/my_tmp/ens_cfg/actuals_' + lbl)

    # selected cfgs
    sel_df = select_cfg(fdf, a_ddf, cutoff_date_)
    sel_df['ts_name'] = ts_name_
    sel_df['cutoff_date'] = cutoff_date_
    p_ut.save_df(sel_df, '~/my_tmp/sel_df__' + lbl)

    # single fcast cfg results
    s_df = single_cfg(fdf, a_ddf, cutoff_date_)                                                             # single cfg results (no opt needed)
    s_df['ts_name'] = ts_name_
    s_df['cutoff_date'] = cutoff_date_
    p_ut.save_df(s_df, '~/my_tmp/singles_df__' + lbl)

    # Lasso and vble selection results
    scores_df = lasso_vble_sel_results(fdf, a_ddf, cutoff_date_, True, 'EN')                                # results from Lasso and variable selection
    scores_df['ts_name'] = ts_name_
    scores_df['cutoff_date'] = cutoff_date_
    p_ut.save_df(scores_df, '~/my_tmp/scores_df__' + lbl)


    rrrrrrrrrrrrrrrrrrr

    # top scores
    etype_arr = err_arr
    b_lasso = scores_df['opt'] == 'lasso'
    ls_f = scores_df[b_lasso].copy()                  # top scores from Lasso + variable selection
    i_df = best_score(scores_df[~b_lasso], 'aic')     # top scores from aic
    all_idf = set_all(i_df, w_df, etype_arr)          # all languages avg for aic
    e_list = [best_score(scores_df[~b_lasso], etype + '_test') for etype in etype_arr]   # top scores for each error type
    all_e_list = [set_all(f, w_df, etype_arr) for f in e_list]                           # language avg for each error type
    top_df = pd.concat(e_list + all_e_list + c_list + [all_idf, i_df, ls_f, s_df, c_aic, all_c], axis=0)        # all top scores
    top_df.sort_values(by=['language'], inplace=True)
    top_df.reset_index(inplace=True, drop=True)
    top_df['cfg_idx'] = top_df['cfg_idx'].apply(lambda x: json.dumps(x))                   # for dedup
    top_df.drop_duplicates(subset=['opt', 'cfg_idx'], inplace=True)
    top_df['cfg_idx'] = top_df['cfg_idx'].apply(lambda x: json.loads(x))                   # for parquet saving and nicer print

    # all scores
    n_df = scores_df[scores_df['opt'] == 'NULL']
    scores_df = pd.concat([top_df, n_df, c_df], axis=0)                                 # all scores
    scores_df['cfg_idx'] = scores_df['cfg_idx'].apply(lambda x: json.dumps(x))    # for dedup
    scores_df.drop_duplicates(subset=['cfg_idx'], keep='last', inplace=True)
    scores_df['cfg_idx'] = scores_df['cfg_idx'].apply(lambda x: json.loads(x))    # for parquet saving

    # save data
    scores_df['ts_name'] = ts_name_
    scores_df['cutoff_date'] = cutoff_date_
    scores_df['horizon_date'] = horizon_date
    scores_df_file = '~/my_tmp/ens_cfg/scores_df_' + lbl
    p_ut.save_df(scores_df, scores_df_file)
    s_ut.my_print('pid: ' + str(os.getpid()) + ' xxxxxxxxxxxxxxx all scores xxxxxxxxxxxxxxx ' + str(num_cfg_))
    print(scores_df.head(10))

    top_df['ts_name'] = ts_name_
    top_df['cutoff_date'] = cutoff_date_
    top_df['horizon_date'] = horizon_date
    top_df_file = '~/my_tmp/ens_cfg/top_df_' + lbl
    p_ut.save_df(top_df, top_df_file)
    s_ut.my_print('pid: ' + str(os.getpid()) + ' xxxxxxxxxxxxxxx top scores xxxxxxxxxxxxxxx ' + str(num_cfg_))
    print(top_df.head(10))

    # map cfg_idx to cfg(str)
    mf = perf_df.copy()
    mf.drop_duplicates(inplace=True)
    mf['ts_name'] = ts_name_
    mf['cutoff_date'] = cutoff_date_
    mf['horizon_date'] = horizon_date
    map_df_file = '~/my_tmp/ens_cfg/map_df_' + lbl
    p_ut.save_df(mf, map_df_file)
    s_ut.my_print('pid: ' + str(os.getpid()) + ' xxxxxxxxxxxxxxx maps xxxxxxxxxxxxxxx ' + str(num_cfg_))
    print(mf.head())

    return {'scores_df': scores_df_file, 'top_df': top_df_file, 'map_df': map_df_file, 'f_df': fdf_file}

