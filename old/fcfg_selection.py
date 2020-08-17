"""

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
import copy
import multiprocessing as mp
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.data import hql_exec as hql
from capacity_planning.utilities import xforms as xf


def get_cfg_data(ts_name, cfg_cols, p_col):
    # read all the cfgs and set the cfg_idx
    t_name = 'sup.fct_cx_forecast_config'
    qry = 'select * from ' + t_name + ';'
    q_file = '/tmp/read_cfg_' + ts_name + '.hql'
    with open(q_file, 'w') as f:
        f.write(qry)
    s_ut.my_print('pid: ' + str(os.getpid()) + ' in query file: ' + q_file)
    fout = None
    ret = hql.run_hql((q_file, q_file), fout)
    if ret == -1:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: Query failed. No configs found')
        sys.exit()
    s_ut.my_print('pid: ' + str(os.getpid()) + ' fcast cfg file: ' + ret)
    cfg_df = p_ut.read_df(ret, sep='\t')
    p_ut.set_week_start(cfg_df, tcol='ds')     # week_starting patch

    if cfg_df is None or len(cfg_df) == 0:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no data for query: ' + str(qry))
        sys.exit()
    dcol = {x: x.replace(t_name.split('.')[-1] + '.', '') for x in cfg_df.columns}
    cfg_df.rename(columns=dcol, inplace=True)
    cfg_df['cutoff'] = pd.to_datetime(cfg_df['cutoff'])
    cfg_df = cfg_df[(cfg_df['ts_name'] == ts_name)]
    cfg_df = cfg_df[cfg_df[p_col] > 0.0].copy()
    cfg_df.fillna('None', inplace=True)
    cfg_df['cfg_str'] = cfg_df.apply(lambda x: json.dumps(x[cfg_cols].to_dict()), axis=1)
    z = cfg_df['cfg_str'].drop_duplicates()
    zf = pd.DataFrame(z)
    zf.reset_index(inplace=True, drop=True)
    zf.reset_index(inplace=True)
    zf.columns = ['cfg_idx', 'cfg_str']
    df = cfg_df.merge(zf, on=['cfg_str'], how='left')
    df['language'].replace(['Mandarin_Offshore', 'Mandarin_Onshore'], 'Mandarin', inplace=True)  # Mandarin need to be fixed later
    df.drop_duplicates(inplace=True)
    p_ut.save_df(df, '~/my_tmp/rk_df_' + ts_name)
    # df = p_ut.read_df('~/my_tmp/rk_df_' + ts_name)
    return df


def set_rank(a_df, p_col):
    def _set_rank(xf, pcol):
        xf.sort_values(by=pcol, inplace=True)
        xf.reset_index(drop=True, inplace=True)
        xf.reset_index(inplace=True)
        xf.rename(columns={'index': 'rank'}, inplace=True)
        xf['rank'] += 1
        xf['rank'] /= (1.0 + xf['rank'].max())                                         # relative daily rank 0 < rank < 1  (helps with xforms!)
        xf['rank_logistic'] = np.log(xf['rank'].values / (1.0 - xf['rank'].values))    # logistic
        xf['rank_log'] = np.log(xf['rank'].values)
        return xf
    g = a_df.groupby(['cutoff', 'language']).apply(_set_rank, pcol=p_col).reset_index(drop=True)
    return g


def ab_func(adb_estimators, max_depth, r, s, lf, X_train, y_train, X_test, y_test, y_perf, topN_list):
    ab_reg = AdaBoostRegressor(n_estimators=adb_estimators, base_estimator=DecisionTreeRegressor(max_depth=max_depth, min_samples_split=s), loss=lf, learning_rate=r)
    try:
        ab_reg.fit(X_train, y_train)
    except ValueError as e:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: ' + str(e))
        s_ut.save_df(pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=0), '~/my_tmp/ab_func_err')
        return [dict()]
    yhat_test = ab_reg.predict(X_test)
    d_list = list()
    d_cfg = {'adb_estimators': adb_estimators, 'max_depth': max_depth, 'learning_rate': r, 'loss': lf, 'min_samples_split': s}

    # for each AdaBoost cfg and topN get the the values of all the loss functions
    for nval in topN_list:
        d_ = copy.deepcopy(d_cfg)
        d_loss = loss_func(y_test, yhat_test, nval, w=y_perf)
        d_.update(d_loss)
        d_list.append(d_)
    return d_list


def loss_func(y, yhat, topN, w=1.0):
    # if i-th cfg placed in topN by AdaBoost but i-th cfg not in topN, loss equal to w_i = f_err(i)
    # y actual rank, yhat predicted rank
    # w = weight (actual performance, not rank)
    # the loss functions must be normalized to be independent of topN
    z = pd.DataFrame({'y': y, 'yhat': yhat})
    if len(z) < topN:
        topN = len(z)
    if topN == 0:
        return dict()
    z['w'] = w                                                  # no need to normalize: w is always the same
    e_hat = z.nsmallest(n=topN, columns=['yhat'])['w'].sum()    # fcast error when choosing topN with yhat
    z.sort_values(by='yhat', inplace=True)
    z.reset_index(drop=True, inplace=True)     # we would select index 0 to topN - 1
    ztop = z.nsmallest(n=topN, columns=['y'])  # any index in ztop < topN is correct
    e_min = ztop['w'].sum()                    # optimal case: fcast error when choosing with y
    ztop.reset_index(inplace=True)             # if index >= topN, error
    ef = ztop[ztop['index'] >= topN]           # selected in error by yhat
    return {'a_loss': ef['w'].mean(),                   # total loss
            'p_loss': len(ef) / topN,                   # <= 1: fraction of bad selections
            'w_loss': ef['w'].sum() / ztop['w'].sum(),  # <= 1: weighted fraction of bad selections
            'r_loss': e_hat / e_min,                    # >= 1: relative loss wrt to min loss
            'topN': topN                                # total selections
            }                               # losses: the smaller the better


def best_regression_cfg(X_train, y_train, X_test, y_test, y_perf, n_good, topN_list, obj_list, used_cpus=0):
    # n_good: nbr of 'good' AdaBoost cfgs to avg on
    # obj_list: loss functions to apply to y col actual and predicted
    ab_cols = ['adb_estimators', 'max_depth', 'learning_rate', 'loss', 'min_samples_split']  # AdaBoost cfg

    estimators_list = [25, 50, 100, 200]
    depth_list = [4, 8, 12, 16]
    learn_rate = [0.5, 1, 1.5, 2]
    min_samples_split_list = [2, 4, 8, 12]
    loss_list = ['linear', 'square', 'exponential']
    ab_cfgs = itertools.product(estimators_list, depth_list, learn_rate, min_samples_split_list, loss_list)
    ab_cfgs = [list(x) + [X_train, y_train, X_test, y_test, y_perf, topN_list] for x in ab_cfgs]
    f_list_ = s_ut.do_mp(ab_func, ab_cfgs, is_mp=True, cpus=None, do_sigkill=True, verbose=False, used_cpus=used_cpus)
    f_list = [x for l in f_list_ for x in l if len(x) > 0]
    f = pd.DataFrame([d for d in f_list if len(d) > 0])
    d_list = list()
    for obj in obj_list:                                     # for each obj func choose the best regressor
        fad = f.nsmallest(n=n_good, columns=[obj])
        if np.isinf(fad[obj].min()) is True:
            continue
        d_adb = {c: fad[c].mode().values[0] for c in ab_cols}  # take the most common among the top n_good
        d_adb['topN'] = fad['topN'].mode().values[0]
        d_adb['obj'] = obj
        d_list.append(d_adb)
    return d_list


def cfg_results(ada_cfg, X_train, y_train, X_pred, X_idx, y_res, xf_obj, y_col, p_col):
    # best AdaBoost regressor results
    topN = ada_cfg['topN']
    obj = ada_cfg['obj']
    ab_reg = AdaBoostRegressor(n_estimators=int(ada_cfg['adb_estimators']),
                               base_estimator=DecisionTreeRegressor(max_depth=int(ada_cfg['max_depth']), min_samples_split=int(ada_cfg['min_samples_split'])),
                               loss=ada_cfg['loss'],
                               learning_rate=ada_cfg['learning_rate'])

    try:
        ab_reg.fit(X_train, y_train)
    except ValueError as e:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: could not fit: ' + str(e))
        p_ut.save_df(X_train, '~/my_tmp/X_train')
        p_ut.save_df(y_train, '~/my_tmp/y_train')
        return dict()

    y_pred = ab_reg.predict(X_pred)                     # predict (transformed) y_col
    lf_out = pd.concat([pd.DataFrame(X_idx, columns=['cfg_idx']), pd.DataFrame(y_pred, columns=[y_col + '_pred'])], axis=1)  # X_train y_train and X_test have same index
    lf_out.drop_duplicates(inplace=True)

    if y_res is not None:
        lf_out = y_res.merge(lf_out, on=['cfg_idx'], how='left')

    l_cfg = lf_out.nsmallest(n=topN, columns=[y_col + '_pred'])                   # can use transformed values because transforms are monotonic increasing
    d_best = {'xform': xf_obj.name, 'topN': topN, 'obj': obj, 'yobj': y_col,
              'yhat': l_cfg[y_col + '_pred'].mean(), 'cfg_idx': [list(l_cfg['cfg_idx'].values)]}
    
    if y_res is not None:
        d_best['y'] = l_cfg[y_col].mean()

    if p_col in l_cfg.columns:
        d_best[p_col] = l_cfg[p_col].mean()
        d_best[p_col + '_max'] = l_cfg[p_col].max()
        d_best.update(ada_cfg)
    return d_best


def cfg_selection_(qtile, y_col, xform,  month_h, obj_list, topN_list, p_col, cutoff_date, df, cpus):
    p = mp.current_process()
    p.daemon = False
    s_ut.my_print('pid: ' + str(os.getpid()) + ' start loop params: y_col: ' + str(y_col) + ' qtile: ' + str(qtile) + ' xform: ' + str(xform) + ' month_h: ' + str(month_h))

    # skip meaningless combinations
    if y_col in ['rank_logistic', 'rank_log']:
        if xform == 'box-cox':
            xform = 'yeo-johnson'
        elif xform == 'logistic':  # logistic on rank_logistic or rank_log makes no sense
            return None
        else:
            pass
    if y_col == 'rank':
        if xform == 'logistic':  # logistic on rank is rank logistic!
            return None

    cutoff_date = pd.to_datetime(cutoff_date)                        # last Sat of the month
    month_start_date = pd.to_datetime(str(cutoff_date.year) + '-' + str(cutoff_date.month) + '-01')  # cutoff date for training
    release_date = month_start_date + pd.DateOffset(months=1)        # fcast release date
    horizon = release_date + pd.DateOffset(months=month_h)           # fcast horizon (12 to 16 weeks counting 4 weeks per month) -horizon the last day of week 16
    df = df[df['cutoff'] <= horizon].copy()
    if y_col == p_col:
        y_col += '_'
        df[y_col] = df[p_col].copy()

    d_list = list()
    for lang, lf in df.groupby('language'):
        if lf.isnull().sum().sum() > 0:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR')
        lf.reset_index(inplace=True, drop=True)

        # if lf['cutoff'].max() >= horizon:
        #     test_date = horizon
        #     train_date = cutoff_date
        # else:
        test_date = release_date
        train_date = test_date - pd.DateOffset(months=month_h)  # at most go back by month_h

        # drop bad train cfgs
        rf = lf[lf['cutoff'] <= train_date][['cfg_idx', y_col, p_col]].drop_duplicates()
        gf = rf.groupby('cfg_idx').apply(lambda x:
                                         pd.DataFrame(
                                             {'avg_' + y_col: [x[y_col].mean()], 'avg_' + p_col: [x[p_col].mean()]}
                                         )).reset_index(level=0).reset_index(drop=True)

        if len(rf) <= 1:
            s_ut.my_print('pid: ' + str(os.getpid()) +
                          ' WARNING: No data for loop params: y_col: ' + str(y_col) + ' qtile: ' + str(qtile) + ' xform: ' + str(xform) + ' month_h: ' + str(month_h))
            continue

        good_cfgs = list(gf[(gf['avg_' + y_col] <= gf['avg_' + y_col].quantile(qtile))]['cfg_idx'].values)   # the higher y_col, the worst the cfg
        lf = lf[lf['cfg_idx'].isin(good_cfgs)].copy()
        lf = lf[~lf.isin([np.nan, np.inf, -np.inf, None])].dropna()  # these values do not work with AdaBoost fit
        lf.drop_duplicates(inplace=True)
        lf.reset_index(inplace=True, drop=True)
        if len(lf) <= 1:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: No data for ' + lang)
            continue

        # encode categorical features
        cat_cols = ['y_mode', 'w_mode', 'r_mode', 'xform']
        num_cols = ['h_mode', 'do_res', 'training', 'changepoint_range', 'n_ds']
        cat_dict_ = {c: [x for x in lf[c].unique() if pd.notna(x)] for c in cat_cols}
        cat_dict = {k: v for k, v in cat_dict_.items() if len(v) > 1}
        in_cols = list(cat_dict.keys())
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        sep = '<::>'
        enc_cols = [c + sep + str(vx) for c, v in cat_dict.items() for vx in v]
        lf_enc = pd.DataFrame(enc.fit_transform(lf[in_cols].values), columns=enc_cols)
        X = pd.concat([lf_enc, lf[num_cols + ['cutoff', 'cfg_idx']]], axis=1)
        rf_cols = list(lf_enc.columns) + num_cols         # regressor columns

        # transform
        y_in = lf[lf['cutoff'] <= train_date][y_col].copy()
        xf_obj = xf.Transform(xform, None)
        _ = xf_obj.fit(y_in.values)                            # use only pre-cutoff for xform fit
        y_vals = lf[y_col].copy()                              # save original data
        lf[y_col] = xf_obj.transform(lf[y_col].values)         # but transform all the range
        if lf[y_col].isnull().sum() > 0:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: ' + str(xf_obj.name) + ' failed for y_col ' + str(y_col) +
                          ' lambda: ' + str(xf_obj.lmbda) + ' Skipping language ' + str(lang))
            continue

        # find the best ada boost cfg
        X_train = X[X['cutoff'] <= train_date][rf_cols].copy()
        y_train = lf[['cutoff', y_col]][lf['cutoff'] <= train_date][y_col].copy()
        X_test = X[X['cutoff'] == test_date][rf_cols].copy()
        y_test = lf[['cutoff', y_col]][lf['cutoff'] == test_date][y_col].copy()
        y_perf = lf[['cutoff', p_col]][lf['cutoff'] == test_date][p_col].copy()
        # list of dict with best (10) cfg for each topN and obj column list
        ada_list = best_regression_cfg(X_train, y_train, X_test, y_test, y_perf, 10, topN_list, obj_list, used_cpus=cpus)            # list of dict

        # predictions at horizon with best regressor
        # new transforms because the dates are different
        lf[y_col] = y_vals.values                               # first reset original values
        y_in = lf[lf['cutoff'] <= test_date][y_col].copy()
        xf_obj = xf.Transform(xform, None)                      # new transform instance because the dates are different
        _ = xf_obj.fit(y_in.values)                             # use only pre-cutoff for xform fit
        lf[y_col] = xf_obj.transform(lf[y_col].values)
        if lf[y_col].isnull().sum() > 0:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: ' + str(xf_obj.name) + ' failed for best regressor and y_col ' + str(y_col) + '. Skipping language ' + str(lang))
            continue

        # train, predict DF and actuals
        X_train = X[X['cutoff'] <= test_date][rf_cols].copy()
        X_idx = X[X['cutoff'] <= test_date]['cfg_idx'].copy()
        y_train = lf[['cutoff', y_col]][lf['cutoff'] <= test_date][y_col].copy()
        X_pred = X_train.copy()                                    # use only configs we trained with
        X_pred['n_ds'] = X_pred['n_ds'].max() + month_h            # for prediction, shift time column to the prediction date

        # actual results
        if lf['cutoff'].max() < horizon:
            y_res = None
        else:     # fcast error at horizon the fcast error in the 4 weeks before horizon with a "cutoff date" of horizon - 112 days
            h = lf['cutoff'] == horizon
            y_res = lf[h][['cfg_idx', y_col, p_col]].copy()

        # get results for the best AdaBoost regressor for each combination of obj and topN
        for ada_cfg in ada_list:                            # best AdaBoost cfg for each obj func and number of fcast cfg's selected
            d_best = cfg_results(ada_cfg, X_train, y_train, X_pred, X_idx, y_res, xf_obj, y_col, p_col)
            if len(d_best) == 0:
                continue
            d_best['qtile'] = qtile
            d_best['month_h'] = month_h
            d_best['language'] = lang
            d_list.append(d_best)
            s_ut.my_print('pid: ' + str(os.getpid()) +
                          ' xform: ' + str(xform) + ' language: ' + lang + ' topN: ' + str(ada_cfg['topN']) +
                          ' qtile: ' + str(qtile) + ' month_h: ' + str(month_h) + ' ycol: ' + str(y_col) + ' obj: ' + str(d_best['obj']) +
                          # ' yhat: ' + str(d_best['yhat']) + ' y_actual: ' + str(d_best['y']) +
                          ' f_err_avg:' + str(d_best.get(p_col, None)) + ' f_err_max: ' + str(d_best.get(p_col + '_max', None)))
    if len(d_list) > 0:
        f_sel = pd.DataFrame(d_list)
        s_ut.my_print('pid: ' + str(os.getpid()) + ' end loop params: y_col: ' + str(y_col) + ' qtile: ' + str(qtile) + ' xform: ' + str(xform) + ' month_h: ' + str(month_h) )
        return f_sel
    else:
        return None


def update_df(adf, ts_name, cutoff_date):
    adf['ts_name'] = ts_name
    adf['cutoff'] = cutoff_date


def cfg_selection(ts_name, cutoff_date, cfg_cols, p_col):
    # finds the best forecast cfgs for each language
    df = get_cfg_data(ts_name, cfg_cols, p_col)    # read all the cfgs and set the cfg_idx
    if df is None or len(df) == 0:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no data return')
        s_ut.my_print('pid: ' + str(os.getpid()) + ': ERROR')
        sys.exit()

    # prepare df for cfg regression: only numerical vals in the cols
    df['h_mode'] = df['h_mode'].astype(int)
    df['do_res'] = df['do_res'].astype(int)

    # set the cutoff dates as increasing integers for the regression
    s = pd.Series(df['cutoff'].unique())
    s.sort_values(inplace=True)
    z = s.dt.to_period('M').diff().apply(lambda x: 0 if pd.isna(x) else x.n)
    fx = pd.DataFrame({'cutoff': s, 'n_ds': z.cumsum()})
    df = df.merge(fx, on='cutoff', how='left')
    df = set_rank(df, p_col)  # set the rank of each cfg for each language and cutoff_date

    q_list = [0.1, 1.0, 0.25]
    y_list = [p_col,'rank_log', 'rank', 'rank_logistic']
    x_list = ['box-cox', None, 'logistic']
    o_list = ['a_loss', 'w_loss', 'p_loss', 'r_loss']
    n_list = [10, 15, 20]
    m_list = [3, 4]
    cpus = len(q_list) * len(y_list) * len(x_list) * len(m_list)
    arg_list_ = itertools.product(q_list, y_list, x_list, m_list)
    arg_list = [list(x) + [o_list, n_list, p_col, cutoff_date, df.copy(), cpus] for x in arg_list_]
    f_list = s_ut.do_mp(cfg_selection_, arg_list, is_mp=True, cpus=None, do_sigkill=True, verbose=False)
    try:
        f_sel = pd.concat([x for x in f_list if x is not None])
    except ValueError as e:
        s_ut.my_print('ERROR: No data returned: ' + str(e))
        sys.exit()
    f_sel['cfg_idx'] = f_sel['cfg_idx'].apply(lambda x: json.dumps([int(y) for y in x[0]]) if isinstance(x, list) and len(x) == 1 else np.nan)
    f_sel.dropna(subset=['cfg_idx'], inplace=True)
    update_df(f_sel, ts_name, cutoff_date)
    f_sel.drop_duplicates(inplace=True)
    f_sel['cfg_idx'] = f_sel['cfg_idx'].apply(lambda x: json.loads(x) if isinstance(x, str) else np.nan)
    f_sel.dropna(subset=['cfg_idx'], inplace=True)

    out_dir = '~/my_tmp/cfg_sel/'
    p_ut.save_df(f_sel, out_dir + 'cfg_sel_' + ts_name + '_' + cutoff_date)

    # select the best for each language
    f_best = f_sel.groupby('language').apply(lambda x: x[x[p_col] == x[p_col].min()]).reset_index(drop=True)                    # select by p_col: optimal but unknown
    f_best = f_best[['xform', 'topN', 'obj', 'yobj', 'cfg_idx', 'f_err', 'f_err_max', 'qtile', 'language', 'ts_name', 'cutoff']].copy()
    p_ut.save_df(f_best, out_dir + 'cfg_best_' + ts_name + '_' + cutoff_date)

    f_idx = df[['ts_name', 'language', 'cfg_idx', p_col, 'cfg_str'] + cfg_cols].copy()
    f_idx['cutoff'] = cutoff_date
    p_ut.save_df(f_idx, out_dir + 'cfg_idx_' + ts_name + '_' + cutoff_date)

    s_ut.my_print('pid: ' + str(os.getpid()) + ': DONE')


if __name__ == '__main__':
    s_ut.my_print('pid: ' + str(os.getpid()) + ' ' + str(sys.argv))
    ts_name_, cutoff_date_ = sys.argv[-2:]
    cfg_cols_ = ['growth', 'y_mode', 'w_mode', 'r_mode', 'xform', 'h_mode', 'training', 'do_res', 'changepoint_range']
    p_col_ = 'f_err'
    cfg_selection(ts_name_, cutoff_date_, cfg_cols_, p_col_)

# from capacity_planning.utilities import pandas_utils as p_ut
# p_col = 'f_err'
# cutoff_date = '2019-08-31'
# for t in ['inbound', 'outbound']:
#     for x in ['aht', 'vol']:
#         ts_name = 'phone-' + t + '-' + x
#         f_sel = p_ut.read_df('~/my_tmp/cfg_sel/cfg_sel_' + ts_name + '_' + cutoff_date)
#         f_best = f_sel.groupby('language').apply(lambda x: x[x[p_col] == x[p_col].min()]).reset_index(drop=True)
#         f_best = f_best[['xform', 'topN', 'obj', 'yobj', 'cfg_idx', 'f_err', 'f_err_max', 'language', 'ts_name', 'cutoff']].copy()
#         p_ut.save_df(f_best, '~/my_tmp/cfg_sel/cfg_best_' + ts_name + '_' + cutoff_date)

