"""
python lang_fcast.py ts_name cutoff_date avg_func
selects best configs and generates forecast based on these configs
- ts_name: phone-inbound-vol, phone-outbound-vol, phone-inbound-aht, phone-outbound-aht
- cutoff_date is the actual cutoff_date, e.g. last saturday of the month. It does not get adjusted by the code.
- avg_func: a, g, h, q, m
- save out put to table sup.fct_ds_interaction_based_forecasts
- the default values of n_cfg,  min_cnt and avg_func must be set up in the code
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
from functools import reduce
import json
import copy

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.forecast.utilities.language import data_prep as dtp
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.data import hql_exec as hql

with s_ut.suppress_stdout_stderr():
    import airpy as ap

DO_MP = True

# ###########################################
# ###########################################
# ###########################################


FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def get_fcast_cfg(ts_name, cutoff_date):
    fdir = '~/my_tmp/cfg_sel/'
    fperf = os.path.expanduser(fdir + 'cfg_sel_' + ts_name + '_' + cutoff_date)
    fidx = os.path.expanduser(fdir + 'cfg_idx_' + ts_name + '_' + cutoff_date)
    df_cfg = p_ut.read_df(fidx)
    p_ut.set_week_start(df_cfg, tcol='ds')  # week_starting patch

    dfp = p_ut.read_df(fperf)
    p_ut.set_week_start(dfp, tcol='ds')  # week_starting patch
    f_list = list()
    for l, f in dfp.groupby('language'):
        tf = f.nsmallest(n=1, columns=['f_err'])
        cfg_list = list(tf.loc[tf.index[0], 'cfg_idx'][0])
        print(l)
        print(cfg_list)
        fi = df_cfg[(df_cfg['language'] == l) & (df_cfg['cfg_idx'].isin(cfg_list))]
        fi.drop('f_err', axis=1, inplace=True)
        fi.drop_duplicates(inplace=True)
        print(fi)
        f_list.append(fi)
    return pd.concat(f_list) if len(f_list) > 0 else None


def main(ts_name, cutoff_date, cfg_cols, to_db=True, df_cfg=None, is_mp=True):   #, is_fcast=True):
    cfg_file = get_fcast_cfg_file()
    with open(os.path.expanduser(cfg_file), 'r') as fp:
        d_cfg = json.load(fp)

    # if is_fcast is False and df_cfg is None:
    #     s_ut.my_print('ERROR: cannot generate ensemble fcasts without fcast configs')
    #     sys.exit()

    perf_df = df_cfg.copy() if df_cfg is not None else get_fcast_cfg(ts_name, cutoff_date)
    # if is_fcast is True and perf_df is None:
    #     s_ut.my_print('ERROR: cannot forecast without fcast configs')
    #     sys.exit()

    if_exists = d_cfg['if_exists']
    upr_horizon, lwr_horizon = d_cfg['upr_horizon_days'], d_cfg['lwr_horizon_days']

    # ##################################
    # ##################################
    # if lang == 'Mandarin_Onshore':
    #     p_df['avg'] = p_df.mean(axis=1)
    #     p_df.sort_values(by='avg', inplace=True)
    #     print('lang: ' + str(lang))
    #     print(p_df.head(1))
    #     cfg_dict[lang] = [p_df.index[0]]
    # ##################################
    # ##################################

    # set up (ts, regressors, ...)
    ts_obj, reg_dict, cfg_dict, _ = dtp.initialize(cfg_file, cutoff_date, ts_name, False, is_mp=is_mp, init_date='2016-01-01')

    # get fcasts
    fcast_list = list()
    ctr = 0
    for l, l_df in ts_obj.df_dict.items():  # by language
        # if l != 'English_NA':
        #     continue
        ctr += 1
        if ctr > 3:
            print(99999999999999999999)
            print('DEBUG @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            break
        if l_df is None or len(l_df) == 0:
            s_ut.my_print('WARNING: no data for language: ' + str(l))
            continue
        pl = perf_df[perf_df['language'] == l].copy()
        if len(pl) == 0:
            s_ut.my_print('WARNING: no fcast cfg data for language: ' + str(l))
            continue
        ql = pl[cfg_cols].copy()
        cfgs = ql.to_dict(orient='records')

        print('\n')
        s_ut.my_print('********************************* starting forecast for language: ' + str(l))
        _ = [print('++ config: ' + str(d)) for d in cfgs]

        arg_list = dtp.prophet_prep(ts_obj, l, reg_dict.get(l, None), cfg_dict, upr_horizon, lwr_horizon, cfgs, False)
        f_list = s_ut.do_mp(dtp.tf, arg_list, is_mp=is_mp, cpus=None, do_sigkill=True)

        s_ut.my_print('********************************* actual forecasts completed for language: ' + str(l) + ': ' + str(len(f_list))) # + ' is_fcast: ' + str(is_fcast))
        if len(f_list) > 0:
            fl = cfg_fcast(f_list, pl, cfg_cols)  # if is_fcast is False else actual_fcast(f_list, pl, avg_func, cutoff_date)
            fl['language'] = l
            fcast_list.append(fl)
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no forecast DF for ' + str(l))

    table = 'sup.fct_ds_interaction_based_forecasts'
    if len(fcast_list) > 0:
        df_out = pd.concat(fcast_list, axis=0)
        file_out = to_table(to_db, table, pd.to_datetime(cutoff_date), ts_name, if_exists, df_out)
    else:
        s_ut.my_print('ERROR: no forecast data generated')
        file_out = None
    return file_out


# def actual_fcast(f_list, pl, avg_func, cutoff_date):
    # generate the final fcast by avg in the ensemble
    # g_list = [f[f['ds'] > cutoff_date][['ds', 'yhat']].copy() for f in f_list]
    # mf = reduce(lambda x, y: x.merge(y, on='ds', how='inner'), g_list)
    # mf.set_index('ds', inplace=True)
    # mf['y'] = set_fcast(mf, avg_func)
    # mf.drop([c for c in mf.columns if 'yhat' in c], axis=1, inplace=True)
    # mf['fcast_count'] = len(pl)
    # mf['op'] = avg_func
    # mf.reset_index(inplace=True)
    # return mf


def cfg_fcast(f_list, pl, cfg_cols):
    # attach fcast_cfg performance data (err, str, cnt)
    all_f = pd.concat(f_list, axis=0)
    print(all_f.head())
    print(pl.head())
    _ = [fx.drop(c, axis=1, inplace=True) for fx in [all_f, pl] for c in ['upr_horizon', 'lwr_horizon'] if c in fx.columns]
    return all_f.merge(pl, on=cfg_cols, how='left')


# def set_fcast(mf, avg_func):
#     if avg_func == 'm':
#         return mf.median(axis=1)  # vol, aht
#     elif avg_func == 'a':
#         return mf.mean(axis=1)  # vol, aht
#     elif avg_func == 'g':
#         return mf.apply(lambda x: np.log(x), axis=1).mean(axis=1).apply(lambda x: np.exp(x))
#     elif avg_func == 'h':
#         return mf.apply(lambda x: 1.0 / x, axis=1).mean(axis=1).apply(lambda x: 1.0 / x)
#     elif avg_func == 'q':
#         return mf.apply(lambda x: x ** 2, axis=1).mean(axis=1).apply(lambda x: np.sqrt(x))
#     else:
#         s_ut.my_print('invalid avg func: ' + str(avg_func))
#         return None


def get_fcast_cfg_file():
    this_file = os.path.basename(__file__)
    cfg_dir = '/'.join(FILE_PATH.split('/')[:-1])
    return os.path.join(cfg_dir, 'config/' + this_file[:-3] + '_cfg.json')


def to_table(to_db, table, cutoff_date, ts_name, if_exists, df_out):
    cu_dt = str(cutoff_date.date())
    df_out['cutoff'] = cutoff_date
    df_out['ts_name'] = ts_name
    file_out = p_ut.save_df(df_out, '~/my_tmp/fcast_df_' + cu_dt + '_' + ts_name)
    if to_db is True:
        partition = {'cutoff': cu_dt, 'ts_name': ts_name}
        df_out['ds'] = df_out['ds'].dt.date.astype(str)
        df_out.drop(['cutoff', 'ts_name'], axis=1, inplace=True)
        s_ut.my_print('Loading data to ' + table + ' for partition: ' + str(partition))
        try:               # presto does not work with a partition argument
            ap.hive.push(df_out, table=table, if_exists=if_exists, partition=partition,
                         table_props={'abb_retention_days': '-1', 'abb_retention_days_reason': 'fact table. No pii'})
        except:
            s_ut.my_print('ERROR: push to ' + table + ' failed for partition: ' + str(partition))
            sys.exit()
    return file_out


if __name__ == '__main__':
    print(sys.argv)
    ts_name_, cutoff_date_ = sys.argv[-2:]
    to_db_ = True
    cfg_cols_ = ['growth', 'y_mode', 'w_mode', 'r_mode', 'xform', 'h_mode', 'training', 'do_res', 'changepoint_range']
    _ = main(ts_name_, cutoff_date_, cfg_cols_,  to_db=to_db_)
    print('DONE')
