"""
$ python lang_ticket_forecast ts_name run_date to_table
run_date: a date. will be reset to the sat prior, i.e. max(saturdays: saturday <= run_date). If missing set to today
ts_name: time series name. Mandatory
to_table: default to 0
generates a bunch of forecasts using all combinations from FCAST_DICT
fcasts get saved for all cfgs with the cfg
if regressors are needed for the time series forecast, it uses (already created) ens fcasts from the regressors
"""
import os
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
import pandas as pd
import numpy as np
import copy
import multiprocessing as mp
import itertools

from capacity_planning.forecast.utilities.language import regressors as regs
from capacity_planning.forecast.utilities.language import time_series as ts
from capacity_planning.forecast.utilities.language import data_prep as dtp
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import time_utils as tm_ut
from capacity_planning.data import hql_exec as hql
from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.utilities import xforms as xf

# pd.set_option('display.max_rows', 100)
# pd.set_option('precision', 4)
# pd.set_option('display.width', 320)
# pd.set_option('display.max_columns', 35)
#

PROPHET_DICT = {
    'prophet_dict': {
        "changepoint_prior_scale": 0.05,
        "changepoint_range": None,
        "changepoints": None,
        "holidays": None,
        "holidays_prior_scale": 10.0,
        "interval_width": 0.8,
        "mcmc_samples": 0,
        "seasonality_prior_scale": 10.0,
        "uncertainty_samples": 200
    }
}

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CFG_COLS = ['growth', 'y_mode', 'w_mode', 'r_mode', 'xform', 'h_mode', 'training', 'do_res', 'changepoint_range', 'outlier_coef']
is_test = True if platform.system() == 'Darwin' else False

if is_test is False:
    FCAST_DICT = {
        'growth': ['linear'],
        'xform': [None, 'box-cox', 'yeo-johnson'],
        'do_res': [True, False],
        'r_mode': [None, 'additive', 'multiplicative'],
        'changepoint_range': [0.6, 0.8],
        'h_mode': [True, False],
        'training': [],
        'y_mode': [None, 'additive', 'multiplicative'],
        'w_mode': [None]
    }
else:
    FCAST_DICT = {
        'growth': ['linear'],
        'xform': ['box-cox'],
        'r_mode': ['multiplicative'],
        'changepoint_range': [0.6],
        'h_mode': [False],
        'do_res': [True],
        'training': [],
        'y_mode': ['additive', None],
        'w_mode': [None]
    }


def main(argv):
    # ###########################
    # parameters
    # ###########################
    time_scale = 'W'  # forecasting time scale reset for daily ticket data
    init_date = pd.to_datetime('2016-01-01')
    froot = '~/my_tmp/fbp/'
    # ###########################
    # ###########################

    print(argv)
    if len(argv) == 2:
        ts_name = argv[-1]
        to_table = False
        run_date = pd.to_datetime('today')
    elif len(argv) == 3:
        ts_name, run_date = argv[-2:]
        try:
            run_date = pd.to_datetime(run_date)
            to_table = False
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, run_date, to_table): ' + str(argv))
            sys.exit()
    elif len(argv) == 4:
        ts_name, run_date, to_table = argv[1:]
        try:
            run_date = pd.to_datetime(run_date)
            to_table = bool(int(to_table))
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, run_date, to_table): ' + str(argv))
            sys.exit()
    else:
        s_ut.my_print('ERROR: invalid arguments (ts_name, run_date, to_table): ' + str(argv))
        sys.exit()

    # data cfg
    cutoff_date = tm_ut.get_last_sat(run_date)  # set to last saturday before run_date or the run_date if a saturday
    ts_cfg, cols = dp.ts_setup(ts_name, cutoff_date, init_date, time_scale)
    FCAST_DICT['outlier_coef'] = ts_cfg.get('outlier_coef', [3.0])

    fcast_days = ts_cfg.get('fcast_days', None)
    if fcast_days is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR" fcast_days must be specified in data_cfg')
        sys.exit()
    else:
        fcast_date = cutoff_date + pd.to_timedelta(fcast_days, unit='D')

    if time_scale == 'W':
        fcast_date = fcast_date - pd.to_timedelta(1 + fcast_date.weekday(), unit='D')  # set to week starting Sunday
        cu = cutoff_date - pd.to_timedelta(1 + cutoff_date.weekday(), unit='D')  # set to week starting Sunday
        fcast_days = (fcast_date - cu).days  # multiple of 7
        upr_horizon = int(fcast_days / 7)  # in time scale units
    elif time_scale == 'D':
        upr_horizon = int(fcast_days)  # in time scale units
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' invalid time scale: ' + str(time_scale))
        sys.exit()

    s_ut.my_print('pid: ' + str(os.getpid()) + ' ------------------------ start language forecast for ' + str(ts_name) + ' from cutoff date '
                  + str(cutoff_date.date()) + ' (excluded) to forecast date ' + str(fcast_date.date()) + '  (included) -----------------------')

    # get actuals
    actuals_df = dp.ts_actuals(ts_name, ts_cfg, cols)  # may have data past cutoff for accuracy checking
    if actuals_df['ds'].max() < cutoff_date:
        s_ut.my_print('ERROR: no actuals available for forecast from cutoff date: ' + str(cutoff_date.date()))
        sys.exit()
    f_actuals_df = actuals_df[actuals_df['ds'] <= cutoff_date].copy()  # actuals for forecast: only use up to cutoff date

    # adjust FCAST_DICT
    if len(FCAST_DICT['do_res']) == 2:  # True, False
        FCAST_DICT['do_res'] = [True]   # MUST overwrite: the False care is always included and otherwise we double count.
    if len(ts_cfg.get('regressors', list())) == 0:
        FCAST_DICT['r_mode'] = [None]
        reg_dict = dict()
    else:
        reg_dict = regs.ens_fcast(ts_name, ts_cfg['regressors'], cutoff_date, time_scale, fcast_days, init_date, f_actuals_df)     # stored by cutoff date on last Sat of the month

    # update init_date
    init_date = max([f_actuals_df['ds'].min()] + [f['ds'].min() for f in reg_dict.values()])
    f_actuals_df = f_actuals_df[f_actuals_df['ds'] >= init_date].copy()
    reg_dict = {lx: f[f['ds'] >= init_date].copy() for lx, f in reg_dict.items()}
    ts_cfg['init_date'] = init_date

    # set the list of fcast cfgs
    tlist = get_f_cfg(FCAST_DICT, cutoff_date, init_date, time_scale)  # list of fcast cfg's
    fix_pars = [f_actuals_df, ts_name, reg_dict, fcast_date, cutoff_date, ts_cfg, time_scale, upr_horizon]
    arg_list = [fix_pars + [tlist[ix]] for ix in range(len(tlist))]  # 2 fcasts are done per input cfg (do_res = true and do_res = false)
    n_fcfg = 2 * len(arg_list)
    s_ut.my_print('pid: ' + str(os.getpid()) + ' ++++++++ there are ' + str(n_fcfg) + ' fcast configs per language **********')

    # ###############################################################################
    # ###############################################################################
    # ###############################################################################
    if is_test:
        df_list_ = s_ut.do_mp(fcast_lang, arg_list, is_mp=False, cpus=None, do_sigkill=True)
    else:
        df_list_ = s_ut.do_mp(fcast_lang, arg_list, is_mp=True, cpus=None, do_sigkill=True)
    # ###############################################################################
    # ###############################################################################
    # ###############################################################################

    # join all the fcasted data into a flat list
    df_list = [f for f in df_list_ if f is not None]
    if len(df_list) > 0:
        ylist, alist = list(), list()
        for fl in df_list:
            if fl is not None:
                fl = set_cfg(fl.copy(), CFG_COLS)
                ylist.append(fl[['ds', 'language', 'yhat', 'ts_name', 'cutoff', 'dim_cfg', 'fcast_date']].copy())
                alist.append(fl)

        # save basic fcast data
        fcast_df = pd.concat(ylist, axis=0)  # now all the list elements have the same columns
        fcast_df.reset_index(inplace=True, drop=True)

        ok_cfg = fcast_df['dim_cfg'].unique()
        s_ut.my_print('pid: ' + str(os.getpid()) + str(len(ok_cfg)) + ' forecasts cfgs available for ' + str(ts_name) + ' from cutoff date '
                      + str(cutoff_date.date()) + ' (excluded) to forecast date ' + str(fcast_date.date()) + '  (included) -----------------------')
        # fcast_df = fcast_df[fcast_df['dim_cfg'].isin(ok_cfg)].copy()
        fname = froot + 'lang_fcast_'
        p_ut.save_df(fcast_df, fname + ts_name + '_' + str(cutoff_date.date()))
        if to_table is True:
            tab_cols = ['ds', 'language', 'dim_cfg', 'yhat']
            partition = {'cutoff': str(cutoff_date.date()), 'ts_name': ts_cfg['ts_key']}
            ret = hql.to_tble(fcast_df, tab_cols, 'sup.cx_language_forecast', partition)
            if ret != 0:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no forecasts loaded to table for ' + str(ts_cfg['ts_key']) +
                              ' and cutoff date ' + str(cutoff_date.date()))
                sys.exit()

        # save all fcast data  (y_upr, y_lwr, ...)
        all_df = pd.concat(alist, axis=0)       # now all the list elements have the same columns
        all_cols = list(set([c for c in all_df.columns if c not in CFG_COLS]))
        all_df.reset_index(inplace=True, drop=True)
        all_df = all_df[all_cols].copy()
        all_df = all_df[all_df['dim_cfg'].isin(ok_cfg)].copy()
        fname = froot + 'fcast_all_'
        p_ut.save_df(all_df, fname + ts_name + '_' + str(cutoff_date.date()))
        if to_table is True:
            all_df.drop(['cutoff', 'ts_name'], axis=1, inplace=True)
            mf = pd.melt(all_df, id_vars=['ds', 'language', 'dim_cfg'], var_name='key', value_name='value')
            mf.dropna(subset=['value'], inplace=True)
            mf = mf[mf['value'] != 0.0].copy()
            partition = {'cutoff': str(cutoff_date.date()), 'ts_name': ts_cfg['ts_key']}
            ret = hql.to_tble(mf, list(mf.columns), 'sup.cx_language_forecast_detail', partition)
            if ret != 0:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no forecasts loaded to table for ' + str(ts_cfg['ts_key']) + ' and cutoff date ' + str(cutoff_date.date()))
                sys.exit()
        print('DONE')
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no forecasts available for ' + str(ts_cfg['ts_key']) + ' from cutoff date '
                      + str(cutoff_date.date()) + ' (excluded) to forecast date ' + str(fcast_date.date()) + '  (included) -----------------------')


def fcast_lang(actuals_df, ts_name, reg_dict_, fcast_date, cutoff_date, ts_cfg, time_scale, upr_horizon, fcast_cfg):
    p = mp.current_process()
    p.daemon = False

    # adjust init date
    reg_dict = dict() if fcast_cfg.get('r_mode', None) is None else copy.deepcopy(reg_dict_)
    ts_cfg['outlier_coef'] = fcast_cfg.get('outlier_coef', 3.0)
    out_list = list()

    # ###############
    if is_test is True:
        ctr = 0
    # ###############

    for lx, t_df in actuals_df.groupby('language'):
        lang = lx[0] if isinstance(lx, (list, tuple)) else lx
        prefix = 'not-' if 'not-' in lang else ''
        if 'Mandarin' in lang:
            lang = prefix + 'Mandarin'

        # ###############
        if is_test is True:
            ctr += 1
            if ctr > 24:
                break
        # ###############

        s_ut.my_print('pid: ' + str(os.getpid()) + ' \n\n\t\t\t************************ starting language:  ' + lx + ' ****************************')
        init_date = t_df['ds'].min()  # some languages/bu appear late
        ts_obj = ts.TimeSeries(ts_name, t_df, fcast_date, cutoff_date, init_date, ts_cfg, time_scale=time_scale)
        if t_df is not None:
            xform_obj = xf.Transform(ts_obj.pre_process, 100, ceiling=ts_obj.ceiling * t_df[ts_obj.ycol].max(), floor=ts_obj.floor * t_df[ts_obj.ycol].min())
            ts_obj.df_dict[lx]['y'] = xform_obj.fit_transform(ts_obj.df_dict[lx]['y'].values)
            arg_list = dtp.prophet_prep(ts_obj, lang, reg_dict.get(lang, None), PROPHET_DICT, upr_horizon, None, [fcast_cfg], False, time_scale=time_scale)
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ************* forecasts for ' + str(lx)
                          + ' with ' + str(fcast_cfg['training']) + ' train ' + str(time_scale) + ' and ' + str(2 * len(arg_list)) + ' configs')

            f_list = dtp.tf(*arg_list)  # arg_list has always len 1
            if f_list is not None and len(f_list) > 0:         # save the fcast configs
                l_df = bxform_df(xform_obj, f_list)
                if l_df is None:
                    s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no results for ' + str(fcast_cfg) + ' for language ' + str(lx))
                    continue
                l_df.reset_index(inplace=True, drop=True)
                l_df['language'] = lang
                l_df['ts_name'] = ts_name
                l_df['cutoff'] = cutoff_date
                l_df['outlier_coef'] = ts_cfg['outlier_coef']   # will be dropped later
                l_df['interval'] = time_scale
                l_df['fcast_date'] = fcast_date
                out_list.append(l_df)
            else:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no results for ' + str(fcast_cfg))
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no t_df DF for ' + str(lx) + ' and fcast cfg ' + str(fcast_cfg))
    if len(out_list) > 0:
        fout = pd.concat(out_list, axis=0)
        return fout
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no forecasts for fcast cfg: ' + str(fcast_cfg))
        return None


def bxform_df(xform_obj, f_list):
    out_list = list()
    for f in f_list:
        y_var = xform_obj.fcast_var(f[['yhat_lower', 'yhat_upper']].copy(), PROPHET_DICT['prophet_dict']['interval_width'])
        for c in ['yhat', 'yhat_upper', 'yhat_lower',
                  'trend', 'trend_upper', 'trend_lower',
                  'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
                  'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper']:
            f[c] = xform_obj.inverse_transform(f[c].values, y_var, lbl=c)
        for c in ['yhat']:  # , 'yhat_upper', 'yhat_lower']:
            if f[c].isnull().sum() > 0:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: nulls in back-transformed values for ' + str(c) + ' Ignoring this forecast cfg')
                break
        else:
            out_list.append(f)
    s_ut.my_print('pid: ' + str(os.getpid()) + ' there is ' + str(len(out_list)) + ' valid fcast cfgs')
    return None if len(out_list) == 0 else pd.concat(out_list, axis=0)


def set_cfg(df, cfg_cols):
    cfg_cols_ = [c for c in df.columns if c in cfg_cols]
    df['dim_cfg'] = df[cfg_cols].apply(lambda x: x.to_json(), axis=1)
    ncols = [c for c in df.columns if c not in cfg_cols_]
    return df[ncols].copy()


def get_f_cfg(cfg_dict, cutoff_date, init_date, time_scale):
    # set the training windows in multiples of year
    if time_scale == 'W':
        periods = (cutoff_date - init_date).days / 7 - 1
        periods = int(np.ceil(periods))
        nperiods = np.floor(periods / 52.25)
        cfg_dict['training'] = [52 * p + 1 for p in range(1, int(nperiods) + 1)]
    elif time_scale == 'D':
        periods = (cutoff_date - init_date).days - 1
        nperiods = np.floor(periods / 365.25)
        cfg_dict['training'] = [365 * p + 1 for p in range(1, int(nperiods) + 1)]
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: unsupported time scale: ' + str(time_scale))
        sys.exit()

    if time_scale == 'W':
        cfg_dict['w_mode'] = [None]
    v_list = list(cfg_dict.values())
    k_list = list(cfg_dict.keys())
    f_list = list(itertools.product(*v_list))
    d_list = [{k_list[i]: x[i] for i in range(len(k_list))} for x in f_list]
    return d_list


if __name__ == '__main__':
    s_ut.my_print(sys.argv)
    main(sys.argv)

