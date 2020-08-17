"""
prepare volume regressors
"""
import pandas as pd
import numpy as np
import os
import sys
from functools import reduce

from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.forecast.utilities import one_forecast as one_fcast
from capacity_planning.forecast.utilities.language import data_processing as d_proc
from capacity_planning.forecast.utilities.language import regressors as regressors


def prepare_regressors(data_cfg, _cfg, d_cfg, cutoff_date, fcast_days, init_date='2016-01-01'):
    s_ut.my_print('************* reading regressors ********************')
    reg_cfg = data_cfg.get('regressors', None)
    if reg_cfg is None:
        return None

    arg_list = [[rname, rcfg, cutoff_date, fcast_days, init_date] for rname, rcfg in reg_cfg.items()]
    rf_list = s_ut.do_mp(prepare_regs, arg_list, is_mp=True, cpus=None, do_sigkill=True)
    arg_list, rcol_list = fcast_prep(rf_list, reg_cfg, cutoff_date, fcast_days, pd.to_datetime(init_date))
    r_list = s_ut.do_mp(fcast_regressors, arg_list, is_mp=True, cpus=None, do_sigkill=True)
    r_list = list(filter(lambda x: x is not None, r_list))         # drop all Nones if any
    reg_fdf = merge_regressors(r_list, rcol_list)                  # merge all regressors in a single DF
    fcast_date = cutoff_date + pd.to_timedelta(fcast_days, unit='D')
    if reg_fdf is not None:
        p_ut.save_df(reg_fdf, '~/my_tmp/reg_df')
        s_ut.my_print('final predicted regressors: fcast date: ' + str(fcast_date.date()) +
                      ' cutoff rate: ' + str(cutoff_date.date()) +
                      ' fcast_days: ' + str(fcast_days) +
                      ' gap: ' + str(max([reg_fdf[reg_fdf['language'] == l]['ds'].diff().dt.days.max() for l in reg_fdf['language'].unique()])) +
                      ' nulls: ' + str(sum([reg_fdf[c].isnull().sum() for c in rcol_list])))
    else:
        s_ut.my_print('WARNING: no regressors available')
    return reg_fdf


def prepare_regs(r_name, rcfg, cutoff_date, fcast_days, init_date):
    s_ut.my_print('pid: ' + str(os.getpid()) + ' preparing regressor ' + str(r_name))
    in_file, r_col_dict, key_cols = rcfg['data_path'], rcfg['r_col'], rcfg.get('key_cols', None)

    # regressors: set deterministic indicators
    if r_name == 'peak':  # peak season indicator. No clean up, imputation or forecast
        r_col = list(r_col_dict.keys())[0]  # peaks
        df = pd.DataFrame({'ds': pd.date_range(start=pd.to_datetime(init_date), end=pd.to_datetime(cutoff_date) + pd.to_timedelta(fcast_days, unit='D'), freq='D')})
        df[r_col] = df['ds'].apply(lambda x: 1 if x.month_name() in ['July', 'August'] else 0)
        regressors.IndicatorRegressor('peak', 'peak', 'ds', init_date, cutoff_date, ['July', 'August'], fcast_days, dim_cols=None)
        return df

    # other regressors: clean up, imputation and forecast (later)
    r_file = d_proc.get_data_file(rcfg['data_path'], cutoff_date)
    if r_file is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: date ' + str(cutoff_date.date()) + ' has no data for regressor ' + r_name)
        return None
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' found data file for regressor ' + r_name + ' and date ' + str(cutoff_date.date()) + ': ' + r_file)

    rdf = p_ut.read_df(r_file)
    p_ut.set_week_start(rdf, tcol='ds')  # week_starting patch
    rdf = rdf[(rdf['ds'] >= pd.to_datetime(init_date)) & (rdf['ds'] <= pd.to_datetime(cutoff_date))].copy()

    if key_cols is not None:  # get only relevant data
        for c, v in key_cols.items():
            rdf = rdf[rdf[c] == v]

    rdf['ceiling'] = rcfg.get('ceiling', 1)
    rdf['floor'] = rcfg.get('floor', 0)

    if r_name == 'contact-rate':
        if len(rdf) > 0:
            dim_cols = 'language' if 'language' in rdf.columns else None
            regressors.Regressor('contact-rate', 'contact_rate', 'ds', rdf, rcfg, init_date, cutoff_date, fcast_days, dim_cols=dim_cols)
        if 'language' in rdf.columns:
            return rdf[['ds', 'language', 'contact_rate', 'ceiling', 'floor']]
        else:
            return rdf[['ds', 'contact_rate', 'ceiling', 'floor']]
    elif r_name == 'tenure':
        if len(rdf) > 0:
            regressors.Regressor('tenure', 'tenure_days', 'ds', rdf, rcfg, init_date, cutoff_date, fcast_days, dim_cols=['language'])
        return rdf[['ds', 'language',  'tenure_days']]
    elif r_name == 'bookings' or r_name == 'checkins':
        if len(rdf) > 0:
            regressors.Regressor(r_name, r_name[:-1] + '_count', 'ds', rdf, rcfg, init_date, cutoff_date, fcast_days, dim_cols=['language'])
        return rdf
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: unknown regressor: ' + str(r_name))
        return None


def merge_regressors(r_list, rcol_list):
    # put all regressors in a single DF
    l_list = [f for f in r_list if f.loc[f.index[0], 'language'] != 'NULL']
    cl_list = list()
    for c in rcol_list:
        v = [f for f in l_list if c in f.columns]
        if len(v) > 0:
            cl_list.append(pd.concat(v, axis=0))
    l_rdf = reduce(lambda x, y: x.merge(y, on=['ds', 'language'], how='inner'), cl_list) if len(cl_list) > 0 else None

    n_list = [f for f in r_list if f.loc[f.index[0], 'language'] == 'NULL']
    _ = [f.drop('language', axis=1, inplace=True) for f in n_list]
    cn_list = list()
    for c in rcol_list:
        v = [f for f in n_list if c in f.columns]
        if len(v) > 0:
            cn_list.append(pd.concat(v, axis=0))
    n_rdf = reduce(lambda x, y: x.merge(y, on='ds', how='inner'), cn_list) if len(cn_list) > 0 else None

    reg_fdf = n_rdf if l_rdf is None else (l_rdf if n_rdf is None else l_rdf.merge(n_rdf, on='ds', how='left'))
    return reg_fdf


def reg_gap_check(reg_df_list_, cutoff_date, init_date, fcast_days):     # check for gaps
    reg_df_list = list()
    for r in reg_df_list_:
        for l in r.language.unique():
            rl = r[r['language'] == l].copy()
            pdict = dict()
            pdict['language'] = l
            for c in ['ceiling', 'floor']:
                if c in rl.columns:
                    pdict[c] = rl.loc[rl.index[0], c]
                    rl.drop(c, axis=1, inplace=True)
            c = [c_ for c_ in rl.columns if c_ != 'ds' and c_ != 'language'][0]
            rl = d_proc.data_check(rl[['ds', c]].copy(), c, 'ds', cutoff_date, init_date, max_int=5, name=l)
            if rl is not None:  # add back language, ceiling and floor
                for k, v in pdict.items():
                    rl[k] = v
                reg_df_list.append(rl)
            else:
                s_ut.my_print('WARNING: regressor ' + str(c) + ' language: ' + l + ' failed data check')
    return reg_df_list


def fcast_prep(rf_list, reg_cfg, cutoff_date, fcast_days, init_date):         # set the arg_list for regressor forecast
    reg_df_list_, reg_df_list, r_cols = list(), list(), list()
    for f in rf_list:
        if f is not None:
            if 'language' not in f.columns:
                f['language'] = 'NULL'
            if len(f) > 0:
                reg_df_list_.append(f.reset_index(drop=True))  # all reg_df have a language column
            else:
                s_ut.my_print('WARNING: regressor ' + str(f.columns) + ' has not data')

    reg_df_list = reg_gap_check(reg_df_list_, cutoff_date, init_date, fcast_days)     # check for gaps
    do_fcast = {list(v['r_col'].keys())[0]: v['do_fcast'] for v in reg_cfg.values()}
    arg_list, rcol_list = list(), list()
    for f in reg_df_list:  # list of regs by reg col and language
        f_cols = [c for c in f.columns if c != 'language']
        rcol = [c for c in f_cols if c not in ['ds', 'ceiling', 'floor']]
        rcol_list += rcol
        lang = f.loc[f.index[0], 'language']
        if len(f) > 0 and len(rcol) == 1:
            arg_list.append([lang, f, 'ds', rcol[0], cutoff_date, fcast_days, do_fcast])
        else:
            s_ut.my_print('WARNING::empty regressor or too many regression columns: ' + str(rcol) + ' language" ' + str(lang) + ' len: ' + str(len(f)))
    rcol_list = list(set(rcol_list))
    return arg_list, rcol_list


def l_group(a_df, count_col):
    r = a_df.loc[a_df.index[0], ]
    lang = r['guest_language']
    b = (a_df['host_language'] == lang).astype(int)
    cnt = (b * a_df[count_col] + 0.5 * (1 - b) * a_df[count_col]).sum()
    ceil = r['ceiling'] if 'ceiling' in a_df.columns else np.nan
    floor = r['floor'] if 'floor' in a_df.columns else np.nan
    f = pd.DataFrame({'ds': [r['ds']], 'language': [lang], count_col: [cnt], 'ceiling': [ceil], 'floor': [floor]})
    _ = [f.drop(c, axis=1, inplace=True) for c in ['ceiling', 'floor'] if f[c].isnull().sum() == len(f)]  # drop floor and ceiling if null
    return f


def fcast_regressors(lang, rf_, tcol, ycol, i_dt, f_days, do_fcast):
    # all regressors are forecasted using the default cfg
    s_ut.my_print('pid: ' + str(os.getpid()) + ' starting regressor fcast for language ' + str(lang) + ' cols: ' + str(list(rf_.columns)))
    if do_fcast.get(ycol, True) is True:
        rg = rf_[rf_['ds'] <= i_dt].copy()
        r = None
        if len(rg) > 0:
            if ycol == 'contact_rate':
                if rg[ycol].min() == 0.0:  # would mean no inbound tickets
                    zmin = rg[rg[ycol] > 0.0][ycol].min() / 10.0
                    rg[ycol].replace(0.0, zmin, inplace=True)
                fcfg = {'xform': 'logistic'}
                pdict = dict()
                for c in ['ceiling', 'floor']:
                    if c in rg.columns:
                        pdict[c] = rg.loc[rg.index[0], c]
            else:
                fcfg = None
                pdict = dict()
            rg.rename(columns={tcol: 'ds', ycol: 'y'}, inplace=True)
            f_obj = one_fcast.OneForecast(rg[['ds', 'y']], None, pdict, fcfg, f_days, time_scale='D', verbose=False)
            r = f_obj.forecast()
        if r is not None:
            r.rename(columns={'yhat': ycol}, inplace=True)
            r['language'] = lang
            if any(np.isinf(r[ycol])):
                s_ut.my_print('ERROR in reg forecast for ' + ycol + ' and language ' + str(lang))
                p_ut.save_df(rg, '~/my_tmp/rg_'+lang + '_' + ycol)
                print(r.describe())
                sys.exit()
    else:  # nothing to fcast
        if rf_[tcol].max() < i_dt + pd.to_timedelta(f_days, unit='D'):
            print('WARNING: no forecast set but missing forecast values for ' + str(ycol) + ' Ignoring')
            r = None
        else:
            r = rf_[rf_[tcol] <= i_dt + pd.to_timedelta(f_days, unit='D')].copy()
            r['language'] = lang
    return r


