"""

"""
import pandas as pd
import numpy as np
import os
import sys
import json
import copy


from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.forecast.utilities import one_forecast as one_fcast

# pd.set_option('display.max_rows', 100)
# pd.set_option('precision', 4)
# pd.set_option('display.width', 320)
# pd.set_option('display.max_columns', 35)


def prophet_prep(ts_obj, lg, reg_df_l, d_cfg, upr_horizon, lwr_horizon, fcast_cfg_list, is_train, time_scale='D'):
    d_cfg['prophet_dict']['holidays'] = ts_obj.hf_dict[lg]
    d_cfg['prophet_dict']['floor'] = ts_obj.floor
    d_cfg['prophet_dict']['ceiling'] = ts_obj.ceiling
    t_df = ts_obj.df_dict[lg]

    if reg_df_l is not None:
        if 'language' in reg_df_l.columns:
            reg_df_l.drop('language', axis=1, inplace=True)
    else:               # no regressors: fix fcast cfg
        c_list = list()
        for c in fcast_cfg_list:
            if c is not None:
                c['r_mode'] = None
                c_list.append(json.dumps(c))
        fcast_cfg_list = [json.loads(c) for c in list(set(c_list))]  # remove dups

    if len(fcast_cfg_list) != 1:
        s_ut.my_print('ERROR: only a single cfg is supported here')
        sys.exit()
    return copy.deepcopy(fcast_cfg_list[0]), t_df, reg_df_l, d_cfg['prophet_dict'], ts_obj.cutoff_date, upr_horizon, lwr_horizon, is_train, time_scale


def tf(f_cfg, tdf, reg_df, p_cfg, cutoff_date, upr_horizon, lwr_horizon, is_train, time_scale):             # training forecasts
    if time_scale not in ['D', 'W']:
        s_ut.my_print('ERROR: invalid time scale: ' + str(time_scale))
        return None
    trn = len(tdf) if f_cfg is None else f_cfg.pop('training')
    ret = do_fcast(f_cfg, tdf, reg_df, p_cfg, cutoff_date, lwr_horizon, upr_horizon, trn, is_train, time_scale=time_scale)        # ret = [r0, r1] and ri = (df, cfg)
    if ret is None:
        return None
    else:  # should be a list of DFs
        f_list = [fcast_(ret[i]) for i in range(len(ret))]
        f_list = [f for f in f_list if f is not None]
        return f_list


def fcast_(r):
    f_df, f_cfg = r
    if f_df is None:
        return None
    else:
        f_out = f_df.copy()
        s_ut.my_print('pid: ' + str(os.getpid()) + ': ' + ' completed fcast for ' + str(f_cfg))
        for c, v in f_cfg.items():
            f_out[c] = v
        return f_out


def do_fcast(fcast_cfg, tdf, reg_df, p_cfg, cutoff_date, lwr_horizon, upr_horizon, trn, is_train, time_scale='D'):
    cu = cutoff_date - pd.to_timedelta((1 + cutoff_date.weekday()) % 7, unit='D') if time_scale == 'W' else cutoff_date  # convert to Sunday (week starting)
    t_start = cu - pd.to_timedelta(trn, unit=time_scale)              # fcast train start date: add 2 to make sure the full trn weeks/days are in
    t_end = cu + pd.to_timedelta(upr_horizon, unit=time_scale)

    if t_start < tdf['ds'].min():
        s_ut.my_print('pid: ' + str(os.getpid()) + ' xxxxxxxxxxxxxx No forecast for cfg due to date ranges (start date = '
                      + str(t_start.date()) + ', min ds: ' + str(tdf['ds'].min().date()) + '): ' + str(fcast_cfg))
        return None

    if is_train is True and tdf['ds'].max() < t_end:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' xxxxxxxxxxxxxxx No forecast for cfg due to date ranges (end date = '
                      + str(t_end.date()) + ', min ds: ' + str(tdf['ds'].max().date()) + '): ' + str(fcast_cfg))
        return None

    if fcast_cfg['xform'] == 'logistic':
        if isinstance(p_cfg.get('ceiling', None), type(None)) or isinstance(p_cfg.get('floor', None), type(None)):
            return None
        elif p_cfg.get('ceiling', 1.0) < p_cfg.get('floor', 0):
            return None
        else:
            pass

    a_df = tdf[(tdf['ds'] <= cutoff_date) & (tdf['ds'] >= t_start)][['ds', 'y']].copy()    # use only data prior to cutoff date for forecast

    # final fcast cfg adjustments
    if fcast_cfg['r_mode'] is None:
        in_reg_df = None
    else:
        if reg_df['ds'].min() > t_start or reg_df['ds'].max() < t_end:        # not enough regressor data
            s_ut.my_print('pid: ' + str(os.getpid()) + ' xxxxxxxxxxxxxxx No forecast for cfg due to missing regressor data for  ' + str(fcast_cfg))
            return None
        else:
            in_reg_df = reg_df[reg_df['ds'] >= t_start].copy()                                            # must carry the regressors fully forecasted
    if fcast_cfg['xform'] == 'box-cox' and a_df['y'].min() < 0.0:
        fcast_cfg['xform'] = 'yeo-johnson'
    p_cfg['n_changepoints'] = 25                                                                          # prophet default

    s_ut.my_print('pid: ' + str(os.getpid()) + ' ************* starting forecast for cfg: ' + str(fcast_cfg))
    fcast_obj = one_fcast.OneForecast(a_df, in_reg_df, p_cfg, fcast_cfg, upr_horizon, time_scale=time_scale, verbose=False)
    f_ret = fcast_obj.forecast()
    if f_ret is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' xxxxxxxxxxxxxxx No forecast for cfg due to OneForecast failure: ' + str(fcast_cfg))
        return None
    else:
        n_res, y_res = f_ret
        fn = n_res.copy() if n_res is not None else None
        fy = y_res.copy() if y_res is not None else None

        # set fn
        cfg_n = copy.deepcopy(fcast_obj.f_cfg)
        cfg_n['do_res'] = False                 # f_cfg has always do_res = True
        cfg_n['training'] = trn
        cfg_n['upr_horizon'] = np.nan if upr_horizon is None else float(upr_horizon)
        cfg_n['lwr_horizon'] = np.nan if lwr_horizon is None else float(lwr_horizon)

        # set fy
        cfg_y = copy.deepcopy(fcast_obj.f_cfg)  # f_cfg has always do_res = True
        cfg_y['training'] = trn
        cfg_y['upr_horizon'] = np.nan if upr_horizon is None else float(upr_horizon)
        cfg_y['lwr_horizon'] = np.nan if lwr_horizon is None else float(lwr_horizon)
        return (fn, cfg_n), (fy, cfg_y)

