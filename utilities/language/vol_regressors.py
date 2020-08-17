"""
prepare volume regressors
"""
import pandas as pd
import numpy as np
import os
import sys

from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.forecast.utilities import one_forecast as one_fcast
from capacity_planning.forecast.utilities.language import data_processing as d_proc
from capacity_planning.forecast.utilities.language import regressors as regressors


def prepare_regressors(data_cfg, _cfg, d_cfg, cutoff_date, fcast_days, int_type, init_date='2016-01-01'):
    s_ut.my_print('************* reading regressors ********************')
    reg_cfg = data_cfg.get('regressors', None)
    if reg_cfg is None:
        return None

    init_date = pd.to_datetime(init_date)
    arg_list = [[rname, rcfg, cutoff_date, fcast_days, int_type, init_date] for rname, rcfg in reg_cfg.items()]
    obj_list = s_ut.do_mp(prepare_regs, arg_list, is_mp=True, cpus=None, do_sigkill=True)   # returns the list of regressor obj's
    reg_fdf = regressors.Regressor.merge_regressors(obj_list)                               # merge all regressors in a single DF
    if reg_fdf is None:
        s_ut.my_print('WARNING: no regressors available')
    return reg_fdf


def prepare_regs(r_name, rcfg, cutoff_date, fcast_days, int_type, init_date):
    s_ut.my_print('pid: ' + str(os.getpid()) + ' preparing regressor ' + str(r_name))
    in_file, r_col_dict, key_cols = rcfg['data_path'], rcfg['r_col'], rcfg.get('key_cols', None)

    # regressors: set deterministic indicators
    if r_name == 'peak':  # peak season indicator. No clean up, imputation or forecast
        r_col = list(r_col_dict.keys())[0]  # peaks
        df = pd.DataFrame({'ds': pd.date_range(start=init_date, end=cutoff_date + pd.to_timedelta(fcast_days, unit='D'), freq='D')})
        df[r_col] = df['ds'].apply(lambda x: 1 if x.month_name() in ['July', 'August'] else 0)
        return regressors.IndicatorRegressor('peak', 'peak', 'ds', init_date, cutoff_date, ['July', 'August'], fcast_days)
    else:    # other regressors
        r_file = d_proc.get_data_file(rcfg['data_path'], cutoff_date)
        if r_file is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: date ' + str(cutoff_date.date()) + ' has no data for regressor ' + r_name)
            return None
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' found data file for regressor ' + r_name + ' and date ' + str(cutoff_date.date()) + ': ' + r_file)
            rdf = p_ut.read_df(r_file)
            p_ut.set_week_start(rdf, tcol='ds')  # week_starting patch
            rdf = rdf[(rdf['ds'] >= pd.to_datetime(init_date)) & (rdf['ds'] <= cutoff_date)].copy()
            if rdf['ds'].max() < pd.to_datetime(cutoff_date):
                s_ut.my_print('WARNING: ' + r_name + ' max date (' + str(rdf['ds'].max().date()) + ') is smaller than cutoff date (' + str(cutoff_date.date()) + ')')
            elif len(rdf) > 0:
                if 'interaction_type' in rdf.columns:
                    rdf = rdf[rdf['interaction_type'] == int_type].copy()
                if r_name == 'contact-rate':
                    if rdf['contact_rate'].min() == 0.0:  # would mean no inbound tickets
                        zmin = rdf[rdf['contact_rate'] > 0.0]['contact_rate'].min() / 10.0
                        rdf['contact_rate'].replace(0.0, zmin, inplace=True)
                    return regressors.Regressor('contact-rate', 'contact_rate', 'ds', rdf[['ds', 'language', 'contact_rate']], rcfg, init_date, cutoff_date, fcast_days)
                elif r_name == 'tenure':
                    rdf = rdf.groupby(['ds', 'language']).agg({'tenure_days': np.sum}).reset_index()
                    return regressors.Regressor('tenure', 'tenure_days', 'ds', rdf, rcfg, init_date, cutoff_date, fcast_days)
                elif r_name == 'bookings' or r_name == 'checkins':
                    return regressors.Regressor(r_name, r_name[:-1] + '_count', 'ds', rdf, rcfg, init_date, cutoff_date, fcast_days)
                else:
                    s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: unknown regressor: ' + str(r_name))
                    return None


