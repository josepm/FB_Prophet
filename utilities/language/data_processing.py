"""

"""
import os
import sys
import json
import pandas as pd
import numpy as np
import os
import copy


from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.forecast.utilities.language import regressors as regs
from capacity_planning.forecast.utilities.tickets import tkt_utils as t_ut
from capacity_planning.utilities import pandas_utils as p_ut


def get_data_file(f_all, cutoff_date):
    if f_all is None:
        return None
    cutoff_date = pd.to_datetime(cutoff_date)
    vf = f_all.split('/')
    d_dir = os.path.expanduser(os.path.dirname('/'.join(vf)))
    fname = vf[-1]
    dt_max, fn = pd.to_datetime('2016-01-01'), None
    for f in os.listdir(d_dir):
        if fname in f:
            try:
                f_date = pd.to_datetime(f.split('.')[0][-10:])
            except ValueError:
                s_ut.my_print('WARNING invalid date in ' + str(f) + ' cutoff date: ' + str(cutoff_date.date()) + ' d_dir: ' + str(d_dir))
                continue
            if f_date >= cutoff_date:                                                # file is acceptable
                if fn is None:                                                       # always update if fn is None
                    dt_max, fn = f_date, f
                elif dt_max < f_date:       # update to a cleaned version if current is cleaned but older
                    dt_max, fn = f_date, f
                else:
                    pass
    if fn is None:
        return None
    else:
        return os.path.join(d_dir, fn)  # return the latest file, cleaned if possible


def get_all_files(f_root, cutoff_date, post_cutoff):
    # all files before cutoff date included
    # f_all is the file path + file pattern
    # f_root: ~/my_tmp/fbp/lang_fcast_bookings
    if f_root is None:
        return None
    cutoff_date = pd.to_datetime(cutoff_date)
    d_dir = os.path.expanduser(os.path.dirname(f_root))
    f_name = f_root.split('/')[-1]
    lf_out = list()
    for f in os.listdir(d_dir):
        if f_name in f:
            f_base, f_ext = os.path.splitext(f)
            try:
                f_date = pd.to_datetime(f_base.split('_')[-1])
            except ValueError:
                s_ut.my_print('WARNING invalid date in ' + str(f) + ' cutoff date: ' + str(cutoff_date.date()) + ' d_dir: ' + str(d_dir))
                continue
            if post_cutoff is True and f_date >= cutoff_date:  # file is acceptable
                lf_out.append(os.path.join(d_dir, f))
            if post_cutoff is False and f_date <= cutoff_date:  # file is acceptable
                lf_out.append(os.path.join(d_dir, f))
    return lf_out


def try_num(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def data_check(data_in, ycol, tcol, end_date, init_date, max_int=5, name=None, unit='D', ctr=0):
    # fill in data gaps between init_date and end_date
    # max_int: max consecutive interpolations
    # ensures no gaps and no NaN
    # returns None on failure
    if unit == 'D':
        ndays = 1
    elif 'W' in unit:
        ndays = 7
    else:
        s_ut.my_print('invalid unit: ' + str(unit))
        return None

    data_ = data_in.copy()
    sdays = (end_date - init_date).days
    if len(data_[(data_[tcol] >= init_date) & (data_[tcol] <= end_date)]) > sdays / ndays and data_[ycol].isnull().sum() == 0:  # DF at least spans the date range and has no nulls
        return data_
    else:
        # find data gaps
        d_data, gap_sz = de_gap(data_.copy(), max_int, ycol, tcol, ndays)    # d_data: DF with date gaps <= max_int * ndays and new max gap = gap_sz

        # make sure enough data at the end and the start
        retry = False
        max_dt_gap = pd.to_timedelta(max_int * ndays, unit='D')  # in days
        if end_date > d_data[tcol].max() + max_dt_gap:  # gap too large at the end
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: data_check failed for name: ' + str(name) +
                          ' and col: ' + str(ycol) + ' large data gap at end:: ' +
                          ' end_date: ' + str(end_date.date()) + ' max ts date: ' + str(d_data[tcol].max().date()) +
                          ' init_date: ' + str(init_date.date()) + ' min ts date: ' + str(d_data[tcol].min().date()) + ' ctr: ' + str(ctr)
                          )
            end_date = d_data[tcol].max() + max_dt_gap
            retry = True
        if init_date < d_data[tcol].min() - max_dt_gap:  # gap too large at start
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: try data_check recovery for name: ' + str(name) +
                          ' and col: ' + str(ycol) + ' large data gap at start:: ' +
                          ' init_date: ' + str(init_date.date()) + ' min ts date: ' + str(d_data[tcol].min().date()) +
                          ' end_date: ' + str(end_date.date()) + ' max ts date: ' + str(d_data[tcol].max().date()) + ' ctr: ' + str(ctr)
                          )
            init_date = d_data[tcol].min() - max_dt_gap
            retry = True

        # try to recover, using data from the earliest possible time
        if retry == False:
            pass
        else:
            if ctr == 0:
                return data_check(d_data, ycol, tcol, end_date, init_date, max_int=max_int, unit=unit, name='retry-' + str(name), ctr=1)
            else:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: data_check recovery for name: ' + str(name) + ' and col: ' + str(ycol) + ' failed')
                return None

        # interpolate
        dr = pd.DataFrame({tcol: pd.date_range(start=data_[tcol].min(), end=data_[tcol].max(), freq=unit)})
        other_cols = list(set(data_.columns) - {ycol, tcol})
        if len(other_cols) > 0:
            o_data = dr.merge(data_[[tcol] + other_cols], on=tcol, how='left')
            m_data = o_data.merge(d_data[[tcol, ycol]], on=tcol, how='left')
        else:
            m_data = dr.merge(d_data, on=tcol, how='left')
        if m_data[ycol].isnull().sum() > 0:
            m_data[ycol].interpolate(method='linear', limit=max_int, limit_direction='both', inplace=True)
        if m_data[ycol].isnull().sum() > 0:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: data_check interpolation failed for name: ' + str(name) +
                          ' and col: ' + ycol + ' with null values:' + str(m_data[ycol].isnull().sum()))
            return None
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' final data_check results OK for name: ' + str(name) +
                          ' gap: ' + str(m_data[tcol].diff().dt.days.max()) + ' nulls: ' + str(m_data[ycol].isnull().sum()) +
                          ' init_date: ' + str(init_date.date()) + ' min ts date: ' + str(m_data[tcol].min().date()) +
                          ' end_date: ' + str(end_date.date()) + ' max ts date: ' + str(m_data[tcol].max().date())
                          )
            return m_data


def de_gap(a_df, max_gap, ycol, tcol, ndays):             # finds the largest NaT or NaN gap in a_df and returns the DF with gaps <= max_gap and the new max gap
    a_df[ycol] = a_df[ycol].apply(lambda x: try_num(x))   # make sure ycol is numeric
    a_df.dropna(subset=[ycol], inplace=True)
    a_df.sort_values(by=tcol, inplace=True)
    a_df.reset_index(inplace=True, drop=True)
    gap_s = a_df[tcol].diff().dt.days
    b = gap_s > max_gap * ndays                                       # all the entries with gaps > max_gap (measured in days)
    tmin = a_df[b][tcol].max() if b.sum() > 0 else a_df[tcol].min()   # last time with gap > max_gap if any
    g_df = a_df[a_df[tcol] >= tmin].copy()                            # DF with no NaNs from tmin onwards
    new_gap = g_df[tcol].diff().dt.days.max()
    return g_df, new_gap / ndays


def set_ts_freq(df, time_scale, agg_dict, t_col='ds'):
    ts = df[t_col].copy()
    ts = ts.drop_duplicates()
    ts.sort_values(inplace=True)
    ts_freq = pd.infer_freq(ts)
    if ts_freq is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no frequency detected time series: ' + str(ts_freq) + ' time_scale: ' + str(time_scale))
        sys.exit()

    df.rename(columns={t_col: 'ds'}, inplace=True)
    ts_freq = ts_freq.split('-')[0]
    if ts_freq == time_scale:
        return df
    else:
        if 'language' not in df.columns:
            df['language'] = 'NULL'
        if ts_freq == 'D':
            if time_scale == 'W':
                return df.groupby([pd.Grouper(key=t_col, freq='W-SUN'), 'language']).agg({k: v for k, v in agg_dict.items()}).reset_index()  # returns ds_week_starting DS
            elif time_scale == 'M':  # M is month end
                return df.groupby([pd.Grouper(key=t_col, freq='M'), 'language']).agg({k: v for k, v in agg_dict.items()}).reset_index()
            else:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: frequency mix not supported: time series: ' + str(ts_freq) + ' time_scale: ' + str(time_scale))
                sys.exit()
        elif ts_freq == 'W-SUN':
            return df
        elif ts_freq == 'W-SAT':  # week ending date: move to week starting
            df['ds'] -= pd.to_timedelta(6, unit='D')
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: frequency mix not supported: time series: ' + str(ts_freq) + ' time_scale: ' + str(time_scale))
            sys.exit()


def ts_setup(ts_name, cutoff_date, init_date, time_scale):
    data_cfg = '~/my_repos/capacity_planning/forecast/config/data_cfg.json'
    with open(os.path.expanduser(data_cfg), 'r') as fptr:
        d_cfg = json.load(fptr)

    ts_cfg = d_cfg[ts_name]
    ts_cfg['cutoff_date'] = cutoff_date
    ts_cfg['init_date'] = init_date
    ts_cfg['time_scale'] = time_scale
    ts_cfg['name'] = ts_name
    bu = ts_cfg.get('bu', '')
    ts_key = ts_cfg['ycol'] if bu == '' else ts_cfg['ycol'] + '_' + bu
    ts_cfg['ts_key'] = ts_key
    cols = ['language'] if bu == '' else ['language', 'business_unit']
    return ts_cfg, cols


def ts_actuals(ts_name, ts_dict, gcols, drop_cols=True, use_cache=None):
    # plain actuals up to cutoff_date (as defined in ts_dict)
    if ts_name == 'ticket_count':
        actuals_df = t_ut.get_actuals(ts_dict, gcols, use_cache=use_cache)
    elif ts_name == 'prod_hours':
        actuals_df = t_ut.get_actuals(ts_dict, gcols, use_cache=use_cache)
    elif ts_name == 'ticket_count_Homes':
        actuals_df = t_ut.get_actuals(ts_dict, gcols, use_cache=use_cache)
    elif ts_name == 'prod_hours_Homes':
        actuals_df = t_ut.get_actuals(ts_dict, gcols, use_cache=use_cache)
    elif ts_name == 'ticket_count_China':
        actuals_df = t_ut.get_actuals(ts_dict, gcols, use_cache=use_cache)
    elif ts_name == 'prod_hours_China':
        actuals_df = t_ut.get_actuals(ts_dict, gcols, use_cache=use_cache)
    elif ts_name == 'ticket_count_Experiences':
        actuals_df = t_ut.get_actuals(ts_dict, gcols, use_cache=use_cache)
    elif ts_name == 'prod_hours_Experiences':
        actuals_df = t_ut.get_actuals(ts_dict, gcols, use_cache=use_cache)
    elif ts_name == 'booking_count':
        actuals_df = regs.get_actuals(ts_dict)
    elif ts_name == 'checkin_count':
        actuals_df = regs.get_actuals(ts_dict)
    elif ts_name == 'tenure':
        actuals_df = regs.get_actuals(ts_dict)
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' invalid ts name: ' + str(ts_name))
        sys.exit()

    if actuals_df is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: no actuals found for ' + str(ts_dict['cutoff_date'].date()) + ' and ts ' + str(ts_name) + ' and gcols ' + str(gcols))
        sys.exit()
    else:
        bu = ts_dict.get('bu', None)
        actuals_df_ = actuals_df.copy() if bu is None else actuals_df[actuals_df['business_unit'] == bu].copy()
        s_ut.my_print('Keep only bu data for bu: ' + str(bu) + ' and ts name: ' + str(ts_name) + ' Rows: ' + str(len(actuals_df_)))
        if drop_cols is True:
            actuals_df_.drop([c for c in gcols if c not in ['ds', 'language']], axis=1, inplace=True)
            actuals_df_ = actuals_df_.groupby(['ds', 'language']).sum().reset_index()
        return set_ts_freq(actuals_df_, ts_dict['time_scale'], ts_dict['agg_dict'], 'ds')


def get_gcols(by):
    if by == 'language':
        gcols = ['language']
    elif by == 'bu':
        gcols = ['business_unit']
    elif by == 'bl':
        gcols = ['business_unit', 'language']
    elif by == 'blc':
        gcols = ['business_unit', 'language', 'channel']
    elif by == 'blcs':
        gcols = ['business_unit', 'language', 'channel', 'service_tier']
    elif by == 'blcsi':
        gcols = ['business_unit', 'language', 'channel', 'service_tier', 'inbound_or_outbound']
    elif by == 'lcs':
        gcols = ['language', 'channel', 'service_tier']
    else:
        s_ut.my_print('ERROR: invalid by-key: ' + str(by))
        sys.exit()
    return gcols

