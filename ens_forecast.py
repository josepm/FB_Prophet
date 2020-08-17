"""
$ python ens_forecast ts_name cutoff_date to_table
cutoff_date: a date. mandatory
ts_name: time series name. Mandatory
to_table: default to 0
builds and saves the ens forecast for a time series
"""

import os
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
import pandas as pd

from capacity_planning.forecast.utilities.language import ens_processing as ep              # must be first: limit on objects loadable with TLS?
from capacity_planning.forecast.utilities.language import fcast_processing as fp
from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.data import hql_exec as hql


# pd.set_option('display.max_rows', 100)
# pd.set_option('precision', 4)
# pd.set_option('display.width', 320)
# pd.set_option('display.max_columns', 35)
#
FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    # ###########################
    # parameters
    # ###########################
    time_scale = 'W'  # forecasting time scale reset for daily ticket data
    init_date = pd.to_datetime('2016-01-01')
    froot = os.path.expanduser('~/my_tmp/fbp/')
    evals = 250
    by_lang = False
    lwr, upr = 9, 12
    # ###########################
    # ###########################

    print(argv)
    if len(argv) == 3:
        ts_name, cutoff_date = argv[-2:]
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
            to_table = False
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
            sys.exit()
    elif len(argv) == 4:
        ts_name, cutoff_date, to_table = argv[1:]
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
            to_table = bool(int(to_table))
        except ValueError:
            s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
            sys.exit()
    else:
        s_ut.my_print('ERROR: invalid arguments (ts_name, cutoff_date, to_table): ' + str(argv))
        sys.exit()

    # data cfg
    ts_cfg, cols = dp.ts_setup(ts_name, cutoff_date, init_date, time_scale)
    fcast_days = ts_cfg.get('fcast_days', None)
    if fcast_days is None:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR" fcast_days must be specified in data_cfg')
        sys.exit()
    else:
        fcast_date = cutoff_date + pd.to_timedelta(fcast_days, unit='D')

    if time_scale == 'W' and fcast_date.weekday() != 6:                 # set fcast date to week starting Sunday unless it is a Sunday already
        fcast_date = fcast_date - pd.to_timedelta(1 + fcast_date.weekday(), unit='D')

    s_ut.my_print('pid: ' + str(os.getpid()) + ' ------------------------ start ens forecast for ' + str(ts_name) + ' from cutoff date '
                  + str(cutoff_date.date()) + ' (excluded) to forecast date ' + str(fcast_date.date()) + '  (included) -----------------------')

    a_df = dp.ts_actuals(ts_name, ts_cfg, cols)                                  # get actuals
    a_df.rename(columns={ts_cfg['ycol']: 'y'}, inplace=True)
    fcast_df = fp.get_lang_fcast(ts_cfg, cutoff_date)      # get fcasts

    if fcast_df is not None and a_df is not None:
        s_ut.my_print(ts_name + ': combining ' + str(fcast_df['dim_cfg'].nunique()) + ' forecast configs')
        xens_ = get_ens(ts_name, cutoff_date)  # ts_cfg['ens'].get(str(cutoff_date.month), ens_dict['default'])

        s_ut.my_print('aggregation for ' + ts_name + ' done with ' + xens_)
        ts_fcast = ep.make_fcast(ts_name, fcast_df, a_df, cutoff_date, fcast_date, xens_, evals, by_lang, (lwr, upr), lwr=lwr, upr=upr)
        ts_fcast['fcast_date'] = fcast_date

        p_ut.save_df(ts_fcast, froot + 'ens_fcast_' + ts_name + '_' + str(cutoff_date.date()))
        if to_table is True:
            cols = ['ds', 'language', 'ens', 'yhat']
            partition = {'cutoff': str(cutoff_date.date()), 'ts_name': ts_cfg['ts_key']}
            ret = hql.to_tble(ts_fcast, cols, 'sup.cx_ens_forecast', partition)
            if ret != 0:
                s_ut.my_print('ERROR: DB write for ' + ts_name + ' ens forecast ' + ' at ' + str(cutoff_date.date()) + ' failed')
                sys.exit()
        print('DONE')
    else:
        s_ut.my_print('ERROR: no actuals or no data for errors of ' + ts_name + ' at ' + str(cutoff_date.date()))


def get_ens(ts_name, cutoff_date, exclude=('2019-11-30', '2019-12-28', '2020-01-25', '2020-02-29', '2020-03-28', '2020-04-25', '2020-05-30', '2020-06-27')):
    df = hql.from_tble('select * from sup.cx_language_forecast_performance_detail where ts_name = \''
                       + ts_name + '\' and cutoff <= \'' + str(cutoff_date.date()) + '\';', ['cutoff', 'upr', 'lwr'])
    df = df[df['language'] == 'All'].copy()
    if len(exclude) != 0:
        df = df[~df['cutoff'].isin(list(exclude))].copy()
    if len(df) > 0:
        gf = df.groupby('ens').apply(lambda x: x['err'].mean()).reset_index()
        s_ut.my_print('+++++++++++++++++ Past performance ++++++++++++++')
        print(gf)
        z = gf.nsmallest(1, columns=[0])
        return z.loc[z.index[0], 'ens']
    else:
        return 'lasso'


if __name__ == '__main__':
    main(sys.argv)
