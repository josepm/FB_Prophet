"""
S python tix_to_table cutoff_date
cutoff_date is the date label in the files in ~/Forecasts/csv/
"""
import airpy as ap

import os
import sys
import pandas as pd

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut


def main(cutoff_date):
    s_ut.my_print('loading to sup.dim_cx_ticket_forecast the forecast with cutoff date ' + str(cutoff_date))
    t_file = os.path.expanduser('~/Forecasts/par/' + 'table_output_' + str(cutoff_date) + '.par')
    s_ut.my_print('table file: ' + str(t_file))

    if os.path.isfile(t_file):
        df = p_ut.read_df(t_file)
        p_ut.set_week_start(df, tcol='ds')  # week_starting patch

        # week_starting patch
        df_cols_ = df.columns
        if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
            df['ds_week_ending'] = pd.to_datetime(df['ds_week_ending'])
            df['ds_week_starting'] = df['ds_week_ending'] - pd.to_timedelta(6, unit='D')

        s_ut.my_print('data file: ' + str(t_file) + ' rows: ' + str(len(df)) + ' to table')
        partition = {'ds': str(cutoff_date)}
        table = 'sup.dim_cx_ticket_forecast'
        ap.hive.push(df, table=table, if_exists='replace', partition=partition,
                     table_props={'abb_retention_days': '-1', 'abb_retention_days_reason': 'fact table. No pii'}
                     )
        return 0
    else:
        s_ut.my_print('ERROR: failed to load: file ' + t_file + ' is missing')
        return -1


def to_table(a_df, cutoff_date_, table_):          # 'josep.dim_ticket_facst_test'  # 'sup.dim_cx_ticket_forecast'
    if len(a_df) > 0:
        s_ut.my_print('pushing ' + str(len(a_df)) + ' rows to table ' + table_)
        partition = {'ds': str(cutoff_date_)}
        # must use hive to support partition arg
        ap.hive.push(a_df, table=table_, if_exists='replace', partition=partition,
                     table_props={'abb_retention_days': '-1', 'abb_retention_days_reason': 'fact table. No pii'}
                     )
        return 0
    else:
        s_ut.my_print('ERROR: failed to load')
        return -1


if __name__ == '__main__':
    cutoff_date = sys.argv[1]
    ret = main(cutoff_date)
    if ret == 0:
        print('DONE')
    else:
        print('ERROR')

    # s_ut.my_print('loading to sup.dim_cx_ticket_forecast the forecast with cutoff date ' + str(cutoff_date))
    # t_file = os.path.expanduser('~/Forecasts/par/' + 'table_output_' + str(cutoff_date) + '.par')
    # s_ut.my_print('table file: ' + str(t_file))
    #
    # if os.path.isfile(t_file):
    #     df = p_ut.read_df(t_file)
    #     s_ut.my_print('data file: ' + str(t_file) + ' rows: ' + str(len(df)))
    #     ap.hive.push(df, table='sup.dim_cx_ticket_forecast', if_exists='replace',
    #                  table_props={'abb_retention_days': '-1', 'abb_retention_days_reason': 'fact table. No pii'}
    #                  )
    #     print('SUCCESS')
    # else:
    #     s_ut.my_print('ERROR: failed to load: file ' + t_file + ' is missing')
