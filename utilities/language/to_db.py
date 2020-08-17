"""

"""
import os
import pandas as pd
import airpy

ts_names = ['tickets', 'tickets_Experience', 'tickets_Homes', 'tickets_China', 'bookings', 'checkins']

f_dir = '~/my_tmp/fbp/'
f_dict = {
    'ens_fcast_': 'sup.cx_weekly_forecasts_',
    'lang_fcast_': 'sup.cx_weekly_forecasts_detail',
    'lang_perf_': 'sup.cx_forecast_performance',
    'hts_fcast_': 'sup.cx_weekly_forecasts'
}
# a_dir = '~/my_tmp/cleaned/'
# a_dict = {
#     'tickets_': 'sup.cx_weekly_actuals',
#     'tickets_Homes_': 'sup.cx_weekly_actuals',
#     'tickets_Experience_': 'sup.cx_weekly_actuals',
#     'tickets_China_': 'sup.cx_weekly_actuals',
#     'checkins_': 'sup.cx_weekly_actuals',
#     'bookings_': 'sup.cx_weekly_actuals',
#     'tenure_': 'sup.cx_weekly_actuals',
#     'phone_vol_': 'sup.cx_weekly_actuals'
# }

for fn, tb in f_dict.items():
    for f in os.listdir(os.path.expanduser(f_dir)):
        if fn in f and 'not-' not in f:
            for ts in ts_names:
                cu = f.split('.')[0].replace(fn + ts + '_', '')
                df = pd.read_parquet(f_dir + f)
                df.drop(['cutoff', 'ts_name'], axis=1, inplace=True)
