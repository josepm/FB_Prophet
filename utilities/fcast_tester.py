"""
runs a basic forecast
"""
import pandas as pd
import fbprophet as fbp

from capacity_planning.forecast.utilities.language import data_prep as dtp
from functools import reduce
from capacity_planning.forecast.utilities import one_forecast as one_fcast
from capacity_planning.forecast.utilities.language import data_processing as d_proc
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import xforms as xf


# parameters
p_cfg = {
    "changepoint_prior_scale": 0.5,
    "changepoint_range": 0.5,
    "changepoints": None,
    "holidays": None,
    "holidays_prior_scale": 10.0,
    "interval_width": 0.8,
    "mcmc_samples": 0,
    "n_changepoints": 25,
    "seasonality_prior_scale": 10.0,
    "uncertainty_samples": 1000,
    "ceiling": 1, "floor": 0
  }
lang = 'English_NA'
cutoff_date = pd.to_datetime('2019-10-01')
agg_dict = {"entered": "sum", "offered": "sum", "accepted": "sum", "short_abandons": "sum", "abandons": "sum"}
check_cols = ["offered", "accepted", "abandons"]
df = pd.read_parquet('~/my_tmp/cleaned/phone-vol_cleaned_2019-11-25.par')
fcast_days = 112
is_train=True
init_date = pd.to_datetime('2016-01-01')
tdays = 1096
upr_horizon, lwr_horizon = 112, 84
# #############################

fcast_date = cutoff_date if is_train is True else cutoff_date + pd.to_timedelta(fcast_days, unit='D')
cutoff_date = cutoff_date if is_train is False else cutoff_date - pd.to_timedelta(fcast_days, unit='D')

ldf = df[df.language == lang].copy()
gf = ldf.groupby(['ds', 'language']).agg(agg_dict).reset_index()
check_list = [d_proc.data_check(gf[['ds', c]].copy(), c, 'ds', cutoff_date, init_date, max_int=5, name=lang) for c in check_cols]
mf = reduce(lambda x, y: x.merge(y, on='ds', how='inner'), check_list) if len(check_list) > 0 else None  # inner merge to get the shortest overlapping times
h_df = dtp.get_holidays(lang, fcast_date, init_date)  # set holidays
d_df = dtp.set_demand(mf, 10, 'phone-inbound-vol')
# t_df = dtp.trim_outliers(d_df.copy(), None, 1.1, ['y'])
t_df = dtp.trim_outliers(d_df.copy(), h_df, 1.1, ['y'])
t_start = cutoff_date - pd.to_timedelta(tdays, unit='D')
a_df = t_df[(t_df['ds'] <= cutoff_date) & (t_df['ds'] >= t_start)][['ds', 'y']].copy()

fcast_cfg = {'xform': None, 'daily': None, 'growth': 'linear', 'h_mode': False, 'r_mode': None, 'y_mode': 'auto', 'w_mode': 'auto', 'do_res': False}
p_cfg['n_changepoints'] = int(12 * tdays / 365)
fcast_obj = one_fcast.OneForecast(a_df, None, p_cfg, fcast_cfg, upr_horizon, ret_cols=['yhat'], time_scale='D', verbose=False)
f_df = fcast_obj.forecast()

upr_dt = cutoff_date + pd.to_timedelta(upr_horizon, unit='D')  # this is the cutoff_date from the CLI
lwr_dt = cutoff_date + pd.to_timedelta(lwr_horizon, unit='D')
fz = f_df[(f_df['ds'] <= upr_dt) & (f_df['ds'] > lwr_dt)].copy()
az = t_df[(t_df['ds'] <= upr_dt) & (t_df['ds'] > lwr_dt)][['ds', 'y']].copy()
e_df = fz.merge(az, on='ds', how='inner')
print(e_df.head())
err, std = dtp.errs(e_df)
print('error: ' + str(err))
e_df.set_index('ds', inplace=True)
e_df.plot(grid=True, style='o-')

x = a_df['y'].values
my_bc = xf.BoxCox()
my_xt = my_bc.fit_transform(x)
print('my lmbda: ' + str(my_bc.lmbda))
my_txt = my_bc.inverse_transform(my_xt)
print(pd.DataFrame({'my_x': x, 'my_txt': my_txt}).head())

nqs = min(100.0, len(a_df) / 4.0)
xform_obj = xf.Transform('quantile', nqs)
tx = xform_obj.fit_transform(a_df['y'].copy().values)
print('lbda: ' + str(xform_obj.lambdas_))
txt = xform_obj.inverse_transform(tx)
zf = pd.DataFrame({'x': x, 'txt': txt})
print(zf.head())

p_cfg.pop('ceiling', None)
p_cfg.pop('floor', None)
p_cfg['n_changepoints'] = int(12 * tdays / 365)

a_df = t_df[(t_df['ds'] <= cutoff_date) & (t_df['ds'] >= t_start)][['ds', 'y']].copy()
az = t_df[(t_df['ds'] <= upr_dt) & (t_df['ds'] > lwr_dt)][['ds', 'y']].copy()
for m in ['box-cox', 'quantile']:
    nqs = min(100.0, len(a_df) / 4.0)
    x_df = pd.DataFrame({'ds': a_df['ds'].values})
    if m is not None:
        xform_obj = xf.Transform(m, nqs)
        # xform_obj = xf.BoxCox(m, nqs)
        yvals = a_df['y'].values
        ty = xform_obj.fit_transform(yvals)
        x_df['y'] = ty
        txt = xform_obj.inverse_transform(ty)
        print(pd.DataFrame({'x': yvals, 'txt': txt, 'm': [str(xform_obj.lmbda)] * len(a_df)}).head())
    else:
        x_df['y'] = a_df['y'].values
    for r in [0.3, 0.4, 0.5, 0.6]:
        p_cfg['changepoint_range'] = r
        n = 24
        # for n in [6, 12, 18, 24]:
        p_cfg['n_changepoints'] = int(n * tdays / 365)
        pobj = fbp.Prophet(**p_cfg)
        with s_ut.suppress_stdout_stderr():
            pobj.fit(x_df)
        future_df = pobj.make_future_dataframe(periods=upr_horizon, freq='D')
        f_df = pobj.predict(future_df)[['ds', 'yhat']]
        if m is not None:
            f_df['yhat'] = xform_obj.inverse_transform(f_df['yhat'].values)
        fz = f_df[(f_df['ds'] <= upr_dt) & (f_df['ds'] > lwr_dt)].copy()
        e_df = fz.merge(az, on='ds', how='inner')
        err, std = dtp.errs(e_df)
        print('error: ' + str(err) + ' r: ' + str(r) + ' n: ' + str(n) + ' m: ' + str(m))

        e_df.set_index('ds', inplace=True)
        e_df.plot(grid=True, style='o-')

