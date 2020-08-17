"""
Only does fcast, assumes regressors and data all set and cleaned (e.g. time col is ds, fcast col is y, ...).
Holidays data if any has also been prepared outside and is in the prophet_dict
"""

# import logging
# logging.getLogger('fbprophet').setLevel(logging.INFO)

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import fbprophet as fbp
import copy
import timeout_decorator

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import xforms as xf

valid_keys = ['growth', 'changepoints', 'n_changepoints',
              'changepoint_range', 'yearly_seasonality',
              'weekly_seasonality', 'daily_seasonality',
              'holidays', 'seasonality_mode', 'seasonality_prior_scale',
              'holidays_prior_scale', 'changepoint_prior_scale',
              'mcmc_samples', 'interval_width', 'uncertainty_samples', 'stan_backend']


class OneForecast(object):
    def __init__(self, data_df, reg_df, prophet_dict, this_cfg, fcast_periods, time_scale='D', verbose=True):
        """
        One forecast, all is ready.
        :param data_df: data with a 'ds' and a 'y' columns
        :param reg_df: regressors DF with a 'ds' and regressor columns. Forecast is already done.
        :param prophet_dict: config for prophet (default priors and hols DF)
        :param this_cfg: fcast cfg info for time series and regressors. (Additive, multiplicative, growth, transforms, ...)
        :param fcast_periods: How many periods to forecast
        :param time_scale: D normally
        :param verbose: print more
        """
        self.data_df = data_df.copy()
        self.reg_df = None if reg_df is None else reg_df.copy()
        self.prophet_dict = dict() if prophet_dict is None else copy.deepcopy(prophet_dict)
        self.ceiling = 1.0 if prophet_dict is None else prophet_dict.get('ceiling', 1.0)     # ceiling is multiplies data.max() when xform is logistic,  g > 1
        self.floor = 0.0 if prophet_dict is None else prophet_dict.get('floor', 0.0)         # floor is multiplies data.min() when xform is logistic, 0 <= g < 1
        self.cutoff_date = self.data_df['ds'].max()
        self.time_scale = time_scale
        self.horizon = fcast_periods
        self.is_default = False                               # indicator for fcast cfg = default fcast cfg
        self.verbose = verbose
        self.prophet_obj = None
        self.cv_df = None
        self.xform_obj = None                                  # transformer object
        self.f_df_n_res = None
        self.f_df_y_res = None
        self.default_cfg = {
            'xform': None,
            'daily': None,
            'growth': 'linear',
            'h_mode': False,
            'r_mode': None,
            'y_mode': 'auto',
            'w_mode': 'auto',
            'do_res': False,
            'changepoint_range': 0.8
        }
        if this_cfg is None:
            self.set_default()
        else:
            self.f_cfg = {k: this_cfg.get(k, v) for k, v in self.default_cfg.items()}   # if a value is missing set to default
        self.check_prophet_dict(self.prophet_dict)

    @staticmethod
    def check_prophet_dict(a_dict):        # drop keys in prophet_dict that are invalid in prophet (that was a hack)
        for k in list(a_dict.keys()):
            if k not in valid_keys:
                # s_ut.my_print('WARNING: dropping key: ' + str(k))
                _ = a_dict.pop(k)

    def set_default(self):
        if self.verbose:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: setting config to default')
        self.is_default = True
        self.f_cfg = copy.deepcopy(self.default_cfg)

    def set_regressors(self, dfy):
        xreg_df = None
        if self.reg_df is not None:
            if self.f_cfg.get('r_mode', None) is None:
                dfc, r_cols = dfy.copy(), None
            else:
                r_cols = [c for c in self.reg_df.columns if c != 'ds']                              # regressor columns
                xreg_df = pd.DataFrame({'ds': self.reg_df['ds'].values})
                drop_cols = list()
                for c in r_cols:
                    if self.reg_df[c].isnull().sum() == 0:
                        if self.reg_df[c].nunique() <= 1:
                            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: dropping regressor: ' + str(c) + ': single value: ' + str(self.reg_df[c].unique()))
                            drop_cols.append(c)
                        elif self.reg_df[c].nunique() == 2:
                            s_ut.my_print('pid: ' + str(os.getpid()) + ' no transform for regressor: ' + str(c))
                            xreg_df[c] = self.reg_df[c].values
                        else:
                            try:
                                s_ut.my_print('pid: ' + str(os.getpid()) + ' do transform for regressor: ' + str(c))
                                rvals = self.reg_df[c].values
                                xreg_df[c] = self.xform_obj.transform(rvals)
                            except:
                                s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: dropping regressor ' + str(c) + ': transform error')
                                drop_cols.append(c)  # drop cols
                    else:
                        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: dropping regressor: ' + str(c) + ': has null values: ' + str(self.reg_df[c].isnull().sum()))
                        drop_cols.append(c)

                r_cols = list(set(r_cols) - set(drop_cols))   # actual regressors
                if len(r_cols) > 0:
                    xreg_df.reset_index(inplace=True, drop=True)
                    dfc = dfy.merge(xreg_df, on='ds', how='left') if len(r_cols) > 0 else dfy.copy()
                    _ = [self.prophet_obj.add_regressor(c, mode=self.f_cfg['r_mode']) for c in r_cols]   # add non-null regressors
                else:
                    dfc = dfy
                    xreg_df, r_cols = None, None
        else:
            dfc, r_cols = dfy.copy(), None

        dfc.reset_index(inplace=True, drop=True)
        return dfc, xreg_df, r_cols

    def set_holidays(self, my_prophet_dict):
        if self.f_cfg.get('h_mode', False) is False:
            my_prophet_dict['holidays'] = None
        else:
            if my_prophet_dict['holidays'] is None or len(my_prophet_dict['holidays']) == 0:
                my_prophet_dict.pop('holidays', None)
            else:
                my_prophet_dict['holidays'].reset_index(inplace=True, drop=True)

    def set_prophet(self, p_dict):
        # seasonalities are set from cfg below
        p_dict['daily_seasonality'] = False
        p_dict['yearly_seasonality'] = False
        p_dict['weekly_seasonality'] = False
        self.check_prophet_dict(p_dict)
        if self.verbose:
            for k, v in p_dict.items():
                if isinstance(v, pd.core.frame.DataFrame):
                    s_ut.my_print('pid: ' + str(os.getpid()) + ' prophet_dict::' + str(k) + ': DF')
                else:
                    s_ut.my_print('pid: ' + str(os.getpid()) + ' prophet_dict::' + str(k) + ': ' + str(v))
            for k, v in self.f_cfg.items():
                s_ut.my_print('pid: ' + str(os.getpid()) + ' fcast_cfg::' + str(k) + ': ' + str(v))
        self.prophet_obj = fbp.Prophet(**p_dict)   # set the prophet obj here

        # set seasonalities from  f_cfg
        if (self.data_df['ds'].max() - self.data_df['ds'].min()).days > 2 * 365:
            ymode = self.f_cfg['y_mode']
            if ymode is not None:
                if ymode in ['additive', 'multiplicative']:
                    self.prophet_obj.add_seasonality('_yearly', period=365, fourier_order=10, mode=ymode)
                else:
                    s_ut.my_print('ERROR: invalid yearly seasonality: ' + str(ymode))
        if (self.data_df['ds'].max() - self.data_df['ds'].min()).days > 4 * 7 and 'W' not in self.time_scale and 'M' not in self.time_scale:
            wmode = self.f_cfg['w_mode']
            if wmode is not None:
                if wmode in ['additive', 'multiplicative']:
                    self.prophet_obj.add_seasonality('_weekly', period=7, fourier_order=5, mode=wmode)
                else:
                    s_ut.my_print('ERROR: invalid weekly seasonality: ' + str(wmode))

        # print prophet cfg
        if self.verbose:
            _ = [s_ut.my_print('attribute: ' + str(c) + ': ' + str(getattr(self.prophet_obj, c, 'Not Valid'))) for c in valid_keys]

    def set_xform(self, method):        # data transformation
        dfy = self.data_df.copy()
        if len(dfy) > 0:
            nqs = min(100.0, len(dfy) / 4.0)
            self.xform_obj = xf.Transform(method, nqs, ceiling=self.ceiling * dfy['y'].max(), floor=self.floor * dfy['y'].min())
            yvals = dfy['y'].copy()
            ty = self.xform_obj.fit_transform(yvals.values)
            if ty is None:
                return None
            else:
                dfy['y'] = ty
                dfy.reset_index(inplace=True, drop=True)
                return dfy
        else:
            return None

    def forecast(self):
        f = self._forecast()
        if f is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: forecast failed with cfg ' + str(self.f_cfg))
            return None
        else:
            vlen = [0 if x is None else len(x) for x in f]
            s_ut.my_print('pid: ' + str(os.getpid()) + ' forecast returned rows(n_res, y_res): ' + str(vlen) + ' for cfg ' + str(self.f_cfg))
            return f   # f_df_n_res, f_df_y_res

    @timeout_decorator.timeout(600, use_signals=True, timeout_exception=StopIteration, exception_message='forecast failed due to timeout')
    def _forecast(self):
        # this decorator assumes one fcast runs on the main thread.  If not, set use_signals=False.
        # see https://github.com/pnpnpn/timeout-decorator
        # timeout is in secs
        ret = self._data_prep()
        if ret is None:
            return None
        else:
            dfc, xreg_df, regs_cols = ret
        try:
            if self.verbose:
                self.prophet_obj.fit(dfc)
            else:
                with s_ut.suppress_stdout_stderr():
                    self.prophet_obj.fit(dfc)
            try:
                future_df = self.prophet_obj.make_future_dataframe(periods=self.horizon, freq=self.time_scale)  # day, week-starting sunday, end of month
                if xreg_df is not None:
                    future_df = future_df.merge(xreg_df, on='ds', how='left')
                    if future_df[regs_cols].isnull().sum().sum() > 0:
                        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: invalid regressors')
                        return None
                f_df = self.prophet_obj.predict(future_df)                            # forecast
            except Exception as e:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: prophet predict future failed with error: ' + str(e))
                return None
        except (Exception, ValueError, KeyError) as e:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: prophet fit failed with error: ' + str(e))
            if dfc is not None:
                print(dfc.head())
                print(dfc.tail())
            else:
                s_ut.my_print('ERROR: fit DF dfc is None')
            return None
        self.resi_fcast(f_df, dfc[['ds', 'y']].copy())  # fcast residuals and save fcast with no residuals
        return self.f_df_n_res, self.f_df_y_res

    def _data_prep(self):         # prepare the data:  min shift,  max scaling, logistic transform,  ts transform
        my_prophet_dict = copy.deepcopy(self.prophet_dict)
        my_prophet_dict['growth'] = self.f_cfg['growth']
        my_prophet_dict['changepoint_range'] = self.f_cfg['changepoint_range']

        dfy = self.set_xform(self.f_cfg['xform'])
        if dfy is None:
            return None
        else:
            self.set_holidays(my_prophet_dict)  # dict gets modified
            self.set_prophet(my_prophet_dict)   # dict gets modified
            dfc, xreg_df, regs_cols = self.set_regressors(dfy)
            return dfc, xreg_df, regs_cols

    def resi_fcast(self, f_df, y_vals):
        # the residuals after the inverse transforms are not normal, we can re-forecast them!
        # forecast residuals:  see https://robjhyndman.com/hyndsight/ljung-box-test/
        self.f_df_n_res = self.back_fcast(f_df.copy())    # forecast with no residual forecast
        self.f_df_y_res = None                            # forecast with residual forecast
        if self.f_df_n_res is None:                       # fcast without residual forecast failed in back transforms
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: forecast without forecasting residuals failed on back transforms')
            return
        elif self.f_cfg['do_res'] is True:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ******************* resi_fcast: starts **************** ')
            n_df = self.resi_fcast_(f_df, y_vals)

            # ###################################################################################################
            # >>>>>>>>>>> only yhat's will be correct in the natural scale for non-trivial transforms <<<<<<<<<<<
            # only additive: yhat = t + s + beta * r = t + a
            # only multiplicative: yhat = (1 + s + beta * r) * t = t * (1 + m)
            # in general, yhat = t * (1 + m) + a
            # we only update yhat, trend (t), additive_terms (a) and multiplicative_terms (m) (all levels)
            # we have, yhat_y = yhat_n + yhat_res = t_y * (1 + m_y) + a_y + t_res * (1 + m_res) + a_res
            # in the end, in order to have the prophet equation for yhat_y, i.e. yhat_y = t_y * (1 + m_y) + a_y
            # we set for yhat_upper, yhat_lower and yhat, the trend, additive and multiplicative terms as,
            # yhat_y = yhat_n + yhat_res
            # t_y = t_n + t_res
            # m_y = (t_n * m_n + t_res * m_res) / (t_n + t_res)
            # a_y = a_n + a_res
            # ###################################################################################################
            levels = ['', '_upper', '_lower']
            if self.verbose:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: only yhat (all levels) back-transforms will be correct')
            if n_df is not None and self.f_df_n_res is not None:
                self.f_df_y_res = self.f_df_n_res.copy()
                for l in levels:
                    self.f_df_y_res['yhat' + l] += n_df['yhat' + l]
                    self.f_df_y_res['trend' + l] += n_df['trend' + l]
                    self.f_df_y_res['additive_terms' + l] += n_df['additive_terms' + l]
                    self.f_df_y_res['multiplicative_terms' + l] = (self.f_df_n_res['multiplicative_terms' + l].copy() * self.f_df_n_res['trend' + l].copy() +
                                                                   n_df['multiplicative_terms' + l].copy() * n_df['trend' + l].copy()) / self.f_df_y_res['trend' + l].copy()
                s_ut.my_print('pid: ' + str(os.getpid()) + ' ******************* resi_fcast: ends **************** ')
            else:     # no residuals fcast
                self.f_cfg['do_res'] = False
            return
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' forecast without forecasting residuals OK')
            return

    def back_fcast(self, f_df):         # undo data transforms
        # >>>>>>>>>>> only yhat's will be correct in the natural scale for non-trivial transforms <<<<<<<<<<<
        if self.verbose:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: only yhat (all levels) back-transforms will be correct')
        levels = ['', '_upper', '_lower']
        for lvl in levels:
            # ###################### should only inverse-xform the terms that exist (eg invert multiplicative_terms only if something is multiplicative)  ##############
            for c in ['yhat', 'trend', 'additive_terms', 'multiplicative_terms']:
                y_var = self.xform_obj.fcast_var(f_df[[c + '_upper', c + '_lower']].copy(), self.prophet_obj.interval_width)   # use the same var for upper and lower
                v = f_df[c + lvl].copy()
                f_df[c + lvl] = self.xform_obj.inverse_transform(v, y_var, lbl=c + lvl)  # only yhat's will be correct in the natural scale for non-trivial transforms

        if f_df['yhat'].isnull().sum() > 0.05 * len(f_df):   # more than pct nulls
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: fcast failed: too many NaNs for cfg ' + str(self.f_cfg))
            s_ut.my_print(str(f_df['yhat'].isnull().sum()) + ' nulls on ' + str(len(f_df)) + ' rows')
            print(f_df.head(10))
            print(f_df.isnull().sum())
            p_ut.save_df(f_df, '~/my_tmp/f_df')
            return None
        else:
            f_df['yhat'].interpolate(inplace=True, limit_direction='both')
            f_df['yhat_upper'].interpolate(inplace=True, limit_direction='both')
            f_df['yhat_lower'].interpolate(inplace=True, limit_direction='both')
            if self.verbose:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' back_fcast: forecast OK with cfg ' + str(self.f_cfg))
        return f_df.copy()

    def resi_fcast_(self, f_df, y_vals):
        # forecast the residuals from self by calling OneForecast again
        # prepare: first, inverse transform actuals and forecasted actuals back to the natural scale
        y = self.xform_obj.inverse_transform(y_vals['y'], 0.0, lbl='y')                                           # inverse transformed actuals
        y_hat_x = f_df[(f_df['ds'] >= y_vals['ds'].min()) & (f_df['ds'] <= y_vals['ds'].max())].copy()             # transformed forecasted actuals
        y_var = self.xform_obj.fcast_var(y_hat_x[['yhat_upper', 'yhat_lower']], self.prophet_obj.interval_width)   # use the same var for upper and lower
        yhat = self.xform_obj.inverse_transform(y_hat_x['yhat'], y_var, lbl='yhat')                          # inverse transformed forecasted actuals
        try:
            res_df = pd.DataFrame({'ds': y_vals['ds'].values, 'y': y - yhat})                                          # residuals DF to forecast in natural scale
        except TypeError as msg:
            s_ut.my_print('WARNING: resi_fcast fails with msg: ' + str(msg))
            print(y_hat_x.head())
            print(y_vals.head())
            return None

        if res_df['y'].isnull().sum() > 0.25 * len(res_df):   # more than pct nulls
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: resi_fcast failed: too many NaNs in res_df: '
                          + str(res_df['y'].isnull().sum()) + ' nulls on ' + str(len(res_df)) + ' rows')
            print(res_df.head(10))
            print(res_df.isnull().sum())
            p_ut.save_df(res_df, '~/my_tmp/res_df')
            return None
        else:
            res_df['y'].interpolate(inplace=True, limit_direction='both')

            # ##################################################################################################
            # ################################### white noise test ##############################################
            # lags = min(10.0, len(y_vals) / 5.0) if self.f_cfg['y_mode'] is None else int(min(2.0 * 365 + 1.0, len(y_vals) / 5.0))
            # is_wn = st_ut.white_noise_test(res_df['y'].values, p_thres=0.05, lags=int(lags), verbose=False)                  # white noise test
            # if is_wn is False:                                                                                               # residuals are not white noise
            # ##################################################################################################
            # ##################################################################################################

            f_cfg = copy.deepcopy(self.f_cfg)                                                                                  # use current cfg but adjust xform,do_res and regs
            f_cfg['xform'] = 'yeo-johnson'                                                                                     # residuals will be negative!
            f_cfg['do_res'] = False                                                                                            # no infinite recursion!
            f_cfg['r_mode'] = None                                                                                             # regs will probably not help here
            s_ut.my_print('pid: ' + str(os.getpid()) + ' resi_fcast: start recursive call: residual forecast')
            res_obj = OneForecast(res_df.copy(), None, copy.deepcopy(self.prophet_dict), f_cfg, self.horizon, time_scale=self.time_scale, verbose=self.verbose)
            ret = res_obj.forecast()                                                                                       # residuals forecast in natural scale
            if ret is None:
                ret = [None]
            s_ut.my_print('pid: ' + str(os.getpid()) + ' resi_fcast: end recursive call: residual forecast')
            return ret[0]  # n_df

