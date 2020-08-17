"""

"""

import os
import platform
from numpy.linalg import inv

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
import pandas as pd
import numpy as np
import copy

from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.forecast.utilities.language import fcast_processing as fp
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut

USE_CACHE = True if platform.system() == 'Darwin' else False


class HTS2(object):  # 2 levels HTS
    # https://otexts.com/fpp2/reconciliation.html
    # assumes top_t = sum_i bottom_ts(i)
    def __init__(self, top_ts, cutoff_date, time_scale='W', init_date='2016-01-01'):
        self.top_ts = top_ts
        self.ts_cfg, self.gcols = dp.ts_setup(self.top_ts, cutoff_date, init_date, time_scale)
        self.init_date = pd.to_datetime(init_date)
        self.time_scale = time_scale
        self.cutoff_date = cutoff_date
        self.init_date = init_date
        self.bottom_ts = list()
        self.n_ts = None
        self.ts_list = list()
        self.w_inv = dict()
        self.w_cols = dict()
        self.g_mtx = dict()
        self.s_mtx = dict()

    def get_forecasts(self, cutoff_date):
        f_df = fp.get_ens_fcast(self.top_ts, self.ts_cfg, cutoff_date)
        f_df.drop('ens', axis=1, inplace=True)
        f_list = [f_df]
        for bu in self.ts_cfg.get('business_units', list()):
            ts_cfg = copy.deepcopy(self.ts_cfg)
            ts_cfg['ts_key'] = ts_cfg['ycol'] if bu == '' else ts_cfg['ycol'] + '_' + bu
            ts_cfg.pop('business_units')
            f_df = fp.get_ens_fcast(ts_cfg['name'], ts_cfg, cutoff_date)
            f_df.drop('ens', axis=1, inplace=True)
            f_list.append(f_df)
        if len(f_list) > 0:
            ff = pd.concat(f_list, axis=0)
            p_ut.save_df(ff, '~/my_tmp/ff')
            p_ff = pd.pivot_table(ff, index=['ds', 'language'], columns='ts_name', values='yhat')
            p_ff.columns = [c + '_hat' for c in p_ff.columns]
            p_ff.reset_index(inplace=True)
            p_ff.fillna(0, inplace=True)
            return p_ff
        else:
            return None

    def get_actuals(self):
        a_df = dp.ts_actuals(self.top_ts, self.ts_cfg, self.gcols, use_cache=USE_CACHE)
        a_df['ts_name'] = self.top_ts
        a_list = [a_df]
        for bu in self.ts_cfg.get('business_units', list()):
            ts = self.top_ts + '_' + bu
            ts_cfg, gcols = dp.ts_setup(ts, self.cutoff_date, self.init_date, self.time_scale)
            f = dp.ts_actuals(ts, ts_cfg, ['language', 'business_unit'], drop_cols=False)
            f['ts_name'] = ts
            a_list.append(f)
            self.bottom_ts.append(ts)
        self.ts_list = [self.top_ts] + self.bottom_ts
        self.n_ts = len(self.ts_list)
        af = pd.concat(a_list, axis=0)
        p_af = pd.pivot_table(af, index=['ds', 'language'], columns='ts_name', values=self.ts_cfg['ycol']).reset_index()
        return p_af

    def set_cov(self, df_train, res_cols, ols=False):
        for lg, fl in df_train.groupby('language'):
            drop_list = [c for c in res_cols if fl[c].sum() == 0]
            self.w_cols[lg] = [c for c in res_cols if c not in drop_list]
            if ols is True:                     # DO NOT USE: will set small time series to 0!
                w = pd.DataFrame(np.diag(np.ones(len(self.w_cols))))
            else:
                w = fl[self.w_cols[lg]].cov()  # DOES MAY NOT WORK when the TS are highly correlated (eg all tickets and Homes: huge inverse
            w_inv = pd.DataFrame(np.linalg.pinv(w.values), w.columns, w.index)  # generalized inverse: w_inv.dot(w) = identity (except for 0 cols)
            self.w_inv[lg] = w_inv                                              # inverted cov for lang lg

    def coherent_fcast(self, df_hat, df_train, res_cols):
        if self.n_ts > 2:
            t_list, cols = list(), ['ds', 'language']
            self.set_cov(df_train, res_cols)
            for lg, fl in df_hat.groupby('language'):
                s_mtx, s_trans = self.set_sum_mtx(len(self.w_cols[lg]) - 1)
                p_mtx = inv(np.dot(np.dot(s_trans, self.w_inv[lg]), s_mtx))                # (S' W^(-1) S)^(-1)
                self.g_mtx[lg] = np.dot(np.dot(p_mtx, s_trans), self.w_inv[lg])            # (S' W^(-1) S)^(-1) S' W^(-1)
                sg_mtx = np.dot(s_mtx, self.g_mtx[lg])                                     # SG
                g_cols = [c.replace('res', 'hat') for c in self.w_cols[lg]]
                fl_tilde = fl[cols + g_cols].apply(self.recombine, g=sg_mtx, cols=g_cols, axis=1)
                fl_tilde['language'] = lg
                t_list.append(fl_tilde)
            fc = pd.concat(t_list, axis=0, sort=True) if len(t_list) > 0 else None
            fc.fillna(0, inplace=True)
        elif self.n_ts == 1:
            fc = df_hat.copy()
            fc.rename(columns={c: c.replace('hat', 'tilde') for c in fc.columns})
        else:
            s_ut.my_print('HTS cannot have 2 time series. Either 1 or more than 2')
            sys.exit()
        if fc is not None:
            for c in self.ts_list:
                try:
                    fc[c + '_tilde'] = fc[c + '_tilde'].astype(int)
                except ValueError:
                    pass
        return fc

    def hts_adj(self, a_df):
        r = a_df[self.top_ts + '_hat'] / a_df[[c + '_hat' for c in self.bottom_ts]].sum(axis=1)
        for ts in self.bottom_ts:
            c = ts + '_hat'
            a_df[c] *= r
            a_df[ts + '_tilde'] = a_df[c].astype(int)
        a_df[self.top_ts + '_tilde'] = a_df[self.top_ts + '_hat']
        a_df.drop([c + '_hat' for c in self.ts_list], axis=1, inplace=True)
        return a_df

    @staticmethod
    def recombine(row, g, cols):          # recombines fcasts to make things add up
        yhat = row[cols].values
        np.reshape(yhat, (-1, len(cols)))
        ytilde = np.dot(g, yhat)           # this is the recombination
        np.reshape(ytilde, np.shape(yhat))
        tcols = [c.replace('_hat', '') + '_tilde' for c in cols]
        dout = {tcols[i]: ytilde[i] for i in range(len(tcols))}
        dout['ds'] = row.loc['ds']
        dout['language'] = row.loc['language']
        return pd.Series(dout)

    @staticmethod
    def set_sum_mtx(n):      # S matrix
        v = [[1] * n]
        for i in range(n):
            z = np.zeros(n)
            z[i] = 1
            v.append(z)
        s_mtx = np.array(v)
        s_trans = np.transpose(s_mtx)
        return s_mtx, s_trans


def main(master_ts, cutoff_date, do_cov=True):
    # ###############################
    # ###############################
    time_scale = 'W'
    init_date = pd.to_datetime('2016-01-01')
    # ###############################
    # ###############################

    cutoff_date = pd.to_datetime(cutoff_date)
    init_date = pd.to_datetime(init_date)
    hts_obj = HTS2(master_ts, cutoff_date, time_scale=time_scale, init_date=init_date)
    a_df = hts_obj.get_actuals()
    f_df = hts_obj.get_forecasts(cutoff_date)

    # merge actuals and forecasts
    df = f_df.merge(a_df, on=['ds', 'language'], how='left')
    df.fillna(0, inplace=True)

    # pure forecast
    df_hat = df[df['ds'] > cutoff_date].copy()
    df_hat = df_hat[['ds', 'language'] + [c for c in df_hat.columns if '_hat' in c]].copy()

    # training data
    df_train = df[df['ds'] <= cutoff_date].copy()
    res_cols = list()
    for ts in hts_obj.ts_list:
        c = ts + '_res'          # residuals
        res_cols.append(c)
        df_train[c] = df_train[ts] - df_train[ts + '_hat']   # used to derive y_tilde
    if len(df_train) == 0:
        s_ut.my_print('ERROR: no train data for hts')
        sys.exit()

    if do_cov is True:   # optimal HTS: Y_tilde = S (S' W^(-1) S)^(-1) S' W^(-1) Y_hat with Y_hat = (tickets_hat, tickets_China_hat, tickets_Homes_hat, tickets_Experiences_hat)
        df_tilde = hts_obj.coherent_fcast(df_hat, df_train, res_cols)
    else:                # cheap and safe: scale to top TS
        df_tilde = hts_obj.hts_adj(df_hat.copy())

    ts_cols = hts_obj.ts_list
    z = df_hat.merge(df_tilde, on=['ds', 'language'], how='left')
    for ts in ts_cols:
        z[ts + '_delta'] = (z[ts + '_tilde'] - z[ts + '_hat']) / z[ts+'_hat']
    p_ut.save_df(z, '~/my_tmp/fbp/hts_all_' + master_ts + '_' + str(cutoff_date.date()))  # save combined and original forecasts and deltas
    return df_tilde, hts_obj.bottom_ts
