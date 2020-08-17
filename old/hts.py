"""
$ python hts.py master_ts cutoff_date adj
hierarchical time series with 2 levels only, to be to recursively build more levels. uses cfg file
adj =1 perform forecast adjustment
https://otexts.com/fpp2/reconciliation.html
"""
import os
import platform
from functools import reduce
from numpy.linalg import inv
import json

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
import pandas as pd
import numpy as np

from capacity_planning.forecast.utilities.language import data_processing as dp
from capacity_planning.forecast.utilities import adj_forecast as adj
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut


class HTS2(object):  # 2 levels HTS
    # https://otexts.com/fpp2/reconciliation.html
    # assumes top_t = sum_i bottom_ts(i)
    def __init__(self, top_ts, time_scale='W', init_date='2016-01-01'):
        self.top_ts = top_ts
        self.init_date = pd.to_datetime(init_date)
        self.time_scale = time_scale
        data_cfg = '~/my_repos/capacity_planning/forecast/config/hts.json'
        try:
            with open(os.path.expanduser(data_cfg), 'r') as fp:
                d_cfg = json.load(fp)
            self.bottom_ts = d_cfg.get(self.top_ts, None)
            if self.bottom_ts is None:
                s_ut.print('ERROR: invalid top ts: ' + str(self.top_ts))
                sys.exit()
            if len(self.bottom_ts) == 1:  # empty bottom ts is OK (adjust)
                s_ut.print('ERROR: invalid bottom ts: ' + str(self.bottom_ts))
                sys.exit()
        except FileNotFoundError:
            self.bottom_ts = list()
        self.n_ts = 1 + len(self.bottom_ts)
        self.ts_list = [self.top_ts] + self.bottom_ts
        self.w_inv = dict()
        self.w_cols = dict()
        self.g_mtx = dict()
        self.s_mtx = dict()

    def get_forecasts(self, cutoff_date, lbl='ens_fcast_'):
        froot = os.path.expanduser('~/my_tmp/fbp/')
        f_list = list()
        for f in os.listdir(froot):
            b_val = any([ts in f for ts in self.ts_list])
            if lbl in f and b_val and str(cutoff_date.date()) in f and 'not-' not in f:
                ff = f.split('.')[0]
                ts_name = ff.replace(lbl, '').replace('_' + str(cutoff_date.date()), '')
                df = p_ut.read_df(froot + f)
                df['ts_name'] = ts_name
                f_list.append(df)
        if len(f_list) > 0:
            ff = pd.concat(f_list, axis=0)
            p_ff = pd.pivot_table(ff, index=['ds', 'language'], columns='ts_name', values='yhat')
            p_ff.columns = [c + '_hat' for c in p_ff.columns]
            p_ff.reset_index(inplace=True)
            p_ff.fillna(0, inplace=True)
            return p_ff
        else:
            return None

    def get_actuals(self, cutoff_date):
        a_list = list()
        for ts in self.ts_list:
            a_df = dp.set_actuals(ts, cutoff_date, self.time_scale, self.init_date)
            a_df['ts_name'] = ts
            a_list.append(a_df)
        af = pd.concat(a_list, axis=0)
        p_af = pd.pivot_table(af, index=['ds', 'language'], columns='ts_name', values='y').reset_index()
        return p_af

    def set_cov(self, df_train, res_cols):
        for lg, fl in df_train.groupby('language'):
            drop_list = [c for c in res_cols if fl[c].sum() == 0]
            self.w_cols[lg] = [c for c in res_cols if c not in drop_list]
            w = fl[self.w_cols[lg]].cov()
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


def main(argv):
    print(argv)
    if len(argv) == 4:
        master_ts, cutoff_date, adj_name = argv[-3:]
    elif len(argv) == 3:
        master_ts, cutoff_date = argv[-2:]
        adj_name = None
    else:
        print('invalid args: ' + str(argv))
        sys.exit()

    # ###############################
    # ###############################
    time_scale = 'W'
    init_date = pd.to_datetime('2016-01-01')
    # ###############################
    # ###############################

    cutoff_date = pd.to_datetime(cutoff_date)
    init_date = pd.to_datetime(init_date)
    hts_obj = HTS2(master_ts, time_scale=time_scale, init_date=init_date)
    f_df = hts_obj.get_forecasts(cutoff_date)
    a_df = hts_obj.get_actuals(cutoff_date)

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

    # cheap HTS: scaling
    # assume top TS is perfect and scale accordingly
    # df_tilde = df_hat.apply(hts_adj, top_col=hts_obj.top_ts + '_hat',  hat_cols=[c + '_hat' for c in hts_obj.bottom_ts], axis=1)
    # df_tilde.rename(columns={c + '_hat': c + '_tilde' for c in hts_obj.ts_list}, inplace=True)

    # optimal HTS
    # Y_tilde = S (S' W^(-1) S)^(-1) S' W^(-1) Y_hat
    # Y_hat = (tickets_hat, tickets_China_hat, tickets_Homes_hat, tickets_Experiences_hat)
    df_tilde = hts_obj.coherent_fcast(df_hat, df_train, res_cols)

    ts_cols = hts_obj.ts_list
    z = df_hat.merge(df_tilde, on=['ds', 'language'], how='left')
    for ts in ts_cols:
        z[ts + '_delta'] = (z[ts + '_tilde'] - z[ts + '_hat']) / z[ts+'_hat']
    p_ut.save_df(z, '~/my_tmp/hts_all_' + master_ts + '_' + str(cutoff_date.date()))

    if adj_name is not None:
        # the master TS is NOT adjusted directly to ensure coherence: it cannot be in the cfg file
        # to adjust master TS, we must set the same adjustment in all components
        m_df = adj.main(df_tilde, adj_name, hts_obj.bottom_ts, master_ts)
    else:
        m_df = df_tilde.copy()

    # save coherent time series
    for ts in hts_obj.ts_list:
        f = m_df[['ds', 'language', ts + '_tilde']].copy()
        f.rename(columns={ts + '_tilde': 'yhat'}, inplace=True)
        p_ut.save_df(f, '~/my_tmp/fbp/hts_fcast_' + ts + '_' + str(cutoff_date.date()))


if __name__ == '__main__':
    s_ut.my_print(sys.argv)
    main(sys.argv)
    print('DONE')




