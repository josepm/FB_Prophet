"""

"""

import os
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import sys
import pandas as pd
import numpy as np
import operator
from functools import reduce

import xgboost as xgb
import sklearn.linear_model as l_mdl
import sklearn.ensemble as sk_ens
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import enet_path
from sklearn.decomposition import FactorAnalysis
from scipy.stats import shapiro as sw_test
from statsmodels.tools import eval_measures as evm

from capacity_planning.utilities import stats_utils as st_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.utilities.language import data_processing as dtp
from capacity_planning.forecast.utilities.language import fcast_processing as fp


from numpy.linalg import inv, LinAlgError
from sklearn.linear_model import ElasticNet
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from scipy.stats import uniform, expon
from itertools import product


# pd.set_option('display.max_rows', 100)
# pd.set_option('precision', 4)
# pd.set_option('display.width', 320)
# pd.set_option('display.max_columns', 35)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
SPACE_DICT = {
    'AdaBoostRegressor': {
        'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1.0))},
    'BaggingRegressor': {
        'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'max_samples': hp.uniform("max_samples", 0.1, 1),
        'max_features': hp.uniform("max_features", 0.1, 1),
        'n_jobs': -1},
    'GradientBoostingRegressor': {
        'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'learning_rate': hp.uniform('learning_rate', 0, 1),
        'min_samples_split': hp.uniform("min_samples_split", 0.1, 1),
        'min_samples_leaf': hp.uniform("min_samples_leaf", 0.1, 0.5),
        'max_depth': hp.quniform("max_depth", 2, 20, 1)},
    'RandomForestRegressor': {
        'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'min_samples_split': hp.uniform("min_samples_split", 0.1, 1),
        'min_samples_leaf': hp.uniform("min_samples_leaf", 0.1, 0.5),
        'max_depth': hp.quniform("max_depth", 2, 20, 1),
        'n_jobs': -1},
    'ExtraTreesRegressor': {
        'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'min_samples_split': hp.uniform("min_samples_split", 0.1, 1),
        'min_samples_leaf': hp.uniform("min_samples_leaf", 0.1, 0.5),
        'max_depth': hp.quniform("max_depth", 2, 20, 1),
        'n_jobs': -1},
    'XGBRegressor': {
        'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'max_depth': hp.quniform("max_depth", 2, 20, 1),
        'gamma': hp.uniform('gamma', 1, 10),
        'reg_alpha': hp.uniform('reg_alpha', 1, 10),
        'reg_lambda': hp.uniform('reg_lambda', 1, 10),
        'learning_rate': hp.uniform('learning_rate', 0, 1),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'nthread': 4 if platform.system() == 'Darwin' else 36
    },
    'Lasso': {
        'alpha': hp.uniform('alpha', 0.1, 10),
    }
}


def gen_output(reg, d_list, best_err, append=False):
    n_features = len(reg.features_)
    if n_features > 0:
        mask_ = reg.mask_
        if append is True:
            s_ut.my_print(reg.name + ' selected ' + str(n_features) + ' features: ' + str(reg.features_))
            d_err = reg.perf_err()
            d_err['best_err'] = best_err
            d_list.append(d_err)
    else:
        s_ut.my_print('WARNING: ' + reg.name + ' failed')
        mask_ = np.array([True] * len(reg.rcols))
    return mask_


def lang_perf(lg, f_data, a_perf, this_cutoff, upr, lwr):
    p_ut.save_df(a_perf, '~/my_tmp/a_perf_' + lg)
    p_ut.save_df(f_data, '~/my_tmp/f_data_' + lg)
    d_list = list()
    a_perf.sort_values(by='a_err', inplace=True)
    a_perf.reset_index(inplace=True, drop=True)
    cfg, b_err = a_perf.loc[a_perf.index[0],]

    # OMP
    # reg_omp = OMP('OMP', f_data, this_cutoff, upr, lwr)
    # _ = gen_output(reg_omp, d_list)

    # lasso
    reg_lasso = LassoRegressor('lasso', f_data, this_cutoff, upr, lwr)
    lasso_mask = gen_output(reg_lasso, d_list, b_err, append=False)

    # ENET
    verbose = False
    reg_enet = EnetOpt('enet', f_data, this_cutoff, upr, lwr, max_evals=100, verbose=verbose)
    enet_mask = gen_output(reg_enet, d_list, b_err, append=False)

    f_mask = enet_mask if np.sum(enet_mask) < len(f_data) else lasso_mask

    # Ridge
    reg_ridge = RidgeOpt('ridge', f_data, this_cutoff, upr, lwr, f_mask, reg_enet.ridge_par)
    _ = gen_output(reg_ridge, d_list, b_err, append=True)

    # 'XGBRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'GradientBoostingRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor'
    # ensembles with ENET selected features
    # for rname in ['XGBRegressor', 'AdaBoostRegressor']:
    #     for loss_type in ['rel', 'abs']:
    #         reg = EnsOpt(rname, fl, cutoff_date, upr, lwr, mask, loss_type=loss_type, max_evals=200)
    #         d_err = reg.perf_err()
    #         d_err['ens'] += '-' + loss_type
    #         d_list.append(d_err)

    return d_list


def data_check(df, name):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    u_vals = [c for c in df.columns if df[c].nunique() <= 1]
    if df.isnull().sum().sum() > 0 or len(u_vals) > 0:
        p_ut.save_df(df, '~/my_tmp/f_data')
        s_ut.my_print('ERROR: invalid data for ' + str(name))
        sys.exit()


class Regressor(object):
    # follow definitions of scikit learn for alpha and lbda (l1_ratio)
    def __init__(self, name, f_data, this_cu, upr, lwr):
        def is_int(x):
            try:
                _ = int(x)
                return True
            except ValueError:
                return False
        self.rcols = [c for c in f_data if is_int(c)]  # same cfgs idx already checked

        self.name = name
        print('--------------- name: ' + str(self.name) + ' cfg list: ' + str(self.rcols))
        self.mdl, self.coef_, self.y_pred, self.features_ = None, None, None, None
        self.mask_ = np.array([True] * len(self.rcols))                 # default value
        self.index = {i: self.rcols[i] for i in range(len(self.rcols))}      # index that maps to col names in df
        self.df = np.nan
        self.alpha, self.lbda = np.nan, np.nan

        # dates
        sun_cu = this_cu - pd.to_timedelta(6, unit='D')  # cutoff date in week starting Sunday
        upr_date = sun_cu + pd.to_timedelta(upr, unit='W')  # horizon for the perf testing at cutoff_date
        lwr_date = sun_cu + pd.to_timedelta(lwr, unit='W')  # lwr date for perf testing window

        # train
        f_train = f_data[(f_data['ds'] > sun_cu - pd.to_timedelta(upr, unit='W')) & (f_data['ds'] <= sun_cu)].copy()
        data_check(f_train.copy(), 'f_train')
        self.y_scaler = StandardScaler()
        self.y_train = self.y_scaler.fit_transform(np.reshape(f_train['y'].values, (-1, 1)))[:, 0]
        self.x_scaler = StandardScaler()
        self.X_train = self.x_scaler.fit_transform(f_train[self.rcols].values)

        # latent variables
        # x_scaler = StandardScaler()
        # X_train = x_scaler.fit_transform(f_train[rcols].values)
        # fa = FactorAnalysis(n_components=1, tol=1.0e-8, iterated_power=5)
        # _ = fa.fit_transform(X_train)  # latent variable
        # F = fa.components_
        # hf = np.dot(h, F)
        # d = hf - X_train
        # zz = pd.DataFrame(fa.noise_variance_, columns=rcols, index=range(41))
        # z_std = pd.DataFrame({'feature': rcols, 'var': fa.noise_variance_})
        # self.rcols = list(z_std.nsmallest(40, columns=['var'])['feature'])
        # self.mask_ = np.array([True] * len(self.rcols))                 # default value
        # self.index = {i: self.rcols[i] for i in range(len(self.rcols))}      # index that maps to col names in df

        # test
        self.f_test = f_data[(f_data['ds'] >= lwr_date) & (f_data['ds'] <= upr_date)].copy()
        data_check(self.f_test.copy(), 'f_test')

        self.X_test = self.x_scaler.transform(self.f_test[self.rcols].values)
        self.y_test = self.y_scaler.transform(np.reshape(self.f_test['y'].values, (-1, 1)))[:, 0]


    def regr_opt(self):
        self._regr_opt()

    def _regr_opt(self):
        pass

    def regr_set(self):
        if self.mdl is None:
            self.features_ = list()
            self.coef_ = None
            self.mask_ = None
        else:
            self.mdl.fit(self.X_train, self.y_train)
            try:
                self.coef_ = self.mdl.coef_
                self.mask_ = np.abs(self.coef_) > np.finfo(self.coef_.dtype).eps      # create or update the mask
            except AttributeError:                                                            # if no coefs (Trees), the there is a non-null mask
                self.coef_ = None
            self._features()   # sets self.features_ list

            # if len(self.features_) > 0:
            #     ys_pred = self.mdl.predict(self.X_train)
            #     y_pred = self.y_scaler.inverse_transform(ys_pred)
            #     y_train = self.y_scaler.inverse_transform(self.y_train)
            #     train_err = self._perf_err(y_train, y_pred)   # in sample error
            #     print('------------ train err: ' + str(train_err))

    def set_cov(self, df_, res_cols, ols=False):
        # df_ has residuals in res_cols
        drop_list = [c for c in res_cols if df_[c].sum() == 0]
        w_cols = [c for c in res_cols if c not in drop_list]
        w = df_[w_cols].cov()  # DOES MAY NOT WORK when the TS are highly correlated (eg all tickets and Homes: huge inverse
        w_inv = pd.DataFrame(np.linalg.pinv(w.values), w.columns, w.index)  # generalized inverse: w_inv.dot(w) = identity (except for 0 cols)
        return w_inv

    def perf_err(self):
        if self.mdl is None:
            return {'ens': self.name, 'err': np.nan, 'n_features': len(self.features_), 'df': self.df, 'alpha': self.alpha, 'l1_ratio': self.lbda}
        else:
            d_mdl = {
                'ens': self.name,
                'n_features': len(self.features_),
                'df': self.df,
                'alpha': self.alpha,
                'l1_ratio': self.lbda
            }

            y_test = self.y_scaler.inverse_transform(self.y_test)  # the quantity to be forecasted
            x_test = self.f_test[self.features_].copy()
            p_ut.save_df(x_test, '~/my_tmp/x_test')

            # actuals regression based error (bad performance)
            # ys_pred = self.mdl.predict(self.X_test)
            # y_pred = self.y_scaler.inverse_transform(ys_pred)
            # d_mdl['mdl_err'] = self._perf_err(y_test, y_pred)

            # plain avg based error
            y_avg = x_test.mean(axis=1)
            d_mdl['avg_err'] = self._perf_err(y_test, y_avg.values)

            # IC based
            for ic in ['aic', 'aicc', 'bic', 'hqic']:
                _arr = np.exp((-0.5) * np.array([self.IC(y_test, self.f_test[c].values, self.df, method=ic) for c in self.features_]))
                weights = _arr / np.sum(_arr)
                _avg = np.average(x_test, axis=1, weights=weights)
                d_mdl[ic + '_err'] = self._perf_err(y_test, _avg)

            # Regression on adj_y_shifted
            # variables need to be centered and scaled?
            lm = l_mdl.LinearRegression(normalize=False).fit(x_test, self.f_test['adj_y_shifted'].values)
            y_pred = lm.predict(x_test)
            d_mdl['mse_err'] = self._perf_err(y_test, y_pred)

            # adj err: use adj_y_shifted as forecast
            d_mdl['adj_err'] = self._perf_err(y_test, self.f_test['adj_y_shifted'].values)
            return d_mdl

    def _perf_err(self, y_test, y_pred):
        ef = pd.DataFrame({'y': y_test, 'yhat': y_pred})
        return st_ut.wmape(ef)

    def _features(self):
        arr = np.array(range(len(self.mask_)))
        if self.coef_ is None:
            self.features_ = list() if self.mask_ is None else [str(self.index[x]) for x in arr[self.mask_]]
        else:
            self.features_ = [str(self.index[x]) for x in arr[self.mask_]]

    @staticmethod
    def IC(y, yhat, df, method='aicc'):
        # returns the IC for the method, i.e. computes the value to be MINIMIZED for the (Gaussian) least squares linear model
        # https://projecteuclid.org/download/pdfview_1/euclid.aos/1194461726
        # http://statweb.stanford.edu/~jtaylo/courses/stats203/notes/selection.pdf
        # http://web.stanford.edu/~rjohari/teaching/notes/226_lecture11_inference.pdf
        # llf = (-n/2) * (1 + log(2*pi) + log(RSS / n)) = (-n/2) * (1 + log(2*pi) + log(MSE))
        # method: aic, aicc, bic, hqic
        n = np.shape(y)[0]               # number of samples
        mse = np.mean((yhat - y) ** 2)
        llf = (-n / 2) * (1 + np.log(2 * np.pi) + np.log(mse))
        try:
            fname = getattr(evm, method)
            return fname(llf, n, df)
        except AttributeError:
            p_ut.my_print('WARNING: invalid method: ' + str(method))
            return np.nan

    @staticmethod
    def deg_freedom(Xa, lbda2):
        # computes the estimated degrees of freedom needed for the IC
        _, p = np.shape(Xa)
        XaT = np.transpose(Xa)
        i_mtx = np.identity(p)
        try:
            m = inv(np.dot(XaT, Xa) + lbda2 * i_mtx)
            h_mtx = np.dot(np.dot(Xa, m), XaT)
            df = np.trace(h_mtx)
            return df
        except LinAlgError:
            return None


class OLS(Regressor):
    def __init__(self, name, f_data, this_cu, upr, lwr):
        super().__init__(name, f_data, this_cu, upr, lwr)
        self.mdl = l_mdl.LinearRegression(normalize=False)
        self.regr_set()
        self.df = np.sum(self.mask_)
        self.alpha = self.mdl.alpha_


class LassoRegressor(Regressor):
    def __init__(self, name, f_data, this_cu, upr, lwr):
        super().__init__(name, f_data, this_cu, upr, lwr)
        self.mdl = l_mdl.LassoLarsIC(criterion='aic', normalize=False)
        self.regr_set()
        self.df = np.sum(self.mask_)
        self.alpha = self.mdl.alpha_


class OMP(Regressor):
    def __init__(self, name, f_data, this_cu, upr, lwr):
        super().__init__(name, f_data, this_cu, upr, lwr)
        self.mdl = l_mdl.OrthogonalMatchingPursuit(normalize=False)
        self.regr_set()
        self.df = np.sum(self.mask_)


class EnetOpt(Regressor):
    def __init__(self, name, f_data, this_cu, upr, lwr, max_evals=1000, verbose=False):
        super().__init__(name, f_data, this_cu, upr, lwr)
        self.max_evals = max_evals
        self.verbose = verbose
        self.iter, self.valid_iter, self.string, self.min_loss = 0, 0, '', np.inf
        self.params, self.loss, self.mdl, self.df, self.l1_ratio = None, None, None, None, None
        self.do_test = True
        self.space_list = self.get_paths()
        self.regr_opt()                    # find opt pars at init time
        if self.params is not None:
            self.alpha = self.params['alpha']
            self.lbda = self.params['l1_ratio']
        self.regr_set()
        if self.params is not None:
            self.ridge_par = 2.0 * len(self.y_train) * self.params['alpha'] * (1 - self.params['l1_ratio'])
        else:
            s_ut.my_print('WARNING: default ridge parameter <<<<<<<<<<<<<<<<<<<<<<<< ')
            self.ridge_par = 1.0
        s_ut.my_print('enet: ' + str(self.params) + ' loss: ' + str(self.loss) + ' df: ' + str(self.df) + ' n_features: ' + str(len(self.features_)))

    def hyperpar_tuning(self, params):
        self.iter += 1
        params['l1_ratio'] = self.l1_ratio
        mdl = ElasticNet(**params, normalize=False)
        loss, df = self.enet_loss(mdl)     # min IC
        if loss < self.min_loss:
            self.min_loss = loss
        if bool(np.isinf(loss)) is False:  # must be ==
            self.valid_iter += 1
        self.string += 'valid_iter: ' + str(self.valid_iter) + ' iter: ' + str(self.iter) + \
                       ' loss: ' + str(loss) + ' params: ' + str(params) + ' df: ' + str(df) + ' min_loss: ' + str(self.min_loss) + '\n'
        return {'loss': loss, 'status': STATUS_OK, 'params': params}

    def _regr_opt(self, evals=None):            # optimize on the last w weeks before cutoff
        if evals is None:
            evals = self.max_evals

        self.iter, self.valid_iter = 0, 0
        best_params = {'alpha': 1.0, 'l1_ratio': 0.5}
        for space_ in self.space_list:
            trials = Trials()
            self.l1_ratio = space_['l1_ratio']
            space = {'alpha': hp.uniform('alpha', space_['alpha_min'], space_['alpha_max'])}
            with s_ut.suppress_stdout_stderr():
                _ = fmin(fn=self.hyperpar_tuning, space=space, algo=tpe.suggest, max_evals=int(evals), trials=trials, show_progressbar=False)
            self.string += '+++++++++++++++++++ iter: ' + str(self.iter) + ' valid iter: ' + str(self.valid_iter) + ' evals: ' + str(int(evals)) + '\n'
            if self.verbose:
                print(self.string)
                self.string = ''
            res = trials.results
            best_res = sorted(res, key=lambda x: x['loss'])[0]
            this_loss = best_res['loss']
            if this_loss <= self.min_loss:
                best_params = best_res['params']
                best_params['l1_ratio'] = self.l1_ratio
                self.min_loss = this_loss
        self.loss = self.min_loss
        self.params = best_params
        self.mdl = ElasticNet(**self.params, normalize=False)
        if bool(np.isinf(self.min_loss)) is True:
            if self.do_test is True:
                s_ut.my_print('ERROR: &&&&&&&&&&&&&&&&&&&&&&& no valid iterations. Turn SW off')
                self.do_test = False
                self._regr_opt()
            else:
                s_ut.my_print('ERROR: &&&&&&&&&&&&&&&&&&&&&&& FATAL: no valid iterations')

    def regr_set(self):
        super().regr_set()
        if self.mdl is None:
            self.df = np.nan
        else:
            Xa = self.X_train[:, self.mask_]
            lbda2 = (1 - self.mdl.l1_ratio) * self.mdl.alpha * len(self.X_train)
            self.df = self.deg_freedom(Xa, lbda2)

    def enet_loss(self, mdl):        # wrapper to compute the loss (IC)
        n = len(self.y_train)
        mdl.fit(self.X_train, self.y_train)
        mask = np.abs(mdl.coef_) > np.finfo(mdl.coef_.dtype).eps
        lbda2 = (1 - mdl.l1_ratio) * mdl.alpha * n
        if np.sum(mask) > 0:
            Xa = self.X_train[:, mask]
            df = self.deg_freedom(Xa, lbda2)
        else:
            df = 0
        if df is None:
            self.string += 'aic: inf, df: None '
            return np.inf, np.nan
        else:
            yhat = mdl.predict(self.X_train)
            pval = self._sw_test(yhat - self.y_train)
            aic_ = self.IC(self.y_train, yhat, df)
            ic = (np.inf if df <= 0.0 else aic_) if pval > 0.05 else np.inf
            self.string += 'pval: ' + str(pval) + ' aic: ' + str(aic_) + ' df: ' + str(df) + ' lbda2: ' + str(lbda2) + ' mask: ' + str(np.sum(mask)) + ' '
            return ic, df

    def _sw_test(self, res):  # residuals should be normal (pval > 0.05) for IC
        return sw_test(res)[1] if self.do_test is True else 1.0

    def get_paths(self, num=10, n_alphas=10):
        l1_ratio_list = np.linspace(max(0.01, 1.0 / num), 1.0, endpoint=False)
        space_list = list()
        for l1 in l1_ratio_list:
            alphas, coefs, _ = enet_path(self.X_train, self.y_train, n_alphas=n_alphas, l1_ratio=l1)
            f = pd.DataFrame(coefs).transpose()            # cols = features, one row per alpha
            g = f.applymap(lambda x: np.abs(x) > 0.0)
            nz = pd.DataFrame({'nz': g.sum(axis=1), 'alpha': alphas})  # number of non-zero coefs and min alpha needed
            nz.sort_values(by='nz', inplace=True)
            nz.reset_index(inplace=True, drop=True)
            alpha_max = nz['alpha'].max()    # if alpha > alpha_max then all coefs are 0
            space_list.append({'l1_ratio': l1, 'alpha_max': alpha_max, 'alpha_min': 0.0})
        return space_list


class RidgeOpt(Regressor):
    def __init__(self, name, f_data, this_cu, upr, lwr, mask, alpha, verbose=False):
        super().__init__(name, f_data, this_cu, upr, lwr)
        if mask is not None and len(mask) > 0:
            self.X_train = self.X_train[:, mask]
            self.X_test = self.X_test[:, mask]
            self.mask_ = mask
        self.mdl = l_mdl.Ridge(alpha, normalize=False)
        self.df = self.deg_freedom(self.X_train, alpha)
        self.alpha = alpha
        self.regr_set()


class EnsOpt(Regressor):
    def __init__(self, name, data, cutoff_date, upr, lwr, mask, loss_type='rel', max_evals=200, verbose=False):
        super().__init__(name, data, cutoff_date, upr, lwr)
        self.max_evals = max_evals
        self.loss_type = loss_type if loss_type in ['abs', 'rel'] else 'rel'
        self.iter, self.string = 0, ''
        if mask is not None and len(mask) > 0:
            self.X_train = self.X_train[:, mask]
            self.X_test = self.X_test[:, mask]
            self.mask_ = mask
        self.space = SPACE_DICT[self.name]
        self.params, self.loss, self.mdl = None, None, None
        try:
            self.rfunc = getattr(sk_ens, name)
        except AttributeError:
            try:
                self.rfunc = getattr(xgb, name)
            except AttributeError:
                s_ut.my_print('ERROR: ' + name + ' not found')
        self.regr_opt()
        self.regr_set()
        if verbose:
            print(self.string)
        s_ut.my_print(self.name + ': ' + str(self.params) + ' loss: ' + str(self.loss) + ' n_features: ' + str(len(self.features_)))

    def hyperpar_tuning(self, params):
        self.iter += 1
        params['n_estimators'] = int(params['n_estimators'])  # always int
        if self.name == 'XGBRegressor':
            params['max_depth'] = int(params['max_depth'])   # must be int for xgb
            params['objective'] = 'reg:squarederror'         # unique to xgb
            params['booster'] = 'gbtree'                     # unique to xgb

        regr = self.rfunc(**params)                          # sets the regressor object instance
        regr.fit(self.X_train, self.y_train)
        y_pred = regr.predict(self.X_test)
        loss = np.mean((self.y_test - y_pred) ** 2) if self.loss_type == 'abs' else np.mean(((self.y_test - y_pred) / self.y_test) ** 2)
        self.string += 'iter: ' + str(self.iter) + ' loss: ' + str(loss) + ' pars: ' + str(params) + '\n'
        return {'loss': loss, 'status': STATUS_OK, 'params': params}

    def _regr_opt(self):  # optimize on the last w weeks before cutoff
        trials = Trials()
        with s_ut.suppress_stdout_stderr():
            _ = fmin(fn=self.hyperpar_tuning, space=self.space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials, show_progressbar=False)
        results = trials.results
        best_res = sorted(results, key=lambda x: x['loss'])[0]
        self.params = best_res['params']
        self.params['n_estimators'] = int(self.params['n_estimators'])
        if self.name == 'XGBRegressor':
            self.params['max_depth'] = int(self.params['max_depth'])
            self.params['objective'] = 'reg:squarederror'
            self.params['booster'] = 'gbtree'
        self.loss = best_res['loss']
        self.mdl = self.rfunc(**self.params)      # optimized regressor obj instance


