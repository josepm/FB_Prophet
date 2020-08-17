"""

"""
import pandas as pd
import os
import sys
from functools import reduce

from capacity_planning.forecast.utilities.holidays import make_holidays as hdays
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.forecast.utilities.language import data_processing as d_proc


class TimeSeries(object):
    def __init__(self, name, data, fcast_date, cutoff_date, init_date, data_cfg, time_scale='D', gcols=('language',)):
        self.name = name
        self.ycol = data_cfg.get('ycol', None)
        self.agg_dict = data_cfg.get('agg_dict', None)
        self.abandon_par = data_cfg.get('abandon_par', None)
        self.cutoff_date = cutoff_date
        self.init_date = init_date
        self.fcast_date = fcast_date
        self.outlier_coef = data_cfg.get('outlier_coef', 3.0)
        self.ceiling = data_cfg.get('ceiling', 1.0)
        self.floor = data_cfg.get('floor', 0.0)
        self.time_scale = time_scale
        self.gcols = list(gcols)
        self.pre_process = data_cfg.get('pre-process', None)
        try:
            self.lang_idx = self.gcols.index('language')
        except ValueError:
            self.lang_idx = None  # no language grouping (no hols)
        self.max_int = 5 if self.time_scale == 'D' else (2 if self.time_scale == 'W' else None)
        self.df_dict, self.hf_dict = self.prepare_data(data)

    def prepare_data(self, data_df):  # clean input data and build holiday DF
        # gcols_ = ['ds', 'language']
        if 'language' in data_df.columns:
            data_df['language'].replace(['Mandarin_Onshore', 'Mandarin_Offshore'], ['Mandarin', 'Mandarin'], inplace=True)
        else:
            data_df['language'] = 'NULL'

        dl_groups = data_df.groupby(['ds'] + self.gcols).agg(self.agg_dict).reset_index()
        df_dict, hf_dict = dict(), dict()
        for tpl, gf in dl_groups.groupby(self.gcols):
            if isinstance(tpl, str):
                tpl = (tpl, )
            s_ut.my_print('************* data checks and hols for group: ' + str(tpl))
            mf = None if self.ycol is None else d_proc.data_check(gf[['ds', self.ycol]].copy(), self.ycol, 'ds', self.cutoff_date, self.init_date,
                                                                  max_int=self.max_int, name=str(tpl), unit=self.time_scale)

            if mf is None:
                s_ut.my_print('WARNING: data_check failed for label ' + str(tpl))
                p_ut.save_df(gf, '~/my_tmp/gf_' + str(tpl))
                continue
            else:
                lang = tpl[self.lang_idx] if self.lang_idx is not None else None
                d_df = self.set_demand(mf)
                h_df = self.get_holidays(lang) if lang is not None else None
                hf_dict[lang] = h_df             # dict of hols DF by language (key)
                t_df = self.trim_outliers(d_df, h_df, self.outlier_coef, ['y'], lbl_dict=tpl)  # trim outliers
                if t_df is not None:
                    for ix in range(len(self.gcols)):  # add the grouping cols
                        t_df[self.gcols[ix]] = tpl[ix]
                    if lang not in df_dict.keys():
                        df_dict[lang] = t_df  # dict of trimmed DF by language
                    else:
                        zx = pd.concat([df_dict[lang].copy(), t_df], axis=0)
                        df_dict[lang] = zx
        return df_dict, hf_dict

    def set_demand(self, t_df):
        # compute the quantity to forecast for each time series
        if self.name == 'phone-inbound-vol':
            # :::::::::::: adjust abandonments with abandonment retries :::::::::::::: #
            # pABN = a / (A + a): observed abandonment prob
            # A: accepted calls
            # a: observed (non-deduped) abandonments
            # r = number of retries per abandonment
            # b = unique (de-duplicated) abandonments: b * (1 + r) = a
            # actual demand D = A + b = A + a / (1 + r)
            # retries model r = r0 * pABN / (1 - pABN) because retries grow with pABN
            # r0 = 10.0  because at pABN ~ 5%, avg retries are about 0.5, ie r0 = 10
            # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
            pABN = t_df['abandons'] / (t_df['accepted'] + t_df['abandons'])
            retries = self.abandon_par * pABN / (1.0 - pABN)
            t_df['y'] = t_df['accepted'] + t_df['abandons'] / (1 + retries)
        elif self.name == 'phone-outbound-vol':  # inbound/outbound data split done after data clean step
            t_df.rename(columns={'calls': 'y'}, inplace=True)
        elif self.name == 'phone-inbound-aht':  # inbound/outbound data split done after data clean step
            t_df = t_df[t_df['calls'] > 0].copy()
            ttl_mins = t_df['agent_mins']  # + t_df['agent_rcv_consult_mins'] + t_df['agent_init_consult_mins']
            t_df['y'] = ttl_mins / t_df['calls']
        elif self.name == 'phone-outbound-aht':  # inbound/outbound data split done after data clean step
            t_df = t_df[t_df['calls'] > 0].copy()
            ttl_mins = t_df['agent_mins']  # + t_df['agent_rcv_consult_mins'] + t_df['agent_init_consult_mins']
            t_df['y'] = ttl_mins / t_df['calls']
        elif self.name == 'deferred':
            pass
        elif self.name == 'deferred_hbat':
            pass
        elif self.name == 'ticket_count':
            t_df['y'] = t_df['ticket_count']
        elif self.name == 'prod_hours':
            t_df['y'] = t_df['prod_hours']
        elif self.name == 'ticket_count_Homes':
            t_df['y'] = t_df['ticket_count']
        elif self.name == 'prod_hours_Homes':
            t_df['y'] = t_df['prod_hours']
        elif self.name == 'ticket_count_China':
            t_df['y'] = t_df['ticket_count']
        elif self.name == 'prod_hours_China':
            t_df['y'] = t_df['prod_hours']
        elif self.name == 'ticket_count_Experiences':
            t_df['y'] = t_df['ticket_count']
        elif self.name == 'prod_hours_Experiences':
            t_df['y'] = t_df['prod_hours']
        elif self.name == 'booking_count':
            t_df['y'] = t_df['booking_count']
        elif self.name == 'checkin_count':
            t_df['y'] = t_df['checkin_count']
        elif self.name == 'prod_hours':
            t_df['y'] = t_df['prod_hours']
        elif self.name == 'prod_hours_Homes':
            t_df['y'] = t_df['prod_hours']
        elif self.name == 'prod_hours_China':
            t_df['y'] = t_df['prod_hours']
        elif self.name == 'prod_hours_Experiences':
            t_df['y'] = t_df['prod_hours']
        elif self.name == 'tenure':
            t_df['y'] = t_df['tenure_days']
        else:
            s_ut.my_print('ERROR: unknown ts_name: ' + str(self.name))
            sys.exit()
        return t_df[['ds', 'y']].copy()

    def get_holidays(self, lang):   # holidays per language
        prefix = 'not-' if 'not-' in lang else ''
        language = prefix + 'Mandarin' if 'Mandarin' in lang else lang
        end_year = self.fcast_date.year
        holidays_df = hdays.get_hols(language, end_year)  # returns None if language not valid (e.g. language = foo), returns all languages if language = None
        if self.time_scale == 'D':
            pass
        elif self.time_scale == 'W':
            gcols = ['language', pd.Grouper(key='ds', freq=self.time_scale)] if 'language' in holidays_df.columns else pd.Grouper(key='ds', freq=self.time_scale)
            holidays_df = holidays_df.groupby(gcols).apply(self.w_hols).reset_index()
            holidays_df.drop('level_2', axis=1, inplace=True)
        else:
            s_ut.my_print('ERROR: invalid time scale: ' + str(self.time_scale))
            sys.exit()

        # set the right time scale for holidays
        if holidays_df is None:
            s_ut.my_print('WARNING: no holidays DF for ' + language)
            return None
        else:
            holidays_df.drop('language', axis=1, inplace=True)
            return holidays_df[(holidays_df['ds'] <= self.fcast_date) & (holidays_df['ds'] >= self.init_date)]

    @staticmethod
    def w_hols(hf):   # drop upper and lower window for weekly
        u_hols = [h.replace('\'', '_').replace(',', '_') for h in hf['holiday'].unique()]
        return pd.DataFrame({'holiday': [','.join(u_hols)], 'upper_window': [0], 'lower_window': [0]})

    @staticmethod
    def trim_outliers(mf_, h_df_, o_coef_, cols_, lbl_dict=None):
        def _ps_outliers(f_, tc_, c_, ocoef, hdates_, lbl_dict_):
            x, _, o_df = p_ut.ts_outliers(f_, tc_, c_, coef=ocoef, verbose=False, replace=True, ignore_dates=hdates_, lbl_dict=lbl_dict_, r_val=0.0)
            return x[[tc_, c_]]

        if h_df_ is not None:
            dates_arr = h_df_.apply(
                lambda x: [x['ds'] + pd.to_timedelta(ix, unit='D') for ix in range(x['lower_window'], x['upper_window'])], axis=1).values  # holiday and window
            h_dates = list(set([dt for dt_list in dates_arr for dt in dt_list]))  # list of hol dates
        else:
            h_dates = list()
        df_list_ = [_ps_outliers(mf_[['ds', c]].copy(), 'ds', c, o_coef_, h_dates, lbl_dict) for c in cols_]
        t_df_out = reduce(lambda x, y: x.merge(y, on='ds', how='inner'), df_list_) if len(df_list_) > 0 else None
        return t_df_out


# used in ticket forecast
class TicketForecast(object):
    def __init__(self, file_path):
        s_ut.my_print('setting forecast from ' + file_path)
        t_info = file_path.split('.')[0].split('/')[-1]
        self.raw = True if 'raw' in t_info else False
        self.adj = not self.raw
        self.rolling = True if '_r_' in t_info else False
        self.cutoff_date = pd.to_datetime(t_info.split('_')[-1])
        self.has_actuals = True if '_xls_' in t_info else False
        self.data = p_ut.read_df(file_path)
        p_ut.clean_cols(self.data, ["language", "service_tier", "channel", "business_unit"],
                        '~/my_repos/capacity_planning/data/config/col_values.json',
                        check_new=False,
                        do_nan=False,
                        rename=True)
        if 'ds_week_ending' in self.data.columns:
            self.data['ds'] = pd.to_datetime(self.data['ds_week_ending']) - pd.to_timedelta(6, unit='D')
            self.data.drop('ds_week_ending', inplace=True, axis=1)
        self.forecast = (self.cutoff_date + pd.to_timedelta(7, unit='D')).month_name()
        self.froot = file_path.split('.')[0][:-10]

