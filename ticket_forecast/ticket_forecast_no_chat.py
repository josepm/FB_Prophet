"""
# put the files in the right place
# ticket data: '~/my_tmp/ticket_by_bus_channel_language_nt.tsv'
# bookings_file = '~/my_tmp/ticket_and_booking.tsv'
# tiers_file = '~/my_tmp/tier_splits_newT.tsv'
$ python ticket_forecast.py run__id run_date rerun
run_id: is not used but must be there for redspot runs. Any number will do
run_date: is the date we launch
rerun: if 1 if we will rerun a done forecast. if 0, do not rerun a completed forecast
"""
from scipy import stats

from scipy.special import boxcox, inv_boxcox
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import pandas as pd

import os
from fbprophet import Prophet
import statsmodels.api as sm
import numpy as np
from itertools import product
import sys

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.ticket_forecast import errors as errs
from capacity_planning.utilities import time_utils as t_ut

with s_ut.suppress_stdout_stderr():
    pass


def get_last_sat(r_date):
    #  get last Saturday date strictly before r_date. If r_date is a Saturday, give the date of the Saturday prior to r_date
    dt = pd.to_datetime(r_date)
    idx = (dt.weekday() + 1) % 7
    sat = dt - pd.to_timedelta(7 + idx - 6, unit='D')
    return sat


def to_pre_week_ending(curr_day):
    if isinstance(curr_day, str):
        curr_day = datetime.strptime(curr_day, '%Y-%m-%d')
    idx = (curr_day.weekday() + 2) % 7
    pre_sat = (curr_day - timedelta(days=idx)).strftime("%Y-%m-%d")
    return pre_sat


def to_x_weeks_ending(curr_day, wks):
    if isinstance(curr_day, str):
        curr_day = datetime.strptime(curr_day, '%Y-%m-%d')
    idx = 7 * wks - (curr_day.weekday() + 2) % 7
    x_wks_sat = (curr_day + timedelta(days=idx)).strftime("%Y-%m-%d")
    return x_wks_sat


def to_curr_week_ending(curr_day):
    if isinstance(curr_day, str):
        curr_day = datetime.strptime(curr_day, '%Y-%m-%d')
    idx = 7 - ((curr_day.weekday() + 2) % 7)
    next_sat = (curr_day + timedelta(days=idx)).strftime("%Y-%m-%d")
    return next_sat


def to_pre_sun(curr_day):
    if isinstance(curr_day, str):
        curr_day = datetime.strptime(curr_day, '%Y-%m-%d')
    idx = (curr_day.weekday() + 1) % 7
    pre_sun = (curr_day - timedelta(days=idx)).strftime("%Y-%m-%d")
    return pre_sun


def to_7d_later(curr_day):
    if isinstance(curr_day, str):
        curr_day = datetime.strptime(curr_day, '%Y-%m-%d')
    next7d = (curr_day + timedelta(days=7)).strftime("%Y-%m-%d")
    return next7d


def idx_to_week_ending(idx):
    idx = idx.to_series().apply(lambda x: to_curr_week_ending(x))
    return idx


def data_clean(dt, pre_sat):
    # sort, truncate and format data
    dt = dt.sort_values(by='ds_week_ending')
    dt = dt[dt['ds_week_ending'] <= pre_sat]
    dt['ds_week_starting'] = dt['ds_week_ending'].apply(lambda x: datetime.strftime(
        (datetime.strptime(x, '%Y-%m-%d') - timedelta(days=6)), '%Y-%m-%d'))
    dt.index = dt['ds_week_starting']
    cols = dt.columns.tolist()
    cols.remove('ds_week_ending')
    cols.remove('ds_week_starting')
    dim_cols = list()
    val_cols = list()
    for col in cols:
        if 'dim' in col:
            dim_cols.append(col)
        else:
            val_cols.append(col)
            dt[col] = pd.to_numeric(dt[col])
    # fill missing week by interpolation
    if len(dim_cols) < 1:
        start_date = dt['ds_week_starting'].iloc[0]
        last_date = dt['ds_week_starting'].iloc[-1]
        end_date = to_pre_sun(pre_sat)
        idata = pd.DataFrame(index=pd.date_range(
            start=start_date, end=end_date, freq='7D'), columns=cols)
        idata.index = idata.index.to_series().dt.strftime('%Y-%m-%d')
        for idx in dt.index:
            idata.loc[idx, cols] = dt.loc[idx, cols]
        if end_date > last_date:
            idata.loc[last_date:end_date, cols] = 0.0
        for col in val_cols:
            idata[col].fillna(0.0, inplace=True)
            # idata[col] = idata[col].astype(float).interpolate()
        return idata

    dim_comb = dict()
    output = list()
    for k in range(len(dim_cols)):
        dim_comb[dim_cols[k]] = dt[dim_cols[k]].unique()
    for comb in product(*dim_comb.values()):
        ds = dt
        for i, col in enumerate(dim_comb.keys()):
            ds = ds[ds[col] == comb[i]]
        if len(ds) > 1:
            start_date = ds['ds_week_starting'].iloc[0]
            last_date = ds['ds_week_starting'].iloc[-1]
            end_date = to_pre_sun(pre_sat)
            ids = pd.DataFrame(index=pd.date_range(
                start=start_date, end=end_date, freq='7D'), columns=cols)
            ids.index = ids.index.to_series().dt.strftime('%Y-%m-%d')
            for idx in ds.index:
                ids.loc[idx, cols] = ds.loc[idx, cols]
            if end_date > last_date:
                ids.loc[last_date:end_date, val_cols] = 0.0
            for col in dim_comb.keys():
                ids[col] = ids[col].fillna(ds[col].any())
            for col in val_cols:
                ids[col].fillna(0.0, inplace=True)
                # ids[col] = ids[col].astype(float).interpolate()
            output.append(ids)
    return pd.concat(output)


def tier_fill(df):
    df['ds_week_starting'] = df.index
    df['dim_tier'] = df['dim_tier'].apply(
        lambda x: 'plus' if x in ['a4w'] else x)
    df['dim_tier'] = df['dim_tier'].apply(
        lambda x: 'safety' if x in ['law-enforcement'] else x)
    df['dim_tier'] = df['dim_tier'].apply(
        lambda x: 'payments' if x in ['payments-and-payouts'] else x)
    df['dim_tier'] = df['dim_tier'].apply(lambda x: 'other' if x in [
                                          'open-homes', 'photo-ops', 'photography', 'risk-regulation', 'trust', 'trust-and-safety'] else x)
    df = df.groupby(['ds_week_starting', 'dim_business_unit',
                     'dim_language', 'dim_tier', 'dim_channel']).sum().reset_index()

    df_exp = df[df['dim_tier'].apply(lambda x: x in ['experiences', 'plus'])]

    df_adj = df[df['dim_tier'].apply(
        lambda x: x not in ['experiences', 'plus'])]
    df_nounk = df_adj[df_adj['dim_tier'] != '-unknown-'].copy()

    dfg_adj = df_adj.groupby(
        ['ds_week_starting', 'dim_business_unit', 'dim_language']).sum().reset_index()
    dfg_nounk = df_nounk.groupby(
        ['ds_week_starting', 'dim_business_unit', 'dim_language']).sum().reset_index()
    merged = pd.merge(dfg_adj, dfg_nounk, on=[
                      'ds_week_starting', 'dim_business_unit', 'dim_language'], suffixes=('_all', '_adj'))
    merged['adj'] = (merged['ticket_count_all'] + 0.01) / \
        (merged['ticket_count_adj'] + 0.01)
    del merged['ticket_count_all'], merged['ticket_count_adj']
    df_nounk = pd.merge(df_nounk, merged, on=[
        'ds_week_starting', 'dim_business_unit', 'dim_language'])
    df_nounk['ticket_count'] = df_nounk['ticket_count'] * df_nounk['adj']
    del df_nounk['adj']

    df_adj = pd.concat([df_nounk, df_exp], ignore_index=True)
    df_adj.index = df_adj['ds_week_starting']
    del df_adj['ds_week_starting']
    df_adj = df_adj.sort_index()
    return df_adj


def group_agg(df, level):
    print('**************::: ' + level)
    print(df.head())
    if level == 'bu':
        l_grouped = df.groupby(
            ['ds_week_starting', 'dim_business_unit']).sum().reset_index()
        l_grouped['pert'] = l_grouped[['ds_week_starting', 'ticket_count']].groupby(
            ['ds_week_starting']).transform(lambda x: x.sum())
    elif level == 'bu_lang':
        l_grouped = df.groupby(
            ['ds_week_starting', 'dim_business_unit', 'dim_language']).sum().reset_index()
        l_grouped['pert'] = l_grouped[['ds_week_starting', 'dim_business_unit', 'ticket_count']].groupby(
            ['ds_week_starting', 'dim_business_unit']).transform(lambda x: x.sum())
    elif level == 'bu_lang_tier':
        l_grouped = df.groupby(
            ['ds_week_starting', 'dim_business_unit', 'dim_language', 'dim_tier']).sum().reset_index()
        l_grouped['pert'] = l_grouped[['ds_week_starting', 'dim_business_unit', 'dim_language', 'ticket_count']].groupby(
            ['ds_week_starting', 'dim_business_unit', 'dim_language']).transform(lambda x: x.sum())
    elif level == 'bu_lang_channel':
        l_grouped = df.groupby(
            ['ds_week_starting', 'dim_business_unit', 'dim_language', 'dim_channel']).sum().reset_index()
        l_grouped['pert'] = l_grouped[['ds_week_starting', 'dim_business_unit', 'dim_language', 'ticket_count']].groupby(
            ['ds_week_starting', 'dim_business_unit', 'dim_language']).transform(lambda x: x.sum())
    elif level == 'bu_lang_tier_channel':
        l_grouped = df.groupby(
            ['ds_week_starting', 'dim_business_unit', 'dim_language', 'dim_tier', 'dim_channel']).sum().reset_index()
        l_grouped['pert'] = l_grouped[['ds_week_starting', 'dim_business_unit', 'dim_language', 'dim_tier', 'ticket_count']].groupby(
            ['ds_week_starting', 'dim_business_unit', 'dim_language', 'dim_tier']).transform(lambda x: x.sum())
    l_grouped['pert'] = (l_grouped['ticket_count']) / \
        (l_grouped['pert'] + 0.00000001)
    l_grouped.index = l_grouped['ds_week_starting']
    l_grouped = l_grouped.sort_index()
    return l_grouped


def bc_trans(data, lbd=None):
    idata = data.sort_index()
    idata = idata + 1.1
    lambda_list = list()
    for tdata in [idata[:-2], idata[:-1], idata]:
        bc_lambda = stats.boxcox_normmax(tdata)
        lambda_list.append(max(min(bc_lambda, 1), 0.5))
    if lbd is None:
        return boxcox(idata, np.mean(lambda_list)), np.mean(lambda_list)
    else:
        return boxcox(idata, lbd), lbd


def phr_forecast_noex_lag(col_lag, data_dict, fcst_end, growth='linear'):
    tdata = data_dict[col_lag[0]]
    lag = col_lag[1]
    data_train = tdata[:-lag] if lag > 0 else tdata
    data_train = data_train.to_frame(name='y')
    data_train['ds'] = data_train.index
    # data_train['ds'] = data_trian.index
    # data_train['y'] = data_train
    # try:
    if 'pert' in col_lag[0] and growth == 'l':
        m = Prophet(growth='logistic')
        data_train['cap'] = max(tdata) * 1.25
    else:
        m = Prophet()
    m.fit(data_train)
    fp = (datetime.strptime(to_pre_sun(fcst_end), '%Y-%m-%d') -
          datetime.strptime(to_pre_sun(data_train.index[-1]), '%Y-%m-%d')).days / 7
    future = m.make_future_dataframe(
        periods=fp, freq='w', include_history=False)
    if 'pert' in col_lag[0] and growth == 'l':
        future['cap'] = max(tdata) * 1.25
    forecast = m.predict(future)
    output = pd.DataFrame(forecast['yhat'].values, index=forecast['ds'])
    return output


def phr_forecast_noex(data, fcst_end, wid, trans='y', tr_lda=None):

        # except:
        #    raise ValueError("Something is wrong when call arima")
    if isinstance(data, pd.Series):
        col_list = [data.name]
        data = data.to_frame()
        data.columns = col_list
    else:
        col_list = data.columns.values.tolist()
    tdata_dict = dict()
    lda_dict = dict()
    if trans == 'y':
        if tr_lda is None:
            for col in col_list:
                tdata_dict[col], lda_dict[col] = bc_trans(data[col])
        else:
            for col in col_list:
                tdata_dict[col], lda_dict[col] = bc_trans(data[col], tr_lda)
    else:
        for col in col_list:
            tdata_dict[col] = data[col]
    col_lag_list = list(product(col_list, range((wid - 1), -1, -1)))
    forecast_dict = dict()
    for col_lag in col_lag_list:
        if trans == 'l':
            forecast_dict[col_lag] = phr_forecast_noex_lag(
                col_lag, tdata_dict, fcst_end, 'l')
        else:
            forecast_dict[col_lag] = phr_forecast_noex_lag(
                col_lag, tdata_dict, fcst_end)
    output_list = list()
    idx = list()
    for col_name in col_list:
        df_col = list()
        for col_lag in col_lag_list:
            if col_lag[0] == col_name:
                df_col.append(forecast_dict[col_lag])
                df = pd.concat(df_col, axis=1).mean(axis=1)[(wid - 1):]
        df.columns = col_name
        df.index = df.index.to_series().dt.strftime('%Y-%m-%d')
        df = pd.concat([tdata_dict[col_name], df])
        if trans == 'y':
            forecast_val = inv_boxcox(df, lda_dict[col_name])
        else:
            forecast_val = df
        forecast_val[forecast_val < 0.0] = 0
        output_list.append(forecast_val)
    output = pd.concat(output_list, axis=1)
    output.columns = col_list
    return output


def phr_forecast_exfcst_lag(lag, tdata, fcst_end, exfcst):
    exog_train = exfcst[:(len(tdata) - lag)]
    exog_fcst = exfcst[(len(tdata) - lag):]

    data_train = tdata[:(len(tdata) - lag)]
    data_train = data_train.to_frame(name='y')
    data_train['ds'] = data_train.index

    m = Prophet()
    for col in exog_fcst.columns:
        data_train[col] = exog_train[col].values
        m.add_regressor(col)

    m.fit(data_train)
    fp = (datetime.strptime(to_pre_sun(fcst_end), '%Y-%m-%d') -
          datetime.strptime(to_pre_sun(data_train.index[-1]), '%Y-%m-%d')).days / 7

    # print('periods: ' + str(fp) + ' type: ' + str(type(fp)))

    future = m.make_future_dataframe(
        periods=int(np.round(fp, 0)), freq='w', include_history=False)
    for col in exog_fcst.columns:
        future[col] = exog_fcst[col].values

    forecast = m.predict(future)
    output = pd.DataFrame(forecast['yhat'].values, index=forecast['ds'])
    return output


def phr_forecast_exfcst(data, fcst_end, wid, exfcst=None, tr_lda=None):

    if exfcst is None:
        return phr_forecast_noex(data, fcst_end, wid)
    if to_pre_sun(fcst_end) != exfcst.index[-1]:
        raise ValueError('exog fcst does NOT match end of fcst date')
    if tr_lda is None:
        tdata, lda = bc_trans(data)
    else:
        tdata, lda = bc_trans(data, tr_lda)
    forecast_list = list()
    for lag in range((wid - 1), -1, -1):
        # print ('fcst lag: ' + str(lag) + ' using exfcst')
        forecast_list.append(phr_forecast_exfcst_lag(
            lag, tdata, fcst_end, exfcst))
    df = pd.concat(forecast_list, axis=1).mean(axis=1)[(wid - 1):]
    if isinstance(data, pd.Series):
        df.name = data.name
    else:
        df.columns = data.columns
    df.index = df.index.to_series().dt.strftime('%Y-%m-%d')
    df = pd.concat([tdata, df])
    return inv_boxcox(df, lda)


def func(x1, a, b, c):
    return a * np.exp(-b * x1) + c


def cr_chg(ds):
    if isinstance(ds, pd.Timestamp) or isinstance(ds, datetime):
        ds = ds.strftime('%Y-%m-%d')
    if ds <= '2016-03-01':
        return 1
    elif (ds > '2016-04-15') and (ds <= '2016-05-16'):
        return 0.875
    elif (ds > '2016-06-03') and (ds <= '2017-05-15'):
        return 0.8
    elif (ds > '2017-07-07') and (ds <= '2017-08-01'):
        return 0.95
    elif (ds > '2017-08-25') and (ds <= '2017-09-20'):
        return 1.05
    elif (ds > '2017-10-01') and (ds <= '2017-11-15'):
        return 1
    elif ds > '2017-12-01' and ds <= '2018-01-05':
        return 1.175
    elif ds > '2018-01-05' and ds <= '2018-05-01':
        return 1
    elif ds > '2018-05-15' and ds <= '2018-08-18':
        return 1.105
    elif ds > '2018-09-14' and ds <= '2019-01-15':
        return 1.25
    elif ds > '2019-02-01':
        return 1.265
    else:
        return np.nan


def cr_forecast(data, fcst_end):
    # Extend the data until fcst_end date
    start_date = data.index[0]
    end_date = data.index[-1]
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime('%Y-%m-%d')

    idata = pd.Series(index=pd.date_range(start=start_date, end=fcst_end, freq='7D').to_pydatetime())
    for idx in data.index:
        idata.loc[idx] = data.loc[idx]

    l = len(idata.loc[:end_date])
    t = np.array(range(len(idata)))

    cr_chg_ts = idata.index.to_series().apply(lambda x: cr_chg(x))
    cr_chg_ts = cr_chg_ts.interpolate()
    apopt, _ = curve_fit(func, (t[:l]), idata[:l] / cr_chg_ts[:l], bounds=(0, [1., 1., 1.]))
    cr_exog = func((np.array(t)), *apopt) * cr_chg_ts
    # plt.plot(cr_exog)
    # plt.plot(idata)
    # plt.show()
    # apopt, _ = curve_fit(func, (t[:l], cr_chg_ts[:l]),
    #                     idata[:l], bounds=(0, [1., 1., 1.]))
    # cr_exog = func((np.array(t), cr_chg_ts), *apopt)
    # plt.plot(idata[:l], 'r-')
    # plt.plot(idata.index, func((np.array(t), cr_exog), *apopt), 'g--')
    data_train = (idata[:l] - cr_exog[:l].values).to_frame(name='y')
    data_train['ds'] = data_train.index.to_series(
    ).dt.strftime('%Y-%m-%d').values

    m = Prophet()
    m.fit(data_train)
    future = m.make_future_dataframe(
        periods=(len(idata) - l), freq='w', include_history=False)
    forecast = m.predict(future)

    cr_fcst = pd.Series(forecast['yhat'].values, index=forecast['ds'])
    cr_fcst = cr_fcst + cr_exog[l:].values
    ds = pd.concat([idata[:l], cr_fcst])
    ds.index = ds.index.to_series().dt.strftime('%Y-%m-%d')
    return ds


def phr_forecast(data, fcst_end, trans='y', exog=None, wid=5, tr_lda=None):
    fcst_end = to_pre_sun(fcst_end)
    if exog is None:
        foutput = phr_forecast_noex(data, fcst_end, wid, trans, tr_lda)
        foutput[foutput < 0.0] = 0
        return foutput

    exog_fcst = pd.DataFrame()
    if fcst_end == exog.index[-1]:
        exog_fcst = exog
    else:
        if len(data) != len(exog):
            raise ValueError(
                'exog length is not the same as input data length')

        if isinstance(exog, pd.Series):
            exog_fcst = phr_forecast_noex(
                exog, fcst_end, wid, trans).to_frame()
        else:
            col_list = exog.columns.values.tolist()
            if 'contact_rate' in col_list:
                exog_fcst['contact_rate'] = cr_forecast(
                    exog['contact_rate'], fcst_end)
                col_list.remove('contact_rate')

            exog_fcst_add = phr_forecast_noex(
                exog[col_list], fcst_end, wid, trans)
            for col in col_list:
                exog_fcst[col] = exog_fcst_add[col]
            if all(x in exog.columns for x in
                   ['contact_rate', 'checkins', 'bookings']):
                exog_fcst['contact_rate'] = exog_fcst['contact_rate'] * \
                    (exog_fcst['checkins'] + exog_fcst['bookings'])
                del exog_fcst['checkins']
                del exog_fcst['bookings']

    foutput = phr_forecast_exfcst(
        data, fcst_end, wid, exog_fcst, tr_lda)
    foutput[foutput < 0.0] = 0
    return foutput


def arima_forecast_noex(data, fcst_end, wid, sea='y', trans='y', tr_lda=None):
    def arima_forecast_noex_lag(col_lag, data_dict, fcst_end, sea):
        sorder = (0, 0, 0, 0)
        tr = [1, 1]
        tdata = data_dict[col_lag[0]]
        lag = col_lag[1]

        data_train = tdata[:-lag] if lag > 0 else tdata
        if len(data_train) > 104 and sea == 'y':  # or (exog is not None):
            sorder = (1, 1, 0, 52)
            tr = None
        try:
            mod = sm.tsa.statespace.SARIMAX(
                data_train, order=(0, 1, 1), seasonal_order=sorder, trend=tr)
            fit_res = mod.fit(disp=False)
            return fit_res.predict(
                start=data_train.index[-1], end=fcst_end, freq='W')[1:]
        except:
            raise ValueError("Something is wrong - switching Prophet")

    if isinstance(data, pd.Series):
        col_list = [data.name]
        data = data.to_frame()
        data.columns = col_list
    else:
        col_list = data.columns.values.tolist()
    tdata_dict = dict()
    lda_dict = dict()
    if trans == 'y':
        if tr_lda is None:
            for col in col_list:
                tdata_dict[col], lda_dict[col] = bc_trans(data[col])
        else:
            for col in col_list:
                tdata_dict[col], lda_dict[col] = bc_trans(data[col], tr_lda)
    else:
        for col in col_list:
            tdata_dict[col] = data[col]
    col_lag_list = list(product(col_list, range((wid - 1), -1, -1)))
    forecast_dict = dict()
    for col_lag in col_lag_list:
        print('fcst ' + col_lag[0] + ' with lag ' + str(col_lag[1]))
        try:
            forecast_dict[col_lag] = arima_forecast_noex_lag(col_lag, tdata_dict, fcst_end, sea)
        except:
            forecast_dict[col_lag] = phr_forecast_noex_lag(col_lag, tdata_dict, fcst_end)
    output_list = list()
    for col_name in col_list:
        df_col = list()
        for col_lag in col_lag_list:
            if col_lag[0] == col_name:
                df_col.append(forecast_dict[col_lag])
                df = pd.concat(df_col, axis=1).mean(axis=1)[(wid - 1):]
        df.columns = col_name
        df.index = df.index.to_series().dt.strftime('%Y-%m-%d')
        df = pd.concat([tdata_dict[col_name], df])
        if trans == 'y':
            forecast_val = inv_boxcox(df, lda_dict[col_name])
        else:
            forecast_val = df
        forecast_val[forecast_val < 0.0] = 0
        output_list.append(forecast_val)
    output = pd.concat(output_list, axis=1)
    output.columns = col_list
    return output


def arima_forecast_exfcst(data, fcst_end, wid, sea='y', exfcst=None, tr_lda=None):
    def arima_forecast_exfcst_lag(lag, wid, tdata, fcst_end, sea, exfcst):
        sorder = (0, 0, 0, 0)
        tr = [1, 1]
        exog = exfcst[:len(tdata)]
        exog_fcst = exfcst[(len(tdata) - wid):]

        data_train = tdata[:-lag] if lag > 0 else tdata
        exog_train = exog[:-lag] if lag > 0 else exog
        if len(data_train) > 104 and sea == 'y':  # or (exog is not None):
            sorder = (1, 1, 0, 52)
            tr = None
        try:
            mod = sm.tsa.statespace.SARIMAX(
                data_train, order=(0, 1, 1), seasonal_order=sorder, trend=tr, exog=exog_train)
            fit_res = mod.fit(disp=False)
        # len of exog_fcst should be 1 less than the length of [start:end] since start is in-sample data
            return fit_res.predict(start=data_train.index[-1], end=fcst_end, exog=exog_fcst[(wid - lag - 1):])[1:]
        except:
            raise ValueError(
                'Something wrong when using exog for forecasting - switching Prophet')

    fcst_end = to_pre_sun(fcst_end)
    if exfcst is None:
        return arima_forecast_noex(data, fcst_end, wid, sea, tr_lda)
    if to_pre_sun(fcst_end) != exfcst.index[-1]:
        raise ValueError('exog fcst does NOT match end of fcst date')
    if tr_lda is None:
        tdata, lda = bc_trans(data)
    else:
        tdata, lda = bc_trans(data, tr_lda)
    forecast_list = list()
    for lag in range((wid - 1), -1, -1):
        try:
            forecast_list.append(arima_forecast_exfcst_lag(
                lag, wid, tdata, fcst_end, sea, exfcst))
        except:
            forecast_list.append(phr_forecast_exfcst_lag(
                lag, tdata, fcst_end, exfcst))
    df = pd.concat(forecast_list, axis=1).mean(axis=1)[(wid - 1):]
    df.index = df.index.to_series().dt.strftime('%Y-%m-%d')
    df = pd.concat([tdata, df])
    return inv_boxcox(df, lda)


def arima_forecast(data, fcst_end, sea='y', trans='y', exog=None, wid=5, tr_lda=None):
    end_date = data.index[-1]
    fcst_end = to_pre_sun(fcst_end)
    if exog is None:
        foutput = arima_forecast_noex(data, fcst_end, wid, sea, trans, tr_lda)
        foutput[foutput < 0.0] = 0
        return foutput

    exog_fcst = pd.DataFrame()
    if fcst_end == exog.index[-1]:
        exog_fcst = exog
    else:
        if len(data) != len(exog):
            raise ValueError(
                'exog length is not the same as input data length')

        if isinstance(exog, pd.Series):
            print('fcst one exog parameter')
            exog_fcst = arima_forecast_noex(exog, fcst_end, wid, sea, trans).to_frame()
        else:
            col_list = exog.columns.values.tolist()
            if 'contact_rate' in col_list:
                print('fcst contact rate')
                exog_fcst['contact_rate'] = cr_forecast(
                    exog['contact_rate'], fcst_end)
                col_list.remove('contact_rate')

            print('fcst multiple exog parameters')
            exog_fcst_add = arima_forecast_noex(exog[col_list], fcst_end, wid, sea, trans)
            for col in col_list:
                exog_fcst[col] = exog_fcst_add[col]
            if all(x in exog.columns for x in
                   ['contact_rate', 'checkins', 'bookings']):
                exog_fcst['contact_rate'] = exog_fcst['contact_rate'] * \
                    (exog_fcst['checkins'] + exog_fcst['bookings'])
                del exog_fcst['checkins']
                del exog_fcst['bookings']
                exog_fcst['contact_rate'] = exog_fcst['contact_rate'].interpolate()
    print('fcst original ts using exog parameters')
    foutput = arima_forecast_exfcst(data, fcst_end, wid, sea, exog_fcst, tr_lda)
    foutput[foutput < 0.0] = 0

    return foutput


def ts_dict_trunc(l):
    s = l.values()[0].index
    for e in l.values():
        s = s.intersection(e.index)
    dict_trunc = dict()
    for n in l.keys():
        dict_trunc[n] = l[n][s]
    return dict_trunc

# def composeof(upper, lower):
#     if upper == 'all' or upper == lower:
#         return True
#     if lower.count('-') == 1:
#         parent = lower.split('-')[0]
#         if upper == parent:
#             return True
#     if lower.count('-') == 2:
#         lowerl = lower.split('-')
#         grandparent = lowerl[0]
#         parent = lowerl[0] + '-' + lowerl[1]
#         if upper == parent or upper == grandparent:
#             return True
#     return False

# def htc(rs_dict):
#     rs = ts_dict_trunc(rs_dict)
#     num_rs = rs.keys()
#     bt = np.max([x.count('-') for x in num_rs])
#     if bt == 0:
#         num_series = list(compress(num_rs, [x != 'all' for x in num_rs]))
#     else:
#         num_series = list(
#             compress(num_rs, [x.count('-') == bt for x in num_rs]))
#
#     a = np.zeros([len(num_rs), len(num_series)])
#     for j, na in enumerate(num_series):
#         i = [i for i, x in enumerate(num_rs) if composeof(x, na)]
#         a[i, j] = 1
#
#     b = rs.values()
#     b_hat = np.linalg.lstsq(a, b)[0]
#     rs_hat = a.dot(b_hat)
#     rs_corr = dict()
#     for i, na in enumerate(num_rs):
#         rs_corr[na] = pd.Series(rs_hat[i, :], rs[na].index)
#     return rs_corr


def forecast_bu(bu_grouped, fcst_end_date, cb):
    rs_bu = dict()
    for bu in bu_grouped['dim_business_unit'].unique():
        input_data = bu_grouped[bu_grouped['dim_business_unit']
                                == bu][['ticket_count']]
        print("processing " + bu)
        if bu in ['Home']:
            tr_lda = None
            wid = 7
            exog = cb
        elif bu in ['China']:
            tr_lda = 0.2
            wid = 7
            exog = None
        elif bu in ['Experiences']:
            tr_lda = 0.4
            wid = 8
            exog = None
        rs_bu[bu] = phr_forecast(
            input_data['ticket_count'], fcst_end_date, wid=wid, exog=exog, tr_lda=tr_lda)
    return rs_bu


def forecast_bu_lang(bl_grouped, fcst_end_date, cb, noex_lang_lst):
    rs_bu_lang = dict()
    exog = None
    tr_lda = None
    wid = 5
    for (bu, lang) in bl_grouped.groupby(['dim_business_unit', 'dim_language']).groups.keys():
        input_data = bl_grouped[(bl_grouped['dim_business_unit']
                                 == bu) & (bl_grouped['dim_language']
                                           == lang)][['ticket_count']]
        print('processing at language level : ' + bu + ' - ' + lang)
        # in case that certain language does have enough long history as checkins
        if len(cb) > len(input_data):
            idx = input_data.index
            cb = cb.loc[idx, ['checkins', 'bookings', 'contact_rate']]
        else:
            input_data = input_data[-len(cb):]

        if bu in ['Homes']:
            tr_lda = None
            wid = 5
            if lang in noex_lang_lst:
                exog = None
            else:
                exog = cb
        elif bu in ['China']:
            if lang == 'chinese':
                tr_lda = 0.15
            else:
                tr_lda = None
            wid = 8
            exog = None
        elif bu in ['Experiences']:
            tr_lda = 0.25
            wid = 8
            exog = None
        else:
            wid = 5
        rs_bu_lang[(bu, lang)] = phr_forecast(
            input_data['ticket_count'], fcst_end_date, wid=wid, exog=exog, tr_lda=tr_lda)
    return rs_bu_lang


def forecast_bu_lang_arima(bl_grouped, fcst_end_date, cb, noex_lang_lst, run_date):
    cut_date = to_pre_sun(to_x_weeks_ending(run_date, 1))
    rs_bu_lang = dict()
    for (bu, lang) in bl_grouped.groupby(['dim_business_unit', 'dim_language']).groups.keys():
        exog = None
        tr_lda = None
        wid = 5
        input_data = bl_grouped[(bl_grouped['dim_business_unit']
                                 == bu) & (bl_grouped['dim_language']
                                           == lang)][['ticket_count']]
        print('processing at langugage level : ' + bu + ' - ' + lang)
        # in case that certain language does have enough long history as checkins
        if len(cb) > len(input_data):
            idx = input_data.index
            cb = cb.loc[idx, ['checkins', 'bookings', 'contact_rate']]
        else:
            input_data = input_data[-len(cb):]

        if bu in ['Homes']:
            tr_lda = None
            wid = 6
            exog = cb
            if lang in noex_lang_lst:
                exog = None
            else:
                exog = cb

            if lang == 'japanese':
                wid = 4
                tr_lda = 0.95
            if lang == 'portuguese':
                wid = 9
                tr_lda = 0.25
            if lang == 'spanish':
                wid = 7
                tr_lda = 0.5
            if lang == 'german':
                wid = 9
                tr_lda = 0.5
            if lang == 'korean':
                wid = 8
                tr_lda = 0.7
            if lang in ['engEMEA']:
                tr_lda = 0.4
                wid = 9
            if lang in ['engNA']:
                tr_lda = 0.5
                wid = 9
            if lang in ['engAPAC']:
                tr_lda = 0.4
                wid = 9
            if lang in ['russian']:
                tr_lda = 0.5
                wid = 5
            if lang in ['italian']:
                tr_lda = 0.25
                wid = 9
            if lang in ['french']:
                tr_lda = 0.35
                wid = 9

        elif bu in ['China']:
            if lang == 'chinese':
                tr_lda = 0.1
            else:
                tr_lda = None
            wid = 6
            exog = None
        elif bu in ['Experiences']:
            tr_lda = 0.25
            wid = 8
            exog = None
        else:
            wid = 5

#        if lang in ['french', 'italian'] and bu == 'Homes':
#            rs_bu_lang[(bu, lang)] = phr_forecast(
#                input_data['ticket_count'], fcst_end_date, wid=wid, exog=exog, tr_lda=tr_lda)
#        else:
        rs_bu_lang[(bu, lang)] = arima_forecast(input_data['ticket_count'], fcst_end_date, wid=wid, exog=exog, tr_lda=tr_lda)[cut_date:fcst_end_date]
    return rs_bu_lang


def forecast_bu_lang_tier(blt_grouped, fcst_end_date, rs_bu_lang, run_date):
    rs_bu_lang_tier = dict()
    cut_date = to_pre_sun(to_x_weeks_ending(run_date, 1))
    width = 10
    for (bu, lang) in blt_grouped.groupby(['dim_business_unit', 'dim_language']).groups.keys():
        print('processing at tier level : ' + bu + ' - ' + lang)
        input_data = blt_grouped[(
            blt_grouped['dim_business_unit'] == bu) & (blt_grouped['dim_language'] == lang)][['dim_tier', 'pert']]
 #        if bu == 'Experiences':
 #            rs_bu_lang_tier[(bu, lang, 'Experiences')
 #                            ] = rs_bu_lang[(bu, lang)].iloc[:, 0]
 #            continue
        if len(input_data) < 3:
            continue
        tr_rs = dict()
        for tier in input_data['dim_tier'].unique():
            tier_pct = input_data[input_data['dim_tier'] == tier]['pert']
            print('process    -----       ' + bu + '_' + lang + '_' + tier + '    ------: ' + str(len(tier_pct)))
            if len(tier_pct) < 26:
                continue
            if tier == 'plus':
                tier_fcst = phr_forecast(
                    tier_pct, fcst_end_date, trans='l', wid=width)
            else:
                tier_fcst = phr_forecast(
                    tier_pct, fcst_end_date, trans='n', wid=width)
            tr_rs[tier] = tier_fcst.iloc[:, 0]
        tr_rs = pd.DataFrame(tr_rs, index=tier_fcst.index)
        tr_rs.fillna(0.0, inplace=True)
        tr_rs[tr_rs < 0.0] = 0.0
        tr_rs = tr_rs.div((tr_rs.sum(axis=1) + 0.0000001), axis=0)
        tr_rs = tr_rs[cut_date:]
        if isinstance(rs_bu_lang[(bu, lang)], pd.Series):
            for ky in tr_rs.columns:
                tr_rs[ky] = (rs_bu_lang[(bu, lang)].values) * tr_rs[ky]
                rs_bu_lang_tier[(bu, lang, ky)] = tr_rs[ky]
        else:
            for ky in tr_rs.columns:
                tr_rs[ky] = (rs_bu_lang[(bu, lang)].iloc[:, 0]) * tr_rs[ky]
                rs_bu_lang_tier[(bu, lang, ky)] = tr_rs[ky]
    return rs_bu_lang_tier


def bu_lang_tier_split(tier_split, fcst_end_date, rs_bu_lang, run_date):
    rs_bu_lang_tier = dict()

    for (bu, lang) in blt_grouped.groupby(['dim_business_unit', 'dim_language']).groups.keys():
        print('processing at tier level : ' + bu + ' - ' + lang)
        if bu == 'Experiences':
            rs_bu_lang_tier[(bu, lang, 'experiences')
                            ] = rs_bu_lang[(bu, lang)].iloc[:, 0]
            continue
        lang_pc = lang if lang not in ('nonChinese') else 'engAPAC'
        tier_pert = tier_split[tier_split['dim_language']
                               == lang_pc][['dim_tier', 'pert']]
        for ky in tier_pert['dim_tier'].unique():
            if isinstance(rs_bu_lang[(bu, lang)], pd.Series):
                rs_bu_lang_tier[(bu, lang, ky)] = (rs_bu_lang[(
                    bu, lang)]) * tier_pert[tier_pert['dim_tier'] == ky]['pert'].values[0]
            else:
                rs_bu_lang_tier[(bu, lang, ky)] = (rs_bu_lang[(
                    bu, lang)].iloc[:, 0]) * tier_pert[tier_pert['dim_tier'] == ky]['pert'].values[0]
    return rs_bu_lang_tier


def list_to_lang_channel(rs_lang_channel):
    # list to dataframe
    rs_lang_channel = pd.concat(rs_lang_channel)
    rs_lang_channel.fillna(0.0, inplace=True)
    # sort the results
    rs_lang_channel['ds_week_ending'] = idx_to_week_ending(
        rs_lang_channel.index)
    rs_lang_channel = rs_lang_channel.reset_index()
    rs_lang_channel = rs_lang_channel.sort_values(
        by=['ds_week_ending', 'dim_language'])
    # arrange the columns of the results
    cols = rs_lang_channel.columns.tolist()
    cols.insert(0, cols.pop(cols.index('dim_language')))
    cols.insert(0, cols.pop(cols.index('ds_week_ending')))
    rs_lang_channel = rs_lang_channel[cols]
    del rs_lang_channel['index']
    return rs_lang_channel


def forecast_bu_lang_tier_ch(bltc_grouped, fcst_end_date, rs_bu_lang_tier):
    fcst_start = to_7d_later(bltc_grouped.index[-1])
    fcst_end = to_pre_sun(fcst_end_date)
    rs_bu_lang_tier_ch = dict()
    pct_bu_lang_tier_ch = dict()
    for (bu, lang, tier) in bltc_grouped.groupby(['dim_business_unit', 'dim_language', 'dim_tier']).groups.keys():
        print('processing at channel level : ' +
              bu + ' - ' + lang + ' - ' + tier)
        input_data = bltc_grouped[(
            bltc_grouped['dim_business_unit'] == bu) & (bltc_grouped['dim_language'] == lang) & (bltc_grouped['dim_tier'] == tier)][['dim_channel', 'pert']]
        if len(input_data) < 3:
            continue
        ch_rs = dict()
        email_rdx = 0.0
        for ch in input_data['dim_channel'].unique():
            ch_pct = input_data[input_data['dim_channel'] == ch]['pert']
            if len(ch_pct) < 3:
                continue
            ch_fcst = pd.Series(index=pd.date_range(
                start=fcst_start, end=fcst_end, freq='7D'))
            if (len(ch_pct) < 1):
                ch_fcst.loc[fcst_start] = 0.0
            elif (lang in ['chinese', 'korean', 'japanese']) and ch == 'directly':
                ch_fcst.loc[fcst_start] = 0.0
            else:
                # average of the last 4 week's of channel split as baseline
                avg_4w = np.mean(
                    ch_pct[-min(len(ch_pct), 4):])
                ch_fcst.loc[fcst_start] = avg_4w
                ch_fcst.iloc[-1] = avg_4w
                # shift from email channel to messaging
                if ch == 'email' and tier in ['community-education', 'resolutions-1', 'resolutions-2']:
                    ch_fcst.iloc[-1] = avg_4w * 0.7
                    email_rdx = avg_4w * 0.3
                if ch == 'messaging' and tier in ['community-education', 'resolutions-1', 'resolutions-2']:
                    ch_fcst.iloc[-1] = avg_4w + email_rdx
                if ch == 'other' and bu == 'Experiences':
                    ch_fcst.iloc[-1] = 0.0
            ch_rs[ch] = ch_fcst.interpolate()

        ch_rs = pd.DataFrame(ch_rs, index=ch_fcst.index)
        ch_rs.fillna(0.0, inplace=True)
        ch_rs[ch_rs < 0.0] = 0.0
        ch_rs = ch_rs.div((ch_rs.sum(axis=1) + 0.0000001), axis=0)
        for ky in ch_rs.columns:
            if (bu, lang, tier) in rs_bu_lang_tier.keys():
                pct_bu_lang_tier_ch[(bu, lang, tier, ky)] = ch_rs[ky]
                rs_bu_lang_tier_ch[(bu, lang, tier, ky)] = (
                    rs_bu_lang_tier[(bu, lang, tier)].values) * ch_rs[ky]
    return rs_bu_lang_tier_ch, pct_bu_lang_tier_ch


def dict_to_table(rs_bu_lang_tier_ch, run_date):
    output = list()
    for ky, fcst in rs_bu_lang_tier_ch.items():
        temp = pd.DataFrame(fcst)
        temp.columns = ['ticket_count']
        temp['time_interval'] = 'week'
        temp['dim_business_unit'] = ky[0]
        temp['dim_language'] = ky[1]
        temp['dim_tier'] = ky[2]
        temp['dim_channel'] = ky[3]
        temp['run_date_inv_ending'] = run_date
        temp['fcst_date_inv_ending'] = idx_to_week_ending(temp.index)
        temp['fcst_horizon'] = pd.to_datetime(temp['fcst_date_inv_ending']) - pd.to_datetime(temp['run_date_inv_ending'])
        temp['fcst_horizon'] = temp['fcst_horizon'].apply(lambda x: x.total_seconds() / (3600 * 24 * 7) - 1)
        output.append(temp)
    output_df = pd.concat(output).reset_index()
    del output_df['index']
    cols = ['dim_business_unit', 'dim_language', 'dim_tier', 'dim_channel',
            'time_interval', 'run_date_inv_ending', 'fcst_date_inv_ending', 'fcst_horizon', 'ticket_count']
    # output_df['fcst_horizon'] = output_df['fcst_horizon']  # .astype(int)
#    output['ticket_count'] = output['ticket_count'].fillna(0.0, inplace=True)
#     output_df['ticket_count'] = output_df['ticket_count']   # .astype(int)
    return output_df[cols]


def pct_to_excel(bltc_grouped, pct_bu_lang_tier_channel, hist_cut_off_date):
    output = list()
    for ky, pct_fcst in pct_bu_lang_tier_channel.items():
        temp = pd.DataFrame(pct_fcst)
        temp.columns = ['pert']
        temp['dim_business_unit'] = ky[0]
        temp['dim_language'] = ky[1]
        temp['dim_tier'] = ky[2]
        temp['dim_channel'] = ky[3]
        temp['ds_week_ending'] = idx_to_week_ending(temp.index)
        output.append(temp)
    fcst_pct = pd.concat(output).reset_index()

    hist_pct = bltc_grouped.copy()
    hist_pct['ds_week_ending'] = hist_pct['ds_week_starting'].apply(lambda x: datetime.strftime(
        (datetime.strptime(x, '%Y-%m-%d') + timedelta(days=6)), '%Y-%m-%d'))
    hist_pct = hist_pct[['dim_business_unit', 'dim_language', 'dim_tier',
                         'dim_channel', 'ds_week_ending', 'pert']]

    output_pct = pd.concat([hist_pct, fcst_pct])
    output_pct = output_pct[output_pct['ds_week_ending'] > hist_cut_off_date]
    output_pct = pd.pivot_table(output_pct, values='pert', index=[
        'dim_business_unit', 'dim_language', 'dim_tier', 'ds_week_ending'], columns=['dim_channel'], aggfunc=sum).reset_index()
    return output_pct


def rs_to_excel(hist_ticket, fcst_ticket, hist_cut_off_date, r_date):
    # all, all - (safety + claims), all - directly, all - (safety + claims + directly)
    def to_speadsheet(df_):
        df_tier = pd.pivot_table(df_, values='ticket_count',
                                 index=['dim_business_unit', 'dim_channel', 'dim_language'],
                                 columns=['ds_week_ending'], aggfunc=sum).reset_index()
        df_tier_all = pd.pivot_table(df_, values='ticket_count',
                                     index=['dim_business_unit', 'dim_language'],
                                     columns=['ds_week_ending'], aggfunc=sum).reset_index()
        df_tier_all['dim_channel'] = 'all'
        output = pd.concat([df_tier_all, df_tier], sort=True)
        cols = output.columns.values.tolist()
        cols.remove('dim_business_unit')
        cols.remove('dim_channel')
        cols.remove('dim_language')
        cols.insert(0, 'dim_language')
        cols.insert(0, 'dim_channel')
        cols.insert(0, 'dim_business_unit')
        return output[cols]

    output = dict()
    hist_ticket = hist_ticket.copy()
    hist_ticket['ds_week_ending'] = hist_ticket['ds_week_starting'].apply(lambda x: datetime.strftime(
        (datetime.strptime(x, '%Y-%m-%d') + timedelta(days=6)), '%Y-%m-%d'))
    hist_ticket = hist_ticket[['dim_business_unit', 'dim_language', 'dim_tier',
                               'dim_channel', 'ds_week_ending', 'ticket_count']]

    df_f = fcst_ticket[
        ['dim_business_unit', 'dim_language', 'dim_tier','dim_channel', 'fcst_date_inv_ending', 'ticket_count']].copy().\
        rename(columns={"fcst_date_inv_ending": "ds_week_ending"})
    df = pd.concat([hist_ticket, df_f])
    df = df[df['ds_week_ending'] >= hist_cut_off_date].copy()
    p_ut.save_df(df, '~/Forecasts/par/xls_df_' + str(r_date.date()))  # save the data that will go to xls in parquet

    output['all'] = to_speadsheet(df)
    df_nodirectly = df[df['dim_tier'] != 'directly']
    output['all - directly'] = to_speadsheet(df_nodirectly)
    df_ = df[~((df['dim_tier'] == 'safety') | (df['dim_tier'] == 'claims'))]
    output['all - (safety&claims)'] = to_speadsheet(df_)
    df_ = df[~((df['dim_tier'] == 'safety') | (df['dim_tier'] == 'claims') | (df['dim_tier'] == 'directly'))]
    output['all - (safety&claims&directly)'] = to_speadsheet(df_)
    for tr in df['dim_tier'].unique():
        output[tr] = to_speadsheet(df[df['dim_tier'] == tr])
    return output


def adjust_ag_tiers(zx, r_date, weeks):
    # adjust dim_tier counts for the last 2 weeks, ie. when ds_week_ending equals runs_date and run_date - 1 week
    # use ratios for run_date - 2 weeks and run_date - 3 weeks to adjust dim_tier counts

    def set_sums(adf, cols_):
        wsum = pd.DataFrame(adf.groupby(cols_).apply(lambda x: x['ticket_count'].sum()))
        wsum.columns = ['ticket_count']
        wsum.reset_index(inplace=True)
        return wsum

    cols = zx.columns
    w_old = zx[(zx['ds_week_ending'] <= r_date - pd.to_timedelta(2, unit='W')) & (zx['ds_week_ending'] >= r_date - pd.to_timedelta(weeks, unit='W'))].copy()
    w = zx[zx['ds_week_ending'] >= r_date - pd.to_timedelta(3, unit='W')].copy()          # data of interest: settled and to adjust
    w_good = w[w['ds_week_ending'] <= r_date - pd.to_timedelta(2, unit='W')].copy()       # good data (settled): weeks -2 and -3
    w_good.drop('ds_week_ending', axis=1, inplace=True)
    if len(w_good) == 0:
        s_ut.my_print('WARNING: no actuals adjustment performed because not enough data')
        return zx

    # get the ratios from the settled weeks
    tsums = set_sums(w_good, ['dim_business_unit', 'dim_language', 'dim_channel', 'dim_tier'])     # tier level sums (group dates)
    lsums = set_sums(tsums, ['dim_business_unit', 'dim_language', 'dim_channel'])                  # lang level sums
    gm = lsums.merge(tsums, on=['dim_business_unit', 'dim_language', 'dim_channel'], how='left')
    gm['ratio'] = gm['ticket_count_y'] / gm['ticket_count_x']
    gm.drop(['ticket_count_x', 'ticket_count_y'], axis=1, inplace=True)
    gm.drop_duplicates(inplace=True)

    # get the (correct) totals (lang level) from the non-settled weeks
    w_bad = w[w['ds_week_ending'] > r_date - pd.to_timedelta(2, unit='W')].copy()                                     # bad data (not settled): weeks 0 and -1
    lsum = set_sums(w_bad, ['ds_week_ending', 'dim_business_unit', 'dim_language', 'dim_channel'])                    # lang level sums (correct)
    bm = w_bad.merge(lsum, on=['ds_week_ending', 'dim_business_unit', 'dim_language', 'dim_channel'], how='left')
    bm.drop('ticket_count_x', axis=1, inplace=True)
    bm.rename(columns={'ticket_count_y': 'total_count'}, inplace=True)                                                # total_count contains correct lang totals

    # adjust
    f = bm.merge(gm, on=['dim_business_unit', 'dim_language', 'dim_channel', 'dim_tier'], how='left')        # match settled ratios to groups
    f['ticket_count'] = np.round(f['total_count'] * f['ratio'], 0)                                           # adjust tix counts
    f = f[f['ticket_count'].notnull()]                                                                       # drop combos missing in the settled weeks. Should be very small
    f['ticket_count'] = f['ticket_count'].astype(int)
    return pd.concat([w_old, f[cols].copy()], axis=0)


if __name__ == '__main__':
    # ######################################
    # Parameters
    # ######################################
    s_ut.my_print(sys.argv)
    HIST_CUT_OFF_DATE = '2016-01-01'
    no_exog_lang = ['korean', 'chinese', 'portuguese']
    if len(sys.argv) == 2:
        run_date = sys.argv[1]   # at least 3 days after last Saturday with actual data
        rerun = 0
        s_ut.my_print('WARNING: rerun not set in command line. Assuming no rerun')
    elif len(sys.argv) == 4:
        _, _, run_date, rerun = sys.argv  # at least 3 days after last Saturday with actual data
    elif len(sys.argv) == 3:
        _, _, run_date = sys.argv  # at least 3 days after last Saturday with actual data
        rerun = 0
        s_ut.my_print('WARNING: rerun not set in command line. Assuming no rerun')
    else:
        print('invalid args: ' + str(sys.argv))
        sys.exit()
    # ######################################
    # ######################################

    run_date = get_last_sat(run_date)   # set to last saturday
    fcst_end_date = run_date + pd.to_timedelta(75, unit='W')
    s_ut.my_print('----- start forecast pipeline from ' + str(run_date.date()) + ' to ' + str(fcst_end_date.date()) + '  -----')

    # output files
    out_file = os.path.expanduser('~/Forecasts/par/' + 'table_output_' + str(run_date.date()) + '.par')
    xls_file = os.path.expanduser('~/Forecasts/xls/_forecast_bu_nt_' + str(run_date.date()) + '.xlsx')
    if os.path.exists(out_file) and os.path.exists(xls_file) and int(rerun) == 0:  # job already ran!
        s_ut.my_print('found output files. exiting')
        print('DONE')
        sys.exit()

    # edit the queries and save to files
    dim_ds = pd.to_datetime('today') - pd.to_timedelta(5, unit='D')       # ${hiveconf:DIM_DS}
    sat_ds = run_date                                     # ${hiveconf:SAT_DS}
    sat_ds2 = run_date - pd.to_timedelta(2, unit='W')     # ${hiveconf:SAT_DS2}

    set_dict = {'DIM_DS':   str(dim_ds.date()),
                'S1_DS':  str(sat_ds.date()),
                'S2_DS':  str(sat_ds2.date())
                }
    p_dict = {'set': '--set', 'SET': '--SET', 'Set': '--Set', 'cs_tickets__cs__hive': 'cs_tickets__cs', 'string': 'VARCHAR'}
    tix_qry_file = os.path.expanduser('/data/vol/ticket_forecast/tickets.hql')
    tiers_qry_file = os.path.expanduser('/data/vol/ticket_forecast/tier_splits.hql')
    b_qry_file = os.path.expanduser('/data/vol/ticket_forecast/bookings_and_checkins.hql')
    zd_tiers_qry_file = os.path.expanduser('/data/vol/ticket_forecast/zendesk-tiers.hql')

    tix_qfiles = hql.gen_query(tix_qry_file, 'tickets_qry', set_dict, date=str(run_date.date()))
    tiers_qfiles = hql.gen_query(tiers_qry_file, 'tiers_qry', set_dict, date=str(run_date.date()))
    b_qfiles = hql.gen_query(b_qry_file, 'bk_qry', set_dict, date=str(run_date.date()))
    z_qfiles = hql.gen_query(zd_tiers_qry_file, 'zd_qry', set_dict, date=str(run_date.date()))

    # run the queries and save to output files
    tickets_file = os.path.expanduser('~/my_tmp/ticket_by_bus_channel_language_nt_' + str(run_date.date()))
    bookings_file = os.path.expanduser('~/my_tmp/ticket_and_booking_' + str(run_date.date()))
    tiers_file = os.path.expanduser('~/my_tmp/tier_splits_newT_' + str(run_date.date()))
    zd_tiers_file = os.path.expanduser('~/my_tmp/zd_tiers_' + str(run_date.date()))

    arg_list = [(tix_qfiles, tickets_file), (tiers_qfiles, tiers_file), (b_qfiles, bookings_file), (z_qfiles, zd_tiers_file)]
    f_list = s_ut.do_mp(hql.run_hql, arg_list, is_mp=True, cpus=None, do_sigkill=True)

    print(" == Cleaning data first == ")
    df, db, tier, zd_df = None, None, None, None
    for f in f_list:
        if f == -1:
            s_ut.my_print('ERROR: some data collection failed: ' + str(f_list))
            sys.exit()
        if 'ticket_by_bus_channel_language_nt' in f:
            df = p_ut.read_df(f)

            # week_starting patch
            df_cols_ = df.columns
            if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
                df['ds_week_ending'] = pd.to_datetime(df['ds_week_ending'])
                df['ds_week_starting'] = df['ds_week_ending'] - pd.to_timedelta(6, unit='D')

        elif 'ticket_and_booking' in f:
            db = p_ut.read_df(f)

            # week_starting patch
            df_cols_ = db.columns
            if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
                db['ds_week_ending'] = pd.to_datetime(db['ds_week_ending'])
                db['ds_week_starting'] = db['ds_week_ending'] - pd.to_timedelta(6, unit='D')

        elif 'tier_splits_newT' in f:
            tier = p_ut.read_df(f)

            # week_starting patch
            df_cols_ = tier.columns
            if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
                tier['ds_week_ending'] = pd.to_datetime(tier['ds_week_ending'])
                tier['ds_week_starting'] = tier['ds_week_ending'] - pd.to_timedelta(6, unit='D')

        elif 'zd_tiers_' in f:
            zd_df_ = p_ut.read_df(f)

            # week_starting patch
            df_cols_ = zd_df_.columns
            if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
                zd_df_['ds_week_ending'] = pd.to_datetime(zd_df_['ds_week_ending'])
                zd_df_['ds_week_starting'] = zd_df_['ds_week_ending'] - pd.to_timedelta(6, unit='D')

            zd_df_['ds_week_ending'] = pd.to_datetime(zd_df_['ds_week_ending'].values)
            p_ut.save_df(zd_df_, '~/my_tmp/pre_adj_df_' + str(run_date.date()))
            zd_df = adjust_ag_tiers(zd_df_, run_date, 16)           # weeks = max weeks back from run_date
            zd_df['ds_week_starting'] = (zd_df['ds_week_ending'] - pd.to_timedelta(6, unit='D')).dt.date.astype(str)
            zd_df.drop('ds_week_ending', axis=1, inplace=True)
            p_ut.save_df(zd_df, '~/my_tmp/adj_df_' + str(run_date.date()))
        else:
            s_ut.my_print('ERROR: invalid file: ' + str(f))
            sys.exit()

    tier_h = tier[tier['dim_business_unit'] == 'HomeAndChina'].copy()
    tier_h['dim_tier'] = tier_h['dim_tier'].apply(lambda x: 'resolutions-1' if x in ('other', 'experiences') else x)
    tier_h.loc[(tier_h['dim_tier'] == 'directly'), 'ticket_count'] = tier_h[tier_h['dim_tier'] == 'directly']['ticket_count'] * 0.9
    tier_h.loc[(tier_h['dim_tier'] == 'plus'), 'ticket_count'] = tier_h[tier_h['dim_tier'] == 'plus']['ticket_count'] * 1.1
    tier_h.loc[(tier_h['dim_tier'] == 'resolutions-2'), 'ticket_count'] = tier_h[tier_h['dim_tier'] == 'resolutions-2']['ticket_count'] * 1.05
    tier_h = tier_h.groupby(['dim_language', 'dim_tier'])['ticket_count'].sum().reset_index()
    tier_h['pert'] = tier_h[['dim_language', 'ticket_count']].groupby(['dim_language']).transform(lambda x: x.sum())
    tier_h['pert'] = (tier_h['ticket_count']) / (tier_h['pert'] + 0.00000001)

    df_clean = data_clean(df, str(run_date.date()))
    df = tier_fill(df_clean)
    db = data_clean(db, str(run_date.date()))

    print('=== Aggregating input data for forecast run ===')
    p_ut.save_df(df, '~/my_tmp/in_df_data_' + str(run_date.date()))
    if 'ds_week_starting' not in df.columns:
        df.reset_index(inplace=True)
    b_grouped = group_agg(df, 'bu')
    bl_grouped = group_agg(df, 'bu_lang')
    blt_grouped = group_agg(df, 'bu_lang_tier')
    bltc_grouped = group_agg(df, 'bu_lang_tier_channel')

    print('==== Generating forecast for different levels ===')

    #############################################
    # ## Testing All The Core Functions ###
    #############################################
    print('=== start bu level forecast ===')
    # rs_bu = forecast_bu(b_grouped, str(fcst_end_date.date()), db[['checkins', 'bookings', 'contact_rate']])

    print('==== start language level forecast ===')
    rs_bu_lang = forecast_bu_lang_arima(bl_grouped, str(fcst_end_date.date()), db[['checkins', 'bookings', 'contact_rate']], no_exog_lang, str(run_date.date()))

    print('==== starting lang tier level forecast ===')

    # rs_bu_lang_tier = forecast_bu_lang_tier(blt_grouped, str(fcst_end_date.date()), rs_bu_lang, str(run_date.date()))

    rs_bu_lang_tier = bu_lang_tier_split(tier_h, str(fcst_end_date.date()), rs_bu_lang, str(run_date.date()))

    print('==== start lang tier channel level forecast  ===')
    rs_bu_lang_tier_channel, pct_bu_lang_tier_channel = forecast_bu_lang_tier_ch(bltc_grouped, str(fcst_end_date.date()), rs_bu_lang_tier)

    # generate fcst table data to be written to a hive table
    # columns: dim_business_unit, dim_language,	dim_tier, dim_channel, time_interval, run_date_inv_ending, fcst_date_inv_ending, fcst_horizon, ticket_count
    table_output = dict_to_table(rs_bu_lang_tier_channel, str(run_date.date()))
    p_ut.save_df(table_output, out_file)
    # ############################################

    # excel file to be shared with CX
    p_ut.save_df(df, '~/my_tmp/final_df_' + str(run_date.date()))   # actuals from beginning of time correct at language level
    start_date = str(zd_df.ds_week_starting.min())
    xls_bu_lang_tier_channel = rs_to_excel(zd_df, table_output, start_date, run_date)
    xls_bltc_pct = pct_to_excel(bltc_grouped, pct_bu_lang_tier_channel, start_date)
    # xls_bu_lang_tier_channel = rs_to_excel(df, table_output, HIST_CUT_OFF_DATE)
    # xls_bltc_pct = pct_to_excel(bltc_grouped, pct_bu_lang_tier_channel, HIST_CUT_OFF_DATE)
    # file_name = os.path.join(
    #     os.path.expanduser('~/Downloads/') + run_date + '/forecast_bu_nt_' + run_date + '.xlsx')
    # xlwriter = pd.ExcelWriter(file_name)
    xlwriter = pd.ExcelWriter(xls_file)
    for ky in xls_bu_lang_tier_channel.keys():
        xls_bu_lang_tier_channel[ky].to_excel(
            xlwriter, ky.replace('/', '-'), index=False)
    xls_bltc_pct.to_excel(xlwriter, 'channel_mix', index=False)
    xlwriter.save()
    print('excel file: ' + xls_file)
    print('DONE')
