"""

"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import pandas as pd
import numpy as np
from isoweek import Week

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.utilities import time_utils as tm_ut

DO_MP = True

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

e_dict = {
    'chinese': {
        'community-education': 0.03,
        'resolutions-1': 0.03,
        'resolutions-2': 0.03
        },
    'nonChinese': {},
    'engAPAC': {},
    'engEMEA': {},
    'engNA': {
        'community-education': 0.03,
        'resolutions-1': 0.03,
        'resolutions-2': 0.03
    },
    'french': {
        'community-education': 0.03,
        'resolutions-1': 0.03,
        'resolutions-2': 0.03
    },
    'german': {},
    'italian': {},
    'japanese': {},
    'korean': {},
    'portuguese': {},
    'russian': {},
    'spanish': {
        'community-education': 0.03,
        'resolutions-1': 0.03,
        'resolutions-2': 0.03
    }
}


# def iso_weeks(yyyy):
#     w = Week(yyyy, 53)
#     if w.year == yyyy:
#         return 53
#     else:
#         return 52
#
#
# def iso_dates(yyyy):
#     weeks = iso_weeks(yyyy)
#     start = pd.to_datetime(Week(yyyy, 1).monday()) - pd.to_timedelta(1, unit='D')
#     end = pd.to_datetime(Week(yyyy, weeks).monday()) - pd.to_timedelta(1, unit='D')
#     return start, end


def to_excel_(xl_file, cutoff_date_, curr_adj_obj, prev_adj_obj, target_year, lang_errs, tier_errs, lact_df):
    """
    build excel output and save it to xls file
    :param xl_file: out file
    :param cutoff_date_: current cutoff date
    :param curr_adj_obj: current fcast adjusted obj
    :param prev_adj_obj: prior fcast adjusted obj
    :param lang_errs: lang errs of the fcast 3 months back
    :param tier_errs: tier errs between the fcast 3 months back
    :param lact_df: lang level actuals 16 weeks back
    :return: _
    """
    # prepare excel output: xls output is adjusted. par output is Not
    fx = os.path.expanduser(xl_file)
    s_ut.my_print('xl file: ' + fx)
    xlwriter = pd.ExcelWriter(fx, engine='xlsxwriter')

    # pull out actuals date indicator
    act_df = curr_adj_obj.data[['ds', 'is_actual']].copy()
    act_df['ds'] = act_df['ds'].dt.date.astype(str)
    act_df.set_index('ds', inplace=True)
    p_ut.save_df(act_df, '~/my_tmp/act_df')
    act_dates = list(act_df[act_df['is_actual'] == 1].index)

    curr_fcast = curr_adj_obj.data   # current forecast
    prev_fcast = prev_adj_obj.data   # previous forecast

    # fcast delta
    fcast_delta(xlwriter, curr_fcast, prev_fcast, 'Forecast Delta')
    for bu in ['Homes', 'Experiences', 'China']:
        c_fcast = curr_fcast[curr_fcast['business_unit'] == bu].copy()
        p_fcast = prev_fcast[prev_fcast['business_unit'] == bu].copy()
        fcast_delta(xlwriter, c_fcast, p_fcast, bu + ' Forecast Delta')

    # target year totals by language-tier-channel
    fcast_totals(xlwriter, curr_fcast, target_year, ' Totals')
    for bu in ['Homes', 'Experiences', 'China']:
        c_fcast = curr_fcast[curr_fcast['business_unit'] == bu].copy()
        c_fcast = c_fcast[(c_fcast['ds'] >= '2019-12-28') & (c_fcast['ds'] <= '2021-01-03')].copy()
        fcast_totals(xlwriter, c_fcast, target_year,  ' ' + bu + ' Totals')

    # language errors wrt actuals
    if lang_errs is not None:
        lang_errs.fillna(0, inplace=True)
        # lang_errs.rename(columns={'language': 'language'}, inplace=True)
        lang_errs['actual_count'] = np.round(lang_errs['actual_count'], 0)
        lang_errs['forecasted_count'] = np.round(lang_errs['forecasted_count'], 0)
        lang_errs.to_excel(xlwriter, 'Language Accuracy', index=False)
        workbook = xlwriter.book
        bold = workbook.add_format({'bold': True})
        worksheet = xlwriter.sheets['Language Accuracy']
        worksheet.set_column('A:ZZ', 20)
        worksheet.set_row(0, None, bold)
        format_key = workbook.add_format({'bg_color': '#E06666', 'font_color': '#660000'})
        worksheet.conditional_format(0, 0, 0, len(lang_errs.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    else:
        s_ut.my_print('No data for ' + 'language accuracy')

    # tier errors wrt actuals
    if tier_errs is not None:
        tier_errs.fillna(0, inplace=True)
        # tier_errs.rename(columns={'language': 'language', 'service_tier': 'service tier'}, inplace=True)
        tier_errs['actual_count'] = np.round(tier_errs['actual_count'], 0)
        tier_errs['forecasted_count'] = np.round(tier_errs['forecasted_count'], 0)
        tier_errs.to_excel(xlwriter, 'Tier Accuracy', index=False)
        workbook = xlwriter.book
        bold = workbook.add_format({'bold': True})
        worksheet = xlwriter.sheets['Tier Accuracy']
        worksheet.set_column('A:ZZ', 20)
        worksheet.set_row(0, None, bold)
        format_key = workbook.add_format({'bg_color': '#E06666', 'font_color': '#660000'})
        worksheet.conditional_format(0, 0, 0, len(tier_errs.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    else:
        s_ut.my_print('No data for ' + 'tier accuracy')

    # channel mix (not very practical to transpose dates as there are 4 values per date)
    cdf = curr_adj_obj.data[curr_adj_obj.data['ds'] > cutoff_date_].copy()
    if cdf is not None:
        cdf['ds'] = cdf['ds'].dt.date.astype(str)
        cdf['ticket_count'] = np.round(cdf['ticket_count'].astype(float).values, 0)
        p_ut.save_df(cdf, '~/my_tmp/cdf')
        channels = cdf['channel'].unique()
        pmix_df = pd.pivot_table(cdf, index=['business_unit', 'language', 'service_tier', 'ds'],
                                 columns='channel', values='ticket_count')
        qmix = pmix_df.div(pmix_df.sum(axis=1), axis=0).reset_index()
        qmix.fillna(0, inplace=True)
        for c in channels:
            qmix[c] = qmix[c].apply(lambda x: np.round(100.0 * x, 1) if isinstance(x, (float, np.float64, int, np.int64)) else x)
        dcol = {c: c + '(%)' for c in channels}
        qmix.rename(columns=dcol, inplace=True)
        qmix.to_excel(xlwriter, 'Channel Mix', index=False)
        workbook = xlwriter.book
        bold = workbook.add_format({'bold': True})
        worksheet = xlwriter.sheets['Channel Mix']
        worksheet.set_column('A:ZZ', 20)
        worksheet.set_row(0, None, bold)
        format_key = workbook.add_format({'bg_color': '#E06666', 'font_color': '#660000'})
        worksheet.conditional_format(0, 0, 0, len(qmix.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    else:
        s_ut.my_print('No data for ' + 'channel mix')

    # language actuals
    if lact_df is not None:
        lact_df['actual_count'] = np.round(lact_df['actual_count'].astype(float).values, 0)  # language level actuals
        lact_df['ds'] = lact_df['ds'].dt.date
        lact_df.to_excel(xlwriter, 'Language Level Actuals', index=False)
        workbook = xlwriter.book
        bold = workbook.add_format({'bold': True})
        worksheet = xlwriter.sheets['Language Level Actuals']
        worksheet.set_column('A:Z', 20)
        worksheet.set_row(0, None, bold)
        format_key = workbook.add_format({'bg_color': '#E06666', 'font_color': '#660000'})
        worksheet.conditional_format(0, 0, 0, len(lact_df.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    else:
        s_ut.my_print('No data for ' + ' language level actuals')

    # add the rest of the tabs
    if curr_adj_obj.data is not None:
        counts_dict = rs_to_excel(curr_adj_obj.data, 'ds')
        for ky, sht in counts_dict.items():
            sht.fillna(0, inplace=True)
            sht = sht.applymap(lambda x: np.round(x, 0) if isinstance(x, (float, np.float64, int, np.int64)) else x)
            tab_name = ky.replace('/', '-')
            sht.to_excel(xlwriter, tab_name, index=False)
            set_colors(xlwriter, tab_name, sht, act_dates)
            workbook = xlwriter.book
            bold = workbook.add_format({'bold': True})
            worksheet = xlwriter.sheets[tab_name]
            worksheet.set_column('B:ZZ', 20, bold)
            worksheet.set_column('A:A', 20, bold)
    else:
        s_ut.my_print('No data for ' + 'forecasts')

    s_ut.my_print('saving xls forecast to ' + fx)
    xlwriter.save()


def totals_tab(fdf, adf, cutoff_date, yyyy):
    start, end = tm_ut.iso_dates(yyyy)
    f_df = fdf[(fdf['ds'] > cutoff_date) & (fdf['ds'] <= end)].copy()  # target year data
    a_df = adf[(adf['ds'] >= start) & (adf['ds'] <= cutoff_date)].copy()  # target year data

    t_df = pd.concat([a_df, f_df], axis=0)
    u_df = _df_prep(t_df, ['language', 'channel'])
    return _get_totals(u_df, 'y')


def _get_totals(t_df, ycol):
    p_df = pd.pivot_table(t_df, index='key', columns='service_tier', values=ycol).reset_index()
    p_df.fillna(0, inplace=True)
    q_df = p_df[[c for c in p_df.columns if c != 'directly']].copy()
    q_df.set_index('key', inplace=True)
    q_df['cs tickets (all - directly)'] = q_df.sum(axis=1)
    q_df.reset_index(inplace=True)
    out_df = p_df.merge(q_df[['key', 'cs tickets (all - directly)']].copy(), on='key', how='left')
    for c in out_df.columns:
        if c != 'key':
            out_df[c] = out_df[c].astype(int)
    lead_cols = ['key', 'cs tickets (all - directly)']
    dl_col = ['directly'] if 'directly' in out_df.columns else []
    cols = lead_cols + [c for c in out_df.columns if c not in lead_cols + ['directly']] + dl_col
    fx = out_df[cols].copy()
    return get_agg(fx)


def get_agg(fx):
    fx.set_index('key', inplace=True)
    fx_all = pd.DataFrame(fx.sum(axis=0)).transpose()
    fx_all.index = ['All']
    fout = pd.concat([fx_all, fx], axis=0, sort=True).reset_index()
    fout.rename(columns={'index': 'key'}, inplace=True)
    return fout


def _df_prep(f, kcols):
    for cx in range(len(kcols)):
        if cx == 0:
            f['key'] = f[kcols[cx]]
        else:
            f['key'] = f['key'] + '-' + f[kcols[cx]]

    f = f.groupby(['key', 'service_tier']).sum(numeric_only=True).reset_index()
    return f


def ff_delta(c_df, p_df, dr, c_cu, p_cu):
    if p_df['ds'].min() > dr.min() or c_df['ds'].min() > dr.min() or p_df['ds'].max() < dr.max() or c_df['ds'].max() < dr.max():
        s_ut.my_print('WARNING: invalid date ranges')
        s_ut.my_print('delta date range: min::' + str(dr.min().date()) + ' max: ' + str(dr.max().date()))
        s_ut.my_print('prior DF dates: min::' + str(p_df['ds'].min().date()) + ' max: ' + str(p_df['ds'].max().date()))
        s_ut.my_print('current DF dates: min::' + str(c_df['ds'].min().date()) + ' max: ' + str(c_df['ds'].max().date()))
        return None

    # previous
    p_f = p_df[p_df['ds'].isin(dr)].copy()
    up_df = _df_prep(p_f, ['language'])
    p_ttl = _get_totals(up_df, 'yhat')

    # current
    c_f = c_df[c_df['ds'].isin(dr)].copy()
    uc_df = _df_prep(c_f, ['language'])
    c_ttl = _get_totals(uc_df, 'yhat')

    # delta
    f_delta = p_ttl.merge(c_ttl, on='key', how='left')
    tcols = [c for c in p_ttl.columns if c != 'key']
    for c in tcols:
        if c + '_y' in f_delta.columns and c + '_x' in f_delta.columns:
            f_delta[c] = np.round(100.0 * (f_delta[c + '_y'] - f_delta[c + '_x']) / f_delta[c + '_x'], 0)
            f_delta[c] = f_delta[c].apply(lambda x: 'N/A' if (pd.isna(x) or np.isinf(x)) else int(x))
        else:
            f_delta[c] = 'N/A'
    dp = {c: ['', ''] for c in tcols}
    p_month = (p_cu + pd.to_timedelta(7, unit='D')).month_name().upper()[:3]
    dp['key'] = ['', p_month]
    pf = pd.DataFrame(dp)
    dc = {c: ['', ''] for c in tcols}
    c_month = (c_cu + pd.to_timedelta(7, unit='D')).month_name().upper()[:3]
    dc['key'] = ['', c_month]
    cf = pd.DataFrame(dc)
    dd = {c: ['', ''] for c in tcols}
    d_lbl = 'DELTA (%)'
    dd['key'] = ['', d_lbl]
    cdel = pd.DataFrame(dd)
    ff = pd.concat([pf, p_ttl, cf, c_ttl, cdel, f_delta[['key'] + tcols].copy()])
    return ff[['key'] + tcols].copy(), p_month, c_month, d_lbl


def set_colors(xlwriter, tab_name, sht, act_dates):
    # https://www.htmlcsscolor.com/
    last_act = max(act_dates)
    try:
        act_idx = list(sht.columns).index(last_act)
    except ValueError:
        act_idx = len(sht.columns)

    workbook = xlwriter.book
    format_fct = workbook.add_format({'bg_color': '#b3cde0', 'font_color': '#03396c'})  # Add a format. Blue
    format_act = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})  # Add a format. Green fill with dark green text.
    format_key = workbook.add_format({'bg_color': '#FFE599', 'font_color': '#7F6000'})
    f_row = 1 + sht[sht['key'] == 'FORMULAS'].index[0]
    try:
        worksheet = xlwriter.sheets[tab_name]
        worksheet.conditional_format(0, 1, 0, act_idx, {'type': 'cell', 'format': format_act, 'criteria': '>=', 'value': '2100-01-01'})
        worksheet.conditional_format(0, act_idx + 1, 0, len(sht.columns) + 1, {'type': 'cell', 'format': format_fct, 'criteria': '>=', 'value': '2100-01-01'})
        worksheet.conditional_format(0, 0, len(sht), 0, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})

        worksheet.conditional_format(f_row, 1, f_row, act_idx, {'type': 'cell', 'format': format_act, 'criteria': '>=', 'value': '2100-01-01'})
        worksheet.conditional_format(f_row, act_idx + 1, f_row, len(sht.columns) + 1, {'type': 'cell', 'format': format_fct, 'criteria': '>=', 'value': '2100-01-01'})
    except AttributeError:
        s_ut.my_print('could not set colors for ' + str(tab_name))


def rs_to_excel(df, pcol, ycol):          # all, all - (safety + claims), all - directly, all - (safety + claims + directly)
    def to_spreadsheet(adf, tcol, ycol_):
        # top DF
        adf_tier = pd.pivot_table(adf, values=ycol_, index=['business_unit', 'channel', 'language'], columns=[tcol], aggfunc=sum).reset_index()
        adf_tier_all = pd.pivot_table(adf, values=ycol_, index=['business_unit', 'language'], columns=[tcol], aggfunc=sum).reset_index()
        adf_tier_all['channel'] = 'all'
        output_ = pd.concat([adf_tier_all, adf_tier], sort=True)
        output_['key'] = output_['business_unit'] + '-' + output_['channel'] + '-' + output_['language']

        # spacer DF
        sf = pd.DataFrame(columns=output_.columns, index=range(3))
        sf.loc[0] = [''] * len(output_.columns)
        sf.loc[2] = [''] * len(output_.columns)
        sf.loc[1] = [c for c in output_.columns]
        sf.loc[1, 'key'] = 'FORMULAS'

        # formulas DF
        pdf_ = pd.pivot_table(adf, values=ycol_, index=['channel', 'language'], columns=[tcol], aggfunc=sum).reset_index()
        pdf_['key'] = pdf_['channel'] + '-' + pdf_['language']
        qdf = pd.DataFrame(pdf_.sum(axis=0)).transpose()
        qdf['key'] = 'All'
        fall = pd.concat([output_, sf, qdf, pdf_], axis=0, sort=True)
        fall.reset_index(inplace=True, drop=True)
        cols = fall.columns.values.tolist()
        _ = [cols.remove(c) for c in ['business_unit', 'channel', 'language', 'key']]
        cols.insert(0, 'key')
        return fall[cols]

    df[pcol] = df[pcol].dt.date.astype(str)
    output = dict()
    output['all'] = to_spreadsheet(df, pcol, ycol)
    df_nodirectly = df[df['service_tier'] != 'directly']
    output['all - directly'] = to_spreadsheet(df_nodirectly, pcol, ycol)
    df_ = df[~((df['service_tier'] == 'safety') | (df['service_tier'] == 'claims'))]
    output['all - (safety&claims)'] = to_spreadsheet(df_, pcol, ycol)
    df_ = df[~((df['service_tier'] == 'safety') | (df['service_tier'] == 'claims') | (df['service_tier'] == 'directly'))]
    output['all - (safety&claims&directly)'] = to_spreadsheet(df_, pcol, ycol)
    for tr in df['service_tier'].unique():

        if tr in ['Community Education', 'Resolutions 1']:
            print(11111111111111111)
            print(ycol)
            print(pcol)
            z = df.groupby(['ds', 'service_tier']).sum().reset_index()
            zz = z[(z['ds'] >= '2020-02-01') & (z['ds'] <= '2020-04-10')]
            print(zz[zz['service_tier'] == tr])

        p_ut.save_df(df[df['service_tier'] == tr], '~/my_tmp/sh_df_' + tr)
        output[tr] = to_spreadsheet(df[df['service_tier'] == tr], pcol, ycol)
    return output


def fcast_delta(xlwriter, c_fcast, p_fcast, dr, c_cu, p_cu, tab_name):
    delta_df, p_month, c_month, d_lbl = ff_delta(c_fcast, p_fcast, dr, c_cu, p_cu)
    delta_df.reset_index(inplace=True, drop=True)
    delta_df.to_excel(xlwriter, tab_name, index=False)
    workbook = xlwriter.book
    bold = workbook.add_format({'bold': True})
    worksheet = xlwriter.sheets[tab_name]
    worksheet.set_column('B:ZZ', 20)
    worksheet.set_column('A:A', 20, bold)
    format_key = workbook.add_format({'bg_color': '#E06666', 'font_color': '#660000'})
    worksheet.conditional_format(0, 0, 0, len(delta_df.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    worksheet.conditional_format(0, 0, len(delta_df), 0, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    worksheet.conditional_format(0, 0, len(delta_df), 0, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    row = delta_df[delta_df['key'] == p_month].index[0] + 1
    worksheet.conditional_format(row, 0, row, len(delta_df.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    row = delta_df[delta_df['key'] == c_month].index[0] + 1
    worksheet.conditional_format(row, 0, row, len(delta_df.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    row = delta_df[delta_df['key'] == d_lbl].index[0] + 1
    worksheet.conditional_format(row, 0, row, len(delta_df.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})


def fcast_totals(xlwriter, c_fcast, act_df, cutoff_date, yyyy, tab_name, bu=None):
    if bu is not None:
        fdf = c_fcast[c_fcast['business_unit'] == bu].copy()
        adf = act_df[act_df['business_unit'] == bu].copy()
    else:
        fdf = c_fcast.copy()
        adf = act_df.copy()
        bu = 'All'
    fdf.rename(columns={'yhat': 'y'}, inplace=True)
    totals_df = totals_tab(fdf, adf, cutoff_date, yyyy)
    t_name = str(yyyy) + ' ' + tab_name + '(' + bu + ')'
    totals_df.to_excel(xlwriter, t_name, index=False)
    workbook = xlwriter.book
    bold = workbook.add_format({'bold': True})
    worksheet = xlwriter.sheets[t_name]
    worksheet.set_column('B:ZZ', 20)
    worksheet.set_column('A:A', 20, bold)
    format_key = workbook.add_format({'bg_color': '#9FC5E8', 'font_color': '#073763'})
    worksheet.conditional_format(0, 0, 0, len(totals_df.columns) - 1, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})
    worksheet.conditional_format(0, 0, len(totals_df), 0, {'type': 'cell', 'format': format_key, 'criteria': '>=', 'value': '""'})


def year_summary(adf, fdf, yyyy, cutoff_date):
    start, end = tm_ut.iso_dates(yyyy)
    s_ut.my_print(str(yyyy) + ' iso start: ' + str(start))
    s_ut.my_print(str(yyyy) + ' iso end: ' + str(end))
    a_df = adf[(adf['ds'] >= start) & (adf['ds'] <= cutoff_date)].copy()
    f_df = fdf[(fdf['ds'] > cutoff_date) & (fdf['ds'] <= end)]
    print(a_df.business_unit.unique())
    f_df.rename(columns={'yhat': 'y'}, inplace=True)
    ttl = a_df.y.sum() + f_df.y.sum()
    s_ut.my_print('+++++++++++++++ Year ' + str(yyyy) + ' Summary +++++++++++++++=')
    s_ut.my_print('\t\tYTD volume: ' + str(a_df.y.sum()))
    s_ut.my_print('\t\tForecasted volume: ' + str(f_df.y.sum()))
    s_ut.my_print('\t\tTotal volume: ' + str(ttl))
    for bu in f_df['business_unit'].unique():
        a = a_df[a_df['business_unit'] == bu]
        f = f_df[f_df['business_unit'] == bu]
        ttl = a.y.sum() + f.y.sum()
        s_ut.my_print('\t\t' + bu + ' YTD volume: ' + str(a.y.sum()))
        s_ut.my_print('\t\t' + bu + ' Forecasted volume: ' + str(f.y.sum()))
        s_ut.my_print('\t\t' + bu + ' Total volume: ' + str(ttl))

