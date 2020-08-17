"""
python ticket_fcast_initiatives.py window cutoff_date
"""
import os
import sys
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'

import pandas as pd

from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut
from capacity_planning.forecast.ticket_forecast import errors as errs
from capacity_planning.forecast.ticket_forecast import to_excel as txl
from capacity_planning.forecast.utilities.language import time_series as ts


def rs_to_excel(df, pcol):
    # all, all - (safety + claims), all - directly, all - (safety + claims + directly)
    def to_speadsheet(adf, nodirectly=False):
        adf_tier = pd.pivot_table(adf, values='ticket_count', index=[
            'dim_business_unit', 'dim_channel', 'dim_language'], columns=[pcol], aggfunc=sum).reset_index()
        adf_tier_all = pd.pivot_table(adf, values='ticket_count', index=[
            'dim_business_unit', 'dim_language'], columns=[pcol], aggfunc=sum).reset_index()
        adf_tier_all['dim_channel'] = 'all'
        output_ = pd.concat([adf_tier_all, adf_tier], sort=True)
        cols = output_.columns.values.tolist()
        cols.remove('dim_business_unit')
        cols.remove('dim_channel')
        cols.remove('dim_language')
        cols.insert(0, 'dim_language')
        cols.insert(0, 'dim_channel')
        cols.insert(0, 'dim_business_unit')
        return output_[cols]

    output = dict()
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


def adjust_fcast(plan_df, fcast_df, bu):
    w_date = plan_df.columns[1:]
    adj_df = pd.melt(plan_df, value_vars=w_date, id_vars=[plan_df.columns[0]])
    adj_df.columns = ['dim_tier', 'ds_week_start', 'multiplier']
    adj_df.fillna(0, inplace=True)
    adj_df['ds_week_start'] = pd.to_datetime(adj_df['ds_week_start'])
    adj_df['ds_week_ending'] = adj_df['ds_week_start'] + pd.to_timedelta(6, unit='D')
    adj_df['dim_business_unit'] = bu
    adj_df['multiplier'] = 1 + adj_df['multiplier']
    df_ = fcast_df[fcast_df['dim_business_unit'] == bu].copy()
    i_df = df_.merge(adj_df, on=['ds_week_ending', 'dim_business_unit', 'dim_tier'], how='left')
    i_df['multiplier'].fillna(1, inplace=True)
    i_df['old_tkt_cnt'] = i_df['ticket_count'].copy()
    i_df['ticket_count'] *= i_df['multiplier']
    if len(i_df[i_df['multiplier'] != 1]) == 0:
        print('WARNING: No changes for ' + str(bu))
        print('bu dim_tiers: ' + str(df_.dim_tier.unique()))
        print('bu plan: ' + str(adj_df.dim_tier.unique()))
    # print(i_df[i_df['multiplier'] != 1].head())
    i_df.drop(['ds_week_start', 'multiplier'], axis=1, inplace=True)
    return i_df


def prepare_plan(fname, cutoff_date_, is_weekly):
    init_dir = '~/Downloads/'
    _plan = pd.read_csv(init_dir + fname)

    # week_starting patch
    df_cols_ = _plan.columns
    if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
        _plan['ds_week_ending'] = pd.to_datetime(_plan['ds_week_ending'])
        _plan['ds_week_starting'] = _plan['ds_week_ending'] - pd.to_timedelta(6, unit='D')

    _plan = _plan[_plan['Snapshot'] == _plan['Snapshot'].max()].copy()  # latest data
    _plan.drop('Snapshot', axis=1, inplace=True)
    if 'all' in _plan['Tier'].unique():
        _plan = _plan[_plan['Tier'] != 'all'].copy()
    _plan.set_index('Tier', inplace=True)
    d_plan = _plan.to_dict()

    # turn to weekly
    if is_weekly is False:
        dr = pd.date_range(start=cutoff_date_, end=cutoff_date_ + pd.to_timedelta(365, unit='D'), freq='W-SUN')
        dh = {d.date(): d_plan[d.month_name()[:3]] if d.year != 2019 else 0.0 for d in dr}
        _plan = pd.DataFrame(dh)  # standard format
    return _plan.reset_index()


if __name__ == '__main__':
    print(sys.argv)
    _, window, cutoff_date_str = sys.argv
    cutoff_date = pd.to_datetime(cutoff_date_str)

    # gsheet: Forecasting/Tix Targets
    # only on rolling forecasts
    fin = '~/Forecasts/rolling/par/adj_r_xls_' + window + '_' + cutoff_date_str + '.par'       # input file
    f_df = pd.read_parquet(fin)                                                                # load last adjusted fcast
    if f_df is None:
        s_ut.my_print('ERROR: could not find ' + fin)
        sys.exit()

    # week_starting patch
    df_cols_ = f_df.columns
    if 'ds_week_ending' in df_cols_ and 'ds_week_starting' not in df_cols_:
        f_df['ds_week_ending'] = pd.to_datetime(f_df['ds_week_ending'])
        f_df['ds_week_starting'] = f_df['ds_week_ending'] - pd.to_timedelta(6, unit='D')

    dates = [str(pd.to_datetime(x).date()) for x in f_df['ds_week_starting'].unique()]

    # China
    print('************** China ***************')
    cn_plan = prepare_plan('Initiatives - China.csv', cutoff_date, True)
    cn_df = adjust_fcast(cn_plan, f_df, 'China')

    # Homes
    print('************** Homes ***************')
    hm_plan = prepare_plan('Initiatives - Homes.csv', cutoff_date, False)
    print('tix: ' + str(f_df[(f_df.dim_business_unit == 'Homes') & (f_df.dim_language == 'engNA') & (f_df.ds_week_starting == '2019-09-29')]['ticket_count'].sum()))
    hm_df = adjust_fcast(hm_plan, f_df, 'Homes')

    # Experiences reduction
    print('************** Experiences Reduction ***************')
    ex_df1 = adjust_fcast(hm_plan, f_df, 'Experiences')

    # Experiences growth
    print('************** Experiences Growth ***************')
    ex_plan = prepare_plan('Initiatives - Experiences.csv', cutoff_date, True)
    ex_df2 = adjust_fcast(ex_plan, ex_df1, 'Experiences')

    # fcast with initiative gains included
    i_df = pd.concat([cn_df, hm_df, ex_df2], axis=0)
    i_df['initiative'] = True
    par_fout = '~/Forecasts/rolling/par/adj_r_fcast_init_' + window + '_' + cutoff_date_str      # par output file
    p_ut.save_df(i_df, par_fout)   # save the adjusted initiatives forecast

    # errors (on actuals)
    adf = errs.get_actuals(cutoff_date)
    print('tix: ' + str(adf[(adf['dim_business_unit'] == 'Homes') & (adf['dim_language'] == 'engNA') & (adf['ds_week_starting'] == '2019-09-29')]['actual_count'].sum()))
    lact_df = errs.c_group(adf)

    fcast_file = errs.get_fcast_file(cutoff_date, '~/Forecasts/rolling/par/raw_r_fcast_' + str(window) + '_', months=3)   # file path from <months> months old forecast from cutoff
    fdf_obj = ts.TicketForecast(fcast_file)                                                                            # fcast obj from 3 months ago
    fdf = fdf_obj.data
    fdf['ds_week_ending'] = pd.to_datetime(fdf['ds_week_ending'])
    fdf.rename(columns={'ticket_count': 'forecasted_count'}, inplace=True)
    # fdf = errs.get_fcast(cutoff_date, '~/Forecasts/rolling/par/adj_r_fcast_init_' + str(window) + '_', months=3)  # get fcast from <months> ago
    if fdf is not None:
        lang_errs, tier_errs, off_df = errs.get_errs(cutoff_date, fdf_obj, adf)
    else:
        lang_errs, tier_errs, off_df = None, None

    # save xls
    xls_fr = '~/Forecasts/rolling/par/adj_r_xls_init_' + str(window) + '_' + str(cutoff_date.date())   # 16 weeks of actuals and forecast.

    b_df = i_df.copy()
    b_df['run_date'] = cutoff_date
    b_df['adj'] = True
    b_df['initiative'] = True
    s_ut.my_print('saving adj data (fcast + actuals) to ' + xls_fr)
    p_ut.save_df(b_df, xls_fr)
    f_obj = ts.TicketForecast(xls_fr)

    # create the excel file
    xls_fout = '~/Forecasts/rolling/xls/adj_r_fcast_init_' + window + '_' + cutoff_date_str + '.xlsx'   # xls output file
    froot = '~/Forecasts/rolling/par/adj_r_xls_init_' + window + '_'
    txl.to_excel_(xls_fout, cutoff_date, f_obj, lang_errs, tier_errs, lact_df)
