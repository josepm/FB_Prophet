"""
compute forecast errors from parquet files for all tickets or by BU
"""

from capacity_planning.forecast import lang_forecast as lf
from capacity_planning.forecast import ens_forecast as e_fc
from capacity_planning.forecast.utilities.tickets import tkt_utils as t_ut

dates = ['2019-07-27', '2019-08-31', '2019-09-28', '2019-10-26', '2019-11-30', '2019-12-28']
ts_names = ['tickets']  #, 'tickets_Homes', 'tickets_China', 'tickets_Experiences']
tfcast_file = '~/my_tmp/fbp/ens_fcast_'
# tfcast_file = '~/Forecasts/rolling/par/adj_r_fcast_4_'
d_list = list()
d_adf, d_fdf = dict(), dict()
by_val = None
for ts_name in ts_names:
    for cutoff_date in dates:
        if cutoff_date not in d_fdf.keys():
            d_fdf[cutoff_date] = dict()
        if ts_name == 'tickets':
            by = by_val
            bu = None
        elif ts_name == 'tickets_Homes':
            by = by_val
            bu = ts_name.split('_')[1]
        elif ts_name == 'tickets_China':
            by = by_val
            bu = ts_name.split('_')[1]
        elif ts_name == 'tickets_Experiences':
            by = by_val
            bu = ts_name.split('_')[1]
        else:
            print('unknown ts name: ' + str(ts_name))
            break

        gcols = t_ut.get_gcols(by)
        if bu == 'Experiences':
            bu = 'Experiences'

        # actuals
        if ts_name in d_adf.keys():
            adf = d_adf[ts_name]
        else:
            adf = e_fc.set_actuals(ts_name, pd.to_datetime(cutoff_date), 'W', 2, pd.to_datetime('2016-01-01'))
            adf.rename(columns={'y': 'ticket_count'}, inplace=True)
            d_adf[ts_name] = adf
        if bu is not None:
            if 'business_unit' in adf.columns:
                adf = adf[adf['business_unit'] == bu].copy()
            else:
                adf['business_unit'] = bu

        # forecasts
        ycol = 'yhat' if tfcast_file == '~/my_tmp/fbp/ens_fcast_' else 'ticket_count'
        data_path = tfcast_file + ts_name if tfcast_file == '~/my_tmp/fbp/ens_fcast_' else tfcast_file + str(pd.to_datetime(cutoff_date).date())
        ts_cfg = {'ycol': ycol, 'agg_dict': {ycol: 'sum'}, 'data_path': data_path, 'check_cols': [ycol], 'cutoff_date': pd.to_datetime(cutoff_date),
                  'init_date': pd.to_datetime('2016-01-01'), 'time_scale': 'W', 'name': ts_name, 'bu': bu}
        fdf = t_ut.get_fcast(ts_cfg, tfcast_file.split('/')[-1], by=by)
        if bu is not None:
            if 'business_unit' in fdf.columns:
                fdf = fdf[fdf['business_unit'] == bu].copy()
            else:
                fdf['business_unit'] = bu
        d_fdf[str(pd.to_datetime(cutoff_date).date())][ts_name] = fdf

        cutoff_date = pd.to_datetime(cutoff_date)
        for c in adf.columns:
            if c != 'ticket_count':
                if c not in gcols:
                    print('adf unexpected column: ' + c)
                    adf = adf.groupby(gcols).sum()   # get rid
        for c in fdf.columns:
            if c != 'ticket_count':
                if c not in gcols:
                    print('tdf unexpected column: ' + c)
                    tdf = tdf.groupby(gcols).sum()  # get rid

        df = adf.merge(fdf, on=gcols, how='left')
        df = df[df['ds'] > cutoff_date].copy()

        other_gcols = list(set(gcols) - {'ds'})

        # one month error
        upr, lwr = 4, 1
        ef = df[(df['ds'] >= cutoff_date + pd.to_timedelta(lwr, unit='W')) & (df['ds'] < cutoff_date + pd.to_timedelta(upr, unit='W'))].copy()
        ef = ef.groupby(other_gcols).sum().reset_index()
        ef['err4'] = np.abs((ef['ticket_count_y'] / ef['ticket_count_x']) - 1)
        all4 = pd.DataFrame({'language': ['All'], 'err4': [np.abs((ef['ticket_count_y'].sum() / ef['ticket_count_x'].sum()) - 1)]})
        ef4 = pd.concat([all4, ef], axis=0)
        # ef = np.abs((ef['ticket_count_y'].sum() / ef['ticket_count_x'].sum()) - 1)

        # 90 days error
        upr, lwr = 12, 8
        ef = df[(df['ds'] >= cutoff_date + pd.to_timedelta(lwr, unit='W')) & (df['ds'] < cutoff_date + pd.to_timedelta(upr, unit='W'))].copy()
        ef = ef.groupby(other_gcols).sum()
        ef = ef.groupby(other_gcols).sum().reset_index()
        ef['err12'] = np.abs((ef['ticket_count_y'] / ef['ticket_count_x']) - 1)
        all12 = pd.DataFrame({'language': ['All'], 'err12': [np.abs((ef['ticket_count_y'].sum() / ef['ticket_count_x'].sum()) - 1)]})
        ef12 = pd.concat([all12, ef], axis=0)
        # err12 = np.abs((ef['ticket_count_y'].sum() / ef['ticket_count_x'].sum()) - 1)

        ef = ef4[['language', 'err4']].merge(ef12[['language', 'err12']], on='language')

        # 2020 totals
        s_date = pd.to_datetime('2019-12-29')
        e_date = pd.to_datetime('2020-12-27')
        vol2020 = fdf[(fdf['ds'] >= s_date) & (fdf['ds'] <= e_date)].groupby('language').sum().reset_index()
        all = pd.DataFrame({'language': ['All'], 'ticket_count': [vol2020['ticket_count'].sum()]})
        vol2020 = pd.concat([all, vol2020], axis=0)

        f = ef.merge(vol2020, on='language', how='inner')
        f['ts_name'] = ts_name
        f['cutoff_date'] = cutoff_date.date()
        d_list.append(f)

        # d_out = dict()
        # d_out['ts_name'] = ts_name
        # d_out['cutoff_date'] = cutoff_date.date()
        # d_out['1st month err'] = err4
        # d_out['90 days err'] = err12
        # d_out['2020 forecast'] = vol2020
        # d_list.append(d_out)


df = pd.DataFrame(d_list)
df = pd.concat(d_list)
print(df)
df.to_csv('~/my_tmp/tix_res.csv', index=False)

from functools import reduce
from scipy.optimize import minimize
def func(z, args):
    q = np.exp(z) / np.sum(np.exp(z))
    diff = 1 - (q[0] * args['homes_tix'] + q[1] * args['china_tix'] + q[2] * args['Experiences_tix']) / args['ticket_count']
    diff = diff ** 2
    return np.mean(diff)
    # y = args[:, 0]
    # w = np.sum(np.array([q[i-1] * args[:, i] / y for i in range(1, 4)]), axis=0)
    # return np.sum((1-w)**2)


cutoff_date = pd.to_datetime('2019-07-27')
z0 = (1, 1, 1)
tix_df = d_fdf[str(cutoff_date.date())]['tickets'].groupby('ds').agg({'ticket_count': np.sum}).reset_index()
homes_df = d_fdf[str(cutoff_date.date())]['tickets_Homes'][['ds', 'ticket_count']]
homes_df.rename(columns={'ticket_count': 'homes_tix'}, inplace=True)
china_df = d_fdf[str(cutoff_date.date())]['tickets_China'][['ds', 'ticket_count']]
china_df.rename(columns={'ticket_count': 'china_tix'}, inplace=True)
Experiences_df = d_fdf[str(cutoff_date.date())]['tickets_Experiences'][['ds', 'ticket_count']]
Experiences_df.rename(columns={'ticket_count': 'Experiences_tix'}, inplace=True)
mf = reduce(lambda x, y: x.merge(y, on='ds', how='inner'), [tix_df, homes_df, china_df, Experiences_df])
args = np.transpose(np.array([mf['ticket_count'].values, mf['homes_tix'].values, mf['china_tix'].values, mf['Experiences_tix'].values]))
z0 = (mf['homes_tix'].sum()/mf['ticket_count'].sum(), mf['china_tix'].sum()/mf['ticket_count'].sum(), mf['Experiences_tix'].sum()/mf['ticket_count'].sum())
res = minimize(func, z0, args=mf, method='Nelder-Mead', tol=1e-6)

q = np.array(np.exp(res.x))/np.sum(np.array(np.exp(res.x)))

v = np.array([1,2,3])
w = np.array

tdf = pd.read_parquet('/Users/josep_ferrandiz/my_tmp/cleaned/tickets_2020-02-15.par')
p_ut.clean_cols(tdf,['dim_language', 'dim_business_unit'], '~/my_repos/capacity_planning/data/config/col_values.json', check_new=False, do_nan=False)
tdf.rename(columns={'dim_language': 'language', 'dim_business_unit': 'business_unit', 'ds_week_starting': 'ds', 'dim_channel': 'channel'}, inplace=True)

tdf = tdf[tdf['language'].isin(['Mandarin', 'English_APAC'])].copy()
adf = adf[adf['language'].isin(['Mandarin', 'English_APAC'])].copy()
fdf = fdf[fdf['language'].isin(['Mandarin', 'English_APAC'])].copy()
ds_min = '2019-07-28'
ds_max = '2020-01-12'
adf.reset_index(inplace=True, drop=True)
tdf.reset_index(inplace=True, drop=True)
fdf.reset_index(inplace=True, drop=True)
df = adf.merge(tdf, on=['language', 'ds'], how='inner')
df = df.merge(fdf, on=['language', 'ds'], how='inner')
df = df[(df['ds'] >= ds_min) & (df['ds'] <= ds_max)].copy()


# tdf = tdf[tdf['business_unit'] == 'China'].copy()
tdf = tdf.groupby(['ds', 'language']).agg({'ticket_count': np.sum}).reset_index()
df = adf.merge(tdf, on=['ds', 'language'], how='inner')
df['diff'] = df['ticket_count_y'] - df['ticket_count_x']
gf = df[df['ds'] >= '2019-06-01'].copy()
gf = gf[gf['language'] == 'Mandarin'].copy()
for l, fl in gf.groupby('language'):
    print(l + ' ' + str((fl['diff'] !=0).sum()))

