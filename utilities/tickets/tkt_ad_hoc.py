# actuals
f = pd.read_parquet('~/my_tmp/in_df_data_2020-01-25.par')
df.reset_index(inplace=True)
df['ds_week_starting'] = pd.to_datetime(df['ds_week_starting'].values)
df['ds_week_ending'] = df['ds_week_starting'] + pd.to_timedelta(6, unit='D')
ym = df['ds_week_ending'].dt.month
ys = ym.apply(lambda x: str(x) if x >= 10 else str('0') + str(x))
df['ds'] = ys + '-' + df['ds_week_ending'].dt.year.astype(str)
df['ds_week_starting'] = df['ds_week_starting'].astype(str)
df['ds_week_ending'] = df['ds_week_ending'].astype(str)

table = 'sup.fct_ticket_forecast_actuals'
if_exists = 'replace'
for ds, fs in df.groupby('ds'):
      gs = fs.copy()
      gs.drop('ds', inplace=True, axis=1)  # not needed to push
      partition_ = {'ds': ds}
      ap.hive.push(gs, table=table, if_exists=if_exists, partition=partition_,
                   table_props={'abb_retention_days': '-1', 'abb_retention_days_reason': 'fact table. No pii'})
      print('data saved to table ' + table + ' for ds ' + ds)

# forecasts
df = pd.read_parquet('~/my_tmp/wf.par')
df['ds_week_ending'] = pd.to_datetime(df['ds_week_ending'].values)
df['ds_week_starting'] = df['ds_week_ending'] - pd.to_timedelta(6, unit='D')
df['ds_week_starting'] = df['ds_week_starting'].astype(str)
df['ds_week_ending'] = df['ds_week_ending'].astype(str)

table = 'sup.fct_ticket_forecast_forecast'
if_exists = 'replace'
for cutoff, fs in df.groupby('cutoff_date'):
      gs = fs.copy()
      gs.drop('cutoff_date', inplace=True, axis=1)  # not needed to push
      partition_ = {'cutoff_date': cutoff}
      ap.hive.push(gs, table=table, if_exists=if_exists, partition=partition_,
                   table_props={'abb_retention_days': '-1', 'abb_retention_days_reason': 'fact table. No pii'})
      print('data saved to table ' + table + ' for ds ' + cutoff)

# errors
# compare queries
qf = pd.read_csv('~/Downloads/sqllab_untitled_query_2_20200211T203522.csv')
qf=qf[['dim_ticket_business_unit','dim_ticket_language_region_group_for_CP', 'dim_channel_group_for_CP', 'dim_ticket_zendesk_group_for_CP', 'ds_week_ending', 'ticket_volume']].copy()
qf.rename(columns={'dim_ticket_business_unit':'dim_business_unit', 'dim_ticket_language_region_group_for_CP': 'dim_language', 'dim_channel_group_for_CP': 'dim_channel', 'dim_ticket_zendesk_group_for_CP': 'dim_tier', 'ticket_volume': 'ticket_count'}, inplace=True)
af['ds_week_starting'] = pd.to_datetime(af['ds_week_starting'])
qf['ds_week_starting'] = pd.to_datetime(qf['ds_week_ending']) + pd.to_timedelta(-6, unit='D')
new_data = p_ut.clean_cols(qf, ['dim_language', 'dim_channel', 'dim_tier'],  '~/my_repos/capacity_planning/data/config/col_values.json', check_new=True, do_nan=True)
qf = qf[(qf['ds_week_starting'] > '2018-12-30') & (qf['ds_week_starting'] < '2020-01-19')].copy()

af = pd.read_parquet('~/my_tmp/in_df_data_2020-01-25.par')
af = af[(af['ds_week_starting'] > '2018-12-30') & (af['ds_week_starting'] < '2020-01-19')].copy()

cols = ['dim_language', 'dim_tier']
d_af = af.groupby(cols).agg({'ticket_count': np.sum}).reset_index()
d_qf = qf.groupby(cols).agg({'ticket_count': np.sum}).reset_index()
df = d_af.merge(d_qf, on=cols, how='inner')
# df['err%'] = 100 * (df['ticket_count_y']-df['ticket_count_x'])/df['ticket_count_x']
z_list = list()
for x in df['dim_language'].unique():
      gf = df[df['dim_language'] == x].copy()
      l_vol = gf['ticket_count_x'].sum()
      for y in gf['dim_tier'].unique():
            hf = gf[gf['dim_tier'] == y].copy()
            t_vol = hf['ticket_count_x'].sum()
            t_err = (hf['ticket_count_y'].sum()-hf['ticket_count_x'].sum())/hf['ticket_count_x'].sum()
            z_list.append(pd.DataFrame({'language': [x], 'tier': [y], 'l_vol': [l_vol], 't%': [100* t_vol / l_vol], 'err%': [100 * t_err] }))
z = pd.concat(z_list, axis=0)
z.sort_values(by=['language', 't%'], ascending=[True, False], inplace=True)
z.reset_index(inplace=True, drop=True)
w = z.groupby('language').apply(lambda x: x['t%'].cumsum()).reset_index(level=0)
w.columns = ['language', 't%_cum']
w.drop('language', axis=1, inplace=True)
f = pd.concat([z, w], axis=1)
f['p'] = f['t%'] * f['err%']
yy = f.groupby('language').apply(lambda x: x['p'].cumsum()).reset_index(level=0)
yy.drop('language', axis=1, inplace=True)
f.drop('p', axis=1, inplace=True)
g = pd.concat([f, yy], axis=1)
g['err%_cum'] =g['p'] / g['t%_cum']
g.drop('p', axis=1, inplace=True)



