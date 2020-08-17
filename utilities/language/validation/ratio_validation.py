"""
computes the month over month weighted change in the fraction of time consumed by each service tier.
If for a month, a language and an interaction type we get delta = 0.1 that means that between the previous month and this month the fraction of time spent in each agent tier
has changed on average by 10%. This average is weighted by the time spend in each service tier
"""


def w_avg(a_df, cols):
    a_df.to_parquet('~/my_tmp/a_df.par')
    ycols = [c + '_y' for c in cols]
    xcols = [c + '_x' for c in cols]
    d_df = a_df[ycols].diff().copy()
    s_df = (a_df[xcols].shift() + a_df[xcols]).copy()
    s_df /= 2.0
    z = pd.concat([d_df, s_df], axis=1)
    z.dropna(inplace=True)
    z['sum'] = z[xcols].sum(axis=1)
    for c in cols:
        z[c] = np.abs(z[c + '_y']) * z[c + '_x'] / z['sum']
    b = z[cols].copy()
    w = b.sum(axis=1)
    f = w.reset_index()
    f.columns = ['ds', 'language', 'interaction_type', 'delta']
    return f


df = pd.read_parquet('/Users/josep_ferrandiz/my_tmp/cleaned/phone-aht_cleaned_2019-12-15.par')
df = df[df['ds'] > '2019-07-31'].copy()
df['w'] = df['calls'] * df['agent_mins']
g = df.groupby([pd.Grouper(key='ds', freq='M'), 'interaction_type', 'language', 'sector']).agg({'calls': np.sum, 'w': np.sum, 'agent_mins': np.sum}).reset_index()
p = pd.pivot_table(g, index=['ds', 'language', 'interaction_type'], columns=['sector'], values=['agent_mins']).reset_index()
p.fillna(0, inplace=True)
p.columns = ['ds', 'language', 'interaction_type', 'Claims', 'Community Education', 'Experiences', 'PST', 'Payments', 'Regulatory Response', 'Resolutions 1', 'Resolutions 2',
             'Safety', 'SafetyHub', 'Social Media']
t_cols = ['Community Education', 'Resolutions 1', 'Resolutions 2']
p.set_index(['ds', 'language', 'interaction_type'], inplace=True)
p['other'] = p[list(set(p.columns) - set(t_cols))].sum(axis=1)
t_cols += ['other']
p = p[t_cols].copy()
p.sort_values(by=['interaction_type', 'language', 'ds'], inplace=True)
q = p.div(p.sum(axis=1), axis=0)
r = p.merge(q, left_index=True, right_index=True, how='inner')
s = r.groupby(['language', 'interaction_type']).apply(w_avg, cols=t_cols)
s.reset_index(inplace=True, drop=True)

