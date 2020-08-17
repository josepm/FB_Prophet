

def get_arrs(f):
    d_dur = (f['ds_solved_at'] - f['ds_created_at']).dt.days
    wh = f['hbat'].sum()
    return pd.DataFrame({
        # 'ds': [f.loc[f.index[0], 'ds_created_at']],
        # 'language': [f.loc[f.index[0], 'dim_language']],
        'arrs': [len(f)],
        # 'arr_is_assigned %': [1 - f['id_agent'].isnull().sum() / len(f)],
        # 'arr_is_reservation %': [1 - f['reservation_id'].isnull().sum() / len(f)],
        # 'arr_agents': [f['id_agent'].nunique()],
        # 'avg_dur': [d_dur.mean()],
        # 'med_dur': [d_dur.median()],
        'wh': [wh]

    })

def get_deps(f):
    m = (f['ds_solved_at'] - f['ds_created_at']).dt.days.median()
    a = (f['ds_solved_at'] - f['ds_created_at']).dt.days.mean()
    return pd.DataFrame({
        # 'ds': [f.loc[f.index[0], 'ds_solved_at']],
        # 'language': [f.loc[f.index[0], 'dim_language']],
        'deps': [len(f)],
        # 'median dur': [m],
        'avg dur': [a],
        # 'dep_is_assigned %': [1 - f['id_agent'].isnull().sum() / len(f)],
        # 'dep_is_reservation %': [1 - f['reservation_id'].isnull().sum() / len(f)],
        'agents': [f['id_agent'].nunique()],
        'hbat': f['hbat'].mean()
    })

def get_queue(fx):
    fx['queue'] = fx['inc'].cumsum()
    return fx

df = ap.hive.query('select '
                   'ctc.id_ticket as id_ticket '
                   ',ctc.id_agent as id_agent '
                   ',ctc.id_reservation as reservation_id '
                   ',ctc.dim_ticket_contact_channel as channel '
                   ',ctc.dim_ticket_ds_created_at as ds_created_at '
                   ',ctc.dim_ticket_ds_solved_at as ds_solved_at '
                   ',ctc.dim_ticket_language as language '
                   ',ctc.dim_ticket_inbound_or_outbound as inbound_or_outbound '
                   ',tc.heartbeat_time_adjusted as hbat '
                   'from metrics.cs_tickets__cs__hive ctc '
                   'LEFT JOIN serv_ex.dim_ticket_cost tc ON tc.ds = \'2020-03-23\' AND CAST(tc.id_ticket AS string) = ctc.id_ticket '
                   'where ctc.cs_tickets = 1 and ctc.dim_ticket_ds_created_at >= \'2019-10-01\' ;')

    # 'SELECT id_zendesk, ds_created_at, ds_solved_at, id_reservation_code, id_assignee, dim_language, dim_ticket_topic_path '
    # 'from core_cx_2.dim_daily_zendesk_tickets where ds = \'2020-03-16\' and ds_created_at >= \'2019-10-01\';')

dr = pd.date_range(start=max(df['ds_created_at'].min(), df['ds_closed_at'].min()), end=min(df['ds_created_at'].max(), df['ds_closed_at'].max()), freq='D')
f_dr = pd.DataFrame({'ds': dr})
for c in ['ds_created_at', 'ds_solved_at']:
    df[c] = pd.to_datetime(df[c], errors='coerce').apply(get_deps)
new_vals = p_ut.clean_cols(df, ["language",  "channel",  "inbound_or_outbound"],
                           '~/my_repos/capacity_planning/data/config/col_values.json', check_new=False, do_nan=True, rename=True)

f_deps = df.groupby(['ds_solved_at', 'language', 'channel']).apply(get_deps).reset_index()
f_deps.rename(columns={'ds_solved_at': 'ds'}, inplace=True)
f_arrs = df.groupby(['ds_created_at', 'language', 'channel']).apply(lambda x: len(x)).reset_index()
f_arrs.rename(columns={'ds_created_at': 'ds'}, inplace=True)

new_vals = p_ut.clean_cols(df, ["language",  "channel",  "inbound_or_outbound"],
                           '~/my_repos/capacity_planning/data/config/col_values.json', check_new=False, do_nan=True, rename=True)
f_a = df.groupby(['ds_created_at', 'language']).apply(get_arrs).reset_index()
f_a.drop('level_2', axis=1, inplace=True)
f_a.rename(columns={'ds_created_at': 'ds'}, inplace=True)
f_d = df.groupby(['ds_solved_at', 'language']).apply(get_deps).reset_index()
f_d.drop('level_2', axis=1, inplace=True)
f_d.rename(columns={'ds_solved_at': 'ds'}, inplace=True)
f = f_a.merge(f_d, on=['ds', 'language'], how='left')
f['inc'] = f['arrs'] - f['deps']
f.sort_values(by='ds', inplace=True)
fq = f.groupby(['language']).apply(get_queue)
fq['arr/ag'] = fq['arrs'] / fq['arr_agents']
fq['dep/ag'] = fq['deps'] / fq['dep_agents']

for lang in ['chinese', 'italian', 'english', 'spanish', 'french']:
    z = fq[fq.language == lang].copy()
    z.set_index('ds', inplace=True)
    z[['arrs', 'deps']].plot(grid=True, title='Daily Ticket Rates ' + lang)
    z[['median dur', 'avg dur']].plot(grid=True, title='Ticket Delay Days ' + lang)
    z[['inc', 'queue']].plot(grid=True, title='Ticket Queue ' + lang)
    z[['arr_is_reservation %', 'dep_is_reservation %']].plot(grid=True, title='Assigments and Reservation Fractions ' + lang)
    z[['arr/ag', 'dep/ag']].plot(grid=True, title='Agent Ratios ' + lang)

