"""
$ python lang_fin_cast.py ts_name cutoff_date
ts_name: phone-inbound-vol, phone-outbound-vol, phone-inbound-aht, phone-outbound-aht, ...
cutoff_data: yyyy-mm-dd
Generates forecasts for all cfgs and given cutoff date.
It measures a cfg performance by,
 - setting a 'fake' cutoff date, fake_cutoff = cutoff_date - upr
 - measuring the performance between fake_cutoff + lwr and fake_cutoff + upr = cutoff_date
Output is a parquet file dated with the input cutoff_date AND table sup.fct_cx_forecast_config
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import platform

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'   # before pandas load
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'    # before pandas load

import pandas as pd

from capacity_planning.forecast.utilities.language import data_prep as dtp
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut


# ###########################################
# ###########################################
# ###########################################


FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def main(argv):
    print('usage: python lang_fcast.py <ts_name> <cutoff_date>')
    print(argv)
    ts_name, cutoff_date = argv
    this_file = os.path.basename(__file__)
    cfg_dir = '/'.join(FILE_PATH.split('/')[:-1])
    cfg_file = os.path.join(cfg_dir, 'config/' + this_file[:-3] + '_cfg.json')

    # validate the data, prepare regressors, holidays DF
    ts_obj, reg_dict, cfg_dict, train_days = dtp.initialize(cfg_file, cutoff_date, ts_name, True, init_date='2016-01-01')
    upr_horizon, lwr_horizon = cfg_dict['upr_horizon_days'], cfg_dict['lwr_horizon_days']
    if_exists = cfg_dict['if_exists']
    cutoff_date = ts_obj.cutoff_date

    out_list = list()
    cu = cutoff_date + pd.to_timedelta(upr_horizon, unit='D')  # actual cutoff date for training
    ds = str(cu.date())
    # ctr = 0
    # train_days = [25, 35]
    for l, t_df in ts_obj.df_dict.items():
        # if l != 'Mandarin':
        #     continue
        s_ut.my_print('\n\n****************************** starting language: ' + str(l))
        lang_list = list()
        if t_df is not None:
            for tdays in train_days:
                tlist = dtp.get_f_cfgs(t_df, l, cutoff_date, tdays, upr_horizon, cfg_dict, is_train=True)
                if tlist is None:
                    s_ut.my_print('WARNING: language ' + str(l) + ' and training cutoff date ' +
                                  str(cutoff_date.date()) + ' and training days ' + str(tdays) + ' has NO fcast configs')
                    continue
                else:
                    arg_list = dtp.prophet_prep(ts_obj, l, reg_dict.get(l, None), cfg_dict, upr_horizon, lwr_horizon, tlist, True)
                    s_ut.my_print('pid: ' + str(os.getpid()) + ' ************* forecasts for ' + str(l)
                                  + ' with ' + str(tdays) + ' train days and ' + str(len(arg_list)) + ' configs')
                    f_list = s_ut.do_mp(dtp.tf, arg_list, is_mp=True, cpus=len(arg_list), do_sigkill=True)
                    if f_list is None:
                        s_ut.my_print('pid: ' + str(os.getpid()) + ': No results with ' + str(tdays) + ' training days')
                        f_list = list()
                    else:
                        s_ut.my_print('pid: ' + str(os.getpid()) + ': ' + str(len(f_list)) + ' results with ' + str(tdays) + ' training days')

                    # save the fcast configs
                    if len(f_list) > 0:
                        s_ut.my_print('pid: ' + str(os.getpid()) + ' concatenating ' + str(len(f_list)) + ' DFs for ' + str(l))
                        l_df = pd.concat([f for f in f_list], axis=0)
                        l_df['language'] = l
                        s_ut.my_print('pid: ' + str(os.getpid()) + ' Language ' + str(l) + ' has ' + str(len(l_df)) + ' fcast cfgs with ' + str(tdays) + ' training days')
                        l_df.reset_index(inplace=True, drop=True)
                        l_df['ds'] = ds                # here we only save cfg's not fcasts. Use ds for partition
                        l_df['ts_name'] = ts_name
                        l_df['cutoff'] = ds
                        lang_list.append(l_df)
                        out_list.append(l_df)
                    else:
                        s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no DF for ' + str(l))
        else:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING: no training DF for ' + str(l))
        # ctr += 1
        # if ctr >= 2:
        #     break
        if len(lang_list) > 0:   # save language level results
            fl = pd.concat(lang_list, axis=0)
            p_ut.save_df(fl, '~/my_tmp/fcast_cfg_v2_' + ds + '_' + ts_name + '_' + l)

    # all training done or this TS. Save data
    if len(out_list) > 0:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' *************** saving training data ***********')
        df_all = pd.concat(out_list, axis=0)
        df_all.drop_duplicates(inplace=True)
        df_all.reset_index(inplace=True, drop=True)
        df_all['ds'] = ds           # here we only save cfg's not fcasts
        df_all['ts_name'] = ts_name
        df_all['cutoff'] = ds
        p_ut.save_df(df_all, '~/my_tmp/fcast_cfg/fcast_cfg_v2_' + ds + '_' + ts_name)
        df_all.drop(['ds', 'ts_name'], inplace=True, axis=1)  # not needed to push
        partition_ = {'ds': ds, 'ts_name': ts_name}
        table = 'sup.fct_cx_forecast_config_v3'
        try:   # only hive works with the partition argument
            with s_ut.suppress_stdout_stderr():
                import airpy as ap
            ap.hive.push(df_all, table=table, if_exists=if_exists, partition=partition_,
                         table_props={'abb_retention_days': '-1', 'abb_retention_days_reason': 'fact table. No pii'})
            s_ut.my_print('data saved to table ' + table + ' for ' + ts_name + ' and ds ' + ds)
            print('DONE')
        except:
            s_ut.my_print('ERROR: could not save to table ' + table + ' for ' + ts_name)
    else:
        s_ut.my_print('ERROR: no output')


if __name__ == '__main__':
    main(sys.argv[-2:])
