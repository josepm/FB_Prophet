"""

"""
import os
import sys
import platform
import pandas as pd

os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'  # before pandas load
os.environ['NUMEXPR_MAX_THREADS'] = '4' if platform.system() == 'Darwin' else '36'  # before pandas load

from capacity_planning.forecast.utilities.language import _ens_cfg as ec
from capacity_planning.utilities import sys_utils as s_ut
from capacity_planning.utilities import pandas_utils as p_ut

if __name__ == '__main__':
    # ############################################
    # parameters
    # ############################################
    w_arr = [1.96]       # weight for the score function
    min_runs_arr = [6]   # min runs for a fcast cfg to be considered
    num_cfg_arr = [128]   # fcast cfgs to retrieve
    p_col = 'f_err'      # column to use to select fcast cfg (smape, mape, lars, mase)

    print(sys.argv)
    if len(sys.argv) == 3:
        idx = 1
    elif len(sys.argv) == 4:
        idx = 2
    else:
        print('invalid arguments: ' + str(sys.argv))
        sys.exit()

    ts_name, cutoff_date = sys.argv[idx:]
    lbl = ts_name + '_' + cutoff_date + '_' + str(max(num_cfg_arr))
    res_dict = dict()
    dict_main = ec.main(ts_name, cutoff_date)
    if dict_main is not None:
        for k, df in dict_main.items():
            if k not in res_dict.keys():
                res_dict[k] = list()
            res_dict[k].append(df)
            # for w in w_arr:
            #     for min_runs in min_runs_arr:
    #         for num_cfg in num_cfg_arr:
    #             s_ut.my_print('meta::starting main: min_runs: ' + str(min_runs) + ' num_cfg: ' + str(num_cfg) + ' w: ' + str(w))
    #             dict_main = ec.main(ts_name, cutoff_date, min_runs, num_cfg, w, p_col)   # {k: DF, ...}
    #             if dict_main is not None:
    #                 for k, df in dict_main.items():
    #                     if k not in res_dict.keys():
    #                         res_dict[k] = list()
    #                     res_dict[k].append(df)

    if len(res_dict) > 0:
        dict_out = {k: pd.concat(arr, axis=0) for k, arr in res_dict.items()}
        _ = [p_ut.save_df(f, '~/my_tmp/ens_cfg/' + k + '__' + lbl) for k, f in dict_out.items()]
        print('DONE')
    else:
        print('ERROR: no data was returned')

