"""
builds language aware holidays DF
https://pypi.org/project/holidays/
adds special periodic events that are not holidays:
- superbowl
- olympics
- uefa, fifa cups

$ python make_holidays.py
"""

# #########################################
import os
import pandas as pd
from datetime import date, timedelta
import copy
import sys

import holidays
from fbprophet import hdays as fbp_holidays
# from fbprophet import make_holidays as mhdays
import json

from capacity_planning.forecast.utilities.holidays import events_and_hols as evh
from capacity_planning.utilities import sys_utils as s_ut

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


country_hols = dict()
from workalendar.usa import UnitedStates
country_hols['UnitedStates'] = UnitedStates()

from workalendar.europe import Russia
country_hols['Russia'] = Russia()
from workalendar.europe import France
country_hols['France'] = France()
from workalendar.europe import Belgium
country_hols['Belgium'] = Belgium()
from workalendar.europe import Spain
country_hols['Spain'] = Spain()
from workalendar.europe import Germany
country_hols['Germany'] = Germany()
from workalendar.europe import Austria
country_hols['Austria'] = Austria()
from workalendar.europe import Italy
country_hols['Italy'] = Italy()
from workalendar.europe import Portugal
country_hols['Portugal'] = Portugal()
from workalendar.europe import UnitedKingdom
country_hols['UnitedKingdom'] = UnitedKingdom()
from workalendar.europe import Ireland
country_hols['Ireland'] = Ireland()
from workalendar.europe import Netherlands
country_hols['Netherlands'] = Netherlands()

from workalendar.asia import China
country_hols['China'] = China()
from workalendar.asia import Japan
country_hols['Japan'] = Japan()
from workalendar.asia import SouthKorea
country_hols['Korea'] = SouthKorea()
# from workalendar.asia import India
# country_hols['India'] = India()
# from workalendar.asia import Thailand
# country_hols['Thailand'] = Thailand()
# from workalendar.asia import Vietnam
# country_hols['Vietnam'] = Vietnam()
# from workalendar.asia import Indonesia
# country_hols['Indonesia'] = Indonesia()

from workalendar.oceania import Australia
country_hols['Australia'] = Australia()

from workalendar.america import Brazil
country_hols['Brazil'] = Brazil()
from workalendar.america import Canada
country_hols['Canada'] = Canada()
from workalendar.america import Mexico
country_hols['Mexico'] = Mexico()


def add_events(countries, years, my_hols):
    # the effect of events that change countries (Olympics, ...) is very hard to model
    # as the interaction impact show at different languages (ie different time series)
    # each time the country changes and it is very hard to move this impact across languages
    for ev_name, ev_list in evh.events.items():
        for c, drg in ev_list:  # country, dates
            if c in countries:
                for d in drg:
                    if d.year in years:
                        my_hols.append((d,  ev_name))


def set_easter(adj_hols, my_hols):
    def set_week(kmin, kmax, hdict, date_):
        for k in range(kmin, kmax + 1):
            nd = date_ - timedelta(k)
            hdict.append((nd, 'Easter_Week'))

    if 'Easter_Week'.lower() in adj_hols:
        if 'Easter Monday'.lower() in adj_hols:
            dates = [x[0] for x in my_hols if x[1].lower() == 'Easter Monday'.lower()]
            _ = [set_week(1, 3, my_hols, d) for d in dates]
            return
        elif 'Easter Sunday'.lower() in adj_hols:
            dates = [x[0] for x in my_hols if x[1].lower() == 'Easter Sunday'.lower()]
            _ = [set_week(1, 2, my_hols, d)for d in dates]
            return
        elif 'Easter'.lower() in adj_hols:
            dates = [x[0] for x in my_hols if x[1].lower() == 'Easter'.lower()]
            _ = [set_week(1, 2, my_hols, d) for d in dates]
            return
        elif 'Good Friday'.lower() in adj_hols:
            dates = [x[0] for x in my_hols if x[1].lower() == 'Good Friday'.lower()]
            _ = [set_week(-2, -1, my_hols, d) for d in dates]
            return
        else:
            return


def expand(adj_hols, hols, language):
    def add(hlist, ex, add_days, lbl=None, start_days=1):
        if lbl is None:
            lbl = ex
        vals = [x[0] for x in hlist if ex.lower() in x[1].lower()]
        for d in vals:
            for idx in range(start_days, add_days + 1):
                hlist.append((d + timedelta(idx), lbl))
        return list(hlist)

    a_hols = [x for x in hols if x[1].lower() in adj_hols]  # trim to tracked hols
    set_easter(adj_hols, a_hols)                     # add easter week (if applicable)

    # add days
    if language == 'Mandarin':
        a_hols = add(a_hols, 'Chinese New Year', 5, 'Spring Festival')
        a_hols = add(a_hols, 'National Day', 7)
    elif language == 'Korean':
        pass
    elif language == 'Japanese':
        a_hols = add(a_hols, 'Showa Day', 3, start_days=-1)
    else:
        pass
    return a_hols


def adjust(hols, language, years):
    if language == 'Mandarin':
        new_hols = [x for x in hols if x[1].lower() not in ['Labor Day'.lower(), 'New Year'.lower()]]
        for y in years:
            new_hols.append((date(y, 5, 1), 'Labor Day'))
            new_hols.append((date(y, 1, 1), 'New Year'))
        return copy.deepcopy(new_hols)
    elif language == 'Korean':
        new_hols = copy.deepcopy(hols)
        for y in years:
            new_hols.append((date(y, 5, 1), 'Labor Day'))
            new_hols.append((date(y, 6, 21), 'Summer Solstice'))
            new_hols.append((date(y, 12, 21), 'Winter Solstice'))
        return copy.deepcopy(new_hols)
    else:
        return copy.deepcopy(hols)


def get_hols(language, end_year, upr_win=1, lwr_win=1):
    init_year = 2016
    countries = get_from_lang_country(language)
    if countries is None:
        s_ut.my_print('holidays:get_hols: no countries found for language ' + str(language))
        return None
    years = list(range(init_year, end_year + 1))
    adj_hols = [x.lower() for x in get_from_lang_holidays(language)]
    my_hols = list()
    for c in countries:
        try:                # workalendar package
            h = [(v[0],  v[1]) for y in years for v in country_hols[c].holidays(y)]
            th = adjust(h, language, years)
            my_hols += expand(adj_hols, th, language)
        except:             # holidays package
            s_ut.my_print('WARNING for calendar ' + c + ' Trying FB calendar')
            try:
                hols_ = getattr(holidays, c)(years=years, expand=False)
                hols = [(k, v) for k, v in hols_.items()]
                th = adjust(hols, language, years)
                my_hols += expand(adj_hols, th, language)
            except AttributeError:
                try:        # fbp_holidays package
                    hols_ = getattr(fbp_holidays, c)(years=years, expand=False)
                    hols = [(k, v) for k, v in hols_.items()]
                    th = adjust(hols, language, years)
                    my_hols += expand(adj_hols, th, language)
                except AttributeError:
                    s_ut.my_print('ERROR: holidays for country ' + c + ' is not available')
                    continue

    my_hols = list(set(my_hols))
    add_events(countries, years, my_hols)
    dout = {'ds': [x[0] for x in my_hols], 'holiday': [x[1] for x in my_hols]}
    hdf = pd.DataFrame(dout)
    hdf = hdf.groupby('ds').apply(lambda x: '+'.join(x['holiday'])).reset_index()
    hdf.columns = ['ds', 'holiday']
    hdf['language'] = language
    hdf['upper_window'] = upr_win
    hdf['lower_window'] = -lwr_win
    hdf['ds'] = pd.to_datetime(hdf['ds'].values)
    return hdf


def get_from_lang_holidays(language):
    return _get_from_dict(language, evh.lang_holidays)


def get_from_lang_country(language):
    return _get_from_dict(language, evh.lang_to_country)


def _get_from_dict(language, a_dict):
    hols = a_dict.get(language, None)
    if hols is not None:
        return hols
    else:
        if language == 'All':
            return list(set([h for v in a_dict.values() for h in v]))
        elif 'not-' in language:
            plang = language.replace('not-', '')
            return list(set([h for lg, lv in a_dict.items() for h in lv if lg != plang]))
        else:
            s_ut.my_print('ERROR: unknown language: ' + str(language))
            return list()


if __name__ == '__main__':
    cfg_file = os.path.join(FILE_PATH, '../../config/w_ts_process_cfg.json')
    with open(cfg_file, 'r') as fp:
        d_cfg = json.load(fp)
    lang = d_cfg.get('language', None)
    year = (pd.to_datetime(d_cfg['issue_date']) + pd.to_timedelta(int(d_cfg['fcast_periods']), unit='D')).year
    hols_file = os.path.join(FILE_PATH, '../../', d_cfg.get('hols_file'))
    df = get_hols(lang, int(year))
    if hols_file is not None:
        s_ut.my_print('saving to ' + hols_file)
        df.to_parquet(hols_file)
    else:
        s_ut.my_print('could not save holidays')

