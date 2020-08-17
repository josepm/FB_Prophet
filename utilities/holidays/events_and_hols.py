"""

"""
import pandas as pd

# Korea is missing: https://en.wikipedia.org/wiki/Public_holidays_in_South_Korea
lang_to_country = {
    'Mandarin': ['China'],
    'Mandarin_Offshore': ['China'],
    'Mandarin_Onshore': ['China'],
    'Russian': ['Russia'],
    'English_NA': ['UnitedStates', 'Canada'],
    'French': ['France', 'Belgium'],
    'English_EMEA': ['France', 'Italy', 'Ireland', 'UnitedKingdom', 'Netherlands', 'Germany', 'Spain'],
    'German': ['Germany', 'Austria'],
    'Italian': ['Italy'],
    'Japanese': ['Japan'],
    'Spanish': ['Spain'],  # 'Argentina', 'Mexico', 'Colombia'],
    'Portuguese': ['Portugal', 'Brazil'],
    'English_APAC': ['Australia', 'Korea', 'Japan'], #, 'India', 'NewZealand', 'SouthAfrica', 'Thailand', 'Vietnam'],
    'Korean': ['Korea'],
    'All': ['China', 'Russia', 'UnitedStates', 'Canada', 'France', 'Belgium', 'Italy', 'Ireland', 'UnitedKingdom', 'Netherlands', 'Germany', 'Spain', 'Germany', 'Austria',
            'Japan', 'Portugal', 'Brazil', 'Australia', 'Korea']
}

name_maps = {
    'Madarin': {}
}


lang_holidays = {
    'French': ['New Year', 'Easter Monday', 'Labour Day', 'Victory in Europe Day', 'Ascension Thursday', 'Whit Monday', 'Bastille Day',
               'Assumption of Mary to Heaven',  'All Saints Day', 'Christmas Day', 'Easter_Week'],
    'Mandarin': ['Chinese New Year', 'Chinese New Year\'s eve', 'Spring Festival',
                 'Ching Ming Festival', 'Labour Day Holiday', 'Dragon Boat Festival', 'Mid-Autumn Festival', 'National Day', 'Tomb-Sweeping Day', 'Labor Day'],
    'Korean': ['Korean New Year\'s Day', 'Independence Day', 'Buddha\'s Birthday', 'Children\'s Day', 'Memorial Day', 'Liberation Day',
               'Midautumn Festival', 'National Foundation Day', 'Hangul Day', 'Christmas Day', 'Winter Solstice', 'Summer Solstice'],
    'Japanese': ['New year', 'Coming of Age Day', 'Foundation Day', "The Emperor's Birthday", 'Vernal Equinox Day',
                 'Showa Day', 'Constitution Memorial Day', "Children's Day", 'Marine Day', 'Health and Sports Day', 'Mountain Day',
                 'Respect-for-the-Aged Day', 'Autumnal Equinox Day', 'Culture Day'],
    'German': ['New year', 'Good Friday', 'Easter Monday', 'Easter_Week', 'Labour Day', 'Ascension Thursday', 'Whit Monday',
               'Day of German Unity', 'Christmas Day', 'Corpus Christi'],
    'Italian': ['New year', 'Easter Monday', 'Easter_Week', 'Liberation Day', "International Workers' Day", 'Republic Day',
                'Assumption of Mary to Heaven', 'All Saints Day', 'Immaculate Conception', 'Christmas Day'],
    'Portuguese': ['New year', 'Good Friday', 'Easter_Week', 'Easter Sunday',
                   'Dia da Liberdade', 'Dia do Trabalhador', 'Dia de Portugal',
                   'Corpus Christi', 'Assunção de Nossa Senhora', 'Implantação da República',
                   'Restauração da Independência', 'Imaculada Conceição', 'Christmas Day'],
    'Russian': ['New year', 'Christmas', 'Defendence of the Fatherland', 'Labour Day',
                'National Day', 'Day of Unity'],
    'Spanish': ['New year', 'Epiphany', 'Good Friday', 'Easter_Week', 'Día del trabajador',
                'Assumption of Mary to Heaven', 'All Saints Day', 'Día de la Constitución Española',
                'Immaculate Conception', 'Christmas Day'],
    'English_NA': ['New year', "Washington's Birthday", 'Memorial Day', 'Independence Day', 'Labor Day',
                   'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Christmas Day'],
    'English_EMEA': ['New Year', 'Easter Sunday', 'Christmas Day', 'Easter Monday', 'Good Friday', 'Whit Monday', 'Easter_Week', 'Saint Patrick\'s Day'],
    'English_APAC': ['New year', 'Australia Day', 'Good Friday', 'Easter_Week', 'Easter Monday', 'Anzac Day', 'Christmas Day']
    # 'All': ['Spring Festival', "Washington's Birthday", 'New year', 'Immaculate Conception', "Chinese New Year's eve", 'Labour Day Holiday', 'Culture Day', 'Labor Day',
    #         'Victory in Europe Day', 'Marine Day', 'Easter_Week', 'Christmas', 'Easter Monday', 'Labour Day', "Korean New Year's Day", 'Dia de Portugal', 'Epiphany',
    #         'Ascension Thursday', "Saint Patrick's Day", 'Anzac Day', 'Memorial Day', 'National Foundation Day', 'Corpus Christi', 'Day of German Unity', 'Ching Ming Festival',
    #         'Veterans Day', "Buddha's Birthday", 'Thanksgiving Day', 'Easter Sunday', 'Assunção de Nossa Senhora', 'Mountain Day', 'Columbus Day', 'Foundation Day', 'New Year',
    #         'Día de la Constitución Española', 'Tomb-Sweeping Day', "Children's Day", 'Bastille Day', 'Good Friday', 'Day of Unity', 'Australia Day', 'Independence Day',
    #         'Hangul Day', 'Winter Solstice', "The Emperor's Birthday", 'Día del trabajador', 'Coming of Age Day', 'Restauração da Independência', 'Defendence of the Fatherland',
    #         'Health and Sports Day', 'Imaculada Conceição', 'Autumnal Equinox Day', 'Whit Monday', 'Mid-Autumn Festival', 'Dragon Boat Festival', 'All Saints Day', 'Republic Day',
    #         'Midautumn Festival', 'National Day', 'Assumption of Mary to Heaven', 'Liberation Day', 'Implantação da República', 'Dia da Liberdade', 'Christmas Day',
    #         'Vernal Equinox Day', 'Summer Solstice', 'Respect-for-the-Aged Day', 'Dia do Trabalhador', 'Constitution Memorial Day', "International Workers' Day",
    #         'Showa Day', 'Chinese New Year']
}

# events that move accross countries cannot be included as they would land in different time series (olympics, soccer cups, ...)
# added _ in country value to avoid picking them up
events = {
    'superbowl': [
        ('UnitedStates', list(map(pd.to_datetime, ['2016-02-07', '2017-02-05',
                                                   '2018-02-04', '2019-02-03', '2020-02-02',
                                                   '2021-02-07', '2022-02-06', '2023-02-05', '2024-02-04'])))
    ],
    'summer_olympics': [
        ('_Brazil', pd.date_range(start='2016-08-05', end='2016-08-21')),
        ('_Japan', pd.date_range(start='2020-07-24', end='2020-08-09')),
        ('_France', pd.date_range(start='2024-07-26', end='2024-08-11'))
    ],
    'winter_olympics': [
        ('_Korea', pd.date_range(start='2018-02-09', end='2018-02-25')),
        ('_China', pd.date_range(start='2022-02-04', end='2022-02-20'))
    ],
    'fifa_cup': [
        ('_Russia', pd.date_range(start='2018-06-14', end='2018-07-15')),
        ('_Qatar', pd.date_range(start='2022-11-21', end='2022-12-18'))
    ],
    'uefa_cup': [
        ('_France', pd.date_range(start='2016-06-10', end='2016-07-10')),
        ('_UnitedKingdom', pd.date_range(start='2020-06-12', end='2020-07-12')),
        ('_Germany', pd.date_range(start='2020-06-12', end='2020-07-12')),
        ('_Italy', pd.date_range(start='2020-06-12', end='2020-07-12')),
        ('_Netherlands', pd.date_range(start='2020-06-12', end='2020-07-12')),
        ('_Ireland', pd.date_range(start='2020-06-12', end='2020-07-12')),
        ('_Spain', pd.date_range(start='2020-06-12', end='2020-07-12')),
        ('_Russia', pd.date_range(start='2020-06-12', end='2020-07-12')),
        ('_Denmark', pd.date_range(start='2020-06-12', end='2020-07-12'))
    ],
    'sundance': [
        ('UnitedStates', pd.date_range(start='2016-01-21', end='2016-01-31')),
        ('UnitedStates', pd.date_range(start='2017-01-19', end='2017-01-29')),
        ('UnitedStates', pd.date_range(start='2018-01-18', end='2018-01-28')),
        ('UnitedStates', pd.date_range(start='2019-01-24', end='2019-02-03')),
        ('UnitedStates', pd.date_range(start='2020-01-23', end='2020-02-02')),
        ('UnitedStates', pd.date_range(start='2021-01-21', end='2021-01-31')),
        ('UnitedStates', pd.date_range(start='2022-01-20', end='2021-01-30'))
    ],
    'sxsw': [
        ('UnitedStates', pd.date_range(start='2016-03-15', end='2016-03-20')),
        ('UnitedStates', pd.date_range(start='2017-03-10', end='2017-03-19')),
        ('UnitedStates', pd.date_range(start='2018-03-09', end='2018-03-18')),
        ('UnitedStates', pd.date_range(start='2019-03-08', end='2019-03-17')),
        ('UnitedStates', pd.date_range(start='2020-03-13', end='2020-03-22')),
        ('UnitedStates', pd.date_range(start='2021-03-12', end='2021-03-22')),
        ('UnitedStates', pd.date_range(start='2022-03-11', end='2021-03-21'))
    ],
    'mardi_grass': [
        ('UnitedStates', list(map(pd.to_datetime, ['2016-02-09', '2017-02-28',
                                                   '2018-02-13', '2019-03-05',
                                                   '2020-02-25', '2021-02-16', '2022-03-01'])))
    ],
    'coachella': [
        ('UnitedStates', pd.date_range(start='2016-04-15', end='2016-04-17')),
        ('UnitedStates', pd.date_range(start='2016-04-22', end='2016-04-24')),
        ('UnitedStates', pd.date_range(start='2017-04-14', end='2017-04-16')),
        ('UnitedStates', pd.date_range(start='2017-04-21', end='2017-04-23')),
        ('UnitedStates', pd.date_range(start='2018-04-13', end='2018-04-15')),
        ('UnitedStates', pd.date_range(start='2018-04-20', end='2018-04-22')),
        ('UnitedStates', pd.date_range(start='2019-04-12', end='2019-04-14')),
        ('UnitedStates', pd.date_range(start='2019-04-19', end='2019-04-21')),
        ('UnitedStates', pd.date_range(start='2020-04-10', end='2020-04-12')),
        ('UnitedStates', pd.date_range(start='2020-04-17', end='2020-04-19')),
        ('UnitedStates', pd.date_range(start='2021-04-09', end='2021-04-11')),
        ('UnitedStates', pd.date_range(start='2021-04-16', end='2021-04-18')),
        ('UnitedStates', pd.date_range(start='2022-04-08', end='2021-04-10')),
        ('UnitedStates', pd.date_range(start='2022-04-15', end='2021-04-17'))
    ],
    'kentucky_derby': [
        ('UnitedStates', list(map(pd.to_datetime, ['2016-05-07', '2017-05-06',
                                                   '2018-05-05', '2019-05-04',
                                                   '2020-05-02', '2021-05.-1', '2022-05-07'])))
    ],
    'bonnaroo': [
        ('UnitedStates', pd.date_range(start='2016-06-09', end='2016-06-12')),
        ('UnitedStates', pd.date_range(start='2017-06-08', end='2017-06-11')),
        ('UnitedStates', pd.date_range(start='2018-06-07', end='2018-06-11')),
        ('UnitedStates', pd.date_range(start='2019-06-13', end='2019-06-16')),
        ('UnitedStates', pd.date_range(start='2020-06-11', end='2020-06-14')),
        ('UnitedStates', pd.date_range(start='2021-06-10', end='2021-06-13')),
        ('UnitedStates', pd.date_range(start='2022-06-09', end='2022-06-12'))
        ],
    'us_open': [
        ('UnitedStates', pd.date_range(start='2016-08-29', end='2016-09-11')),
        ('UnitedStates', pd.date_range(start='2017-08-28', end='2017-09-10')),
        ('UnitedStates', pd.date_range(start='2018-08-27', end='2018-09-09')),
        ('UnitedStates', pd.date_range(start='2019-08-26', end='2019-09-08')),
        ('UnitedStates', pd.date_range(start='2020-08-31', end='2020-09-13')),
        ('UnitedStates', pd.date_range(start='2021-08-30', end='2021-09-12')),
        ('UnitedStates', pd.date_range(start='2022-08-29', end='2021-09-11'))
    ],
    'governors_ball': [
        ('UnitedStates', pd.date_range(start='2016-06-03', end='2016-06-05')),
        ('UnitedStates', pd.date_range(start='2017-06-02', end='2017-06-05')),
        ('UnitedStates', pd.date_range(start='2018-06-01', end='2018-06-04')),
        ('UnitedStates', pd.date_range(start='2019-05-31', end='2019-06-02')),
        ('UnitedStates', pd.date_range(start='2020-06-05', end='2020-06-07')),
        ('UnitedStates', pd.date_range(start='2021-06-04', end='2021-06-06')),
        ('UnitedStates', pd.date_range(start='2022-06-03', end='2022-06-05'))
    ],
    'montreal_jazz': [
        ('Canada', pd.date_range(start='2016-06-29', end='2016-07-09')),
        ('Canada', pd.date_range(start='2017-06-28', end='2017-07-08')),
        ('Canada', pd.date_range(start='2018-06-28', end='2018-07-07')),
        ('Canada', pd.date_range(start='2019-06-27', end='2019-07-06')),
        ('Canada', pd.date_range(start='2020-06-25', end='2020-07-04')),
        ('Canada', pd.date_range(start='2021-07-01', end='2021-07-10')),
        ('Canada', pd.date_range(start='2022-06-28', end='2022-07-09'))
    ],
    'firefly_music': [
        ('UnitedStates', pd.date_range(start='2016-06-16', end='2016-06-19')),
        ('UnitedStates', pd.date_range(start='2017-06-15', end='2017-06-18')),
        ('UnitedStates', pd.date_range(start='2018-06-14', end='2018-06-17')),
        ('UnitedStates', pd.date_range(start='2019-06-21', end='2019-06-24')),
        ('UnitedStates', pd.date_range(start='2020-06-19', end='2020-06-22')),
        ('UnitedStates', pd.date_range(start='2021-06-18', end='2021-06-21')),
        ('UnitedStates', pd.date_range(start='2022-06-17', end='2022-06-20'))

    ],
    'french_open': [
        ('France', pd.date_range(start='2016-05-22', end='2016-06-05')),
        ('France', pd.date_range(start='2017-05-22', end='2017-06-11')),
        ('France', pd.date_range(start='2018-05-27', end='2018-06-10')),
        ('France', pd.date_range(start='2019-05-26', end='2019-06-09')),
        ('France', pd.date_range(start='2020-05-24', end='2020-06-07')),
        ('France', pd.date_range(start='2021-05-24', end='2021-06-06')),
        ('France', pd.date_range(start='2022-05-23', end='2022-06-05'))
    ],
    'oktoberfest': [
        ('Germany', pd.date_range(start='2016-09-17', end='2016-10-03')),
        ('Germany', pd.date_range(start='2017-09-16', end='2017-10-03')),
        ('Germany', pd.date_range(start='2018-09-22', end='2018-10-07')),
        ('Germany', pd.date_range(start='2019-09-21', end='2019-10-06')),
        ('Germany', pd.date_range(start='2020-09-19', end='2020-10-04')),
        ('Germany', pd.date_range(start='2021-09-18', end='2021-10-03')),
        ('Germany', pd.date_range(start='2022-09-17', end='2022-10-03'))
    ],
    'brazil_carnival': [
        ('Brazil', pd.date_range(start='2016-02-07', end='2016-02-12')),
        ('Brazil', pd.date_range(start='2017-02-26', end='2016-03-03')),
        ('Brazil', pd.date_range(start='2018-02-09', end='2018-02-14')),
        ('Brazil', pd.date_range(start='2019-03-01', end='2019-03-06')),
        ('Brazil', pd.date_range(start='2020-02-21', end='2020-02-26')),
        ('Brazil', pd.date_range(start='2021-02-12', end='2021-02-27')),
        ('Brazil', pd.date_range(start='2022-02-25', end='2022-03-02'))
    ],
    'nxne': [
        ('Canada', pd.date_range(start='2016-06-13', end='2016-06-19')),
        ('Canada', pd.date_range(start='2017-06-16', end='2017-06-25')),
        ('Canada', pd.date_range(start='2018-06-08', end='2018-06-17')),
        ('Canada', pd.date_range(start='2019-06-07', end='2019-06-16')),
        ('Canada', pd.date_range(start='2020-06-12', end='2020-06-21')),
        ('Canada', pd.date_range(start='2021-06-11', end='2021-06-20')),
        ('Canada', pd.date_range(start='2022-06-10', end='2022-06-19')),
    ],
    'tulip_festival': [],
    'copenhaguen_jazz': [],
    'canadian_music_week': [],
    'root_picnic_music_festival': [],
    'eu_rugby': [],                       # probably in multiple countries
    'summer_paralympic_games': [],        # probably in multiple countries
    'winter_paralympic_games': [],        # probably in multiple countries
    'spring_break': []                   # varies by college a lot?
}


