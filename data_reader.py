import argparse
import csv
import datetime
import math
import random
import pickle

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from utils import geo
                                                                                       # Distinct                    Min              Max                   Avg                 Std
cols_raw =[ 'event-id',                                                                # 0 65555   | type='int'     | 1327259329.0  | 1327356134.0 | 1327309855.51       |  28237.9522391     |
            'visible',                                                                 # 1 1       | type='str'     | NA            |              |                     |                    |
            'timestamp',                                                               # 2 64885   | type='str'     | NA            |              |                     |                    |
            'location-long',                                                           # 3 16800   | type='int'     | -178.925      | 179.211      | -159.141198368      |  26.7476752401     |
            'location-lat',                                                            # 4 8342    | type='int'     | 55.037        | 71.626       | 66.5375792236       |  3.4426639343      |
            'algorithm-marked-outlier',                                                # 5 1       | type='str'     | NA            |              |                     |                    |
            'argos:altitude',                                                          # 6 219     | type='float'   | 0.0           | 3314.0       | 6.77235908779       |  51.7691380718     |
            'argos:best-level',                                                        # 7 29      | type='float'   | -140.0        | -109.0       | -130.743528335      |  3.15205129717     |
            'argos:calcul-freq',                                                       # 8 26209   | type='float'   | 401.6395579   | 401.6799106  | 401.657870843       |  0.0142823820079   |
            'argos:iq',                                                                # 9 38      | type='int'     | 0.0           | 68.0         | 42.0671344672       |  26.6204604347     |
            'argos:lat1',                                                              # 10 8776   | type='float'   | 46.021        | 89.883       | 66.5835557318       |  3.60882477897     |
            'argos:lat2',                                                              # 11 31336  | type='float'   | -7.992        | 89.974       | 68.3377295553       |  12.0038691772     |
            'argos:lc',                                                                # 12 7      | type='int'     | Error         |              |                     |                    |
            'argos:lon1',                                                              # 13 17225  | type='float'   | -179.987      | 179.929      | -158.123607444      |  31.1820782735     |
            'argos:lon2',                                                              # 14 56757  | type='float'   | -179.995      | 180.0        | -21.0334114408      |  130.646980196     |
            'argos:nb-mes',                                                            # 15 15     | type='int'     | 1.0           | 15.0         | 4.97464724277       |  2.40393451078     |
            'argos:nb-mes-120',                                                        # 16 5      | type='int'     | 0.0           | 4.0          | 0.00335596064373    |  0.0697863513026   |
            'argos:nopc',                                                              # 17 5      | type='int'     | 0.0           | 4.0          | 2.68397528793       |  0.975451222738    |
            'argos:pass-duration',                                                     # 18 728    | type='float'   | 0.0           | 999.0        | 341.046968195       |  169.374704975     |
            'argos:sensor-1',                                                          # 19 174    | type='int'     | 0.0           | 255.0        | 142.032812142       |  82.297515194      |
            'argos:sensor-2',                                                          # 20 255    | type='int'     | 0.0           | 255.0        | 155.064007322       |  85.0454146916     |
            'argos:sensor-3',                                                          # 21 251    | type='int'     | 0.0           | 255.0        | 65.3964304782       |  79.1712241477     |
            'argos:sensor-4',                                                          # 22 257    | type='str'     | NA            |              |                     |                    |
            'argos:valid-location-manual',                                             # 23 3      | type='str'     | NA            |              |                     |                    |
            'manually-marked-outlier',                                                 # 24 1      | type='str'     | NA            |              |                     |                    |
            'manually-marked-valid',                                                   # 25 1      | type='str'     | NA            |              |                     |                    |
            'sensor-type',                                                             # 26 1      | type='str'     | NA            |              |                     |                    |
            'individual-taxon-canonical-name',                                         # 27 1      | type='str'     | NA            |              |                     |                    |
            'tag-local-identifier',                                                    # 28 78     | type='int'     | 2521.0        | 41691.0      | 21281.058592        |  14214.950019      |
            'individual-local-identifier',                                             # 29 110    | type='str'     | NA            |              |                     |                    |
            'study-name',                                                              # 30 1      | type='str'     | NA            |              |                     |                    |
            ]
                                                                                       # Distinct                      Min              Max             Avg                      Std
cols_env = ['event-id',                                                                # 0 65555   | <type 'int'>   | 1327259329.0  | 1327356134.0 | 1327309855.51       |  28237.9522391     |
            'timestamp',                                                               # 1 64885   | <type 'str'>   | NA            |              |                     |                    |
            'location-long',                                                           # 2 16800   | <type 'int'>   | -178.925      | 179.211      | -159.141198368      |  26.7476752401     |
            'location-lat',                                                            # 3 8342    | <type 'int'>   | 55.037        | 71.626       | 66.5375792236       |  3.4426639343      |
            'individual-local-identifier',                                             # 4 110     | <type 'str'>   | NA            |              |                     |                    |
            'ECMWF Interim Full Daily SFC-FC Evaporation',                             # 5 6017    | <type 'int'>   | -0.002839196  | 0.001169741  | -0.000209237272422  |  0.000253724189994 |
            'ECMWF Interim Full Daily SFC Temperature (2 m above Ground)',             # 6 65547   | <type 'int'>   | 241.4055473   | 292.6229989  | 273.644842504       |  8.35396648946     |
            'ECMWF Interim Full Daily SFC-FC Sunshine Duration',                       # 7 45189   | <type 'int'>   | 0.0           | 42333.0      | 7521.45597249       |  8667.6764868      |
            'NCEP NARR SFC Visibility at Surface',                                     # 8 65306   | <type 'int'>   | 4.447330214   | 20023.39613  | 12498.8428409       |  7303.46229224     |
            'ECMWF Interim Full Daily SFC Charnock Parameter',                         # 9 51157   | <type 'int'>   | 0.010075927   | 0.065644898  | 0.0176142772015     |  0.00248376142306  |
            'ECMWF Interim Full Daily PL V Velocity',                                  # 10 65551  | <type 'int'>   | -21.08058671  | 20.16687464  | -1.02472339275      |  4.86479531666     |
            'ECMWF Interim Full Daily PL U Velocity',                                  # 11 65551  | <type 'int'>   | -20.81070009  | 20.24135953  | -1.4506310121       |  5.24345092172     |
            'ECMWF Interim Full Daily SFC Total Cloud Cover',                          # 12 60327  | <type 'int'>   | 1e-12         | 1.0          | 0.743102277598      |  0.293911600156    |
            'ECMWF Interim Full Daily SFC Surface Air Pressure',                       # 13 65513  | <type 'int'>   | 94959.48236   | 104380.6282  | 100349.698093       |  1158.53176885     |
            'ECMWF Interim Full Daily SFC Total Atmospheric Water',                    # 14 65551  | <type 'int'>   | 0.70862954    | 39.72950247  | 11.1945769725       |  6.41966390837     |
            'ECMWF Interim Full Daily SFC Snow Temperature',                           # 15 65543  | <type 'int'>   | 241.0419503   | 290.9746729  | 273.17023822        |  7.54046645446     |
            'NASA Distance to Coast (Signed)',                                         # 16 59658  | <type 'int'>   | -34.62561917  | 132.3779905  | 4.60725164282       |  7.99626514017     |
            'GlobCover 2009 2009 Land-Cover Classification',                           # 17 16     | <type 'int'>   | 60.0          | 230.0        | 203.139501182       |  20.2297682809     |
            'ECMWF Interim Full Daily SFC-FC Runoff',                                  # 18 4350   | <type 'int'>   | 0.0           | 0.005736834  | 8.03360337718e-05   |  0.000220902290779 |
            'ECMWF Interim Full Daily SFC Ice Temperature at 0-7 cm',                  # 19 44104  | <type 'int'>   | 247.408705    | 273.16082    | 270.482584781       |  3.31512118564     |
            'ECMWF Interim Full Daily SFC Soil Temperature at 1-7 cm',                 # 20 65535  | <type 'int'>   | 248.0458487   | 291.0741865  | 273.841026536       |  6.96357059547     |
            'NCEP NARR SFC Snow Cover at Surface',                                     # 21 28228  | <type 'int'>   | 0.0           | 1.0          | 0.644213099549      |  0.410414792385    |
            'ECMWF Interim Full Daily SFC Volumetric Soil Water Content at 1-7 cm'     # 22 63902  | <type 'int'>   | -9.26e-22     | 0.363463317  | 0.0916834456686     |  0.069889765194    |
            ]

base_time = '2000-06-26 00:22:57.000'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


def parse_time(data):
    return datetime.datetime.strptime('{}000'.format(data), TIME_FORMAT)

def date_of_year(date):
    return int(date.strftime('%j'))

base_date = parse_time(base_time)

def parse_row(row, keep_cols = False):
    row[2] = parse_time(row[2]) # Parse time
    row[3] = float(row[3]) # Longitude
    row[4] = float(row[4]) # Latitude

    if not keep_cols:
        return (row[2], row[3], row[4], row[-2])
    else:
        return row

def events_in_month(events):
    m = events[0][1].month
    es_in_m = []
    for i in range(0,12):
        es_in_m.append((m,[event for event in events if event[1].month==m]))
        m += 1
        if m > 12:
            m = 1
    return es_in_m

def average(events):
    es_in_m = events_in_month(events)
    new_events = []
    for e_in_m in es_in_m:
        if e_in_m[1] == []:
            continue
        event_base = list(e_in_m[1][0])
        for i,v in enumerate(event_base):
            if i > 4:
                values = [event[i] for event in e_in_m[1]]
                event_base[i] = sum(values) / float(len(values))
        longs = [event[2] for event in e_in_m[1]]
        lats = [event[3] for event in e_in_m[1]]
        av_long, av_lat = geo.average_positions(longs, lats)
        event_base[2] = av_long
        event_base[3] = av_lat
        new_events.append(event_base)
    return new_events


def average_birds(birds):
    for bird in birds:
        bird['events'] = average(bird['events'])


def parse_row_ENV(row):
    row[1] = parse_time(row[1]) # Parse time

    row[2] = float(row[2]) # Longitude
    row[3] = float(row[3]) # Latitude
    for i,r in enumerate(row):
        if i>2 and not i==4:
            row[i] = float(r)

    return row


def load_data_with_ENV():
    data = []
    with open('data/data_with_ENV.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        for index, row in enumerate(reader):
            if index == 0:
                continue
            row = parse_row_ENV(row)
            data.append(row)
    return data

def get_birds():
    data = load_data_with_ENV()
    birds = []
    bird = {}
    bird['id']=''
    bird['events'] = []
    for d in data:
        if d[4]!= bird['id']:
            birds.append(bird)
            bird = {}
            bird['id'] = d[4]
            bird['events'] = []
        bird['events'].append(d)
    birds.append(bird)
    del birds[0]
    return birds

def find_bird(birds):
    for i, bird in enumerate(birds):
        tm = max([event[2] for event in bird['events']])
        if tm > 0:
            return i
    return -1

def plot_bird(bird):
    import draw_map
    e = bird['events'][0]
    z = [(event[1]-e[1]).days for event in bird['events']]
    x = [event[2] for event in bird['events']]
    y = [event[3] for event in bird['events']]

    draw_map.plot(y, x, z, title = 'Position of a single bird')


def simple_x_y_plot(xs, ys, zs = None, title = 'No title'):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca()

    if zs:
        ax.plot(xs, ys, zs,'o-')
    else:
        ax.plot(xs, ys, 'o-')

    plt.title(title)
    plt.show()

def load_raw_data(keep_cols = False, find_max = False):
    data = []

    with open('data/data.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        for index, row in enumerate(reader):
            if index == 0:
                continue
            row = parse_row(row, keep_cols = keep_cols)
            data.append(row)

    if find_max:
        # Find max
        maxes = {}
        for row in data:
            name = row[-1]
            value = row[0]

            if name not in maxes or maxes[name][0] < value:
                maxes[name] = row

        data = [row for k, row in maxes.iteritems()]

    return data

def draw_globe_from_raw_data(data):
    import draw_map
    longs = [row[1] for row in data]
    lats = [row[2] for row in data]
    z = np.array([date_of_year(row[0]) for row in data])

    draw_map.plot(lats, longs, z, save = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Kmean')
    parser.add_argument('-t', '--task', dest = 'task', default = 0, help = 'Choose which task to run', type = int)
    args = parser.parse_args()

    if args.task == 0:
        birds = get_birds()
        plot_bird(birds[28])
        # events = birds[0]['events']

    if args.task == 1:
        with open('kmean/kmean_detailed_results.pickle', 'r') as f:
            detailed_results = pickle.load(f)
            means = detailed_results[11][1]

    if args.task == 2:
        data = load_raw_data(keep_cols = True)
        # data = load_data_with_ENV()

        len_to_loop = len(cols_raw)
        for i in xrange(len_to_loop):

            to_test = data[0][i]
            data_type = None
            try:
                int(to_test)
                data_type = int
            except:
                try:
                    float(to_test)
                    data_type = float
                except:
                    data_type = str

            if data_type is int or data_type is float:
                try:
                    values = [float(row[i]) for row in data]
                except:
                    print 4454, i, "Error"
                    continue
                maxx = np.max(values)
                minn = np.min(values)
                avg = np.average(values)
                std = np.std(values)

                print 4454, i, 4312, minn, 5246, maxx, 3769, avg, 9851, std
            else:
                print 4454, i, "NA"



    # months = {}
    # for event in events[:]:
    #     date = event[1]
    #     month = date.month
    #     if month not in months:
    #         months[month] = []

    #     months[month].append(event)

    # for month, value in months.iteritems():
    #     print month, len(value)
