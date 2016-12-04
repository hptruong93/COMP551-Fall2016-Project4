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

cols_raw =[
'event-id',
'visible',
'timestamp',
'location-long',
'location-lat',
'algorithm-marked-outlier',
'argos:altitude',
'argos:best-level',
'argos:calcul-freq',
'argos:iq',
'argos:lat1',
'argos:lat2',
'argos:lc',
'argos:lon1',
'argos:lon2',
'argos:nb-mes',
'argos:nb-mes-120',
'argos:nopc',
'argos:pass-duration',
'argos:sensor-1',
'argos:sensor-2',
'argos:sensor-3',
'argos:sensor-4',
'argos:valid-location-manual',
'manually-marked-outlier',
'manually-marked-valid',
'sensor-type',
'individual-taxon-canonical-name',
'tag-local-identifier',
'individual-local-identifier',
'study-name',
]


cols_env = [
'event-id',
'timestamp',
'location-long',
'location-lat',
'individual-local-identifier',
'ECMWF Interim Full Daily SFC-FC Evaporation',
'ECMWF Interim Full Daily SFC Temperature (2 m above Ground)',
'ECMWF Interim Full Daily SFC-FC Sunshine Duration',
'NCEP NARR SFC Visibility at Surface',
'ECMWF Interim Full Daily SFC Charnock Parameter',
'ECMWF Interim Full Daily PL V Velocity',
'ECMWF Interim Full Daily PL U Velocity',
'ECMWF Interim Full Daily SFC Total Cloud Cover',
'ECMWF Interim Full Daily SFC Surface Air Pressure',
'ECMWF Interim Full Daily SFC Total Atmospheric Water',
'ECMWF Interim Full Daily SFC Snow Temperature',
'NASA Distance to Coast (Signed)',
'GlobCover 2009 2009 Land-Cover Classification',
'ECMWF Interim Full Daily SFC-FC Runoff',
'ECMWF Interim Full Daily SFC Ice Temperature at 0-7 cm',
'ECMWF Interim Full Daily SFC Soil Temperature at 1-7 cm',
'NCEP NARR SFC Snow Cover at Surface',
'ECMWF Interim Full Daily SFC Volumetric Soil Water Content at 1-7 cm'
]

cols_human = [
'event-id','visible',
'timestamp',
'location-long',
'location-lat',
'algorithm-marked-outlier',
'argos:altitude',
'argos:best-level',
'argos:calcul-freq',
'argos:iq',
'argos:lat1',
'argos:lat2',
'argos:lc',
'argos:lon1',
'argos:lon2',
'argos:nb-mes',
'argos:nb-mes-120',
'argos:nopc',
'argos:pass-duration',
'argos:sensor-1',
'argos:sensor-2',
'argos:sensor-3',
'argos:sensor-4',
'argos:valid-location-manual',
'manually-marked-outlier',
'manually-marked-valid',
'sensor-type',
'individual-taxon-canonical-name',
'tag-local-identifier',
'individual-local-identifier',
'study-name',
'ECMWF Interim Full Daily SFC-FC Evaporation',
'ECMWF Interim Full Daily SFC Temperature (2 m above Ground)',
'ECMWF Interim Full Daily SFC-FC Sunshine Duration',
'NCEP NARR SFC Visibility at Surface',
'ECMWF Interim Full Daily SFC Charnock Parameter',
'NCEP NARR 3D Cloud Water',
'ECMWF Interim Full Daily SFC Sea Ice Cover',
'ECMWF Interim Full Daily PL V Velocity',
'ECMWF Interim Full Daily PL U Velocity',
'ECMWF Interim Full Daily SFC Total Cloud Cover',
'SEDAC GRUMP v1 2000 Population Density Adjusted',
'ECMWF Interim Full Daily SFC Surface Air Pressure',
'ECMWF Interim Full Daily SFC Total Atmospheric Water'
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

    if args.task == 3:
        import metadata
        metadata.print_metadata('data/aggregated_data.csv')

    if args.task == 2:
        all_data = []
        all_cols = []

        def add_col(data, col_index):
            for index, row in enumerate(data):
                if len(all_data) < len(data):
                    all_data.append([])
                    assert len(all_data) == index + 1

                all_data[index].append(row[col_index])


        cols = [cols_raw, cols_env, cols_human]
        for findex, name in enumerate(['data.csv', 'data_with_ENV.csv', 'data_with_human_ENV.csv']):
            with open('data/%s' % name, 'r') as f:
                reader = csv.reader(f, delimiter = ',')

                loaded = [row for row in reader]
                for index, col_title in enumerate(cols[findex]):
                    if col_title not in all_cols:
                        all_cols.append(col_title)

                        add_col(loaded, index)
                    else:
                        continue

        with open('data/aggregated_data.csv', 'w') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(all_cols)
            writer.writerows(all_data)


        print len(all_cols)
        print len(all_data)

    if args.task == 1:
        with open('kmean/kmean_detailed_results.pickle', 'r') as f:
            detailed_results = pickle.load(f)
            means = detailed_results[11][1]

    if args.task == 0:
        birds = get_birds()
        plot_bird(birds[28])
        # events = birds[0]['events']




    # months = {}
    # for event in events[:]:
    #     date = event[1]
    #     month = date.month
    #     if month not in months:
    #         months[month] = []

    #     months[month].append(event)

    # for month, value in months.iteritems():
    #     print month, len(value)
