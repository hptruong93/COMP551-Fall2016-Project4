import csv
import datetime
import math
import random

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from nltk.cluster.kmeans import KMeansClusterer

cols_raw =[ 'event-id',
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
            'study-name']

base_time = '2000-06-26 00:22:57.000'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

def globe_distance(origin, destination):
    """
        Return distance in km.
        https://gist.github.com/rochacbruno/2883505

        Haversine formula example in Python
        Author: Wayne Dyck
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km
    #
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    #
    return d

def average_positions(longs, lats):
    assert len(longs) == len(lats)
    n = len(lats)

    sum_x = 0
    sum_y = 0
    sum_z = 0

    longs = [math.radians(l) for l in longs]
    longs = [(math.sin(l), math.cos(l)) for l in longs]

    lats = [math.radians(l) for l in lats]
    lats = [(math.sin(l), math.cos(l)) for l in lats]

    # x += Math.Cos(latitude) * Math.Cos(longitude);
    # y += Math.Cos(latitude) * Math.Sin(longitude);
    # z += Math.Sin(latitude);

    sum_x = sum([lats[i][1] * longs[i][1] for i in xrange(n)])
    sum_y = sum([lats[i][1] * longs[i][0] for i in xrange(n)])
    sum_z = sum([lats[i][0] for i in xrange(n)])

    # x = x / total;
    # y = y / total;
    # z = z / total;

    sum_x /= n
    sum_y /= n
    sum_z /= n

    # var centralLongitude = Math.Atan2(y, x);
    # var centralSquareRoot = Math.Sqrt(x * x + y * y);
    # var centralLatitude = Math.Atan2(z, centralSquareRoot);

    result = math.atan2(sum_y, sum_x), math.atan2(sum_z, math.sqrt(sum_x**2 + sum_y**2))
    return math.degrees(result[0]), math.degrees(result[1])


def parse_time(data):
    return datetime.datetime.strptime('{}000'.format(data), TIME_FORMAT)

def date_of_year(date):
    return int(date.strftime('%j'))

base_date = parse_time(base_time)

def parse_row(row):
    row[2] = parse_time(row[2]) # Parse time
    #
    row[3] = float(row[3]) # Longitude
    row[4] = float(row[4]) # Latitude
    #
    return (row[2], row[3], row[4], row[-2])

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
        av_long, av_lat = average_positions(longs, lats)
        event_base[2] = av_long
        event_base[3] = av_lat
        new_events.append(event_base)
    return new_events
        



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
    with open('data_with_ENV.csv', 'r') as f:
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
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    e = bird['events'][0]
    z = [(event[1]-e[1]).days for event in bird['events']]
    x = [event[2] if event[2] <= 0 else event[2] - 360 for event in bird['events']]
    y = [event[3] for event in bird['events']]
    ax.plot(x, y, z,'o-')
    plt.show()

def simple_x_y_plot(xs, ys, zs = None):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca()

    if zs:
        ax.plot(xs, ys, zs,'o-')
    else:
        ax.plot(xs, ys, 'o-')

    plt.show()

def load_raw_data(find_max = False):
    data = []

    with open('data.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        for index, row in enumerate(reader):
            if index == 0:
                continue
            row = parse_row(row)
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
    # birds = get_birds()
    # plot_bird(birds[1])
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

    longs = [0.0, 0.0]
    lats = [0.0, 90.0]
    
    print average_positions(longs, lats)

    # print "There are {} data points".format(len(data))
    # # data = data[:200]
    # data = np.array([(row[1], row[2]) for row in data])

    # results = []
    # for cluster_count in xrange(10, 50, 1):
    #     kclusterer = KMeansClusterer(cluster_count, distance=globe_distance, avoid_empty_clusters = True, repeats=10)
    #     assigned_clusters = kclusterer.cluster(data, assign_clusters=True)

    #     means = kclusterer.means()

    #     print "Count = {} and means are {}".format(cluster_count, means)

    #     error = sum([globe_distance(means[point], data[i]) for i, point in enumerate(assigned_clusters)])
    #     results.append((cluster_count, error))

    # with open('kmean_result.csv', 'w') as f:
    #     writer = csv.writer(f, delimiter = ',')

    #     for row in results:
    #         writer.writerow(row)