#!/usr/bin/python2

import argparse
import csv
import datetime
import math
import random
import pickle
from collections import Counter


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn import utils as sklearn_utils
import metadata
from utils import geo

cols_raw = metadata.get_cols('data/data.csv')
cols_env = metadata.get_cols('data/data_with_ENV.csv')
cols_human = metadata.get_cols('data/data_with_human_ENV.csv')

cols_aggregate = metadata.get_cols('data/aggregated_data.csv')

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

def plot_bird(bird, save_fig = False):
    import draw_map
    e = bird['events'][0]
    z = [(event[1]-e[1]).days for event in bird['events']]
    x = [event[2] for event in bird['events']]
    y = [event[3] for event in bird['events']]

    draw_map.plot(y, x, z, save = save_fig, title = 'Position of a single bird')

def old_plot_bird(bird):
   import matplotlib as mpl
   from mpl_toolkits.mplot3d import Axes3D
   import numpy as np
   import matplotlib.pyplot as plt
   mpl.rcParams['legend.fontsize'] = 10
   fig = plt.figure()
   ax = fig.gca(projection='3d')
   e = bird['events'][0]
   z = [(event[0]-e[0]).days for event in bird['events']]
   x = [event[1] for event in bird['events']]
   y = [event[2] for event in bird['events']]
   ax.plot(x, y, z,'o-')
   plt.show()

def to_cluster(event):
    longitude = event[2]
    latitude = event[3]
    from kmean import kmean_utils
    # Load means
    with open('kmean/kmean_detailed_results.pickle', 'r') as f:
        detailed_results = pickle.load(f)
        chosen_cluster_count = 11
        means = detailed_results[chosen_cluster_count][1]
    mean_index, d = kmean_utils.closest_mean(longitude, latitude, means)
    return mean_index


def simple_x_y_plot(xs, ys, zs = None, title = 'No title', xtitle = None, ytitle = None):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca()

    if zs:
        ax.plot(xs, ys, zs,'o-')
    else:
        ax.plot(xs, ys, 'o-')

    if xtitle:
        plt.xlabel(xtitle)
    if ytitle:
        plt.ylabel(ytitle)

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

def load_aggregate_data():
    with open('data/aggregated_data.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        data = [row for row in reader][1:]

        # Have to parse before remove otherwise lost track of the indices
        data = generic_parse_row(data, metadata.metadata_aggregated)
        return filter_irrelevant_data(data, metadata.metadata_aggregated, cols_aggregate)

def generic_parse_row(data, the_metadata):
    # Parse data based on metadata

    for row_index, row in enumerate(data):
        for index, col_metadata in enumerate(the_metadata):
            data_type = col_metadata[1]

            if data_type in [int, float]:
                try:
                    row[index] = data_type(row[index])
                except:
                    print "Unable to parse row %s of column %s. Value is %s" % (row_index, index, row[index])
            elif data_type is datetime.datetime:
                row[index] = parse_time(row[index])

    return data


def filter_irrelevant_data(data, the_metadata, col_titles):
    removing_indices = set()

    for index, col_metadata in enumerate(the_metadata):
        # Remove too few occurrences
        if col_metadata[0] < 50:
            # print "Removing %s due to occurrence = %s" % (col_titles[index], col_metadata[0])
            removing_indices.add(index)

        # Remove nan
        if col_metadata[6] > 0:
            # print "Removing %s due to nan" % col_titles[index]
            removing_indices.add(index)

    # Manual filtering
    removing_indices.add(col_titles.index('event-id'))
    removing_indices.add(col_titles.index('argos:lat1')) # Repeated long
    removing_indices.add(col_titles.index('argos:lon1')) # Repeated lat
    removing_indices.add(col_titles.index('argos:sensor-4'))
    removing_indices.add(col_titles.index('tag-local-identifier')) # Identifier

    # Now proceed to remove columns
    # Remove the last index first to preserve index for next removal if need be
    for index in sorted(removing_indices, reverse = True):
        for row in data:
            del row[index]

    print "Reduced from %s columns to %s columns. Removed %s columns." % (len(col_titles), len(data[0]), len(col_titles) - len(data[0]))

    new_titles = [title for i, title in enumerate(col_titles) if i not in removing_indices]
    new_metadata = [old_metadata for i, old_metadata in enumerate(the_metadata) if i not in removing_indices]

    return data, new_metadata, new_titles

def transform_data(data, the_metadata, col_titles):
    """
        Calculate new features from current set of features. Losing metadata in the process since we do not do any analysis.
    """

    new_titles = col_titles[:]

    # First calculate month since beginning and month
    replacing_index = col_titles.index('timestamp')

    del new_titles[replacing_index]
    new_titles.insert(replacing_index, 'Group by qtt since beginning')
    new_titles.insert(replacing_index, 'Month')

    # Secondly convert long lat to x y z
    long_index = new_titles.index('location-long')
    del new_titles[2]
    del new_titles[2]

    new_titles.insert(2, 'Z')
    new_titles.insert(2, 'Y')
    new_titles.insert(2, 'X')


    for row in data:
        current_date = row[replacing_index]
        month = current_date.month
        # month_since_beginning = (current_date.year - base_date.year) * 12 + current_date.month - base_date.month
        time_since_beginning = (current_date - base_date).days / 30

        del row[replacing_index]
        row.insert(replacing_index, time_since_beginning)
        row.insert(replacing_index, month)

        x, y, z = geo.to_3d_representation(row[2], row[3])
        del row[2]
        del row[2]

        row.insert(2, z)
        row.insert(2, y)
        row.insert(2, x)

    # Make identifier to be readable
    identifier_index = new_titles.index('individual-local-identifier')
    for row in data:
        to_replace = int(row[identifier_index].replace('-', ''))
        row[identifier_index] = to_replace

    all_ids = sorted(list(set(row[identifier_index] for row in data)))
    for row in data:
        row[identifier_index] = all_ids.index(row[identifier_index])

    return data, new_titles


def normalize(data, cols):
    # If cols = -1 then do all cols
    #uniform normlzn btwn 0,1: x = (x - min)/ (max - min)

    if (cols == -1): cols = xrange(len(data[0]))

    for i in cols:
        minn = np.min(data[:,i])
        maxx = np.max(data[:,i])

        data[:,i] -= minn
        data[:,i] /= float(maxx - minn)

    return data

def group_by_time_period(data, col_titles, y = None):
    """
        For now group by column 'Group by qtt' since beginning
    """
    current_individual = 0
    current_month = 0

    id_index = col_titles.index('individual-local-identifier')
    month_index = col_titles.index('Group by qtt since beginning')

    untouched_columns = ['individual-local-identifier']
    untouched_column_indices = [col_titles.index(col) for col in untouched_columns]

    all_rows = []
    all_ys = []

    output = []
    output_y = []

    for row_index, row in enumerate(data):
        individual_id = row[id_index]
        month = row[month_index]

        if month == current_month and individual_id == current_individual:
            all_rows.append(row)

            if y is not None:
                all_ys.append(y[row_index])
        else:
            if len(all_rows) != 0:
                new_row = []

                # Now do average for each column
                for column_index in xrange(len(col_titles)):
                    if column_index in untouched_column_indices:
                        new_row.append(all_rows[0][column_index]) # Use the first value
                        continue

                    avg = np.average([row_of_month[column_index] for row_of_month in all_rows])
                    new_row.append(avg)

                output.append(new_row)

                # Major vote for y
                if y is not None:
                    counter = Counter(all_ys) # To count major vote
                    output_y.append(counter.most_common(1)[0][0]) # Get the value only. Don't care about frequency.

            all_rows = [row]
            if y is not None:
                all_ys = [y[row_index]]

            current_month = month
            current_individual = individual_id

    return output, output_y


def load_y(file_name = 'data/y_11.csv'):
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        data = [int(row[1]) for row in reader] # Column 0 is the index, which we don't care
        return data

def load_and_preprocess(y_data_file = 'data/y_11.csv'):
    data, new_metadata, new_titles = load_aggregate_data()
    data, new_titles = transform_data(data, new_metadata, new_titles)

    y = load_y(y_data_file)

    original_len = len(y)
    data, y = group_by_time_period(data, new_titles, y)
    print "Grouped from {} down to {} rows.".format(original_len, len(y))

    # with open('data/test1.csv', 'w') as f:
    #     writer = csv.writer(f, delimiter = ',')
    #     writer.writerow(new_titles)
    #     writer.writerows(data)

    data = np.array(data)
    y_data = np.array(y)

    group_by_feature_index = new_titles.index('Group by qtt since beginning')
    max_group_by_feature_value = int(np.max(data[:, group_by_feature_index]))

    import serialize
    serialized_data = serialize.serialize(data, y_data, new_titles.index('individual-local-identifier'), group_by_feature_index, max_group_by_feature_value)
    X, y = serialize.flatten_chunk(serialized_data, new_titles, time_period = 3)
    X = normalize(X, -1)

    # count = 0
    # for i, value in enumerate(y[:-1]):
    #     if value == y[i +1]:
    #         count += 1

    # print float(count) / len(y)

    print y[0]
    sklearn_utils.shuffle(X, y)

    return X, y, new_titles