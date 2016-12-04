import argparse
import csv
import matplotlib.pyplot as plt
import data_reader
import numpy as np
import pickle

from sklearn.cluster import KMeans

def plot_kmean_error(results):
    x = [p[0] for p in results]
    y = [p[1] for p in results]
    plt.plot(x, y, 'ro', markersize=6)

    plt.savefig('kmean.png')

def plot_kmean_error_from_file():
    results = []
    with open('kmean_result.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader:
            results.append(row)

    plot_kmean_error(results)

def kmean(data, longitude_index = 1, latitude_index = 2):
    print "There are {} data points".format(len(data))
    data = [(row[longitude_index], row[latitude_index]) for row in data]
    data = np.array([data_reader.to_3d_representation(row[0], row[1]) for row in data])

    results = []
    detailed_results = {}
    old_error = None

    for cluster_count in xrange(2, 50, 1):
        kmeans = KMeans(n_clusters=cluster_count, n_jobs = -2)
        kmeans.fit(data)

        reduction_factor = 'unknown'
        if old_error is not None:
            reduction_factor = 100 * (old_error - kmeans.inertia_) / old_error

        detailed_results[cluster_count] = (reduction_factor, kmeans.cluster_centers_)

        old_error = kmeans.inertia_
        print "Count = {} and error is {}. Reduction factor is {}".format(cluster_count, kmeans.inertia_, reduction_factor)

        results.append((cluster_count, kmeans.inertia_))

    with open('kmean_result.csv', 'w') as f:
        writer = csv.writer(f, delimiter = ',')

        for row in results:
            writer.writerow(row)

    with open('kmean_detailed_results.pickle', 'w') as f:
        pickle.dump(detailed_results, f)

def plot_means(detailed_results):
    for k in [5, 10, 15, 20]:
        if k not in detailed_results:
            continue

        longs = []
        lats = []
        for center in detailed_results[k][1]:
            longitude, latitude = data_reader.to_long_lat(center[0], center[1], center[2])

            longs.append(longitude)
            lats.append(latitude)

        draw_map.plot(lats, longs, title = 'Means. k = %s' % k)
        # data_reader.simple_x_y_plot([l if l <= 0 else l - 360 for l in longs], lats, title = 'For k = %s' % k)

def plot_reduction_factor(detailed_results):
    xs = []
    ys = []
    for k in xrange(50):
        if k not in detailed_results:
            continue

        xs.append(k)

        reduction_factor = detailed_results[k][0]
        ys.append(100 if reduction_factor == 'unknown' else reduction_factor)

    data_reader.simple_x_y_plot(xs, ys, title = 'Reduction factor in percent against min.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Kmean')
    parser.add_argument('-t', '--task', dest = 'task', default = 0, help = 'Choose which task to run', type = int)
    args = parser.parse_args()

    if args.task == 0:
        data = data_reader.load_raw_data()
        kmean(data)

    if args.task == 1:
        data = data_reader.get_birds()
        data_reader.average_birds(data)
        data = [(event[2], event[3]) for bird in data for event in bird['events']]
        kmean(data, 0, 1)

    if args.task == 2:
        import draw_map
        plot_kmean_error_from_file()
        with open('kmean_detailed_results.pickle', 'r') as f:
            detailed_results = pickle.load(f)
            plot_means(detailed_results)

    if args.task == 3:
        with open('kmean_detailed_results.pickle', 'r') as f:
            detailed_results = pickle.load(f)
            plot_reduction_factor(detailed_results)