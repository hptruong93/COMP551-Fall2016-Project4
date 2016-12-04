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
    for cluster_count in xrange(10, 50, 1):
        kmeans = KMeans(n_clusters=cluster_count, n_jobs = -2)
        kmeans.fit(data)

        detailed_results[cluster_count] = kmeans.cluster_centers_
        print "Count = {} and error is {}".format(cluster_count, kmeans.inertia_)

        results.append((cluster_count, kmeans.inertia_))

    with open('kmean_result.csv', 'w') as f:
        writer = csv.writer(f, delimiter = ',')

        for row in results:
            writer.writerow(row)

    with open('kmean_detailed_results.pickle', 'w') as f:
        pickle.dump(detailed_results, f)

def plot_means():
    with open('kmean_detailed_results.pickle', 'r') as f:
        detailed_results = pickle.load(f)

    for k in [20, 30, 40]:
        
        longs = []
        lats = []
        for center in detailed_results[k]:
            longitude, latitude = data_reader.to_long_lat(center[0], center[1], center[2])

            longs.append(longitude)
            lats.append(latitude)

        data_reader.simple_x_y_plot([l if l <= 0 else l - 360 for l in longs], lats, title = 'For k = %s' % k)

if __name__ == "__main__":
    # data = data_reader.load_raw_data()
    # kmean(data)

    # data = data_reader.get_birds()
    # data_reader.average_birds(data)
    # data = [(event[2], event[3]) for bird in data for event in bird['events']]
    # kmean(data, 0, 1)

    plot_kmean_error_from_file()
    plot_means()