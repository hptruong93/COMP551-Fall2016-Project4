#!/usr/bin/python2
import argparse
import csv
import datetime
import math
import random
import pickle
from collections import Counter

from data_reader import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Kmean')
    parser.add_argument('-t', '--task', dest = 'task', default = 0, help = 'Choose which task to run', type = int)
    args = parser.parse_args()


    if args.task == 7: # Serializing and thing
        data, new_metadata, new_titles = load_aggregate_data()
        data, new_titles = transform_data(data, new_metadata, new_titles)

        y = load_y('data/y_11.csv')

        data, y = group_by_time_period(data, new_titles, y)
        data = np.array(data)
        y_data = np.array(y)

        group_by_feature_index = new_titles.index('Week since beginning')

        max_group_by_feature_value = int(np.max(data[:, group_by_feature_index]))

        import serialize
        serialized_data = serialize.serialize(data, y_data, new_titles.index('individual-local-identifier'), group_by_feature_index, max_group_by_feature_value)
        X, y = serialize.flatten_chunk(serialized_data)

        from sklearn import linear_model, svm
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cross_validation import cross_val_score


        models = [linear_model.LinearRegression(), linear_model.Lasso(alpha=0.1), linear_model.Ridge(alpha = 1.0, max_iter = None, tol = 0.001),
                    svm.SVC(kernel = 'linear'), svm.SVC(kernel = 'rbf'),
                    RandomForestClassifier(n_estimators = 20, n_jobs = -1)]

        for model in models:
            scores = cross_val_score(model, X, y, cv = 10, n_jobs = 6)
            print type(model), np.average(scores)

    if args.task == 6: # Group by
        data, new_metadata, new_titles = load_aggregate_data()
        data, new_titles = transform_data(data, new_metadata, new_titles)

        data, _ = group_by_time_period(data, new_titles)

        # Write to file
        with open('data/test1.csv', 'w') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(new_titles)
            writer.writerows(data)

    if args.task == 5: # Convert from aggregated to filter data: all floats
        data, new_metadata, new_titles = load_aggregate_data()
        data, new_titles = transform_data(data, new_metadata, new_titles)

        # Write to file
        with open('data/filtered_data.csv', 'w') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(new_titles)
            writer.writerows(data)

    if args.task == 4: # Cluster long lat into cluster for each point
        from kmean import kmean_utils
        data = load_raw_data()

        # Load means
        with open('kmean/kmean_detailed_results.pickle', 'r') as f:
            detailed_results = pickle.load(f)
            chosen_cluster_count = 11

            means = detailed_results[chosen_cluster_count][1]

        ys = []
        for row_index, row in enumerate(data):
            longitude, latitude = row[1], row[2]
            mean_index, d = kmean_utils.closest_mean(longitude, latitude, means)

            ys.append((row_index, mean_index))

        print Counter([y[1] for y in ys])
        with open('data/y_%s.csv' % chosen_cluster_count, 'w')  as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerows(ys)


    if args.task == 3: # Test of printing metadata
        metadata.print_metadata('data/aggregated_data.csv')

    if args.task == 2: # Aggregate the data from different csv
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

                loaded = [row for row in reader][1:] # Remember to remove headers
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

    if args.task == 1: # Loading means from kmean result
        with open('kmean/kmean_detailed_results.pickle', 'r') as f:
            detailed_results = pickle.load(f)
            means = detailed_results[11][1]

    if args.task == 0:
        birds = get_birds()
        plot_bird(birds[28])
        # events = birds[0]['events']
