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
        X, y, titles = load_and_preprocess('data/y_11.csv')
        print "Identified {} data samples.".format(X.shape)

        from sklearn import linear_model, svm
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cross_validation import cross_val_score

        # model = linear_model.Lasso(alpha=0.1)
        # model.fit(X, y)
        # print "Scores are {}".format(model.score(X, y))

        # c = model.coef_
        # irr = []

        # for row_index, row in enumerate(c):
        #     print "Starting with cluster {}".format(row_index)

        #     for col_index, val in enumerate(row):
        #         if abs(val) < 1:
        #             irr.append(titles[col_index % len(titles)])
        #             print "Feature {} is irrelevant.".format(titles[col_index % len(titles)])

        # common = Counter(irr).most_common()
        # print len(common), common

        models = [
                    # linear_model.LinearRegression(),
                    # linear_model.Lasso(alpha=0.1),
                    # linear_model.Ridge(alpha = 0.1, max_iter = None, tol = 0.00001),
                    linear_model.LogisticRegression(),
                    # svm.SVC(kernel = 'linear'),
                    # svm.SVC(kernel = 'rbf'),
                    RandomForestClassifier(n_estimators = 100, n_jobs = 1)
                    ]

        for model in models:
            scores = cross_val_score(model, X, y, cv = 10, n_jobs = 7)
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
        data = load_raw_data(keep_cols = True)

        # Load means
        with open('kmean/kmean_detailed_results.pickle', 'r') as f:
            detailed_results = pickle.load(f)
            chosen_cluster_count = 30

            means = detailed_results[chosen_cluster_count][1]

        current_bird = None
        current_clusters = []
        lens = []

        ys = []
        for row_index, row in enumerate(data):
            longitude, latitude = row[3], row[4]
            mean_index, d = kmean_utils.closest_mean(longitude, latitude, means)

            ys.append((row_index, mean_index))

            bird = row[29]
            if bird == current_bird:
                current_clusters.append(mean_index)
            else:
                lens.append(len(set(current_clusters)))
                print Counter(current_clusters).most_common()

                current_clusters = [mean_index]
                current_bird = bird

        print np.average(lens)
        print Counter([y[1] for y in ys])
        # with open('data/y_%s.csv' % chosen_cluster_count, 'w')  as f:
        #     writer = csv.writer(f, delimiter = ',')
        #     writer.writerows(ys)


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
