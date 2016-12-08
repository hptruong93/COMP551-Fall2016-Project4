#!/usr/bin/python2
import argparse
import csv
import datetime
import math
import random
import pickle
from collections import Counter

from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

from utils import geo
from data_reader import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Kmean')
    parser.add_argument('-t', '--task', dest = 'task', default = 0, help = 'Choose which task to run', type = float)
    args = parser.parse_args()

    if args.task == 8.5: # Plot the REGRESSION result in pairs
        import draw_map

        names = ['Ridge']
        for name in names:
            with open('data/{}_result.csv'.format(name), 'r') as f:
                reader = csv.reader(f, delimiter = ',')

                data = [row for row in reader]
                longs_prediction = [float(row[0]) for row in data]
                lats_prediction = [float(row[1]) for row in data]
                longs_actual = [float(row[2]) for row in data]
                lats_actual = [float(row[3]) for row in data]

            zs = [geo.globe_distance((lats_prediction[i], longs_prediction[i]), (lats_actual[i], longs_actual[i])) for i in xrange(len(data))]
            draw_map.plot(lats_actual, longs_actual, z = zs, title = 'Prediction errors per point for {}'.format(name), z_title = 'Prediction error for the point (km)')


    if args.task == 8: # Plot the REGRESSION result
        import draw_map

        names = ['Ridge', 'Lasso', 'LinearRegression']
        for name in names:
            with open('data/{}_result.csv'.format(name), 'r') as f:
                reader = csv.reader(f, delimiter = ',')

                data = [row for row in reader]
                longs_prediction = [float(row[0]) for row in data]
                lats_prediction = [float(row[1]) for row in data]
                longs_actual = [float(row[2]) for row in data]
                lats_actual = [float(row[3]) for row in data]

            draw_map.plot(  lats_prediction + lats_actual,
                            longs_prediction + longs_actual,
                            z = [0 for i in xrange(len(data))] + [1 for i in xrange(len(data))],
                            discrete_z = True, title = 'Result for {}'.format(name), z_title = 'Blue: prediction; Red: actual')


    if args.task == 7: # Serializing and thing
        X, y, titles = load_and_preprocess('data/y_11.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

        print "Identified {} data samples. Test size is {}.".format(X_train.shape, X_test.shape)

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

        def get_model_name(model):
            full_name = str(type(model))
            return full_name[full_name.index("'") + 1 : full_name.rindex("'")].split('.')[-1]

        models = [
                    linear_model.LinearRegression(),
                    linear_model.Lasso(alpha=0.1),
                    linear_model.Ridge(alpha = 0.1, max_iter = None, tol = 0.00001),
                    # linear_model.LogisticRegression(),
                    # svm.SVC(kernel = 'linear'),
                    # svm.SVC(kernel = 'rbf'),
                    # RandomForestClassifier(n_estimators = 100, n_jobs = 1)
                    ]

        for model in models:
            scores = cross_val_score(model, X, y, cv = 10, n_jobs = 7)
            print get_model_name(model), np.average(scores)

            model.fit(X_train, y_train)
            print "Test score is {}".format(model.score(X_test, y_test))

            x_index = titles.index('X')
            y_index = titles.index('Y')
            z_index = titles.index('Z')
            prediction = model.predict(X_test)

            # longs_lats = [geo.to_long_lat(row[x_index], row[y_index], row[z_index] for row in X_test)]
            ys = [geo.to_long_lat(*row) for row in y_test]
            y_pred = [geo.to_long_lat(*row) for row in prediction]

            # Write test result to file
            with open('data/{}_result.csv'.format(get_model_name(model)), 'w') as f:
                writer = csv.writer(f, delimiter = ',')

                to_write = [pred + ys[i] for i, pred in enumerate(y_pred)]
                writer.writerows(to_write)


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

    if args.task == 4.5: # Cluster visualization
        from kmean import kmean_utils
        data = load_raw_data()

        # Load means
        with open('kmean/kmean_detailed_results.pickle', 'r') as f:
            detailed_results = pickle.load(f)
            chosen_cluster_count = 11

            means = detailed_results[chosen_cluster_count][1]

        longs = []
        lats = []
        zs = []
        for row_index, row in enumerate(data):
            longitude, latitude = row[1], row[2]
            longs.append(longitude)
            lats.append(latitude)

            mean_index, d = kmean_utils.closest_mean(longitude, latitude, means)

            zs.append(mean_index)

        import draw_map
        draw_map.plot(lats, longs, z = zs, discrete_z = True, title = 'K-mean cluster visualization for k = {}'.format(chosen_cluster_count), z_title = 'Cluster')


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
