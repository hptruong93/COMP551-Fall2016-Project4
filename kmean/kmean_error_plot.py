import csv
import matplotlib.pyplot as plt


def plot_kmean_error(results):
    x = [p[0] for p in results]
    y = [p[1] for p in results]
    plt.plot(x, y, 'ro', markersize=6)
    plt.show()


if __name__ == "__main__":
	results = []
	with open('../kmean_result.csv', 'r') as f:
		reader = csv.reader(f, delimiter = ',')
		for row in reader:
			results.append(row)

	plot_kmean_error(results)