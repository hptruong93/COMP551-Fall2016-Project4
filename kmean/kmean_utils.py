from utils import geo

def closest_mean(longitude, latitude, means):
	smallest = 99999999999
	smallest_index = None

	for index, mean in enumerate(means):
		mx, my, mz = mean[0], mean[1], mean[2]
		x, y, z = geo.to_3d_representation(longitude, latitude)

		d = (x - mx)**2 + (y - my)**2 + (z - mz)**2

		if d < smallest:
			smallest = d
			smallest_index = index

	return smallest_index, smallest
