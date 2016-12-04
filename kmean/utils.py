from multiprocessing import Pool
from utils import geo

p = Pool(7)

def closest_mean(longitude, latitude, means):
	smallest = 99999999999
	smallest_mean = None

	for mean in means:
		d = geo.globe_distance((longitude, latitude), (m_longitude, m_latitude))

		if d < smallest:
			smallest = d
			smallest_mean = mean

	return smallest_mean, smallest
