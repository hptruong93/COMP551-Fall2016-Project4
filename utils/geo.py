import math
import numpy as np

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

def to_3d_representation(longs, lats):
    longs = math.radians(longs)
    lats = math.radians(lats)

    long_coses = np.cos(longs)
    lat_coses = np.cos(lats)
    long_sines = np.sin(longs)
    lat_sines = np.sin(lats)

    x = lat_coses * long_coses
    y = lat_coses * long_sines
    z = lat_sines

    return (x, y, z)

def to_long_lat(x, y, z):
    longitude = math.atan2(y, x)
    latitude = math.atan2(z, math.sqrt(x**2 + y**2))

    return math.degrees(longitude), math.degrees(latitude)