#!/usr/bin/python2
import numpy as np

class Bird:
    def __init__(self, ID, max_month):
        self.ID = ID
        self.chunks = [None for i in range(max_month)]

class Month_Chunk:
    def __init__(self, month, features, cluster):
        self.month = month
        self.x = features
        self.y = cluster

def serialize(dataX, dataY, ID_col, month_col, max_month):
    #organizes data into object oriented series
    #max_month should be max(year*12 + month)

    #sort.dataX(dataX[ID_col])
    birds = []
    IDs = []

    for i, data_x in enumerate(dataX):
        ID = int(data_x[ID_col])

        if not ID in IDs:
            IDs.append(int(ID))
            birds.append(Bird(ID, max_month+1))

        month_value = int(data_x[month_col])

        if (birds[IDs.index(ID)].chunks[month_value] != None): print("WARNING: multiple entries per month in a single individual.")
        birds[IDs.index(ID)].chunks[month_value] = Month_Chunk(month_value, data_x, dataY[i])

    return birds

def flatten_chunk(birds, titles, time_period = 3):
    """
        Return tuple of X and y
    """
    output_x = []
    y = []

    x_index = titles.index('X')
    y_index = titles.index('Y')
    z_index = titles.index('Z')

    for bird in birds:
        for i in xrange(len(bird.chunks) - (time_period + 1)):
            try:
                new_row = np.hstack([bird.chunks[i + j].x for j in xrange(time_period)])
                # y.append(bird.chunks[i + (time_period + 1)].y)
                y.append((bird.chunks[i + (time_period + 1)].x[x_index],
                            bird.chunks[i + (time_period + 1)].x[y_index],
                            bird.chunks[i + (time_period + 1)].x[z_index]))

                output_x.append(new_row)
            except:
                # print "Skipping over missing data point"
                pass

    X =  np.vstack(output_x)
    y = np.array(y)

    print "Flattened into {} and {}".format(X.shape, y.shape)

    return X, y
    # for i, c in enumerate(birds[15].chunks):
    #     print i, c

    # return np.array([])


if __name__ == "__main__":
    X = np.array([[1,1,1,1],
        [1,2,6,8],
        [1,3,7,8],
        [2,1,1,5],
        [2,2,2,2],
        [2,3,5,6]])
    Y = np.array([1,2,3,1,2,3])
    series = serialize(X,Y,0,1,3)
    print(series[1].chunks[1].x)