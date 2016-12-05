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

    for i in range(len(dataX)):
        ID = dataX[i][ID_col]
        if not ID in IDs:
            IDs.append(int(ID))
            birds.append(Bird(ID, max_month+1))
        if (birds[IDs.index(ID)].chunks[dataX[i][month_col]] != None): print("WARNING: multiple entries per month in a single individual.")
        birds[IDs.index(ID)].chunks[dataX[i][month_col]] = Month_Chunk(dataX[i][month_col], dataX[i], dataY[i])

    return birds


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