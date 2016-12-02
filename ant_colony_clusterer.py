import random as rd
import multiprocessing as mp
import plot_clusterer, math


class Datapt:
    def __init__(self, longitude, latitude):
        self.x = longitude
        self.y = latitude

class Ant:
    def __init__(self, longitude, latitude):
        self.x = longitude
        self.y = latitude
        self.data = None
        self.vel_modifier = rd.random()*10

    def drop(self, grid):
        if (self.data == None):
            print("ERROR: ant has no data to drop.")
            return
        if (grid[self.x][self.y] != None):
            print("ERROR: grid does not have space to drop.")
            return
        grid[self.x][self.y] = self.data
        self.data = None



def cluster_ant_colonies(X, num_ants, iterations, vel, output_freq, nghSize, fileName, alpha, s, k1, k2, slowdown):
    #X is a list of tuples as the features [longitude, latidude]
    #can have more, but only first two are used for clustering

    grain = 10


    max_long, max_latd, min_long, min_latd = -100000,-10000,10000,10000
    for location in X:
        if (float(location[0])> 0):
            location[0] = -360 + float(location[0])
        max_long = max(float(location[0]), max_long)
        max_latd = max(float(location[1]), max_latd)
        min_long = min(float(location[0]), min_long)
        min_latd = min(float(location[1]), min_latd)
    max_long = int((max_long)*grain)
    max_latd = int((max_latd)*grain)
    min_long = int((min_long)*grain)
    min_latd = int((min_latd)*grain)


    #max_long, min_long, max_latd, min_latd = int(-107.18*grain), int(-191.657*grain), int(71.626*grain), int(55.037*grain)
    print(max_long, min_long, max_latd, min_latd)

    long_range = max_long - min_long+1
    latd_range = max_latd - min_latd+1

    datapts = []
    ants = []
    grid = [[None for i in range(latd_range)] for j in range(long_range)]

    for location in X:
        if (float(location[0])> 0):
            location[0] = -360 + float(location[0])
        location[0] = (int((float(location[0]))*grain)-min_long)/long_range
        location[1] = (int((float(location[1]))*grain)-min_latd)/latd_range
        datapts.append(Datapt(location[0], location[1]))

    #curr scatter datapts, can try placing as are to start
    for pt in datapts:
        long = rd.randint(0, long_range-1)
        latd = rd.randint(0, latd_range-1)
        grid[int(pt.x)][int(pt.y)] = pt
        #print(pt.x, pt.y, pt)
        grid[long][latd] = pt
    for i in range(num_ants):
        long = rd.randint(0, long_range-1)
        latd = rd.randint(0, latd_range-1)
        ants.append(Ant(long,latd))

    file_index=0


    for i in range(0, iterations):
        ant = ants[rd.randint(0,len(ants)-1)]
        dyn_nghSize = int(nghSize*slowdown/(slowdown+math.log(i+1)))
        dyn_vel = ant.vel_modifier*vel*slowdown/(slowdown+math.log(i+1))

        #ant mvmt w/ wrap around boundaries
        ant.x = int((ant.x+int(rd.choice([1,-1])*rd.random()*dyn_vel)) % long_range)
        ant.y = int((ant.y + int(rd.choice([1,-1])*rd.random()*dyn_vel)) % latd_range)

        dpt = grid[ant.x][ant.y]

        if (ant.data == None and dpt != None):
            if (rd.random() <= P_pickup(dpt,grid,ant,dyn_nghSize, long_range, latd_range, alpha, s, k1) ):
                #(datapoint, grid, ant, nghSize)
                ant.data = dpt
                grid[ant.x][ant.y] = None

        elif (ant.data != None and dpt == None):
            if (rd.random() <= P_drop(ant.data,grid,ant,dyn_nghSize,long_range, latd_range, alpha, s, k2) ):
                grid[ant.x][ant.y] = ant.data
                ant.data = None

        if (i % output_freq == 0):
            #print(max_long, max_latd, min_long, min_latd)
            file = fileName + "000_" + str(file_index)
            plot_clusterer.grid_plot(grid, file, max_long, max_latd)
            #print("Finished iteration " + str(i))
            file_index += 1


def P_drop(datapoint, grid, ant, nghSize, long_range, latd_range, alpha, s, k2):
    #where nghSize is direction in both directions
    dist,numNgh, sum =0,0,0
    expo = 1
    for i in range(-nghSize,nghSize):
        for j in range(-nghSize,nghSize):
            #boundary check:
            if (ant.x+i+1 < len(grid) and ant.x+i > 0):
                if (ant.y + j + 1 < len(grid[ant.x + i]) and ant.y + j > 0):
                    if (grid[ant.x+i][ant.y+j] != None):
                        x_dist = math.pow(abs(math.pow(grid[ant.x+i][ant.y+j].x,2)-math.pow(datapoint.x,2)),.5) /math.pow(long_range, expo)
                        y_dist = math.pow(abs(math.pow(grid[ant.x+i][ant.y+j].y,2)-math.pow(datapoint.y,2)),.5) /math.pow(latd_range, expo)
                        dist += x_dist + y_dist
                        sum += 1-dist/alpha
                        numNgh += 1
    #print(sum, numNgh, ant.x, ant.y, nghSize)
    if (numNgh != 0):     funct = max(0, sum / numNgh)
    else:       funct = max(0, sum)
    #if (dist > 1 or dist < 0): print ("ERR in dist = " + str(dist))
    if (funct<k2): score = 2*funct
    else: score = 1
    #print("chance drop = " + str(score))

    return score

def P_pickup(datapoint, grid, ant, nghSize, long_range, latd_range, alpha, s, k1):
    #where nghSize is direction in both directions
    expo = 1
    dist,numNgh,sum=0,0,0
    for i in range(-nghSize,nghSize):
        for j in range(-nghSize,nghSize):
            #boundary check:
            if (ant.x+i+1 < len(grid) and ant.x+i > 0):
                if (ant.y + j + 1 < len(grid[ant.x + i]) and ant.y + j > 0):
                    if (grid[ant.x+i][ant.y+j] != None):
                        x_dist = math.pow(abs(math.pow(grid[ant.x + i][ant.y + j].x, 2) - math.pow(datapoint.x, 2)),.5)  /math.pow(long_range, expo)
                        y_dist = math.pow(abs(math.pow(grid[ant.x + i][ant.y + j].y, 2) - math.pow(datapoint.y, 2)),.5)  /math.pow(latd_range, expo)
                        dist += x_dist + y_dist
                        sum += 1 - dist / alpha
                        numNgh += 1

    #print(sum, numNgh)
    if (numNgh != 0):     funct = max(0, sum / numNgh)
    else:       funct = max(0, sum)
    #if (dist > 1 or dist < 0): print ("ERR in dist = " + str(dist))

    score = k1/(k1+funct)
    #print("\t\t\tchance pickup = " + str(score))
    return score


if __name__ == "__main__":

    #X = [[0,1],[1,0],[0,0],[1,1]]
    numAnts = 100
    vels = [1000,100,10] #starting vec, dyn slows
    nghSizes = [50,100] #starting size, dyn shrinks
    iterations = 5000000
    data_cutoff = [1000, 10000,0]
    alpha, k1, k2 = .5, .1, .15 #recommended params
    #k1 is essentially pickup chance, k2 drop chance
    s = 0 #curr unused
    slowdowns = [5,10,15]
    cutoffs = [1000, 100000, 0]

    for l in range(len(cutoffs)):
        for j in range(len(vels)):
            for k in range(len(nghSizes)):
                for i in range(len(slowdowns)):
                    fileName = "_cutoff" + str(cutoffs[l]) + "_vel" + str(vels[j]) + "_nghSize" + str(nghSizes[k]) + "_slowdown" + str(slowdowns[i])
                    X = plot_clusterer.parse(cutoffs[l])
                    #print("Finished parsing.")
                    cluster_ant_colonies(X, numAnts, iterations, vels[j], int(iterations/5), nghSizes[k], fileName, alpha, s, k1, k2, slowdowns[i])
                    #(X, num_ants, iterations, vel, output_freq, nghSize)
                    print("Finished ant clustering with param #s " + str(l) + ", " + str(j) + ", " + str(k) + "," + str(i))
    print("Done.")