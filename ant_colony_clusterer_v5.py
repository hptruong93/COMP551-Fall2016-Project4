#!/usr/bin/python3

import random as rd
import multiprocessing as mp
import plot_clusterer, math ,csv 
import metadata, datetime
import numpy as np
import queue
import geo
import AC_data_reader
#from sklearn.model_selection import cross_val_score


class Datapt:
    def __init__(self, x, y):
        #where x is an array of features and y is the soln
        self.x = x
        self.y = y
        self.cluster = None

class Ant:
    def __init__(self, x, y):
        #where x and y are coordinates on the grid
        self.x = x
        self.y = y
        self.data = None
        self.vel_modifier = rd.random()*10
        self.behavior = 0
        #0 - construct clusters, 1 - destroy clusters
        self.velocity = 0

        self.STM = queue.Queue(maxsize=8)

    def drop(self, grid):
        if (self.data == None):
            print("ERROR: ant has no data to drop.")
            return
        if (grid[self.x][self.y] != None):
            print("ERROR: grid does not have space to drop.")
            return
        grid[self.x][self.y] = self.data
        self.data = None




def assign_cluster(Y, grid, datapoints, nghSize, outdir, fileName):
    num_clusters = 0
    cluster_y = []

    sizes = []
    scores = []
    overlaps = []
    gridSize=len(grid)
    for x in range(gridSize):
        for y in range(gridSize):
            if (grid[x][y] != None):
                dpt = grid[x][y]
                if (dpt.cluster == None):
                    y_choices = []
                    y_votes = []

                    y_choices.append(dpt.y)
                    y_votes.append(1)

                    dpt.cluster = num_clusters
                    dist,numNgh,summ=0,0,0
                    for i in range(-nghSize,nghSize):
                        for j in range(-nghSize,nghSize):
                            nghx = (x+i)%gridSize
                            nghy = (y+j)%gridSize
                            ngh_pt = grid[nghx][nghy]
                            if (ngh_pt != None):
                                dist = dist_btwn_pts(dpt, ngh_pt)
                                summ += 1-dist/alpha
                                numNgh += 1
                                dpt.cluster = num_clusters
                                #if (grid[nghx][nghy].cluster != None):   LATER HANDLE TO COUNT OVERLAPS
                                #print("WARNING: multiple cluster IDs possible.")
                                ngh_pt.cluster = num_clusters
                                if (ngh_pt.y != None):
                                    if ngh_pt.y not in y_choices:
                                        y_choices.append(ngh_pt.y)
                                        y_votes.append(1)
                                    else:
                                        idx = y_choices.index(ngh_pt.y)
                                        y_votes[idx] += 1

                    if (numNgh != 0):     funct = max(0, summ / math.pow((2*nghSize),2))
                    else:       funct = max(0, summ)
                    sizes.append(numNgh)
                    scores.append(funct)

                    top_choice = 0
                    top_votes = 0
                    for i in range(len(y_choices)):
                        if (y_votes[i] > top_votes):
                            top_votes = y_votes[i]
                            top_choice = y_choices[i]
                    cluster_y.append(top_choice)
                    num_clusters += 1

    err = 0
    num_unknown = 0
    for d in range(len(datapoints)):
        dpt = datapoints[d]
        if (dpt.y == None):
            dpt.y = cluster_y[dpt.cluster]
            if (dpt.y != Y[d]): err += 1
            num_unknown +=1
    err /= num_unknown


    ''' AS preprocessing
    dpts_out = outdir + "/" + fileName + "_cluster_IDs.csv"
    cluster_info = outdir + "/" + fileName + "_cluster_info.csv"
    with open(dpts_out, 'w') as csv_out:
        csv_out.write("Datapoint #, Cluster #\n")
        for d in range(len(datapoints)):
            csv_out.write(str(datapoints[d].cluster) + "\n")
    with open(cluster_info, 'w') as info_out:
        info_out.write("Avg Dist, # Clusters\n")
        if (num_clusters != 0): average = sum(scores)/num_clusters
        else: average = sum(scores)
        info_out.write(str(average) + "," + str(num_clusters) + "\n")
    '''

    return err


def batch(X, num_runs):
    num_ants = 1000
    vel = 100 
    ngh_size = 50
    iters = 10000000 
    slowdown = 100
    cutoff = 0 
    alpha, k1, k2 = 1000, .1, .15
    num_threads = mp.cpu_count()
    pool = mp.Pool(num_threads)

    worker_args = []

    for i in range(num_runs):
        fileName = "run_" + str(i)
        for n in range(num_threads):
            fileName += "_worker_" + str(n)
            args = [X, numAnts, iterations, vel, iterations, ngh_size, fileName, alpha, k1, k2, slowdown, cutoff]
            worker_args.append(args)

    pool.starmap(cluster_ant_colonies, worker_args)


def cluster_ant_colonies(x_train, y_train, x_test, Y, num_ants, iterations, vel, output_freq, nghSize, fileName, alpha, k1, k2, slowdown):

    #if (len(X) != len(Y)): print("ERROR in cluster_ant_colonies(): X and Y data are not same length")
    grain = 10

    datapts = []
    ants = []

    gridSize = int(math.sqrt((len(x_train)+len(x_test))*10))
    grid = [[None for i in range(gridSize)] for j in range(gridSize)]

    for i in range(len(x_train)):
        features = []
        for feature in x_train[i]:
            features.append(feature)

        datapts.append(Datapt(features, y_train[i]))

    for i in range(len(x_test)):
        features = []
        for feature in x_test[i]:
            features.append(feature)

        datapts.append(Datapt(features, None))

    #initialize datapoints and ants on the grid
    for pt in datapts:
        x = rd.randint(0,gridSize-1)
        y = rd.randint(0,gridSize-1)
        while (grid[x][y] != None):
            x = rd.randint(0,gridSize-1)
            y = rd.randint(0,gridSize-1)
        grid[x][y] = pt

    for i in range(num_ants):
        x = rd.randint(0,gridSize-1)
        y = rd.randint(0,gridSize-1)
        ants.append(Ant(x,y))

    file_index=0


    for i in range(0, iterations):
        ant = ants[rd.randint(0,len(ants)-1)]
        #dyn_nghSize = int(nghSize*slowdown/(slowdown+math.log(i+1)))
        dyn_nghSize=nghSize
        dyn_vel = ant.vel_modifier*vel*slowdown/(slowdown+math.log(i+1))
        ant.velocity = dyn_vel

        #ant mvmt w/ wrap around boundaries

        x = int(rd.choice([1,-1])*rd.random())
        y = int(rd.choice([1,-1])*rd.random())
        if (not(ant.STM.empty()) and ant.data!=None):
            minDist = 10000000
            x,y = 0,0
            '''
            for i in range(8):
                past_pt = ant.STM.get()
                ant.STM.put(past_pt)
                dist = dist_btwn_pts(past_pt[0], ant.data)
                if (dist < minDist):
                    minDist = dist
                    x = past_pt[1]-ant.x
                    y = past_pt[2]-ant.y
            '''
        ant.x = int((ant.x+(x*dyn_vel)) % gridSize)
        ant.y = int((ant.y + (y*dyn_vel)) % gridSize)

        dpt = grid[ant.x][ant.y]

        if (ant.data == None and dpt != None):
            if (rd.random() <= P_pickup(dpt,grid,gridSize,ant,dyn_nghSize, alpha, k1,vel) ):
                ant.data = dpt
                if (ant.STM.full()): ant.STM.get(block=False)
                ant.STM.put([dpt,ant.x,ant.y])
                grid[ant.x][ant.y] = None

        elif (ant.data != None and dpt == None):
            if (rd.random() <= P_drop(ant.data,grid,gridSize,ant,dyn_nghSize, alpha, k2,vel) ):
                grid[ant.x][ant.y] = ant.data
                ant.data = None

        if (i % output_freq == 0):
            file = fileName + "_" + str(file_index)
            plot_clusterer.grid_plot(grid, file, gridSize, gridSize)
            file_index += 1

    err = assign_cluster(Y, grid, datapts, nghSize, "/home/2014/choppe1/Documents/COMP551/output/csv", fileName)
    return err


def dist_btwn_pts(dpt1, dpt2):
    dist = 0
    for i in range(len(dpt1.x)):
        dist += math.pow(abs(math.pow(float(dpt1.x[i]),2)-math.pow(float(dpt2.x[i]),2)),.5)
    return dist

def k_fold_validation(k, num_ants, iterations, vel, output_freq, nghSize, fileName, alpha, k1, k2, slowdown):
    
    X, Y, titles = AC_data_reader.load_and_preprocess()

    '''
    X, metadata, titles = AC_data_reader.load_aggregate_data()  
    X, titles = AC_data_reader.transform_data(X, metadata, titles)
    X = np.array(X)
    X = AC_data_reader.normalize(X, -1)
    X = 
    Y = AC_data_reader.load_y()
    '''
    X_t = np.transpose(X)
    Y_t = np.transpose(Y)
    err = []
    for i in range(k):
        cutoff = int((len(X))/k)
        x_train = np.transpose(np.vstack((X_t[:i*cutoff,],X_t[(i+1)*cutoff:,])))
        y_train = np.transpose(np.vstack((Y_t[:i*cutoff,],Y_t[(i+1)*cutoff:,])))
        print(np.shape(x_train), np.shape(y_train), i*cutoff, (i+1)*cutoff, np.shape(X_t), np.shape(X_t[:i*cutoff,]))

        x_test = X[i*cutoff:(i+1)*cutoff]
        y_test = Y[i*cutoff:(i+1)*cutoff]

        vald_err = cluster_ant_colonies(x_train, y_train, x_test, Y, num_ants, iterations, vel, output_freq, nghSize, fileName, alpha, k1, k2, slowdown)
        err.append(vald_err)
        
    total_err = sum(err)
    err.append(total_err)
    print(err)
    return err



def P_drop(datapoint, grid, gridSize, ant, nghSize, alpha, k2,vel):
    #where nghSize is direction in both directions
    dist,numNgh, summ =0,0,0
    for i in range(-nghSize,nghSize):
        for j in range(-nghSize,nghSize):
            nghx = (ant.x+i)%gridSize
            nghy = (ant.y+j)%gridSize
            if (grid[nghx][nghy] != None):
                dist = dist_btwn_pts(datapoint, grid[nghx][nghy])
                summ += 1-dist/((alpha+alpha*(ant.velocity-1)/(vel*10)))
                numNgh += 1
    funct = max(0, summ / math.pow((2*nghSize),2))
    if (funct<k2): score = 2*funct
    else: score = 1
    #print("chance drop = " + str(score))

    return score

def P_pickup(datapoint, grid, gridSize, ant, nghSize, alpha, k1,vel):
    #where nghSize is direction in both directions
    dist,numNgh,summ=0,0,0
    for i in range(-nghSize,nghSize):
        for j in range(-nghSize,nghSize):
            nghx = (ant.x+i)%gridSize
            nghy = (ant.y+j)%gridSize
            if (grid[nghx][nghy] != None):
                dist = dist_btwn_pts(datapoint, grid[nghx][nghy])
                summ += 1-dist/((alpha+alpha*(ant.velocity-1)/(vel*10)))
                numNgh += 1
    funct = max(0, summ / math.pow((2*nghSize),2))
    score = k1/(k1+funct)
    #print("\t\t\tchance pickup = " + str(score))
    return score



if __name__ == "__main__":

    #X = [[0,1],[1,0],[0,0],[1,1]]
    num_ants = 1000
    vel = 100 #starting vec, dyn slows
    nghSize = 5 #starting size, dyn shrinks
    iterations = 10
    output_freq = iterations/5
    alpha, k1, k2 = 100, .1, .15 #recommended params
    #k1 is essentially pickup chance, k2 drop chance
    s = 0 #curr unused
    slowdown = 100 #1,5,10

    fileName = "predict"

    k_fold_validation(3, num_ants, iterations, vel, output_freq, nghSize, fileName, alpha, k1, k2, slowdown)

    ''' #PARALLEL IMPLEMENTATION
    pool = mp.Pool(mp.cpu_count())
    argz = []

    for j in range(len(vels)):
        for k in range(len(nghSizes)):
            for i in range(len(slowdowns)):
                for l in range(len(cutoffs)):
                    for m in range(len(k2s)):
                        fileName = "_vel" + str(vels[j]) + "_nghSize" + str(nghSizes[k]) + "_slowdown" + str(slowdowns[i]) + "_cutoffs " + str(cutoffs[l])
                        #paramSet = mp.Process(target=cluster_ant_colonies, args=(numAnts, iterations, vels[j], int(iterations/5), nghSizes[k], fileName, alpha, s, k1, k2, slowdowns[i], cutoffs[l]))
                        args = [numAnts, iterations, vels[j], int(iterations / 10), nghSizes[k], fileName, alpha, k1, k2, slowdowns[i], cutoffs[l]]
                        argz.append(args)
                        #cluster_ant_colonies(numAnts, iterations, vels[j], int(iterations / 5), nghSizes[k], fileName, alpha, s, k1, k2, slowdowns[i], cutoffs[l])
    pool.starmap(cluster_ant_colonies, argz)
    #cluster_ant_colonies(numAnts, iterations*10, 1000, int(iterations/10), 10, "allData", alpha, s, k1, k2, 5, 0)
    '''
    print("Done.")

