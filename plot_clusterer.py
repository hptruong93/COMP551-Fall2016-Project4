import matplotlib
matplotlib.use('Agg') # This must be done before importing matplotlib.pyplot
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, random
from scipy.stats import itemfreq
import csv, datetime
import numpy as np

def parse(cutoff):
    #where pts are longitude, latitude, time
    eider_file_name = "/Users/Crbn/Desktop/McG Fall '16/Comp 551/ass4/input/Common Eider Petersen Alaska 2000-2009.csv"
    eider_file = open(eider_file_name, 'r')
    eider_file.readline()
    lines = eider_file.readlines()

    data = []
    if (cutoff==0): cutoff = len(lines)

    for line in lines[:cutoff]:
        parts = line.split(',')
        time_parts = parts[2].split('-')

        #time = int(time_parts[1])
        time = (int(time_parts[0]) - 2000)*12 + int(time_parts[1])      #num months since study started (2000-01)
        longitude = parts[3]
        latitude = parts[4]
        data.append([longitude, latitude, time])

    eider_file.close()

    return data


def xyz_plot(data):
    #assumes data is of triples
    output_dir =  "/Users/Crbn/Desktop/McG Fall '16/Comp 551/ass4/output"
    x=[]
    y=[]
    z=[]
    for d in range(len(data)):
        x_data = float(data[d][0])
        if (x_data > 0):
            break
            #x_data -= 400   #to put all points on same side
        x.append(x_data)
        y.append(data[d][1])
        z.append(data[d][2])

    colorplot = plt.scatter(x, y, c=z)
    plt.colorbar(colorplot)
    #plt.interactive(False)
    #plt.show()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Months (in color) of Bird Locations")
    destn = output_dir+"/time_scatter.png"
    plt.savefig(destn)


def grid_plot(grid, file_index, long_max, latd_max):
    output_file =  "/Users/Crbn/Desktop/McG Fall '16/Comp 551/ass4/output/ACO_plot_" + str(file_index) + ".png"
    x = []
    y = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (grid[i][j] != None):
                x.append(i)  #+int(long_min))
                y.append(j) #+int(latd_min))
    xyplot = plt.scatter(x, y)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Location of Data Points")

    max_long = 300 #TEMP FIX
    max_latd = 70
    xticks = []
    yticks = []
    for i in range(11):
        xticks.append(i*max_long/10)
        yticks.append(i*max_latd/10)

    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.savefig(output_file)
    plt.close()

'''

X = parse(1000)
xyz_plot(X)
'''