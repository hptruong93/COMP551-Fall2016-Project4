# Import matplotlib and Basemap
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def _setup():
    # Create a figure of size (i.e. pretty big)
    fig = plt.figure(figsize=(20,10))

    # Create a map, using the Gall-Peters projection
    the_map = Basemap(projection='lcc',
                  # with low resolution,
                  resolution = 'l',
                  area_thresh = 100000.0, # And threshold 100000

                  # llcrnrlat=50, llcrnrlon=180,
                  # urcrnrlat=75, urcrnrlon=-180,
                  width = 4000000,
                  height = 5000000,

                  # lat_0=0, lon_0=0) # Centered at 0,0 (i.e null island)
                  lat_0=64.2008, lon_0=-149.4937) # Alaska

    # Draw the coastlines on the map
    the_map.drawcoastlines(linewidth=2)

    # Draw country borders on the map
    the_map.drawcountries(linewidth=2)

    # Fill the land with grey
    the_map.fillcontinents(color = '#888888')

    # Draw the map boundaries
    the_map.drawmapboundary(fill_color='#f4f4f4')

    # draw parallels.
    parallels = np.arange(0.,90,2.)
    the_map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    # draw meridians
    meridians = np.arange(180.,360.,10.)
    the_map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

    return the_map

def _finish(title = 'Title', save = False):
    plt.title(title)

    if save:
        plt.savefig('map.png')
    else:
        # Show the map
        plt.show()

def plot(lats, longs, z = None, discrete_z = False, save = False, title = 'Eagle\'s positions over the year', z_title = 'Date of year'):
    """
        Example from http://chrisalbon.com/python/matplotlib_plot_points_on_map.html
    """
    the_map = _setup()

    # Define our longitude and latitude points
    x, y = the_map(longs, lats)

    # Plot them using round markers of size 6
    if z is None:
        the_map.plot(x, y, 'ro', markersize = 6)
    else:
        scatter = the_map.scatter(x, y, c = z)

        if discrete_z:
            # define the colormap
            cmap = plt.cm.bwr
            # extract all colors from the color map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry to be grey
            cmaplist[0] = (.5,.5,.5,1.0)
            # create the new map
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            z_count = len(set(z))
            bounds = np.linspace(0,z_count,z_count + 1)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

            color_bar = the_map.colorbar(scatter, location = 'bottom', pad = '5%', cmap = cmap, spacing = 'proportional', ticks = bounds, boundaries = bounds, format = '%1i')
        else:
            color_bar = the_map.colorbar(scatter, location = 'bottom', pad = '5%', cmap = plt.cm.bwr)

        color_bar.set_label(z_title)

    _finish(title, save)

def plot_paired_paths(lats_1, longs_1, lats_2, longs_2, title = 'Path', save = False):
    the_map = _setup()
    x1, y1 = the_map(longs_1, lats_1)
    x2, y2 = the_map(longs_2, lats_2)

    for i, x in enumerate(x1):
        xs = [x, x2[i]]
        ys = [y1[i], y2[i]]

        the_map.plot(xs, ys, 'o-', markersize = 6)

    _finish(title, save)


def plot_path(lats, longs, title = 'Path', save = False):
    the_map = _setup()

    # Define our longitude and latitude points
    x, y = the_map(longs, lats)
    the_map.plot(x, y, 'o-', markersize = 6)

    _finish(title, save)


if __name__ == "__main__":
    # lats = [27.173968, 27.164328, 20.930555, 31.784217, 29.935895]
    # longs = [78.037519, 78.015289, 49.948611, 35.134277, 29.935895]

    # plot(lats, longs)

    longs = [-148.305, -148.278, -148.409, -148.375, -148.251, -148.307, -148.326, -148.174, -148.296, -148.251, -148.307, -148.288, -147.86, -147.925, -147.92]
    lats = [70.434, 70.433, 70.426, 70.473, 70.466, 70.433, 70.442, 70.45, 70.459, 70.421, 70.399, 70.46, 70.476, 70.448, 70.465]

    zs = [1 for i in xrange(len(lats))]
    plot_path(lats, longs)