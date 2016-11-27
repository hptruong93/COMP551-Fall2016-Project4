# Import matplotlib and Basemap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def plot(lats, longs, z = None, save = False, title = 'Eagle\'s positions over the year'):
    """
        Example from http://chrisalbon.com/python/matplotlib_plot_points_on_map.html
    """

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
    the_map.drawcoastlines()

    # Draw country borders on the map
    the_map.drawcountries()

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

    # Define our longitude and latitude points
    x, y = the_map(longs, lats)

    # Plot them using round markers of size 6
    if z is None:
        the_map.plot(x, y, 'ro', markersize=6)
    else:
        scatter = the_map.scatter(x, y, c = z)
        color_bar = the_map.colorbar(scatter, location = 'bottom', pad = '5%')
        color_bar.set_label('Date of year')

    plt.title(title)

    if save:
        plt.savefig('map.png')
    else:
        # Show the map
        plt.show()

if __name__ == "__main__":
    lats = [27.173968, 27.164328, 20.930555, 31.784217, 29.935895]
    longs = [78.037519, 78.015289, 49.948611, 35.134277, 29.935895]

    plot(lats, longs)