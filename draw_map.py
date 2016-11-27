# Import matplotlib and Basemap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot(lats, longs):
    """
        Example from http://chrisalbon.com/python/matplotlib_plot_points_on_map.html
    """

    # Create a figure of size (i.e. pretty big)
    fig = plt.figure(figsize=(20,10))

    # Create a map, using the Gall-Peters projection
    the_map = Basemap(projection='gall',
                  # with low resolution,
                  resolution = 'l',
                  area_thresh = 100000.0, # And threshold 100000

                  # llcrnrlat=50, llcrnrlon=165,
                  # urcrnrlat=75, urcrnrlon=-135,

                  lat_0=0, lon_0=0) # Centered at 0,0 (i.e null island)
                  # lat_0=64.2008, lon_0=-149.4937) # Alaska

    # Draw the coastlines on the map
    the_map.drawcoastlines()

    # Draw country borders on the map
    the_map.drawcountries()

    # Fill the land with grey
    the_map.fillcontinents(color = '#888888')

    # Draw the map boundaries
    the_map.drawmapboundary(fill_color='#f4f4f4')

    # Define our longitude and latitude points
    x, y = the_map(longs, lats)

    # Plot them using round markers of size 6
    the_map.plot(x, y, 'ro', markersize=6)

    # Show the map
    plt.show()

if __name__ == "__main__":
    lats = [27.173968, 27.164328, 20.930555, 31.784217, 29.935895]
    longs = [78.037519, 78.015289, 49.948611, 35.134277, 29.935895]

    plot(lats, longs)