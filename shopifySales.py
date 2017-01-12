################################################################################
# Shopify Data Choropleth Using Matplotlib/Basemap
# Adapted from Choropleth tutorial 
#
# Gavin Summers , 2017
#
###############################################################################
#Lots of imports
import json
from pprint import pprint
from lxml import etree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from pysal.esda.mapclassify import Natural_Breaks as nb
from descartes import PolygonPatch
import fiona
from itertools import chain

#Open Json Data file for reading
def openFile():
    #Open json data file to read from
    with open('shopify.json') as data_file:
        file = json.load(data_file)
        
    return file

#Calculate total revenue from file
#As per shopify API: 
#total_price: The sum of all the prices of all the items in the order,
# taxes and discounts included (must be positive).
def getRevenue(data):
    x = 0
    revenue = 0
    for key in data["orders"]:
        orderRev =  float(data["orders"][x]["total_price"])
        revenue = revenue + orderRev
        x = x + 1

    return revenue

#Create and populate dict, pull relevant fields from the json file
def makeDict(data):
    y = 0
    #Create Dict to grab relevant fields from shopify json
    output = dict()
    output['id'] = []
    output['state'] = []
    output['dollarValue'] = []
    output['lon'] = []
    output['lat'] = []

    #Grab said fields from shopify.json and populate dict
    for key in data["orders"]:
        output['id'].append(data["orders"][y]["id"])
        output['state'].append((data["orders"][y]["billing_address"]["province"]))
        output["dollarValue"].append(float(data["orders"][y]["total_price"]))
        output["lon"].append((data["orders"][y]["billing_address"]["longitude"]))
        output["lat"].append((data["orders"][y]["billing_address"]["latitude"]))
        y+=1

    return output

#Create pandas dataframe & Ensure lat / lon co-ords are float type
def makeDataFrame(orderDict):
    #create pandas dataframe
    df = pd.DataFrame(orderDict)
    df = df.dropna()

    #Ensure latitude and longitude values are numerical
    df[['lon', 'lat']] = df[['lon', 'lat']].astype(float)

    return df

#Create Matplotlib Basemap using shapefile of US counties 
#Shapefile retrieved from census.gov
def makeBase():  
    #Create basemap 
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    m.readshapefile(
        'data/cb_2015_us_county_5m',
        'America',
        color='none',
        zorder=2)

    return m

#Create Pandas dataframe to house info and polygon data from shapefile
def makeMapFrame(m):
    df_map = pd.DataFrame({
        'poly': [Polygon(xy) for xy in m.America],
        'statefp': [state["STATEFP"] for state in m.America_info],
        'state_name': [state['NAME'] for state in m.America_info]})
    df_map['area_m'] = df_map['poly'].map(lambda x: x.area)
    df_map['area_km'] = df_map['area_m'] / 100000

    return df_map

# Convenience functions for working with colour ramps and bars
#Takes a standard colour ramp, and discretizes it,
#then draws a colour bar with correctly aligned labels
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

#Returns a discrete colormap from continuous colormap cmap
def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

#Calculate and set sales densities per county based on sale instances per county area
def setDensity():
    global df_map, breaks
    df_map['count'] = df_map['poly'].map(lambda x: int(len(filter(prep(x).contains, ldn_points))))
    df_map['density_m'] = df_map['count'] / df_map['area_m']
    df_map['density_km'] = df_map['count'] / df_map['area_km']
    df_map.replace(to_replace={'density_m': {0: np.nan}, 'density_km': {0: np.nan}}, inplace=True)

    # Calculate Jenks natural breaks for density
    breaks = nb(
        df_map[df_map['density_km'].notnull()].density_km.values,
        initial=300,
        k=5)

    # the notnull method lets us match indices when joining
    jb = pd.DataFrame({'jenks_bins': breaks.yb}, index=df_map[df_map['density_km'].notnull()].index)
    df_map = df_map.join(jb)
    df_map.jenks_bins.fillna(-1, inplace=True)

#Create Densities Bar Label based on jenkins breaks to chunk sales density levels
def makeDensityLabel(breaks):
    #Make labels for bar chart label
    jenks_labels = ["<= %0.5f/km$^2$(%s counties)" % (b, c) for b, c in zip(
        breaks.bins, breaks.counts)]
    jenks_labels.insert(0, 'No orders (%s counties)' % len(df_map[df_map['density_km'].isnull()]))

    return jenks_labels

#Calculate buyer location points in reference to location boarders
#This should be a function but it has to return or modify globals
#map_points, buyer_points, states_polygon, ldn_points
def mapBuyerLocations():
    global map_points, buyer_points, states_polygon, ldn_points
    #pandas.core.series.Series
    map_points = pd.Series([Point(base(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(df['lon'], df['lat'])])
    #shapely.geometry.multipoint.MultiPoint
    buyer_points = MultiPoint(list(map_points.values))
    #shapely.prepared.PreparedGeometry
    states_polygon = prep(MultiPolygon(list(df_map['poly'].values)))
    #list
    ldn_points = filter(states_polygon.contains, buyer_points)

#Render, and plot map / density colours + render additional labels and features
def renderMap():
    #Start ploting map
    plt.clf()
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    # use a blue colour ramp - we'll be converting it to a map using cmap()
    # draw wards with grey outlines
    cmap = plt.get_cmap('Blues')
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#787878', fc='#555555', lw=.2, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)

    # impose our colour map onto the patch collection
    norm = Normalize()
    pc.set_facecolor(cmap(norm(df_map['jenks_bins'].values)))
    ax.add_collection(pc)

    # Add a colour bar
    cb = colorbar_index(ncolors=len(label), cmap=cmap, shrink=0.7, labels=label)
    cb.ax.tick_params(labelsize=10)

    #Label for revenue
    revenueLabel = 'Total Revenue of Orders:\n\n' + "$" +str(getRevenue(data))

    #Draw revenue label
    revenueInfo = cb.ax.text(
        -1., 0 - 0.007,
        revenueLabel,
        ha='right', va='bottom',
        size=13,
        color='#555555')

    # Draw a map scale meter
    base.drawmapscale(
        (-110),  25,
        -64, 49,
        1000.,
        barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555',
        fontcolor='#555555',
        zorder=5)

    # this will set the image width to 722px at 100dpi
    plt.title("Shopify Sales Density (orders per km squared by county)")
    plt.tight_layout()
    plt.savefig('data/shopifySales.png', dpi=100, alpha=True)
    plt.show()

#Initialize global objects
data = openFile()
orderDict = makeDict(data)
base = makeBase()
df = makeDataFrame(orderDict)
df_map = makeMapFrame(base)
breaks = nb
#dummy polygon to feed to states_polygon
polygon = Point(0,0)
map_points = pd.Series()
buyer_points = MultiPoint()
states_polygon = prep(polygon)
ldn_points = list()

#setup mapping and features
mapBuyerLocations()
setDensity()
label = makeDensityLabel(breaks)

#choropleth'd!
renderMap()