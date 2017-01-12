# Shopify Cloropleth Map
Simple cloropleth map rendering using Matplotlib and Basemap.
## Installation
I will assume you already have pip & python 2.7 installed
- make a project directory wherever you would like
- Make a virtualenv (recomended but not required), and install requirements with pip
```
mkvirutalenv shopify
pip install matplotlib
pip install fiona
pip install shapely
pip install descartes
pip install pysal
pip install lxml
pip install pandas
```
- Installing basemap (not so fun part)
```
wget -O basemap.tar.gz http://downloads.sourceforge.net/project/matplotlib/matplotlib-toolkits/basemap-1.0.7/basemap-1.0.7.tar.gz?r=&ts=1484241851&use_mirror=superb-dca2
```
- Extract files
```
tar -xvzf basemap.tar.gz
```
- Install Geod 3.3.3 and install
```
cd basemap-1.0.7/
cd geos-3.3.3/
# Recomend home directory to avoid permissions issues in /usr/local
export GEOS_DIR=<wherever you want the libs to go>
./configure --prefix=$GEOS_DIR
make; make install
cd ..
python setup.py install
```
- Test that basemap is installed by running python or ipython and importing
```
>>> mpl_toolkits.basemap import Basemap
```
- Clean up 
```
rm basemap.tar.gz
sudo rm -R basemap-1.0.7/
```
Choropleth program should now be ready to go!
## Usage
Running the python file will create a semi interactive cloropleth as well as save a png of the map to that data/ dir
```
python shopifySales.py
```
## TODO
- Calculate translation on Map data and sales data localized in Alaska or Hawaii to move both the rendered states and associated sales points below the main rendering to represent all data within the frame
- Modify Density calculations and labels to display based on dollar values instead of # of orders
- Add mouse over info per state (eg: Sales in $ value)


![Alt text](/data/shopifySales.png?raw=true)