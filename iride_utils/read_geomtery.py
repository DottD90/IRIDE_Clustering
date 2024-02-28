import geopandas as gpd

# - SHAPEFILE Case
# Load the shapefile
gdf = gpd.read_file('your_shapefile.shp')

# Convert the CRS to EPSG:4326
gdf = gdf.to_crs("EPSG:4326")

# Get the bounding box of the GeoDataFrame
bbox = gdf.total_bounds

# Print the bounding box
print(f"Bounding Box: {bbox}")

from shapely.geometry import Polygon

# Get the bounding box of the GeoDataFrame
bbox = gdf.total_bounds

# Create a Polygon from the bounding box
polygon = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]),
                   (bbox[2], bbox[3]), (bbox[2], bbox[1])])

# - CSV Case
import pandas as pd
import geopandas as gpd

# Load the CSV file
df = pd.read_csv('your_file.csv')

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude,
                                                       df.latitude))

# Set the CRS to EPSG:4326 (WGS84)
gdf.crs = "EPSG:4326"

# Print the GeoDataFrame
print(gdf.head())