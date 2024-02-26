#!/usr/bin/env python3
u"""
index_tiles.py
Written by: Enrico Ciraci' - February 2024

Identify IRIDE S3-01-SNT-03 Tiles covering the selected area of interest.
Save the generated index to an ESRI shapefile.

usage: index_bursts.py [-h] [-T TILE_DIR] burst_file aoi

Identify bursts of the same satellite track covering the selected area
of interest.

positional arguments:
  tile_file            Path to the Sentinel-1 burst index shapefile.
  aoi                   Path to the area of interest shapefile.

options:
  -h, --help            show this help message and exit
  -T TILE_DIR, --tile_dir TILE_DIR
                        Directory containing the IRIDE S3-01-SNT-03 Tiles

Python Dependencies:
- geopandas: Python tools for working with geospatial data in python.
    https://geopandas.org

"""
# - Python modules
import os
import argparse
import re
from datetime import datetime
from pathlib import Path
# - External modules
import numpy as np
import geopandas as gpd
from xml_utils import extract_xml_from_zip


def main() -> None:
    # - Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Identify S3-01-SNT-03 Tiles covering the selected "
                    "area of interest."
    )
    # - burst_file: Sentinel-1 burst index file file
    parser.add_argument('tile_file', type=str,
                        help='Path to the S3-01-SNT-03 Tiles index shapefile.')
    # - aoi: area of interest shapefile
    parser.add_argument('aoi', type=str,
                        help='Path to the area of interest shapefile.')
    # - burst directory: directory containing the burst files
    parser.add_argument('-T', '--tile_dir', type=str,
                        default=os.getcwd(), help='Directory containing the '
                                                  'Tiles files.')

    args = parser.parse_args()

    # - Verify if reference file exists
    if not os.path.exists(args.aoi):
        raise FileNotFoundError(f"# - AOI File {args.aoi} does not exist.")
    if not os.path.isdir(args.tile_dir):
        raise ValueError(f"# - Data directory: {args.tile_dir} not found.")

    # - Read the input file with geopandas
    tile_gdf = gpd.read_file(args.tile_file).to_crs("EPSG:4326")

    # - Import aoi boundaries shapefile with geopandas
    aoi_file = args.aoi
    aoi_gdf = gpd.read_file(aoi_file).to_crs("EPSG:4326")

    # - Find the intersection between the aoi and the bursts shapefiles
    print("# - Looking for Tiles covering the selected area of interest.")
    aoi_tiles = gpd.sjoin(tile_gdf, aoi_gdf,
                          predicate="intersects", how="inner")
    aoi_tiles = aoi_tiles[tile_gdf.columns]

    # - Loop though the generated dataframe and verify if the file relative
    # - to the tile exists in the tile directory.
    tile_dir_content = os.listdir(args.tile_dir)
    path_to_tiles = []
    start_date = []
    end_date = []

    if len(aoi_tiles) == 0:
        raise ValueError("# - No tiles found covering "
                         "the selected area of interest.")
    aoi_tiles = aoi_tiles.reset_index(drop=True).to_crs("EPSG:3035")

    for index, row in aoi_tiles.iterrows():
        # - Extract tile identifier
        x_c, y_c = row['geometry'].exterior.coords[0]
        xc_str = str(int(np.floor(x_c/1e5)))
        yc_str = str(int(np.ceil(y_c/1e5)))
        print(f"# - Looking for tile E{xc_str}N{yc_str}")
        re_pattern = re.compile(f"E{xc_str}N{yc_str}")
        found_tile = [tile for tile in tile_dir_content if
                      re.search(re_pattern, tile) and tile.endswith(".zip")]

        if len(found_tile) > 0:
            path_to_tiles.append(os.path.join(args.tile_dir,
                                              found_tile[0]))
            # - Extract xml metadata file from the zip file
            meta_dict \
                = extract_xml_from_zip(os.path.join(args.tile_dir,
                                                    found_tile[0]))[0]
            # -
            try:
                start_date.append(meta_dict['start_date'])
                end_date.append(meta_dict['end_date'])
            except KeyError:
                # - If the metadata file does not contain the start and end
                # - date, extract it from thr filename.
                # - NOTE - This should be a temporary solution.
                start_date.append(found_tile[0].split('_')[2])
                end_date.append(found_tile[0].split('_')[3])
        else:
            path_to_tiles.append('None')
            start_date.append('None')
            end_date.append('None')

    # - Add the path to the bursts to the dataframe
    aoi_tiles["Path"] = path_to_tiles
    aoi_tiles["start_date"] = start_date
    aoi_tiles["end_date"] = end_date
    # - Create directory to save the shapefile
    out_dir = Path(args.tile_dir).parent / Path('AOIs_tiles')
    os.makedirs(out_dir, exist_ok=True)

    # - Save the dataframe to a shapefile
    out_shp = out_dir / Path(aoi_file).name
    # -  if a column named 'overlap' exists, remove it
    if 'overlap' in aoi_tiles.columns:
        aoi_bursts = aoi_tiles.drop(columns=['overlap'])
    aoi_tiles.to_file(str(out_shp), driver="ESRI Shapefile")
    print(f"# - Shapefile saved to {out_shp}")


if __name__ == '__main__':
    main()
