#!/usr/bin/env python3
u"""
index_bursts.py
Written by: Enrico Ciraci' - February 2024

Identify Sentinel-1 bursts covering the selected area of interest.
Save the generated index to an ESRI shapefile.

usage: index_bursts.py [-h] [-D BURST_DIR] burst_file aoi

Identify bursts of the same satellite track covering the selected area
of interest.

positional arguments:
  burst_file            Path to the Sentinel-1 burst index shapefile.
  aoi                   Path to the area of interest shapefile.

options:
  -h, --help            show this help message and exit
  -D BURST_DIR, --burst_dir BURST_DIR
                        Directory containing the Sentinel-1 bursts files

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
import geopandas as gpd
from xml_utils import extract_xml_from_zip


def main() -> None:
    # - Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Identify bursts of the same satellite "
                    "track covering the selected area of interest."
    )
    # - burst_file: Sentinel-1 burst index file file
    parser.add_argument('burst_file', type=str,
                        help='Path to the Sentinel-1 burst index shapefile.')
    # - aoi: area of interest shapefile
    parser.add_argument('aoi', type=str,
                        help='Path to the area of interest shapefile.')
    # - burst directory: directory containing the burst files
    parser.add_argument('-D', '--burst_dir', type=str,
                        default=os.getcwd(), help='Directory containing the '
                                                  'Sentinel-1 bursts files.')

    args = parser.parse_args()

    # - Verify if reference file exists
    if not os.path.exists(args.aoi):
        raise FileNotFoundError(f"# - AOI File {args.aoi} does not exist.")
    if not os.path.isdir(args.burst_dir):
        raise ValueError(f"# - Data directory: {args.burst_file} not found.")

    # - Read the input file with geopandas
    burst_gdf = gpd.read_file(args.burst_file).to_crs("EPSG:4326")

    # - Import aoi boundaries shapefile with geopandas
    aoi_file = args.aoi
    aoi_gdf = gpd.read_file(aoi_file).to_crs("EPSG:4326")

    # - Find the intersection between the aoi and the bursts shapefiles
    print("# - Looking for bursts covering the selected area of interest.")
    aoi_bursts = gpd.sjoin(burst_gdf, aoi_gdf,
                           predicate="intersects", how="inner")
    aoi_bursts = aoi_bursts[burst_gdf.columns]

    # - Loop though the generated dataframe and verify if the file relative
    # - to the burst exists in the burst directory.
    burst_dir_content = os.listdir(args.burst_dir)
    name_list = []              # - List of burst names
    track_list = []             # - List of track numbers
    burst_list = []             # - List of burst numbers
    subswath_list = []          # - List of subswath numbers
    geometry_list = []          # - List of geometries
    path_to_bursts = []         # - Path to the bursts
    start_date = []             # - Start date of the burst
    end_date = []               # - End date of the burst
    c_type = []                 # - Calibrated or not

    if len(aoi_bursts) == 0:
        raise ValueError("# - No bursts found covering "
                         "the selected area of interest.")

    for index, row in aoi_bursts.iterrows():
        track = row["Track"]
        burst = row["Burst"]
        subswath = row["Subswath"]
        re_pattern = re.compile(f"{track}.*{burst}{subswath}")
        found_bursts = [bst for bst in burst_dir_content if
                        re.search(re_pattern, bst) and bst.endswith(".zip")]

        if len(found_bursts) > 0:
            for bi in range(len(found_bursts)):
                # - Append the burst info to the lists
                name_list.append(row["Name"])
                track_list.append(row["Track"])
                burst_list.append(row["Burst"])
                subswath_list.append(row["Subswath"])
                geometry_list.append(row["geometry"])
                # - Add other info
                path_to_bursts.append(os.path.join(args.burst_dir,
                                                   found_bursts[bi]))
                # - Extract xml metadata file from the zip file
                meta_dict \
                    = extract_xml_from_zip(os.path.join(args.burst_dir,
                                                        found_bursts[bi]))[0]
                # -
                start_date.append(meta_dict['start_date'])
                end_date.append(meta_dict['end_date'])

                # - Valid only for TRE-A data
                # - Extract Input Product type from file name
                p_f_name = found_bursts[bi].split('_')
                if p_f_name[4].endswith('B'):
                    # - Basic Products
                    c_type.append('B')
                else:
                    # - Calibrated Products
                    c_type.append('C')
        else:
            # - Append the burst info to the lists
            name_list.append(row["Name"])
            track_list.append(row["Track"])
            burst_list.append(row["Burst"])
            subswath_list.append(row["Subswath"])
            geometry_list.append(row["geometry"])
            # - Set other info to None if the burst file is not found
            path_to_bursts.append('None')
            start_date.append('None')
            end_date.append('None')
            c_type.append('None')

    # - Create a new dataframe with the burst info
    aoi_bursts = gpd.GeoDataFrame({
        "Name": name_list,
        "Track": track_list,
        "Burst": burst_list,
        "Subswath": subswath_list,
        "c_type": c_type,
        "geometry": geometry_list,
        "Path": path_to_bursts,
        "start_date": start_date,
        "end_date": end_date,
    })

    # - set the crs of the dataframe
    aoi_bursts.crs = "EPSG:4326"

    # - Create directory to save the shapefile
    out_dir = Path(args.burst_dir).parent / Path('AOIs_bursts')
    os.makedirs(out_dir, exist_ok=True)

    # - Save the dataframe to a shapefile
    out_shp = out_dir / Path(aoi_file).name
    # -  if a column named 'overlap' exists, remove it
    if 'overlap' in aoi_bursts.columns:
        aoi_bursts = aoi_bursts.drop(columns=['overlap'])
    aoi_bursts.to_file(str(out_shp), driver="ESRI Shapefile")
    print(f"# - Shapefile saved to {out_shp}")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")
