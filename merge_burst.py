#!/usr/bin/env python3
"""
Written by Enrico Ciraci' - February 2024
Merge burst data relative to a single AOI into a single file.

usage: merge_burst.py [-h] [-D OUT_DIR] index_file

Merge GSP Bursts belonging to track into a single Product.

positional arguments:
  index_file            Index file containing the list of bursts available
                        over the AOI.

options:
  -h, --help            show this help message and exit
  -D OUT_DIR, --out_dir OUT_DIR
                        Output directory where the results will be saved.

Python Dependencise:
    - pandas:Python Data Analysis Library
        https://pandas.pydata.org/
    - geopandas: Python tools for working with geospatial data in python
        https://geopandas.org/
    - numpy: The fundamental package for scientific computing with Python
        https://numpy.org/
    - zipfile: Work with zip archives
    - xml.etree.ElementTree: XML parsing and generation
    - xml.dom.minidom: Pretty print xml files
    - typing: Type hints
    - lxml: XML validation against a schema
    - pathlib: Work with file paths
"""
# - Python modules
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import zipfile
# - External modules
import geopandas as gpd
from xml_utils import extract_xml_from_zip
from iride_utils.aoi_info import get_aoi_info
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

from active_areas_clustering import read_as_geodataframe
from iride_utils.gsp_description import gsp_description, gsp_metadata
from iride_utils.add_meta_field import add_meta_field

# - Set logging level and message format
log_format = "%(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

SENSOR = 'SNT'    # - Sentinel-1


def main() -> None:
    # - Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Merge GSP Bursts belonging to track into "
                    "a single Product.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # - burst_file: Sentinel-1 burst index file file
    parser.add_argument('index_file', type=str,
                        help='Index file containing the list of'
                             'bursts available over the AOI.')

    # - Optional arguments
    # - Output directory
    parser.add_argument("-D", "--out_dir", type=str,
                        default=os.getcwd(),
                        help='Output directory where '
                             'the results will be saved.')
    args = parser.parse_args()

    # - Verify if reference file exists
    if not os.path.exists(args.index_file):
        logging.error(f"# - Index file {args.index_file} does not exist.")
        return

    # - Create the output directory if it does not exist yet
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        logging.info(f"# - Created output directory {args.out_dir}")

    # - Load the index file
    logging.info(f"# - Loading index file {args.index_file}")
    gdf = gpd.read_file(args.index_file)

    # - Extract AOI info
    mask_name = args.index_file.split('\\')[-1].replace('.shp', '')
    aoi_info = get_aoi_info(mask_name)
    # - Update Output Directory
    out_dir = os.path.join(args.out_dir, str(aoi_info['aoi_tag']))
    os.makedirs(out_dir, exist_ok=True)

    # - Get the list of tracks
    tracks = gdf['Track'].unique()

    # - Loop over the tracks
    for track in tracks:
        # - Filter the GeoDataFrame by track
        track_gdf = gdf[gdf['Track'] == track]
        logging.info(f"# - Processing track {track} with "
                     f"{len(track_gdf)} bursts")

        # - Select all the lines with Path value different from None
        track_gdf = track_gdf[track_gdf['Path'] != 'None']
        # - verify if there are bursts to merge
        if len(track_gdf) == 0:
            logging.warning(f"# - No bursts found for track {track}")
            continue

        # - Load the first burst
        fb_path = Path(track_gdf.iloc[0]['Path'])
        gdf_brst = read_as_geodataframe(fb_path)

        # - Extract Metadata from the first burst archive
        xml_dicts_ref = extract_xml_from_zip(str(fb_path))[0]
        # - Extract fields from the metadata
        in_prod_id = xml_dicts_ref['product_id']
        in_prod_id_ns = xml_dicts_ref['product_id'].replace('-', '')
        in_start_date = xml_dicts_ref['start_date'].replace('-', '')
        in_end_date = xml_dicts_ref['end_date'].replace('-', '')
        provider_ref = xml_dicts_ref['provider']

        # - Loop over the remaining bursts and merge them
        for idx in range(1, len(track_gdf)):
            brst_path = Path(track_gdf.iloc[idx]['Path'])
            gdf_brst = gdf_brst._append(read_as_geodataframe(brst_path))

        # - Remove duplicates based on Longitude and Latitude coordinates
        gdf_brst = gdf_brst.drop_duplicates(subset=['latitude', 'longitude'])
        print(f"# - Number of bursts merged: {len(gdf_brst)}")
        print(f"# - Number of unique pid: {len(gdf_brst['pid'].unique())}")

        # - Output File Name
        out_f_name = (f"ISS_{in_prod_id_ns}_{in_start_date}"
                      f"_{in_end_date}_{track}"
                      f"{aoi_info['aoi_tag']}F_01")
        out_file = os.path.join(out_dir, out_f_name + '.csv')
        logging.info(f"# - Saving merged file {out_file}")
        gdf_brst.to_csv(out_file, index=False)

        # - Covert the Detaframe Geometry to EPSG:4326 (WGS84)
        gdf_brst = gdf_brst.to_crs("EPSG:4326")

        # - Generate the metadata file
        metadata_f_name = (f"ISS_{in_prod_id_ns}_{in_start_date}"
                           f"_{in_end_date}_{track}"
                           f"{aoi_info['aoi_tag']}F_01.xml")
        metadata_file = os.path.join(out_dir, metadata_f_name)
        logging.info(f"# - Generating metadata file {metadata_file}")

        # - Extract Bounding Box and Perimeter
        x_min, y_min, x_max, y_max = gdf_brst.total_bounds
        # - extract dataset envelope polygon
        envelope \
            = gpd.GeoSeries([gdf_brst.unary_union.envelope],
                            crs=gdf_brst.crs).iloc[0].exterior.coords.xy
        xs = envelope[0]
        ys = envelope[1]
        crd = list(zip(xs, ys))

        env_geometry = {
            "type": "Polygon",
            "coordinates": [crd],
        }

        # - Add XML File
        # - Create the root element
        root = ET.Element('GSP')

        # - Add child elements to the root
        gsp_id = ET.SubElement(root, 'gsp_id')
        gsp_id.text = in_prod_id
        product_id = ET.SubElement(root, 'product_id')
        product_id.text = out_f_name.replace('.csv', '')
        description = ET.SubElement(root, 'description')
        description.text = gsp_description(in_prod_id)

        sensor = ET.SubElement(root, 'sensor_id')
        sensor.text = SENSOR

        track_id = ET.SubElement(root, 'track_id')
        track_id.text = track

        provider = ET.SubElement(root, 'provider')
        provider.text = provider_ref

        # - Add Processing, Start, and End Date
        production_date = ET.SubElement(root, 'production_date')
        production_date.text \
            = xml_dicts_ref['production_date'].replace('-', '')

        start_date = ET.SubElement(root, 'start_date')
        start_date.text \
            = xml_dicts_ref['start_date'].replace('-', '')

        end_date = ET.SubElement(root, 'end_date')
        end_date.text \
            = xml_dicts_ref['end_date'].replace('-', '')

        # - Add AOI
        aoi = ET.SubElement(root, 'aoi')
        aoi.text = aoi_info['aoi_tag']

        # - Add Bounding Box and Perimeter
        bbox = ET.SubElement(root, 'bbox')
        bbox.text = f"{x_min} {y_min} {x_max} {y_max}"
        geometry = ET.SubElement(root, 'geometry')
        geometry.text = str(env_geometry)

        # - Add CRS
        crs = ET.SubElement(root, 'crs')
        crs.text = xml_dicts_ref['crs']

        # - Create a new element for the dataset
        dataset = ET.SubElement(root, 'dataset')

        # - Add metadata relative to other EO and Non-EO input data
        meta_data = gsp_metadata(in_prod_id)
        for md in meta_data:
            add_meta_field(dataset, md)

        # - Loop over the remaining bursts and merge them
        for idx in range(len(track_gdf)):
            meta_path = Path(track_gdf.iloc[idx]['Path'])
            xml_dict_gsp = extract_xml_from_zip(str(meta_path))[0]

            # - Valid only for TRE-A data
            # - Extract Input Product type from file name
            p_f_name = Path(track_gdf.iloc[idx]['Path']).stem.split('_')
            if p_f_name[4].endswith('B'):
                # - Single Geometry Deformation
                gsp_id_in = f'S301{SENSOR}01'
            else:
                # - Single Geometry Calibrated Deformation
                gsp_id_in = f'S301{SENSOR}02'

            # - Extract Geospatial Product ID
            gsp = ET.SubElement(dataset, 'gsp')
            gsp_id = ET.SubElement(gsp, 'gsp_id')
            gsp_id.text = gsp_id_in

            # - Extract Product ID - input file name
            product_id_in = Path(track_gdf.iloc[idx]['Path']).stem

            # - Convert the input file name to the original
            # - one produced by SVC01
            product_id_in\
                = product_id_in.replace(p_f_name[1], gsp_id_in)
            product_id_in \
                = product_id_in.replace(p_f_name[4], p_f_name[4][:-1])

            product_id = ET.SubElement(gsp, 'product_id')
            product_id.text = product_id_in

            burst_id = ET.SubElement(gsp, 'burst_id')
            burst_id.text = xml_dict_gsp['burst_id']

            description = ET.SubElement(gsp, 'description')
            description.text = gsp_description(gsp_id_in)

        # - Save Formatted XML to File
        xml_string = ET.tostring(root, encoding='utf-8', method='xml')
        # - Use minidom to prettify the xml string
        dom = parseString(xml_string)
        pretty_xml = dom.toprettyxml()
        # - Write the pretty xml string to file
        with open(os.path.join(out_dir, metadata_file), 'w') as f:
            f.write(pretty_xml)

        # - Save results inside a compressed archive
        zip_name\
            = os.path.join(out_dir,
                           metadata_file).replace('.xml', '.zip')
        # - Create a new zipfile
        print(f"# - Creating new zipfile {zip_name}\n")
        with zipfile.ZipFile(zip_name, 'w') as zip_ref:
            zip_ref.write(os.path.join(out_dir, metadata_file),
                          metadata_f_name)
            zip_ref.write(out_file, os.path.basename(out_file))
        # - Remove the temporary files
        os.remove(os.path.join(out_dir, metadata_file))
        os.remove(out_file)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")
