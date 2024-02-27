"""
Written by Enrico Ciraci' - February 2024

Metadata Template for commonly used datasets
"""
# - Python Modules
import xml.etree.ElementTree as ET


def add_meta_field(tree: ET.Element, dataset: str) -> None:
    """
    Add metadata fields to the XML tree
    :param tree: pointer to the XML tree
    :param dataset: dataset name
    :return: None
    """
    if dataset in ['TINITALY', 'Tinitaly', 'Tinitaly-10']:
        # - Other EO Non-EO Input Data
        input_x = ET.SubElement(tree, 'input')
        input_id = ET.SubElement(input_x, 'input_id')
        input_id.text = 'S3-NEO-I01'
        version = ET.SubElement(input_x, 'version')
        version.text = 'Tinitaly-10'
        description = ET.SubElement(input_x, 'description')
        description.text \
            = ("Tarquini S., I. Isola, M. Favalli, A. Battistini,"
               "G. Dotta (2023). TINITALY, a digital elevation model "
               "of Italy with a 10 meters cell size (Version 1.1). "
               "Istituto Nazionale di Geofisica e Vulcanologia (INGV). "
               "https://doi.org/10.13127/tinitaly/1.1.")

    elif dataset in ['OpenStreetMap', 'OSM']:
        # - Other EO Non-EO Input Data
        input_x = ET.SubElement(tree, 'input')
        input_id = ET.SubElement(input_x, 'input_id')
        input_id.text = 'S3-NEO-I09'
        version = ET.SubElement(input_x, 'version')
        version.text = 'OpenStreetMap'
        description = ET.SubElement(input_x, 'description')
        description.text \
            = ("OpenStreetMap (Version 1.0). OpenStreetMap "
               "Foundation. https://doi.org/10.13127/osm/1.0.")

    elif dataset in ['Copernicus', 'CopDem']:
        # - Copernicus DEM
        input_x = ET.SubElement(tree, 'input')
        input_id = ET.SubElement(input_x, 'input_id')
        input_id.text = 'S3-NEO-I01'
        version = ET.SubElement(input_x, 'version')
        version.text = 'Cop-DEM - Resolution (m) 30 x 30'
        description = ET.SubElement(input_x, 'description')
        description.text \
            = ("Copernicus Digital Elevation Model (DEM) (Version 1.0). "
               "https://spacedata.copernicus.eu/collections/"
               "copernicus-digital-elevation-model.")
    else:
        raise ValueError(f"Dataset {dataset} not found.")