"""
Set of utilities used to generate anc package geospatial products for
the IRIDE Service Segment - Lot 2ù.
"""
from typing import List


def gsp_description(gsp_id: str) -> str:
    """
    Returns the description for the GSP product included in the IRIDE
    Service Segment - Lot 2
    :param gsp_id: str - GSP product identifier
    """

    if gsp_id in ['S3-01-SNT-01', 'S3-01-CSM-01', 'S3-01-SAO-01',
                  'S301SNT01', 'S301CSM01', 'S301SAO01']:
        return "Single Geometry Deformation."

    elif gsp_id in ['S3-01-SNT-02', 'S3-01-CSM-02', 'S3-01-SAO-02',
                    'S301SNT02', 'S301CSM02', 'S301SAO02']:
        return "Single Geometry Calibrated Deformation."

    elif gsp_id in ['S3-01-SNT-03', 'S3-01-CSM-03', 'S3-01-SAO-03',
                    'S301SNT03', 'S301CSM03', 'S301SAO03']:
        return "2D Deformation East-West and Vertical Components."

    elif gsp_id in ['S3-02-SNT-02', 'S3-02-CSM-02', 'S3-02-SAO-02',
                    'S302SNT02', 'S302CSM02', 'S302SAO02']:
        return "LOS velocities projected along the maximum slope."

    elif gsp_id in ['S3-02-SNT-04', 'S3-02-CSM-04', 'S3-02-SAO-04',
                    'S302SNT04', 'S302CSM04', 'S302SAO04']:
        return "Temporal Anomaly Maps."

    elif gsp_id in ['S3-02-SNT-05', 'S3-02-CSM-05', 'S3-02-SAO-05',
                    'S302SNT05', 'S302CSM05', 'S302SAO05']:
        return "Automatic identification of unstable slopes."
    else:
        return ""


def gsp_metadata(gsp_id: str) -> List[str]:
    """
    Returns a list of metadata objects used to create a certains GSP product-
    :param gsp_id: GSP product identifier
    :return: List of metadata objects
    """
    if gsp_id in ['S3-01-SNT-01', 'S3-01-CSM-01', 'S3-01-SAO-01',
                  'S301SNT01', 'S301CSM01', 'S301SAO01']:
        return ['CopDem']

    elif gsp_id in ['S3-01-SNT-02', 'S3-01-CSM-02', 'S3-01-SAO-02',
                    'S301SNT02', 'S301CSM02', 'S301SAO02']:
        return ['CopDem']

    elif gsp_id in ['S3-01-SNT-03', 'S3-01-CSM-03', 'S3-01-SAO-03',
                    'S301SNT03', 'S301CSM03', 'S301SAO03']:
        return ['CopDem']

    elif gsp_id in ['S3-01-SNT-04', 'S3-01-CSM-04', 'S3-01-SAO-04',
                    'S301SNT04', 'S301CSM04', 'S301SAO04']:
        return ['CopDem', 'OpenStreetMap']

    elif gsp_id in ['S3-02-SNT-02', 'S3-02-CSM-02', 'S3-02-SAO-02',
                    'S302SNT02', 'S302CSM02', 'S302SAO02']:
        return ['Tinitaly-10']

    elif gsp_id in ['S3-02-SNT-04', 'S3-02-CSM-04', 'S3-02-SAO-04',
                    'S302SNT04', 'S302CSM04', 'S302SAO04']:
        return ['Tinitaly-10']

    elif gsp_id in ['S3-02-SNT-05', 'S3-02-CSM-05', 'S3-02-SAO-05',
                    'S302SNT05', 'S302CSM05', 'S302SAO05']:
        return ['Tinitaly-10']

    else:

        return []
