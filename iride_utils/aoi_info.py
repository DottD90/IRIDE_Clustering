"""
Written by Enrico Ciraci' - February 2024
Return IRIDE AOIs information
"""


def aoi_info(aoi: str) -> dict:
    """
    Return AOI information
    :param aoi: AOI Name [can be the extended name or aoi tag]
    :return: Dictionary with AOI information
    """

    if aoi in ['Nocera_Terinese', '']:
        aoi_tag = 'NTR'
        aoi_name = 'Nocera Terinese'
    else:
        raise ValueError(f"AOI {aoi} not found.")

    return {'aoi_tag': aoi_tag, 'aoi_name': aoi_name}
