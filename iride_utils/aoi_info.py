"""
Written by Enrico Ciraci' - February 2024
Return IRIDE AOIs information
"""


def get_aoi_info(aoi: str) -> dict:
    """
    Return AOI information
    :param aoi: AOI Name [can be the extended name or aoi tag]
    :return: Dictionary with AOI information
    """
    # - convert to all lower case
    aoi = aoi.lower()

    if aoi in ['nocera_terinese', 'ntr', 'nocera terinese']:
        aoi_tag = 'NTR'
        aoi_name = 'Nocera Terinese'

    elif aoi in ['palermo', 'pal']:
        aoi_tag = 'PAL'
        aoi_name = 'Palermo'

    elif aoi in ['brennero', 'bre']:
        aoi_tag = 'BRN'
        aoi_name = 'Brennero'

    elif aoi in ['cortina', 'crt']:
        aoi_tag = 'CRT'
        aoi_name = 'Cortina'

    elif aoi in ['norcia', 'nri']:
        aoi_tag = 'NRI'
        aoi_name = 'Norcia'

    elif aoi in ['pistoia', 'pst']:
        aoi_tag = 'PST'
        aoi_name = 'Pistoia'

    else:

        raise ValueError(f"AOI {aoi} not found.")

    return {'aoi_tag': aoi_tag, 'aoi_name': aoi_name}
