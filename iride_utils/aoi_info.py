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

    # - Italian Regions
    elif aoi in ['calabria', 'cal']:
        aoi_tag = 'CAL'
        aoi_name = 'Calabria'

    elif aoi in ['sicilia', 'sic']:
        aoi_tag = 'SIC'
        aoi_name = 'Sicilia'

    elif aoi in ['basilicata', 'bas']:
        aoi_tag = 'BAS'
        aoi_name = 'Basilicata'

    elif aoi in ['puglia', 'pug']:
        aoi_tag = 'PUG'
        aoi_name = 'Puglia'

    elif aoi in ['campania', 'cam']:
        aoi_tag = 'CAM'
        aoi_name = 'Campania'

    elif aoi in ['molise', 'mol']:
        aoi_tag = 'MOL'
        aoi_name = 'Molise'

    elif aoi in ['abruzzo', 'abr']:
        aoi_tag = 'ABR'
        aoi_name = 'Abruzzo'

    elif aoi in ['lazio', 'laz']:
        aoi_tag = 'LAZ'
        aoi_name = 'Lazio'

    elif aoi in ['umbria', 'umb']:
        aoi_tag = 'UMB'
        aoi_name = 'Umbria'

    elif aoi in ['marche', 'mar']:
        aoi_tag = 'MAR'
        aoi_name = 'Marche'

    elif aoi in ['emilia_romagna', 'emr']:
        aoi_tag = 'EMR'
        aoi_name = 'Emilia Romagna'

    elif aoi in ['toscana', 'tos']:
        aoi_tag = 'TOS'
        aoi_name = 'Toscana'

    elif aoi in ['lombardia', 'lom']:
        aoi_tag = 'LOM'
        aoi_name = 'Lombardia'

    elif aoi in ['piemonte', 'pie']:
        aoi_tag = 'PIE'
        aoi_name = 'Piemonte'

    elif aoi in ['trentino', 'tnt']:
        aoi_tag = 'TNT'
        aoi_name = 'Trentino'

    elif aoi in ['veneto', 'ven']:
        aoi_tag = 'VEN'
        aoi_name = 'Veneto'

    elif aoi in ['friuli_venezia_giulia', 'fvg']:
        aoi_tag = 'FVG'
        aoi_name = 'Friuli Venezia Giulia'

    elif aoi in ['liguria', 'lig']:
        aoi_tag = 'LIG'
        aoi_name = 'Liguria'

    elif aoi in ['valle_d_aosta', 'vda']:
        aoi_tag = 'VDA'
        aoi_name = 'Valle d\'Aosta'
    else:

        raise ValueError(f"AOI {aoi} not found.")

    return {'aoi_tag': aoi_tag, 'aoi_name': aoi_name}
