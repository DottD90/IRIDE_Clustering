#!/usr/bin/env python3
"""
Written by: Filippo Santarelli - 02/2024
Edited by Enrico Ciraci' - 02/2024

Compute active areas perimeter (polygons) from a set of points/persistent
scatterers. The algorithm is based on a combination of K-means and DBSCAN
clustering algorithms. The output is a shapefile containing the polygons
with the following fields:
    - pid: product ID
    - n_points: number of points inside the polygon
    - density: density of points inside the polygon
    - vel: average velocity of points inside the polygon
    - max_vel: maximum velocity of points inside the polygon
    - min_vel: minimum velocity of points inside the polygon
    - vel_std: standard deviation of velocity of points inside the polygon

The input file can be a shapefile, a CSV file or a ZIP file containing a CSV.

Script usage:
usage: Active Areas Deformation Clustering
    [-h] [-K KMEANS_CLASSES [KMEANS_CLASSES ...]] [-T VEL_THRESHOLD
    [VEL_THRESHOLD ...]] [-E EPS [EPS ...]] [-M MIN_SAMPLES [MIN_SAMPLES ...]]
    [-P PID_BASE [PID_BASE ...]] [-B BUFFER [BUFFER ...]]
    [-I IN_FIELD [IN_FIELD ...]] [-O OUT_FIELD [OUT_FIELD ...]] [-J JOBS]
    in_path [in_path ...]

positional arguments:
  in_path      ESRI shapefiles, CSV or ZIP containing the CSV
   to be processed at once

options:
  -h, --help            show this help message and exit
  -K KMEANS_CLASSES [KMEANS_CLASSES ...], --kmeans-classes KMEANS_CLASSES
            [KMEANS_CLASSES ...]
             Number of classes for k-Means velocity clustering
             (broadcast to the size of input paths)

  -T VEL_THRESHOLD [VEL_THRESHOLD ...], --vel-threshold
            VEL_THRESHOLD [VEL_THRESHOLD ...]
            Velocity threshold applied to clusters
            (broadcast to the size of input paths)

  -E EPS [EPS ...], --eps EPS [EPS ...]
            epsilon parameter for DBSCAN algorithm and alphashape
            (broadcast to the size of input paths)

  -M MIN_SAMPLES [MIN_SAMPLES ...], --min-samples MIN_SAMPLES [MIN_SAMPLES ...]
            minimum number of samples, fed to the DBSCAN algorithm
            (broadcast to the size of input paths)
  -P PID_BASE [PID_BASE ...], --pid-base PID_BASE [PID_BASE ...]
            base string for PID construction
            (broadcast to the size of input paths)

  -B BUFFER [BUFFER ...], --buffer BUFFER [BUFFER ...]
        buffer used to dilate polygons
        (broadcast to the size of input paths)

  -I IN_FIELD [IN_FIELD ...], --in-field IN_FIELD [IN_FIELD ...]
        input field used for clustering (broadcast to the size of input paths)

  -O OUT_FIELD [OUT_FIELD ...], --out-field OUT_FIELD [OUT_FIELD ...]
        name of the field saved on the output shapefile
        (broadcast to the size of input paths)

  -j JOBS, --jobs JOBS  Number of concurrent jobs

Python Dependencies:
    - pandas:Python Data Analysis Library
        https://pandas.pydata.org/
    - geopandas: Python tools for working with geospatial data in python
        https://geopandas.org/
    - numpy: The fundamental package for scientific computing with Python
        https://numpy.org/
    - scikit-learn: Machine Learning in Python
        https://scikit-learn.org/stable/
     - alphashape: Alpha Shape Toolbox
        https://pypi.org/project/alphashape/
    - fsspec: Filesystem interfaces for Python
        https://filesystem-spec.readthedocs.io/en/latest/
    - shapely: Manipulation and analysis of geometric objects
        in the Cartesian plane.
        https://pypi.org/project/shapely/
    - tqdm: A Fast, Extensible Progress Bar for Python and CLI
        https://tqdm.github.io/
"""
# - Python modules
import argparse
import logging
import sys
from traceback import format_exception
from datetime import datetime
import fsspec
import pandas as pd
import numpy as np
import scipy as sp
from pathlib import Path
import concurrent.futures as cf
from typing import Optional
# - Third-party modules
from sklearn.cluster import DBSCAN, KMeans
import alphashape
import geopandas as gpd
import shapely.geometry
from tqdm import tqdm


schema = {
    "geometry": "Polygon",
    "properties": {
        "pid": "str:10",
        "n_points": "int:10",
        "density": "float:10.2",
        "vel": "float:10.1",
        "max_vel": "float:10.1",
        "min_vel": "float:10.1",
        "vel_std": "float:10.1",
    },
}

BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def encode(n, p, pad=None):
    """
    Encode a number in base 62
    :param n: input number
    :param p: base
    :param pad: padding
    :return: base 62 encoded number
    """
    if n < 0:
        raise ValueError("Cannot encode negative numbers.")
    res = ""
    while True:
        n, r = divmod(n, p)
        res = BASE62[r] + res
        if n == 0:
            break
    if pad is None:
        return res
    return res.rjust(pad, BASE62[0])


def read_csv_as_geodataframe(path: Path, **kwargs) -> gpd.GeoDataFrame:
    """
    Read a CSV file as a GeoDataFrame
    :param path: absolute path to the file
    :param kwargs: dictionary of parameters passed to the read_csv method
    :return: gpd.GeoDataFrame
    """
    df = pd.read_csv(path, **kwargs)
    return gpd.GeoDataFrame(
        data=df,
        geometry=gpd.points_from_xy(df.easting, df.northing, crs=3035),
        crs=3035,
    )


def read_as_geodataframe(path: Path, **kwargs) -> Optional[gpd.GeoDataFrame]:
    """
    Read a file as a GeoDataFrame
    :param path: absolute path to the file
    :param kwargs: dictionary of parameters passed to the read_csv method
    :return: gpd.GeoDataFrame or None if file extension is not recognised
    """
    path = Path(path)
    try:
        match path.suffix:
            case ".zip":
                with fsspec.open("zip://*.csv::" + path.as_posix()) as of:
                    return read_csv_as_geodataframe(of, **kwargs)
            case ".csv":
                return read_csv_as_geodataframe(path, **kwargs)
            case ".shp":
                df = gpd.read_file(Path(path))
                df.crs = 4326
                return df.to_crs(3035)
            case _:
                logging.error(f"File extension {path.suffix} not recognised")
                return None
    except Exception as e:
        logging.error(f"Error reading file {path}: {e}")
        return None


def process_burst(
        in_path: Path | str,
        kmeans_classes: int, in_field_threshold: float,
        eps: float, min_samples: int, pid_base: int,
        buffer: float, in_field: str, out_field: str,
        ) -> gpd.GeoDataFrame | None:
    """
    Process a single file
    :param in_path: absolute path to the file
    :param kmeans_classes: K-means number of classes
    :param in_field_threshold: input field threshold - used to filter out
        polygons with average velocity lower than this value.
    :param eps: DBSCAN epsilon parameter
    :param min_samples: minimum number of samples for defining a cluster
    :param pid_base: base number for product PID computation
    :param buffer: polygon buffer size in meters
    :param in_field: input reference field name (e.g. "mean_vel")
    :param out_field: output reference field name (e.g. "mean_vel")
    :return: gpd.GeoDataFrame or None if file extension is not recognised
    """
    # - Load input data as a GeoDataFrame
    logging.info(f"# - Processing file {in_path}")
    df = read_as_geodataframe(in_path)
    if df is None:
        return None

    # - create classes according to velocity
    kmeans_algorithm = KMeans(n_clusters=kmeans_classes, n_init="auto")
    vel_clusters \
        = kmeans_algorithm.fit_predict(df[in_field].values.reshape(-1, 1))
    mask_clusters \
        = np.abs(kmeans_algorithm.cluster_centers_) > in_field_threshold
    (cluster_idx,) = mask_clusters.ravel().nonzero()

    # - Velocity Clustering
    clst = np.sum(vel_clusters[None, :]
                  == cluster_idx[:, None], axis=0).astype(bool)
    vel_clusters[clst] = 1
    vel_clusters[~clst] = -1
    df["label"] = vel_clusters

    # - Apply Density-based clustering (DBSCAN)
    ps_coords = np.stack((df.geometry.x, df.geometry.y),
                         axis=-1)[df.label == 1, :]
    spatial_clustering = DBSCAN(eps=eps, min_samples=min_samples)
    polygons_ids = spatial_clustering.fit_predict(ps_coords)

    df.loc[df.label == 1, "polygon_id"] = polygons_ids
    df.loc[df.label != 1, "polygon_id"] = -1

    # - Compute Polygons
    poly_geoms = []
    data = {
        "pid": [],
        "n_points": [],
        "density": [],
        "vel": [],
        "max_vel": [],
        "min_vel": [],
        "vel_std": [],
    }

    for id_pt in df.polygon_id.unique():
        if id_pt == -1:
            continue
        mask = df.polygon_id == id_pt
        sub_df = df.loc[mask, :]
        x_s = np.stack((sub_df.geometry.x, sub_df.geometry.y), axis=-1)

        try:
            # - This Exception should be further investigated
            poly = alphashape.alphashape(x_s, alpha=1 / eps)
        except sp.spatial._qhull.QhullError:
            continue
        if isinstance(poly, gpd.GeoDataFrame):
            poly = poly.dissolve().geometry[0]
        elif isinstance(poly, (shapely.geometry.Point,
                               shapely.geometry.LineString)):
            continue
        elif poly.is_empty:
            continue
        poly = poly.buffer(buffer)
        centroid = poly.centroid
        num = int(centroid.y) * 2**32 + int(centroid.x)
        pid = pid_base + encode(n=num, p=62, pad=6)
        inside_points = gpd.clip(df, poly)
        vel = inside_points[out_field].mean()
        vel_std = inside_points[out_field].std()
        if abs(vel) + vel_std < in_field_threshold:
            continue
        data["pid"].append(pid)
        data["n_points"].append(inside_points[out_field].count())
        data["density"].append(inside_points[out_field].count()
                               / (poly.area / 1e6))
        data["vel"].append(vel)
        data["max_vel"].append(inside_points[out_field].max())
        data["min_vel"].append(inside_points[out_field].min())
        data["vel_std"].append(vel_std)
        poly_geoms.append(poly)

    # - Return the generated polygon
    gdf = gpd.GeoDataFrame(data=data, geometry=poly_geoms, crs=3035)
    return gdf


def main() -> None:
    parser = argparse.ArgumentParser(
        "Active Areas Deformation Clustering",
        description="Compute active areas perimeter (polygons) from a set of "
                    "points/persistent scatterers."
    )
    # - Positional arguments
    parser.add_argument(
        "in_path",
        nargs="+",
        type=Path,
        help="List of ESRI Shapefiles, CSV or ZIP "
             "containing the CSV to be processed at once",
    )
    # - Optional arguments
    parser.add_argument(
        "-K",
        "--kmeans-classes",
        type=int,
        default=21,
        nargs="+",
        help="Number of classes for k-Means velocity clustering "
             "(broadcast to the size of input paths)",
    )
    parser.add_argument(
        "-T",
        "--vel-threshold",
        type=float,
        default=1.25,
        nargs="+",
        help="Velocity threshold applied to clusters "
             "(broadcast to the size of input paths)",
    )
    parser.add_argument(
        "-E",
        "--eps",
        type=float,
        default=25,
        nargs="+",
        help="epsilon parameter for DBSCAN algorithm and "
             "alphashape (broadcast to the size of input paths)",
    )
    parser.add_argument(
        "-M",
        "--min-samples",
        type=int,
        default=10,
        nargs="+",
        help="minimum number of samples, fed to the DBSCAN algorithm "
             "(broadcast to the size of input paths)",
    )
    parser.add_argument(
        "-P",
        "--pid-base",
        type=str,
        default="bDG1",
        nargs="+",
        help="base string for PID construction "
             "(broadcast to the size of input paths)",
    )
    parser.add_argument(
        "-B",
        "--buffer",
        type=float,
        default=12.5,
        nargs="+",
        help="buffer used to dilate polygons "
             "(broadcast to the size of input paths)",
    )
    parser.add_argument(
        "-I",
        "--in-field",
        type=str,
        default="mean_vel",
        nargs="+",
        help="input field used for clustering "
             "(broadcast to the size of input paths)",
    )
    parser.add_argument(
        "-O",
        "--out-field",
        type=str,
        default="mean_vel",
        nargs="+",
        help="name of the field saved on the output shapefile "
             "(broadcast to the size of input paths)",
    )
    parser.add_argument("-J", "--jobs", type=int,
                        help="Number of concurrent jobs")
    args = parser.parse_args()

    all_params = list(
        zip(
            args.in_path,
            np.broadcast_to(args.kmeans_classes, shape=(len(args.in_path))),
            np.broadcast_to(args.vel_threshold, shape=(len(args.in_path))),
            np.broadcast_to(args.eps, shape=(len(args.in_path))),
            np.broadcast_to(args.min_samples, shape=(len(args.in_path))),
            np.broadcast_to(args.pid_base, shape=(len(args.in_path))),
            np.broadcast_to(args.buffer, shape=(len(args.in_path))),
            np.broadcast_to(args.in_field, shape=(len(args.in_path))),
            np.broadcast_to(args.out_field, shape=(len(args.in_path))),
        )
    )

    with cf.ProcessPoolExecutor(args.jobs) as pool:
        future_to_path = {}
        for params in all_params:
            future = pool.submit(process_burst, *params)
            future_to_path[future] = params[0]

    for future in tqdm(
        cf.as_completed(future_to_path.keys()),
        desc="# - Clustering",
        total=len(future_to_path),
    ):
        path = future_to_path[future]
        try:
            future.result()
        except Exception as exc:
            logging.critical(
                "# - %s generated an exception:\n%s",
                path.stem,
                "\n".join(format_exception(exc)),
            )
            sys.exit(127)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")
