import argparse
import concurrent.futures as cf
import logging
import pickle
import re
import sys
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from traceback import format_exception
from typing import Tuple, Union

import fsspec
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
from dask_ml.wrappers import ParallelPostFit
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel,
                                              _check_length_scale)
from tqdm import tqdm


class Sign(IntEnum):
    POSITIVE: 1
    NEGATIVE: -1


def read_as_geodataframe(path: Path | str, **kwargs) -> gpd.GeoDataFrame:
    """Read CSV, ZIP or SHP into a `geopandas.GeoDataFrame`.

    Args:
        path (Path | str): path to the file with geospatial data

    Raises:
        NotImplementedError: for unsupported file format

    Returns:
        gpd.GeoDataFrame: with the geospatial content of the provided file.

    Note:
        The input ZIP or CSV must have `easting` and `northing` fields; the
        input SHP must be in EPSG:4326.
    """
    path = Path(path)
    match path.suffix:
        case ".zip":
            with fsspec.open("zip://*.csv::" + path.as_posix()) as of:
                df = pd.read_csv(of, **kwargs)
            df = gpd.GeoDataFrame(
                data=df,
                geometry=gpd.points_from_xy(df.easting, df.northing, crs=3035),
                crs=3035,
            )
        case ".csv":
            df = pd.read_csv(path, **kwargs)
            df = gpd.GeoDataFrame(
                data=df,
                geometry=gpd.points_from_xy(df.easting, df.northing, crs=3035),
                crs=3035,
            )
        case ".shp":
            df = gpd.read_file(Path(path))
            df.crs = 4326
            df = df.to_crs(3035)
        case _:
            raise NotImplementedError(f"File extension {path.suffix} not recognised")
    return df


class ExpDecayKernel(RBF):
    def __call__(
        self, X: npt.ArrayLike, Y: npt.ArrayLike = None, eval_gradient: bool = False
    ) -> Union[npt.ArrayLike, Tuple[npt.ArrayLike, npt.ArrayLike]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Args:
            X (npt.ArrayLike): array of shape (n_samples_X, n_features),
                left argument of the returned kernel k(X, Y)
            Y (npt.ArrayLike: array of shape (n_samples_Y, n_features),
                right argument of the returned kernel k(X, Y). If None, k(X, X)
                if evaluated instead. (Defaults to None)
            eval_gradient (bool): determines whether the gradient with respect to the log of
                the kernel hyperparameter is computed. Only supported when Y is None. (Defaults to False)

        Returns:
            K (npt.ArrayLike): array of shape (n_samples_X, n_samples_Y), the kernel k(X, Y)
            K_gradient (npt.ArrayLike): array of shape (n_samples_X, n_samples_X, n_dims),
                the gradient of the kernel k(X, X) with respect to the log of the
                hyperparameter of the kernel. Only returned when `eval_gradient`
                is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="euclidean")
            K = np.exp(-dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
            K = np.exp(-dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) / length_scale
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K


def prepare_gnss_interpolation(
    source: Path | str,
    dest: Path | str,
) -> Tuple[ParallelPostFit, ParallelPostFit, ParallelPostFit]:
    """Precomputes the interpolators for kriging.

    The interpolators are stored in the destination file and reused
    if already available.

    Args:
        source (Path|str): GNSS file.
        dest (Path|str): file where the interpolators will be stored.

    Returns:
        Three `dask_ml.wrappers.ParallelPostFit` corresponding to the
        interpolators for the south-north, east-west and vertical
        components, respectively.
    """
    if Path(dest).is_file():
        with Path(dest).open("rb") as file:
            kriging_N, kriging_E, kriging_Up = pickle.load(file)
    else:
        if not Path(source).is_file():
            raise FileNotFoundError("GNSS source not found")
        gnss = pd.read_csv(source, usecols=["easting", "northing", "N", "E", "Up"])
        _xy = ["easting", "northing"]

        # infittisco i punti del GNSS in corrispondenza del SAR
        kernel = ConstantKernel(0.23) * ExpDecayKernel(90_000)
        kriging_E = ParallelPostFit(
            estimator=GaussianProcessRegressor(kernel, optimizer=None)
        )
        kriging_E.fit(gnss[_xy], gnss.E)
        kriging_N = ParallelPostFit(
            estimator=GaussianProcessRegressor(kernel, optimizer=None)
        )
        kriging_N.fit(gnss[_xy], gnss.N)
        kernel = ConstantKernel(0.78) * ExpDecayKernel(130_000)
        kriging_Up = ParallelPostFit(
            estimator=GaussianProcessRegressor(kernel, optimizer=None)
        )
        kriging_Up.fit(gnss[_xy], gnss.Up)

        with Path(dest).open("wb") as file:
            pickle.dump((kriging_N, kriging_E, kriging_Up), file)

    return kriging_N, kriging_E, kriging_Up


def gnss_calibration(
    df: gpd.GeoDataFrame,
    key: str,
    component: str,
    gnss_path: Path | str,
    sign: Sign = Sign.POSITIVE,
):
    """Add or remove the GNSS component from the input timeseries.

    Args:
        path (Path | str): _description_
        out_dir (Path | str): _description_
        key (str): mean velocity field, that will be changed by this fuction.
        component (str): either "vertical" or "east-west".
        gnss_path (Path | str): GNSS file.
        sign (Sign): whether the GNSS trend should be added or removed.

    Raises:
        NotImplementedError: if the requested component is not supported.
    """
    kriging_N, kriging_E, kriging_Up = prepare_gnss_interpolation(
        source=gnss_path, dest=Path(gnss_path).with_suffix(".pkl")
    )

    x, y = df.geometry.x, df.geometry.y
    match component:
        case "vertical":
            predictor = kriging_Up
        case "east-west":
            predictor = kriging_E
        case _:
            raise NotImplementedError
    velocity_gnss = predictor.predict(np.stack((x, y), axis=-1))

    pattern = re.compile(r"\d+")
    dates_fields = sorted(filter(pattern.fullmatch, df.columns))
    dates = [datetime.strptime(col, "%Y%m%d") for col in dates_fields]
    intervals = np.cumsum(np.diff(dates))
    intervals = [0] + [dt.days for dt in intervals]
    mean_v = velocity_gnss / 365.0  # in giorni
    sp_fixes = np.array([mean_v * dt for dt in intervals]).T
    df[dates_fields] += sign.value * sp_fixes
    df[key] += sign.value * velocity_gnss

    return df


def gnss_calibration_from_path(
    path: Path | str,
    out_dir: Path | str,
    key: str,
    component: str,
    gnss_path: Path | str,
    sign: Sign = Sign.POSITIVE,
):
    """Add or remove the GNSS component from the input timeseries.

    Args:
        path (Path | str): file with timeseries.
        out_dir (Path | str): output folder, where the resulting timeseries will be saved.
        key (str): mean velocity field, that will be changed by this fuction.
        component (str): either "vertical" or "east-west".
        gnss_path (Path | str): GNSS file.
        sign (Sign): whether the GNSS trend should be added or removed.

    Raises:
        NotImplementedError: if the requested component is not supported.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    df = read_as_geodataframe(path)

    df = gnss_calibration(df, key, component, gnss_path, sign)

    df.to_csv(out_dir / path.name, index=False)


def main():
    parser = argparse.ArgumentParser(
        "detrend", description="Removes GNSS contribution from L3 results"
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="List of space-separated shapefiles, CSV or ZIP containing the CSV which the GNSS shall be removed from (accepts a list of space-separated paths)",
    )
    parser.add_argument(
        "out_dir", type=Path, help="Output directory, where the files will be stored"
    )
    parser.add_argument(
        "component", choices=["vertical", "east-west"], help="Component to be removed"
    )
    parser.add_argument(
        "-g",
        "--gnss_estimator_path",
        type=Path,
        default="data/gnss/EGMS_AEPND_V2023.0.csv",
        help="Path to the GNSS file",
    )
    parser.add_argument(
        "-f",
        "--velocity-field",
        type=str,
        default="mean_velocity",
        help="Velocity field name that will be affected by the GNSS removal",
    )
    parser.add_argument("-j", "--jobs", type=int, help="Number of concurrent jobs")
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)

    with cf.ProcessPoolExecutor(args.jobs) as pool:
        future_to_path = {}
        for path in args.paths:
            future = pool.submit(
                gnss_calibration_from_path,
                path,
                args.out_dir,
                key=args.velocity_field,
                component=args.component,
                gnss_estimator_path=args.gnss_estimator_path,
                sign=Sign.NEGATIVE,
            )
            future_to_path[future] = path
    for future in tqdm(
        cf.as_completed(future_to_path.keys()),
        desc="Clustering",
        total=len(future_to_path),
    ):
        path = future_to_path[future]
        try:
            future.result()
        except Exception as exc:
            logging.critical(
                "%s generated an exception:\n%s",
                path.stem,
                "\n".join(format_exception(exc)),
            )
            sys.exit(127)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")
