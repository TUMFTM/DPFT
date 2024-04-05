from typing import Tuple

import numpy as np

from dprt.utils.misc import as_dtype, round_perc


@round_perc
@as_dtype(dtype=float)
def polar2cart_rad(r: np.ndarray,
                   phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y


def polar2cart(r: np.ndarray,
               phi: np.ndarray,
               degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the Cartesian coordinates for the given polar coordinates.

    Convention of the polar angle phi in respect to the Cartesian axis.

                90
          135    y    45
                 |
        180      ---x    0

          -135       -45
                -90

    Arguments:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the range. The angle is zero on the x-axis
            increases mathematical positivly.

    Returns:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
    """
    if degrees:
        return polar2cart_rad(r, np.deg2rad(phi))

    return polar2cart_rad(r, phi)


@round_perc
@as_dtype(dtype=float)
def cart2polar_rad(x: np.ndarray,
                   y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Convert coordinates
    r = np.linalg.norm(np.vstack((x, y)), axis=0)
    phi = np.arctan2(y, x)

    return r, phi


def cart2polar(x: np.ndarray,
               y: np.ndarray,
               degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the polar coordinates for the given Cartesian coordinates.

    Convention of the polar angle phi in respect to the Cartesian axis.

                90
          135    y    45
                 |
        180      ---x    0

          -135       -45
                -90

    Arguments:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.

    Returns:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the range. The angle is zero on the x-axis
            increases mathematical positivly.
    """
    r, phi = cart2polar_rad(x, y)

    if degrees:
        return r, np.rad2deg(phi)

    return r, phi


@round_perc
@as_dtype(dtype=float)
def spher2cart_rad(r: np.ndarray,
                   phi: np.ndarray,
                   roh: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = r * np.cos(phi) * np.cos(roh)
    y = r * np.sin(phi) * np.cos(roh)
    z = r * np.sin(roh)

    return x, y, z


def spher2cart(r: np.ndarray,
               phi: np.ndarray,
               roh: np.ndarray,
               degrees: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the Cartesian coordinates for the given spherical coordinates.

    Convention of the spherical angle phi (left) and roh (right)
    in respect to the Cartesian axis.

                90                          90
          135    y    45              45     z    45
                 |                           |
        180      ---x    0          0        ---y    0

          -135       -45              -45        -45
                -90                         -90

    Arguments:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the y-z-plane. The angle is zero on the x-axis
            increases mathematical positivly.
        roh: Elevation (inclination) angle. Angle between the
            z-axis and the x-y-plane. The angle is zero on the
            x-y-plane, positive towards the positive z-axis and
            negative towards the negative z-axis.

    Returns:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
        z: Values of the z-dimension in Cartesian coordinates.
    """
    if degrees:
        return spher2cart_rad(r, np.deg2rad(phi), np.deg2rad(roh))

    return spher2cart_rad(r, phi, roh)


@round_perc
@as_dtype(dtype=float)
def cart2spher_rad(x: np.ndarray,
                   y: np.ndarray,
                   z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert coordinates
    r = np.linalg.norm(np.vstack((x, y, z)), axis=0)
    phi = np.arctan2(y, x)
    roh = np.arcsin(np.divide(z, r, out=np.zeros_like(z), where=r != 0))

    return r, phi, roh


def cart2spher(x: np.ndarray,
               y: np.ndarray,
               z: np.ndarray,
               degrees: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the spherical coordinates for the given Cartesian coordinates.

    Convention of the spherical angle phi (left) and roh (right)
    in respect to the Cartesian axis.

                90                          90
          135    y    45              45     z    45
                 |                           |
        180      ---x    0          0        ---y    0

          -135       -45              -45        -45
                -90                         -90

    Arguments:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
        z: Values of the z-dimension in Cartesian coordinates.

    Returns:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the y-z-plane. The angle is zero on the x-axis
            increases mathematical positivly.
        roh: Elevation (inclination) angle. Angle between the
            z-axis and the x-y-plane. The angle is zero on the
            x-y-plane, positive towards the positive z-axis and
            negative towards the negative z-axis.
    """
    r, phi, roh = cart2spher_rad(x, y, z)

    if degrees:
        phi = np.rad2deg(phi)
        roh = np.rad2deg(roh)

    return r, phi, roh
