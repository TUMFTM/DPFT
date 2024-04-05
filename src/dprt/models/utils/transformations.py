from typing import Tuple

import torch

from torch import nn


def cart2polar(x: torch.Tensor,
               y: torch.Tensor,
               degrees: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
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
        degrees: Whether the angular values are given in
            degrees (True) or radians (False).

    Returns:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the range. The angle is zero on the x-axis
            increases mathematical positivly.
    """
    # Convert coordinates
    r = torch.linalg.norm(torch.dstack((x, y)), axis=0)
    phi = torch.arctan2(y, x)

    if degrees:
        phi = torch.rad2deg(phi)

    return r, phi


class Cart2Polar(nn.Module):
    def __init__(self,
                 dim: int = -1,
                 degrees: bool = True,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.degrees = degrees

    def forward(self, batch: torch.Tensor):
        """Returns the polar coordinates for the given Cartesian coordinates.

        Arguments:
            batch: Batch of points in cartesian
                coordinates with shape
                (batch, points, 2)

        Returns:
            batch: Batch of points in polar
                coordinates with shape
                (batch, points, 2)
        """
        return torch.cat(cart2polar(*batch.split(1, self.dim), self.degrees), self.dim)


def cart2spher(x: torch.Tensor,
               y: torch.Tensor,
               z: torch.Tensor,
               degrees: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        degrees: Whether the angular values are given in
            degrees (True) or radians (False).

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
    # Convert coordinates
    r = torch.linalg.norm(torch.dstack((x, y, z)), axis=-1).reshape_as(x)
    phi = torch.arctan2(y, x)

    # Avoid devision by zero
    c = torch.zeros_like(z)
    mask = (r != 0)
    c[mask] = z[mask] / r[mask]

    roh = torch.arcsin(c)

    if degrees:
        phi = torch.rad2deg(phi)
        roh = torch.rad2deg(roh)

    return r, phi, roh


class Cart2Spher(nn.Module):
    def __init__(self,
                 dim: int = -1,
                 degrees: bool = True,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.degrees = degrees

    def forward(self, batch: torch.Tensor):
        """Returns the spherical coordinates for the given Cartesian coordinates.

        Arguments:
            batch: Batch of points in cartesian
                coordinates with shape
                (batch, points, 3)

        Returns:
            batch: Batch of points in spherical
                coordinates with shape
                (batch, points, 3)
        """
        return torch.cat(cart2spher(*batch.split(1, self.dim), self.degrees), self.dim)


def polar2cart(r: torch.Tensor,
               phi: torch.Tensor,
               degrees: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
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
        degrees: Whether the angular values are given in
            degrees (True) or radians (False).

    Returns:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
    """
    if degrees:
        x = r * torch.cos(torch.deg2rad(phi))
        y = r * torch.sin(torch.deg2rad(phi))
    else:
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)

    return x, y


class Polar2Cart(nn.Module):
    def __init__(self,
                 dim: int = -1,
                 degrees: bool = True,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.degrees = degrees

    def forward(self, batch: torch.Tensor):
        """Returns the Cartesian coordinates for the given polar coordinates.

        Arguments:
            batch: Batch of points in polar
                coordinates with shape
                (batch, points, 2)

        Returns:
            batch: Batch of points in cartesian
                coordinates with shape
                (batch, points, 2)
        """
        return torch.cat(polar2cart(*batch.split(1, self.dim), self.degrees), self.dim)


def spher2cart(r: torch.Tensor,
               phi: torch.Tensor,
               roh: torch.Tensor,
               degrees: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        degrees: Whether the angular values are given in
            degrees (True) or radians (False).

    Returns:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
        z: Values of the z-dimension in Cartesian coordinates.
    """
    if degrees:
        x = r * torch.cos(torch.deg2rad(phi)) * torch.cos(torch.deg2rad(roh))
        y = r * torch.sin(torch.deg2rad(phi)) * torch.cos(torch.deg2rad(roh))
        z = r * torch.sin(torch.deg2rad(roh))
    else:
        x = r * torch.cos(phi) * torch.cos(roh)
        y = r * torch.sin(phi) * torch.cos(roh)
        z = r * torch.sin(roh)

    return x, y, z


class Spher2Cart(nn.Module):
    def __init__(self,
                 dim: int = -1,
                 degrees: bool = True,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.degrees = degrees

    def forward(self, batch: torch.Tensor):
        """Returns the Cartesian coordinates for the given spherical coordinates.

        Arguments:
            batch: Batch of points in spherical
                coordinates with shape
                (batch, points, 3)

        Returns:
            batch: Batch of points in cartesian
                coordinates with shape
                (batch, points, 3)
        """
        return torch.cat(spher2cart(*batch.split(1, self.dim), self.degrees), self.dim)


def build_transformation(name: str, *args, **kwargs):
    if name is None:
        return None
    if 'polar2cart' in name.lower():
        return Polar2Cart(*args, **kwargs)
    if 'spher2cart' in name.lower():
        return Spher2Cart(*args, **kwargs)
    if 'cart2polar' in name.lower():
        return Cart2Polar(*args, **kwargs)
    if 'cart2spher' in name.lower():
        return Cart2Spher(*args, **kwargs)
