import numpy as np

from scipy.spatial.transform import Rotation


def get_transformation(translation: np.ndarray = None,
                       rotation: np.ndarray = None,
                       degrees: bool = False,
                       inverse: bool = False,
                       dtype: str = 'float32') -> np.ndarray:
    """Returns a homogeneous transformation matrix given
    a translation and rotation.

    Arguments:
        translation: Translation values (x, y, z) with shape (3,).
        rotation: Rotation values given as euler angles or
            quaternion. Euler agles should be given as (x, y, z)
            with shape (3,), while quaternions should be given
            as (x, y, z, w) with shape (4,).
        degrees: Whether the rotation values are given in
            degrees instead of radians.
        inverse: Whether to return the inverse transformation
            matrix.
        dtype: Data type of the transformation matrix.

    Returns:
        transformation: A homogeneous transformation matrix
            with shape (4, 4).
    """
    # Initialize transformation values
    transformation = np.eye(4, dtype=np.dtype(dtype))
    translation = translation if translation is not None else np.zeros(3)
    rotation = rotation if rotation is not None else np.zeros(3)

    # Get rotation matrix
    if rotation.size == 3:
        # Convert euler angles to (3, 3) rotation matrix
        rotation_matrix = Rotation.from_euler('xyz', rotation, degrees=degrees).as_matrix()

    elif rotation.size == 4:
        # Convert quaternion to (3, 3) rotation matrix
        rotation_matrix = Rotation.from_quat(rotation).as_matrix()

    else:
        raise ValueError(
            f"The rotation must be either given as euler angles"
            f"with shape (3,) or quaternion with shape (4,) but"
            f"an input value with shape {rotation.shape()} was given."
        )

    # Construct homogeneous (4, 4) transformation matrix
    if inverse:
        rot_inv = rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        transformation[:3, :3] = rot_inv
        transformation[:3, 3] = rot_inv.dot(trans)
    else:
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = np.transpose(np.array(translation))

    return transformation


def get_box_corners(boxes: np.ndarray,
                    wlh_factor: float = 1.0,
                    wlh_offset: float = 0.0) -> np.ndarray:
    """ Returns eight corners of the given M bounding boxes with
    three coordinates per corner.

    Bounding boxes must be provided in the following fromat:
    x, y, z, theta, l, w, h, ...

    The returnd corners are given in the following order:
        5------4
       /|     /|
      6------7 |
      | 1----|-0
      |/     |/
      2------3

    Arguments:
        boxes: Bounding boxes given as numerical array of shape (M, ...).
            The second dimension can be of arbitrary size as long as the
            x, y, z, theta, l, w, h values represent the first 7 values
            of this dimension and are provided in the right order.
        wlh_factor: Scale factor for the bounding box dimensions.
        wlh_offset: Offset for the bounding box dimensions.

    Returns:
        corners: Bounding box corners (x, y, z) with dimensions (M, 8, 3).
    """
    # Ensure boxes shape for single bounding boxes
    boxes = np.atleast_2d(boxes)

    # Get number of boxes
    M = boxes.shape[0]

    # Apply wlh factor and offset
    boxes[:, 4:7] = boxes[:, 4:7] * wlh_factor
    boxes[:, 4:7] = boxes[:, 4:7] + wlh_offset

    # Get 3D bounding box corners
    x_corners = (boxes[:, 4] / 2)[:, None] * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = (boxes[:, 5] / 2)[:, None] * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = (boxes[:, 6] / 2)[:, None] * np.array([0, 0, 0, 0, 1, 1, 1, 1])
    corners = np.swapaxes(np.dstack((x_corners, y_corners, z_corners)), 1, 2)

    # Create rotation matrices from rotation angle
    rotation_matrices = np.stack([np.eye(3, 3)] * M)
    rotation_matrices[:, 0, 0] = np.cos(boxes[:, 3])
    rotation_matrices[:, 0, 1] = -np.sin(boxes[:, 3])
    rotation_matrices[:, 1, 0] = np.sin(boxes[:, 3])
    rotation_matrices[:, 1, 1] = np.cos(boxes[:, 3])

    # Rotate 3D bounding box corners
    corners = np.einsum('...jk, ...km', rotation_matrices, corners)

    # Translate 3D bounding box corners
    corners[:, 0, :] = corners[:, 0, :] + boxes[:, 0, None]
    corners[:, 1, :] = corners[:, 1, :] + boxes[:, 1, None]
    corners[:, 2, :] = corners[:, 2, :] + boxes[:, 2, None]

    return np.swapaxes(corners, 1, 2)


def transform_boxes(boxes: np.ndarray,
                    transformation: np.ndarray) -> np.ndarray:
    """Returns the transformed bounding boxes.

    Arguments:
        boxes: M bounding boxes given as numerical array of shape (M, ...).
            The second dimension can be of arbitrary size as long as the
            x, y, z, theta values represent the first 4 values
            of this dimension and are provided in the right order.
        transformation: A homogeneous transformation matrix
            with shape (4, 4).

    Returns:
        boxes: Transformed bounding boxes given as numerical
            array of shape (M, ...).
    """
    # Get number of boxes
    M = boxes.shape[0]

    # Convert boxes coordinates to homogeneous coordinates of shape (M, 4)
    center = np.column_stack((boxes[:, :3], np.ones(M)))

    # TODO: Include rotation

    # Transform bounding box corners (cornerwise dot product)
    boxes[:, :3] = np.einsum('ij,...j->...i', transformation, center)[:, :3]

    return boxes


def transform_points(points: np.ndarray,
                     transformation: np.ndarray) -> np.ndarray:
    """Returns the transformed point cloud.

    Arguments:
        points: N points given as numerical array of shape (N, ...).
            The second dimension can be of arbitrary size as long as the
            x, y, z values represent the first 3 values of this
            dimension and are provided in the right order.
        transformation: A homogeneous transformation matrix
            with shape (4, 4).

    Returns:
        boxes: Transformed point cloud given as numerical
            array of shape (N, ...).
    """
    # Get number of boxes
    N = points.shape[0]

    # Convert point coordinates to homogeneous coordinates of shape (M, 4)
    coord = np.column_stack((points[:, :3], np.ones(N)))

    # Transform point coordinates (pointwise dot product)
    points[:, :3] = np.einsum('ij,...j->...i', transformation, coord)[:, :3]

    return points
