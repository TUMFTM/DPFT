from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

from dprt.utils.geometry import get_box_corners, get_transformation
from dprt.utils.project import cart2spher, polar2cart, spher2cart


# Define TUM color map
TUMCM = LinearSegmentedColormap.from_list(
    'tum', [[0.0, 0.2, 0.34901960784313724], [1.0, 1.0, 1.0]], N=100
)


def get_tum_accent_cm() -> Colormap:
    return ListedColormap(np.array([
        [162, 173, 0],
        [227, 114, 34],
        [152, 198, 234],
        [218, 215, 203]
    ]) / 255)


def scalar2rgba(scalars: np.ndarray, cm: Colormap = None, norm: bool = True) -> np.ndarray:
    """Returns RGBA values for scalar input values accoring to a colormap.

    Arguments:
        scalars: Scalar values with shape (n,) to map to
            RGBA values.
        cm: Colormap to map the scalar values to.
        norm: Whether to normalize the scalar values.

    Returns:
        rgba: Red, green, blue, alpha values with
            shape (n, 4) for all scalar values.
    """
    # Get data normalization function
    if norm:
        norm = Normalize(vmin=np.min(scalars), vmax=np.max(scalars), clip=True)
    else:
        norm = None

    # Define color map
    mapper = ScalarMappable(norm=norm, cmap=cm)

    # Map scalars to rgba values
    rgba = mapper.to_rgba(scalars.flatten())

    return rgba


def visu_camera_data(img, dst: str = None) -> None:
    """Visualizes a given image.

    Arguments:
        img: Image data with shape (w, h, 3).
        dst: Destination filename to save the figure
            to. If provided, the figure will not be
            displayed but only saved to file.
    """
    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot image to axes
    ax.imshow(img)

    if dst is not None:
        fig.savefig(dst)
    else:
        fig.show()


def visu_lidar_data(pc: np.ndarray,
                    boxes: np.ndarray = None,
                    xlim: Tuple[float, float] = None,
                    ylim: Tuple[float, float] = None,
                    cm: Colormap = None) -> None:
    """Visualizes a given lidar point cloud.

    Arguments:
        pc: Lidar point cloud with shape (N, ...).
            The second dimension of the point cloud
            can be of arbitrary size as long as the
            first 4 values represent the x, y, z and
            intensity values.
        boxes: Bounding boxes given as numerical array of shape (M, ...).
            The second dimension can be of arbitrary size as long as the
            x, y, z, theta, l, w, h, class values represent the first 8
            values of this dimension and are provided in the right order.
        xlim: Tuple (min, max) of x-coordinate limits
            to restrict the point cloud to.
        ylim: Tuple (min, max) of y-coordinate limits
            to restrict the point cloud to.
        cm: Color map to visualize the point
            cloud values.
    """
    # Get colormap
    if cm is None:
        cm = TUMCM

    # Limit the point cloud
    if xlim is not None:
        pc = pc[np.logical_and(pc[:, 0] > xlim[0], pc[:, 0] < xlim[1])]

    if ylim is not None:
        pc = pc[np.logical_and(pc[:, 1] > ylim[0], pc[:, 1] < ylim[1])]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    # Get colors based on intensity
    rgb = scalar2rgba(pc[:, 3], cm=cm, norm=True)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Create a visualization object and window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Display lidar point cloud
    vis.add_geometry(pcd)

    # Add bounding boxes
    if boxes is not None:
        cm = get_tum_accent_cm()

        for box in boxes:
            # Create bounding box
            bbox = o3d.geometry.OrientedBoundingBox()
            bbox.center = box[:3]
            bbox.extent = box[4:7]
            bbox.R = get_transformation(
                rotation=np.array([0.0, 0.0, box[3]]), degrees=True
            )[:3, :3]
            bbox.color = np.asarray(cm(int(box[7])))[:3]

            # Visualize bounding box
            vis.add_geometry(bbox)

    # Block thread until window is closed
    vis.run()


def visu_2d_lidar_points(ax: plt.Axes,
                         points: np.ndarray,
                         dims: Tuple[int, int],
                         roi: Tuple[float, float, float, float] = None,
                         cart: bool = True,
                         r_max: float = None,
                         flip: bool = True) -> None:
    """Visualizes the given points on the specified axis.

    Arguments:
        ax: Axis to visualize the points on.
        points: Points with shape (M, ...). The second
            dimension can be of arbitrary size as long as
            the first 4 values represent the x, y, z,
            intensity values.
        roi: Region of interest given as min and max
            values for both dimensions. The order is
            (min1, max1, min2, max2).
        cart: Wheter to visualize the provided points in
            cartesian coordinates or polar coordinates.
    """
    # Remove everything except x, y, z and intensity
    points = points[:, :4]

    # Limit point cloud to a given region of interest
    if roi is not None:
        points[:, 0], points[:, 1], points[:, 2] = \
            cart2spher(points[:, 0], points[:, 1], points[:, 2], degrees=True)
        points[:, 0] = np.ones_like(points[:, 0]) * r_max if r_max is not None else points[:, 0]
        points = points[np.logical_and(points[:, dims[0]] > roi[0], points[:, dims[0]] < roi[1])]
        points = points[np.logical_and(points[:, dims[1]] > roi[2], points[:, dims[1]] < roi[3])]
        points[:, 0], points[:, 1], points[:, 2] = \
            spher2cart(points[:, 0], points[:, 1], points[:, 2], degrees=True)

    if not cart:
        points[:, 0], points[:, 1], points[:, 2] = \
            cart2spher(points[:, 0], points[:, 1], points[:, 2], degrees=True)
        points[:, 0] = np.ones_like(points[:, 0]) * r_max if r_max is not None else points[:, 0]

    if flip:
        ax.scatter(points[:, dims[0]], points[:, dims[1]], s=0.2, c='black')
    else:
        points[:, dims[0]] *= -1
        ax.scatter(points[:, dims[0]], points[:, dims[1]], s=0.2, c='black')


def visu_3d_radar_data(cube: np.ndarray,
                       dims: str,
                       raster: List[np.ndarray] = None,
                       cart: bool = False,
                       cm: Colormap = None,
                       **kwargs) -> None:
    """Visualizes a given 3D radar cube.

    Arguments:
        cube: 3D cube representing a section of the 4D radar
            tesseract with shape (N, M, K).
        raster: Rasterization values (grid points) of
            of the associated radar dimensions with
            shape (N, M, K).
        cart: Wheter to project the provided
            grid values from polar to cartesian
            coordinates. If True, the second
            raster dimension has to represent the
            angular values.
        cm: Color map to visualize the gird values.
        jv: Whether to use a jupyter visualizer for
            visualizations within a jupyter notebook.
    """
    # Check input dimensions
    if cart and dims != 'rae':
        raise ValueError(
            f"A cartesian transformation is only possible "
            f"if the data is provided in a 'rae' order. "
            f"However, the data was given as {dims}."
        )

    # Mesh grid based on sensor specifications
    if raster is not None:
        x, y, z = np.meshgrid(raster[0], raster[1], raster[2], indexing='ij')
    else:
        x, y, z = np.meshgrid(np.arange(cube.shape[0]), np.arange(cube.shape[1]),
                              np.arange(cube.shape[2]), indexing='ij')

    # Convert spherical to cartesian coordinate values
    if cart:
        x_shape, y_shape, z_shape = x.shape, y.shape, z.shape
        x, y, z = spher2cart(x.flatten('F'), y.flatten('F'), z.flatten('F'), degrees=True)
        x = x.reshape(x_shape, order='F')
        y = y.reshape(y_shape, order='F')
        z = z.reshape(z_shape, order='F')

    # Create point cloud representing the voxel center points
    xyz = np.zeros((np.size(x), 3))
    xyz[:, 0] = np.reshape(x, -1)
    xyz[:, 1] = np.reshape(y, -1)
    xyz[:, 2] = np.reshape(z, -1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Get radar rcs values
    rcs = 10 * np.log10(cube)

    # Map rcs values to color
    rgb = scalar2rgba(rcs, cm=cm, norm=True)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Create voxel grid from grid points
    # voxel_size = np.min([np.min(np.diff(r)) for r in raster])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)

    # Visualize 3d radar cube
    o3d.visualization.draw_geometries([voxel_grid])


def visu_2d_boxes(ax: plt.Axes,
                  boxes: np.ndarray,
                  dims: Tuple[int, int],
                  cart: bool = True,
                  r_max: float = None,
                  flip: bool = False) -> None:
    """

    Order of the 2D bounding box corners.
        3------2
        |      |
        |      |
        0------1

    """
    # Get number of boxes
    M = boxes.shape[0]
    dims = sorted(list(dims))

    # Get 3D box corners form bounidng boxes
    corners3d = get_box_corners(boxes)

    # Project bounding boxes to 2D (but maintain the third coordinate for transformation)
    if 0 in dims:
        corners2d = corners3d[:, :4, :]
    else:
        corners2d = np.zeros((M, 4, 3))
        corners2d[:, 0, :] = \
            corners3d[np.arange(M), np.argmin(corners3d[:, :4, dims[0]], axis=-1), :]
        corners2d[:, 1, :] = \
            corners3d[np.arange(M), np.argmax(corners3d[:, :4, dims[0]], axis=-1), :]
        corners2d[:, 2, :] = \
            corners3d[np.arange(M), 4 + np.argmax(corners3d[:, -4:, dims[0]], axis=-1), :]
        corners2d[:, 3, :] = \
            corners3d[np.arange(M), 4 + np.argmin(corners3d[:, -4:, dims[0]], axis=-1), :]

    if flip:
        corners2d[:, :, 1] *= -1

    # Define box edges with shape (boxes, edges, dimensions, resolution)
    res = 50
    edges = np.zeros((M, 4, 3, res))
    for i in range(4):
        edges[:, i, 0, :] = \
            np.linspace(corners2d[:, i % 4, 0], corners2d[:, (i + 1) % 4, 0], num=res).T
        edges[:, i, 1, :] = \
            np.linspace(corners2d[:, i % 4, 1], corners2d[:, (i + 1) % 4, 1], num=res).T
        edges[:, i, 2, :] = \
            np.linspace(corners2d[:, i % 4, 2], corners2d[:, (i + 1) % 4, 2], num=res).T

    # Convert cartesian box edges to spherical box edges
    for i in range(4):
        x = edges[:, i, 0, :].flatten()
        y = edges[:, i, 1, :].flatten()
        z = edges[:, i, 2, :].flatten()
        r, phi, roh = cart2spher(x, y, z, degrees=True)
        r = np.ones_like(r) * r_max if r_max is not None else r
        edges[:, i, 0, :] = r.reshape((M, res))
        edges[:, i, 1, :] = phi.reshape((M, res))
        edges[:, i, 2, :] = roh.reshape((M, res))

    if cart:
        for i in range(4):
            r = edges[:, i, 0, :].flatten()
            phi = edges[:, i, 1, :].flatten()
            roh = edges[:, i, 2, :].flatten()
            x, y, z = spher2cart(r, phi, roh, degrees=True)
            edges[:, i, 0, :] = x.reshape((M, res))
            edges[:, i, 1, :] = y.reshape((M, res))
            edges[:, i, 2, :] = z.reshape((M, res))

    # Get colormap
    cm = get_tum_accent_cm()

    # Plot boxes
    for box_edges, box in zip(edges, boxes):
        for edge in box_edges:
            if flip:
                ax.plot(edge[dims[1], :], edge[dims[0], :], color=cm(int(box[-2])))
            else:
                ax.plot(edge[dims[0], :], edge[dims[1], :], color=cm(int(box[-2])))


def visu_2d_radar_grid(ax: plt.Axes,
                       grid: np.ndarray,
                       raster: List[np.ndarray] = None,
                       cart: bool = False,
                       dims: str = 'ra',
                       r_max: float = 1.0,
                       cm: Colormap = None,
                       flip: bool = False):
    """
    """
    # Swap axis
    if flip:
        grid = grid.T
        raster = list(reversed(raster))

    # Mesh grid based on sensor specifications
    if raster is not None:
        x_mesh, y_mesh = np.meshgrid(raster[0], raster[1])
    else:
        x_mesh, y_mesh = np.meshgrid(np.arange(grid.shape[0]+1), np.arange(grid.shape[1]+1))

    if cart and dims in {'ra', 'ar'}:
        # Convert polar to cartesian coordinate values
        x_shape, y_shape = x_mesh.shape, y_mesh.shape
        x_mesh, y_mesh = polar2cart(x_mesh.flatten(), y_mesh.flatten(), degrees=True)
        x_mesh, y_mesh = x_mesh.reshape(x_shape), y_mesh.reshape(y_shape)

    if cart and dims in {'ae', 'ea'}:
        # Convert spherical to cartesian coordinate values
        x_shape, y_shape = x_mesh.shape, y_mesh.shape
        _, y_mesh, x_mesh = spher2cart(
            np.ones_like(x_mesh).flatten() * r_max,
            y_mesh.flatten(),
            x_mesh.flatten(),
            degrees=True
        )
        x_mesh, y_mesh = x_mesh.reshape(x_shape), y_mesh.reshape(y_shape)

    # Get radar rcs values (dB)
    rcs = 10 * np.log10(grid)

    # Plot radar data
    if flip:
        y_mesh *= -1
        p = ax.pcolormesh(y_mesh, x_mesh, rcs.T, cmap=cm, shading='nearest')
    else:
        p = ax.pcolormesh(x_mesh, y_mesh, rcs.T, cmap=cm, shading='nearest')

    # Add colorbar to the plot
    plt.colorbar(p, ax=ax, label='Power in dB')


def visu_2d_radar_data(grid: np.ndarray,
                       dims: str,
                       boxes: np.ndarray = None,
                       points: np.ndarray = None,
                       raster: List[np.ndarray] = None,
                       roi: bool = True,
                       label: Tuple[str, str] = None,
                       cart: bool = False,
                       r_max: float = 1.0,
                       cm: Colormap = None,
                       dst: str = None,
                       **kwargs) -> None:
    """Visualizes a given 2D radar grid.

    Arguments:
        grid: 2D grid representing a slice of the 4D radar
            tesseract with shape (N, M).
        dims:
        boxes:
        points:
        raster: Rasterization values (grid points) of
            of the associated radar dimensions with
            shape (N, M).
        roi:
        label: Description of the provided radar
            dimensions.
        cart: Wheter to project the provided
            grid values from polar to cartesian
            coordinates. If True, the second
            raster dimension has to represent the
            angular values.
        cm: Color map to visualize the gird values.
        dst: Destination filename to save the figure
            to. If provided, the figure will not be
            displayed but only saved to file.
    """
    # Check input data
    valid_dims = {'ra', 'ar', 'ae', 'ea'}
    if cart and dims not in valid_dims:
        raise ValueError(
            f"It is only possible to visualize projections "
            f"of spatial and non perpendicular dimensions. "
            f"Therefore, you can only visualize the "
            f"{valid_dims} dimensions but {dims} was given."
        )
    # Initialize parameters
    flip = False
    dims_to_xyz = {'r': 0, 'a': 1, 'e': 2}
    xyz = tuple((dims_to_xyz[d] for d in dims))

    # Adjust parameters
    if dims in {'ar', 'ea'}:
        flip = True

    if 'e' not in dims:
        r_max = None

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cm = cm if cm is not None else 'viridis'

    # Visualize 2D radar grid
    visu_2d_radar_grid(ax=ax, grid=grid, raster=raster, cart=cart,
                       dims=dims, r_max=r_max, cm=cm, flip=flip)

    if roi:
        roi = (np.min(raster[0]), np.max(raster[0]),
               np.min(raster[1]), np.max(raster[1]))

    # Visualize 2D point cloud
    if points is not None:
        visu_2d_lidar_points(ax, points, dims=xyz, roi=roi,
                             cart=cart, r_max=r_max, flip=not flip)

    # Visualize 2D bounding boxes
    if boxes is not None:
        visu_2d_boxes(ax, boxes, dims=xyz, cart=cart, r_max=r_max, flip=flip)

    # Add axis label
    if label is not None:
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])

    # Set equal ax ratios
    ax.axis('equal')

    if dst is not None:
        fig.savefig(dst)
    else:
        fig.show()


def visu_radar_tesseract(tesseract: np.ndarray,
                         dims: str,
                         raster: Dict[str, np.ndarray] = None,
                         aggregation_func: Callable = np.max,
                         **kwargs) -> None:
    """Visualizes the specified dimensions of a given radar tesseract.

    Arguments:
        tesseract: Data of the 4D radar tesseract with shape
            (doppler, range, elevation, azimuth).
        dims: Dimensions to visualize. Can be any combination of
            either two or three dimensions expressed by thier
            abbriviation (r: range, d: doppler, a: azimuth,
            e: elevation), e.g. 'ra'.
        raster: Dictionary specifying the rasterization values
            (grid points) of all radar dimensions.
        aggregation_func: Aggregation function for the
            reduction of the radar dimensions.
    """
    # Get maximum range
    r_max = max(raster['r'])

    # Get raster of the data distribution
    if raster is not None:
        raster = [raster[d] for d in dims]

    # Map dim abbreviations to data dimensions
    dim_order = {'d': 0, 'r': 1, 'e': 2, 'a': 3}
    dim_names = {'d': 'doppler', 'r': 'range', 'e': 'elevation', 'a': 'azimuth'}
    names = [dim_names[d] for d in dims]
    dim_idx = [dim_order[d] for d in dims]

    # Reduce radar data dimensions
    data = aggregation_func(
        tesseract,
        axis=tuple(set(dim_order.values()).difference(set(dim_idx)))
    )

    # rcs = tesseract[0, ...]
    # doppler = doppler_raster[np.argmax(tesseract, axis=0)]

    # Restructure data accoring to the given order in dims
    data = np.moveaxis(data, np.arange(data.ndim), np.argsort(dim_idx))

    # Select plot based on the number of dimensions
    if not 1 < len(dims) < 4:
        raise ValueError(
            f"There must be either two or three dimensions "
            f"selected for visualization but {len(dims)} "
            f"were given!"
        )

    # Visualize 3D radar data
    if len(dims) == 3:
        visu_3d_radar_data(cube=data, dims=dims, raster=raster, cm=TUMCM, **kwargs)

    # Visualize 2D radar data
    if len(dims) == 2:
        visu_2d_radar_data(grid=data, dims=dims, raster=raster,
                           r_max=r_max, label=names, cm=TUMCM, **kwargs)
