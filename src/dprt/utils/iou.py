import torch
import torch.nn.functional as F

from pytorch3d.ops import box3d_overlap

from dprt.utils import bbox


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    # Define box planes
    _box_planes = [
        [0, 1, 2, 3],
        [3, 2, 6, 7],
        [0, 1, 5, 4],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
        [4, 5, 6, 7],
    ]

    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)

    return (mat1.bmm(mat2).abs() < eps).squeeze()


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    # Define box triangles
    _box_triangles = [
        [0, 1, 2],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [1, 5, 6],
        [1, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [3, 2, 6],
        [3, 6, 7],
        [0, 1, 5],
        [0, 4, 5],
    ]

    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    return torch.all((face_areas > eps), dim=1)


def iou3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Returns the intersection over union between the given boxes.

    Backward is not supported.

    Arguments:
        boxes1: Bounding box corners with shape (B, N, 8, 3).
        boxes2: Bounding box corners with shape (B, M, 8, 3).

    Returns:
        iou: Intersection over union with shape (B, N, M)
    """
    # Get input shapes: B, N, and M
    B = boxes1.shape[0]
    N = boxes1.shape[1]
    M = boxes2.shape[1]

    # Flatten inputs (B, N, 8, 3) -> (B * N, 8, 3)
    boxes1 = boxes1.flatten(0, 1)
    boxes2 = boxes2.flatten(0, 1)

    # Initialize iou
    iou_3d = torch.zeros((B * N, B * M), dtype=boxes1.dtype, device=boxes1.device)

    # Check if inputs are empty
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return iou_3d.reshape((B, N, M))

    # Get mask for invalid boxes
    mask_1 = torch.logical_and(_check_nonzero(boxes1), _check_coplanar(boxes1))
    mask_2 = torch.logical_and(_check_nonzero(boxes2), _check_coplanar(boxes2))
    mask = torch.logical_and(*torch.meshgrid(mask_1, mask_2, indexing='ij'))

    # Check if inputs contain valid entries
    if not mask.any():
        return iou_3d.reshape((B, N, M))

    # Get intersection over union
    _, iou_3d_valid = box3d_overlap(boxes1[mask_1], boxes2[mask_2])

    # Insert valid iou values
    iou_3d[mask] = iou_3d_valid.flatten()

    # Reconstruct input shape
    iou_3d = iou_3d.reshape((B, N, M))

    return iou_3d


def giou3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Returns the generalized intersection over union between boxes.

    The Generalized Intersection over Union is given as
    GIoU = |A n B| / |A u B| - |C / (A u B)| / |C| =
           IoU - |C / (A u B)| / |C|

    with boxes A and B as well as thier minumin enclosing box C.

    Reference: https://giou.stanford.edu/

    Note: Backward is not supported.

    Arguments:
        boxes1: Bounding box corners with shape (B, N, 8, 3).
        boxes2: Bounding box corners with shape (B, M, 8, 3).

    Returns:
        giou: Generalized intersection over union with shape (B, N, M)
    """
    # Get input shapes: B, N, and M
    B = boxes1.shape[0]
    N = boxes1.shape[1]
    M = boxes2.shape[1]

    # Get minimal enclosing boxes (B, N, M, 8, 3)
    C = bbox.get_minimum_enclosing_box_corners(boxes1, boxes2)

    # Flatten inputs (B, N, 8, 3) -> (B * N, 8, 3)
    boxes1 = boxes1.flatten(0, 1)
    boxes2 = boxes2.flatten(0, 1)

    # Flatten minimal enclosing boxes (B, N, M, 8, 3) -> (B * N * M, 8, 3)
    C = C.flatten(0, 2)

    # Initialize iou, volume and union
    iou_3d = torch.zeros((B * N, B * M), dtype=boxes1.dtype, device=boxes1.device)
    vol_3d = torch.zeros((B * N, B * M), dtype=boxes1.dtype, device=boxes1.device)
    uni_3d = torch.zeros((B * N, B * M), dtype=boxes1.dtype, device=boxes1.device)

    # Initialize enclosing volume
    evol_3d = -torch.ones((B * N, B * M), dtype=boxes1.dtype, device=boxes1.device)

    # Check if inputs are empty
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return evol_3d.reshape((B, N, M))

    # Get mask for invalid boxes
    mask_1 = torch.logical_and(_check_nonzero(boxes1), _check_coplanar(boxes1))
    mask_2 = torch.logical_and(_check_nonzero(boxes2), _check_coplanar(boxes2))
    mask = torch.logical_and(*torch.meshgrid(mask_1, mask_2, indexing='ij'))

    # Check if inputs contain valid entries
    if not mask.any():
        return evol_3d.reshape((B, N, M))

    # Get intersection over union
    vol_3d_valid, iou_3d_valid = box3d_overlap(boxes1[mask_1], boxes2[mask_2])

    # Insert valid values
    iou_3d[mask] = iou_3d_valid.flatten()
    vol_3d[mask] = vol_3d_valid.flatten()

    # Calculate union
    zero_mask = (iou_3d != 0)
    uni_3d[zero_mask] = vol_3d[zero_mask] / iou_3d[zero_mask]

    # Reconstruct input shape
    iou_3d = iou_3d.reshape((B, N, M))
    vol_3d = vol_3d.reshape((B, N, M))
    uni_3d = uni_3d.reshape((B, N, M))

    # Calculate enclosing volume
    evol_3d_valid = bbox.get_box_volume_from_corners(C[mask.flatten()])

    # Insert valid values
    evol_3d[mask] = evol_3d_valid

    # Reconstruct input shape
    evol_3d = evol_3d.reshape((B, N, M))

    # Initialize giou
    giou = torch.zeros((B, N, M), dtype=boxes1.dtype, device=boxes1.device)

    # Calculate giou
    zero_mask = (evol_3d != 0)
    giou[zero_mask] = \
        iou_3d[zero_mask] - (evol_3d[zero_mask] - uni_3d[zero_mask]) / evol_3d[zero_mask]

    return giou
