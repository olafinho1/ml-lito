#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements connection between our pipeline (based on open3d convensions)
# and blender pipeline.

import typing as T

import numpy as np

# should not import libraries like torch that are not available in blender.


def convert_blender_camera_to_open3d(
    H_c2w: np.ndarray,
    intrinsic: np.ndarray,
    width_px: int,
    height_px: int,
):
    """
    Convert blender camera pose (H_c2w)
    Args:
        H_c2w:
            (*, 4, 4) blender camera pose.
            Right-handed:  x to right of image, y to top of image, z to us.
            However, the camera looks at -z_c and the up of the image is +y_c.
            In other words, the origin of the camera coordinate is at the bottom left of the image corner.
            Even though the camera looks at -z_c, the depth map (z_c) returned by the renderer
            is always positive
        intrinsic:
            (*, 3, 3) blender camera intrinsic
            Note the camera coordinate origin is at the bottom left corner for blender (see above).

    Returns:
        H_c2w:
            (*, 4, 4) open3d camera pose.
            Right-handed:  x to right of image, y to bottom of image, z to far.
            In other words, the origin of the camera coordinate is at the top left of the image corner.
            The camera looks toward +z_c, which is different from blender camera convention.
        intrinsic:
            (*, 3, 3) open3d camera intrinsic
            Note the camera coordinate origin is at the bottom left corner for blender (see above).
    """
    # Ensure input matrices are numpy arrays
    H_c2w = np.asarray(H_c2w)
    intrinsic = np.asarray(intrinsic)

    # Conversion matrix to account for the coordinate system differences
    # Blender -> Open3D: flip y and z axes
    conversion_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )  # (*, 4, 4)
    conversion_matrix = np.broadcast_to(conversion_matrix, H_c2w.shape)  # (*, 4, 4)

    # Convert the camera pose
    H_c2w_o3d = H_c2w @ conversion_matrix  # (*, 4, 4)

    # Adjust the intrinsic matrix for the new coordinate system
    intrinsic_o3d = intrinsic.copy()
    intrinsic_o3d[..., 1, 2] = height_px - intrinsic[..., 1, 2]

    return dict(
        H_c2w=H_c2w_o3d,
        intrinsic=intrinsic_o3d,
        width_px=width_px,
        height_px=height_px,
    )


def convert_open3d_camera_to_blender_H_c2w(
    H_c2w: np.ndarray,
):
    """
    Convert Open3D camera pose (H_c2w) to Blender format.

    Args:
        H_c2w: (*, 4, 4) Open3D camera pose.

    Returns:
        H_c2w: (*, 4, 4) Blender camera pose.
    """
    # Ensure input matrices are numpy arrays
    H_c2w = np.asarray(H_c2w)

    # Conversion matrix to account for the coordinate system differences
    # Open3D -> Blender: flip y and z axes
    conversion_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=H_c2w.dtype,
    )  # (4, 4)
    conversion_matrix = np.broadcast_to(conversion_matrix, H_c2w.shape)  # (*, 4, 4)

    # Convert the camera pose
    H_c2w_blender = H_c2w @ conversion_matrix

    return H_c2w_blender


def convert_blender_camera_to_open3d_H_c2w(H_c2w: np.ndarray):
    # symmetric to convert_open3d_camera_to_blender_H_c2w
    return convert_open3d_camera_to_blender_H_c2w(H_c2w)


def convert_open3d_camera_to_blender_intrinsic(
    intrinsic: np.ndarray,
    height_px: int,
):
    """
    Convert Open3D camera intrinsic matrix to Blender format.

    Args:
        intrinsic:
            (*, 3, 3) Open3D camera intrinsic matrix.

    Returns:
        intrinsic:
            (*, 3, 3) Blender camera intrinsic matrix.
    """
    assert intrinsic.shape[-2:] == (3, 3), f"{intrinsic.shape=}"

    # Adjust the intrinsic matrix for the new coordinate system
    intrinsic_blender = intrinsic.copy()
    intrinsic_blender[..., 1, 2] = height_px - intrinsic[..., 1, 2]
    return intrinsic_blender


def convert_blender_camera_to_open3d_intrinsic(
    intrinsic: np.ndarray,
    height_px: int,
):
    """
    Convert Blender camera intrinsic matrix to open3d format.

    Args:
        intrinsic:
            (*, 3, 3) Blender camera intrinsic matrix.

    Returns:
        intrinsic:
            (*, 3, 3) Open3d camera intrinsic matrix.
    """
    # symmetric to convert_open3d_camera_to_blender_intrinsic
    return convert_open3d_camera_to_blender_intrinsic(intrinsic, height_px)


def convert_open3d_camera_to_blender(
    H_c2w: np.ndarray,
    intrinsic: np.ndarray,
    width_px: int,
    height_px: int,
):
    """
    Convert Open3D camera pose (H_c2w) and intrinsic matrix to Blender format.

    Args:
        H_c2w:
            (*, 4, 4) Open3D camera pose.
        intrinsic:
            (*, 3, 3) Open3D camera intrinsic matrix.

    Returns:
        H_c2w:
            (*, 4, 4) Blender camera pose.
        intrinsic:
            (*, 3, 3) Blender camera intrinsic matrix.
    """
    # Ensure input matrices are numpy arrays
    H_c2w = np.asarray(H_c2w)
    intrinsic = np.asarray(intrinsic)
    assert H_c2w.shape[:-3] == intrinsic.shape[:-3]

    H_c2w_blender = convert_open3d_camera_to_blender_H_c2w(H_c2w)

    intrinsic_blender = convert_open3d_camera_to_blender_intrinsic(intrinsic, height_px)

    return dict(
        H_c2w=H_c2w_blender,
        intrinsic=intrinsic_blender,
        width_px=width_px,
        height_px=height_px,
    )


def convert_open3d_H_c2w_to_blender(
    H_c2w: np.ndarray,
):
    """
    Convert Open3D camera pose (H_c2w) to Blender format.

    Args:
        H_c2w: (*, 4, 4) Open3D camera pose.

    Returns:
        H_c2w: (*, 4, 4) Blender camera pose.
    """
    return convert_open3d_camera_to_blender_H_c2w(H_c2w=H_c2w)
