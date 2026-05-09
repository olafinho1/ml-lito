#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements util functions to use blender for rendering.

import argparse
import copy
import json
import math
import os
import pathlib
import pdb
import platform
import random
import re
import shutil
import sys
import typing as T
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set

import bpy
from mathutils import Matrix, Vector
import numpy as np
import numpy.typing as nptyping

# Add src/ to sys.path so `from blender_rendering import ...` works in Blender's Python
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent))
from blender_rendering import blender_open3d_utils
from plibs import json_utils

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    # "obj": bpy.ops.import_scene.obj,
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.wm.usd_import,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.wm.stl_import,
    "usda": bpy.ops.wm.usd_import,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.wm.ply_import,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.open_mainfile,
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene = bpy.context.scene
    scene.camera = new_camera
    scene.camera.data.sensor_fit = "HORIZONTAL"

    # set gamma for srgb
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.gamma = 1


def reset_scene(
    remove_light: bool = False,
    remove_camera: bool = False,
) -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """

    safe_types = set()
    if not remove_light:
        safe_types.add("LIGHT")
    if not remove_camera:
        safe_types.add("CAMERA")

    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in safe_types:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def reset_world():
    """Reset the world shading node."""
    # Access the world settings
    world = bpy.context.scene.world
    # Enable node-based shading for the world
    world.use_nodes = True
    # Access the world node tree
    node_tree = world.node_tree
    nodes = node_tree.nodes
    # Clear existing nodes
    nodes.clear()


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    context = bpy.context
    scene = context.scene

    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz (need https://github.com/robmcrosby/BlenderUSDZ)
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        print(f"object_path: {object_path}")
        print(f"import_function: {import_function}")
        assert os.path.exists(object_path)
        import_function(filepath=object_path, load_ui=False)

        # Step 2: Delete all cameras and lights
        bpy.ops.object.select_all(action="DESELECT")
        for obj in [o for o in bpy.data.objects if o.type in {"CAMERA", "LIGHT"}]:
            obj.select_set(True)
        bpy.ops.object.delete()

        # Step 3: Select all the remaining objects
        bpy.ops.object.select_all(action="SELECT")
        remaining_objects = list(bpy.context.selected_objects)

        # Step 4: Create an empty to act as the root
        # bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
        # root_empty = bpy.context.active_object
        # root_empty.name = "Root"

        root_empty = bpy.data.objects.new("Root", None)
        bpy.context.scene.collection.objects.link(root_empty)

        # get the root node of the remaining object
        remaining_root_objs = list({get_root_parent(obj) for obj in remaining_objects})
        print(f"remaining_root_objs: {remaining_root_objs}")

        # Step 5: Parent all remaining root objects to the root empty
        for obj in remaining_root_objs:
            obj.parent = root_empty

        # add a camera to the scene "Cemera"
        reset_cameras()

        # Deselect everything first
        bpy.ops.object.select_all(action="DESELECT")

        # Select the root empty
        root_empty.select_set(True)

        # Make the root empty the active object
        bpy.context.view_layer.objects.active = root_empty

    elif file_extension in {"glb", "gltf"}:
        import_function(
            filepath=object_path, merge_vertices=True, guess_original_bind_pose=False, bone_heuristic="TEMPERANCE"
        )
        # # CHECK THIS IS WORKING
        # bpy.ops.object.select_all(action="DESELECT")
        # for obj in [o for o in bpy.data.objects if o.type in {"LIGHT", "CAMERA", "EMPTY"}]:
        #     obj.select_set(True)
        # pdb.set_trace()
        # bpy.ops.object.delete()
        # bpy.ops.object.select_all(action="SELECT")
    else:
        import_function(filepath=object_path)


def delete_missing_textures() -> Dict[str, Any]:
    """
    Delete individual object's textures if their texture images cannot be found
    and replace the texture with a random color.
    This is to avoid undefined behavior in blender.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs["Base Color"].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def cut_outside_aabb_open(
    obj: bpy.types.Object,
    aabb_center: T.List[float] = (0.0, 0.0, 0.0),
    aabb_radius: T.List[float] = (1.0, 1.0, 1.0),
):
    """
    Cut a mesh to avoid going outside of the aabb bounding box.
    It leaves the mesh open after cut.

    Args:
        obj:
            the mesh to be cut
        aabb_center:
            (3,) center of the aabb
        aabb_radius:
            float, radius (half width) of the aabb
    """
    assert aabb_center is not None
    assert aabb_radius is not None

    def bisect_mesh(mesh_obj, plane_co, plane_no, clear_side):
        """
        Cut the mesh with a plane

        Args:
            mesh_obj:
            plane_co:
                (3,) a point on the plane
            plane_no:
                (3,) normal of the plane
            clear_side:
                'INNER': remove geometry behind inner
                'OUTER': remove geometry in front of the plane
        """
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")

        # Perform the bisect, choosing to clear either the inner or outer part
        bpy.ops.mesh.bisect(
            plane_co=plane_co,
            plane_no=plane_no,
            use_fill=False,
            clear_inner=(clear_side == "INNER"),
            clear_outer=(clear_side == "OUTER"),
            threshold=0.0,
        )

        bpy.ops.object.mode_set(mode="OBJECT")

        # Deselect all to clean up selection states
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.update()

    def clip_mesh(mesh_obj):
        if mesh_obj.type != "MESH":
            return
        directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for direction in directions:
            for factor in [-1.0, 1.0]:
                # Calculate plane position
                plane_co = [aabb_center[i] + aabb_radius[i] * factor * direction[i] for i in range(3)]
                plane_no = [direction[i] * (1 if factor > 0 else -1) for i in range(3)]
                # Determine which side to clear based on the direction and factor
                clear_side = "OUTER"  #  if factor > 0 else 'INNER'
                bisect_mesh(mesh_obj, plane_co, plane_no, clear_side)

    # Iterate over all objects in the scene and apply clipping to mesh objects
    child_objs = obj.children  # list
    if len(child_objs) == 0:
        clip_mesh(obj)
    else:
        for child in child_objs:
            cut_outside_aabb_open(
                obj=child,
                aabb_center=aabb_center,
                aabb_radius=aabb_radius,
            )


def cut_outside_aabb_close(
    obj: bpy.types.Object,
    aabb_center: T.List[float] = (0.0, 0.0, 0.0),
    aabb_radius: T.List[float] = (1.0, 1.0, 1.0),
):
    """
    Cut a mesh to avoid going outside of the aabb bounding box.
    It tries to close the mesh after cut.

    Args:
        obj:
            the mesh to be cut
        aabb_center:
            (3,) center of the aabb
        aabb_radius:
            float, radius (half width) of the aabb
    """
    assert aabb_center is not None
    assert aabb_radius is not None

    # Function to apply Boolean operation
    def clip_mesh(mesh_obj):
        if mesh_obj.type != "MESH":
            return mesh_obj

        # print(
        #     f'clipping {mesh_obj.name} to aabb centered at '
        #     f'{aabb_center} with radius {aabb_radius}', flush=True)

        # Create the clipping cube
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0.0, 0.0, 0.0))
        clipping_cube = bpy.context.object
        clipping_cube.name = "ClippingCube"
        clipping_cube.scale.x = 2 * aabb_radius[0]  # Scale along x-axis
        clipping_cube.scale.y = 2 * aabb_radius[1]  # Scale along x-axis
        clipping_cube.scale.z = 2 * aabb_radius[2]  # Scale along x-axis
        clipping_cube.location.x = aabb_center[0]  # Optional offset along x-axis
        clipping_cube.location.x = aabb_center[1]  # Optional offset along x-axis
        clipping_cube.location.x = aabb_center[2]  # Optional offset along x-axis

        # # the clipping happens at the object coordinate,
        # # so we need to transform the cube to the obj's coodinate
        # # Calculate the relative transformation
        # H_c2o = mesh_obj.matrix_world.inverted() @ clipping_cube.matrix_world
        # clipping_cube.matrix_world = mesh_obj.matrix_world @ H_c2o

        bool_modifier = mesh_obj.modifiers.new(name="ClipModifier", type="BOOLEAN")
        bool_modifier.operation = "INTERSECT"
        bool_modifier.object = clipping_cube
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)
        bpy.ops.object.modifier_apply(modifier=bool_modifier.name)

        # delete the cube if it's no longer needed
        bpy.data.objects.remove(clipping_cube, do_unlink=True)

        # Deselect all to clean up selection states
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.update()
        return mesh_obj

    # Iterate over all objects in the scene and apply clipping to mesh objects
    child_objs = obj.children  # list
    if len(child_objs) == 0:
        clip_mesh(obj)
    else:
        for child in child_objs:
            cut_outside_aabb_close(
                obj=child,
                aabb_center=aabb_center,
                aabb_radius=aabb_radius,
            )


def get_active_world_output_node():
    """
    Get the current active world output node, create one
    if not existed.

    Returns:
        the current active world output node
    """

    # Access the current scene's world
    world = bpy.context.scene.world

    # Ensure that nodes are enabled for the world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes

    # Retrieve all ShaderNodeOutputWorld nodes
    output_nodes = [node for node in nodes if node.type == "OUTPUT_WORLD"]

    if output_nodes:
        # Check if there is an active output node
        active_output_node = next((node for node in output_nodes if node.is_active_output), None)

        if active_output_node:
            pass
        else:
            # No active output node found. Setting the first one as active.
            output_nodes[0].is_active_output = True
            active_output_node = output_nodes[0]
    else:
        # No ShaderNodeOutputWorld nodes found. Creating one.
        # Create a new ShaderNodeOutputWorld node
        active_output_node = nodes.new(type="ShaderNodeOutputWorld")
        active_output_node.is_active_output = True

    return active_output_node


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_root_parent(obj: bpy.types.Object) -> bpy.types.Object:
    while obj.parent is not None:
        obj = obj.parent
    return obj


def get_bbox(
    obj: T.Optional[bpy.types.Object] = None,
) -> T.Tuple[Vector, Vector]:
    """
    Returns the bounding box of the obj in the world coordinate

    Args:
        obj:
            if None, compute the aabb bbox of the entire scene

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """

    bpy.context.view_layer.update()

    if obj is None:
        child_objs = bpy.context.scene.objects
    else:
        child_objs = obj.children  # list

    bbox_min_w = (math.inf,) * 3
    bbox_max_w = (-math.inf,) * 3

    if len(child_objs) == 0:
        if obj is not None:
            # single object and no children
            for coord in obj.bound_box:
                # coord is a corner of bbox in the obj coordinate
                coord = Vector(coord)
                coord = obj.matrix_world @ coord  # from object-space to world-space
                bbox_min_w = tuple(min(x, y) for x, y in zip(bbox_min_w, coord))
                bbox_max_w = tuple(max(x, y) for x, y in zip(bbox_max_w, coord))

            return Vector(bbox_min_w), Vector(bbox_max_w)
        else:
            # no object in the scene
            raise RuntimeError("no objects in scene to compute bounding box for")
    else:
        # go through the bbox of each child
        for child_obj in child_objs:
            child_bbox_min, child_bbox_max = get_bbox(obj=child_obj)
            bbox_min_w = tuple(min(x, y) for x, y in zip(bbox_min_w, child_bbox_min))
            bbox_max_w = tuple(max(x, y) for x, y in zip(bbox_max_w, child_bbox_max))
        return Vector(bbox_min_w), Vector(bbox_max_w)


def get_bbox_of_sequence(
    end_frame_idx: int,  # included
    obj=None,
    ignore_matrix=False,
    start_frame_idx: int = 1,
):
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min_w = (math.inf,) * 3
    bbox_max_w = (-math.inf,) * 3

    for i in range(start_frame_idx, end_frame_idx + 1):
        bpy.context.scene.frame_set(i)
        frame_bbox_min, frame_bbox_max = get_bbox(obj=obj)
        bbox_min_w = tuple(min(x, y) for x, y in zip(bbox_min_w, frame_bbox_min))
        bbox_max_w = tuple(max(x, y) for x, y in zip(bbox_max_w, frame_bbox_max))

    return Vector(bbox_min_w), Vector(bbox_max_w)


# def get_scene_meshes():
#     """Returns all meshes in the scene.

#     Yields:
#         Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
#     """
#     for obj in bpy.context.scene.objects.values():
#         if isinstance(obj.data, (bpy.types.Mesh)):
#             yield obj

# def get_bbox(
#     obj = None, ignore_matrix = False
# ):
#     """Returns the bounding box of the scene.

#     Taken from Shap-E rendering script
#     (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

#     Args:
#         single_obj (Optional[bpy.types.Object], optional): If not None, only computes
#             the bounding box for the given object. Defaults to None.
#         ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
#             to False.

#     Raises:
#         RuntimeError: If there are no objects in the scene.

#     Returns:
#         Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
#     """
#     bbox_min = (math.inf,) * 3
#     bbox_max = (-math.inf,) * 3
#     found = False
#     # for i in range(num_frames):
#     #     bpy.context.scene.frame_set(i * args.downsample)
#     for obj in get_scene_meshes() if obj is None else [obj]:
#         found = True
#         for coord in obj.bound_box:
#             coord = Vector(coord)
#             if not ignore_matrix:
#                 coord = obj.matrix_world @ coord
#             bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
#             bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

#     if not found:
#         raise RuntimeError("no objects in scene to compute bounding box for")

#     return Vector(bbox_min), Vector(bbox_max)


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_scale(obj: bpy.types.Object) -> np.ndarray:
    """
    Get the scale of each dimension of the obj.

    Args:
        obj:

    Returns:
        (3xyz,) scale for each dimension
    """
    scale = obj.scale
    return np.array([scale.x, scale.y, scale.z])


def get_H_c2w(
    obj: bpy.types.Object,
    type: str = "numpy",
) -> T.Union[Matrix, np.ndarray]:
    """
    Get the camera pose (H_c2w) from the blender camera object.
    Scale is not included.  Note that the returned H_c2w includes
    all the transformation in the parent object (coordinate systems).

    Args:
        obj:
            blender camera
        type:
            'numpy'
            'blender'

    Returns:
        H_c2w:
            (4, 4) camera coordinate and position in the world. The matrix
            maps 3d points in camera coordinate to the world coordinate.

    Note:
        (1) The function ignores the scaling.
        (2) The H_c2w is set as is. In other words, the camera-up direction is simply H_c2w[:3, 1].
        For example, if H_c2w is eye(4), the camera pinhole will be placed at (0, 0, 0),
        the x-axis of the camera will be (1, 0, 0) (right), y-axis of camera will be (0, 1, 0) (up), and
        z-axis of camera will (0, 0, 1) (to us). This is different from what we use in open3d where z is
        to far.
    """
    t_c2w, rotation_q, scale = obj.matrix_world.decompose()  # vector, quaternion, vector
    R_c2w = rotation_q.to_matrix()

    H_c2w = Matrix(
        [R_c2w[0][:] + (t_c2w[0],), R_c2w[1][:] + (t_c2w[1],), R_c2w[2][:] + (t_c2w[2],), [0.0, 0.0, 0.0, 1.0]]
    )

    if type == "numpy":
        H_c2w = np.array(H_c2w)
    return H_c2w


def set_H_c2w(
    obj: bpy.types.Object,
    H_c2w: np.ndarray,
    scale: np.ndarray = None,
):
    """
    Set the pose (rotation and translation) of a blender object.

    Args:
        obj:
            any blender object (but mostly camera)
        H_c2w:
            (4, 4) camera coordinates in the world coordinate.
            Scale is not included.
        scale:
            (3xyz,) the target scale for each axis in the object coordinate (ie, before transformed by H_c2w),
            If None, use the current scale.

    Note:
        (1) The function ignores the scaling.
        (2) The H_c2w is set as is. In other words, the camera-up direction is simply H_c2w[:3, 1].
        For example, if H_c2w is eye(4), the camera pinhole will be placed at (0, 0, 0),
        the x-axis of the camera will be (1, 0, 0) (right), y-axis of camera will be (0, 1, 0) (up), and
        z-axis of camer
        (3) It might be easier to use `set_look_at`
    """

    if scale is None:
        scale = get_scale(obj)  # (3,)

    H_c2w = H_c2w.copy()  # (4, 4)
    for i in range(3):
        H_c2w[:, i] *= scale[i]

    if isinstance(H_c2w, np.ndarray):
        H_c2w = Matrix(H_c2w)

    obj.matrix_world = H_c2w


def set_look_at(
    obj: bpy.types.Object,
    location: np.ndarray,
    look_at: np.ndarray,
    up_dir: np.ndarray = (0.0, 1.0, 0.0),
):
    """
    Set the pose of a blender object.

    Args:
        obj:
            any blender object (but mostly camera)
        location:
            (3,) the location of the object
        look_at:
            (3,) a 3d point where the object looks at (not the direction)
        up_dir:
            (3,) a suggested up direction
    """

    if isinstance(location, (list, tuple)):
        location = np.array(location)
    if isinstance(look_at, (list, tuple)):
        look_at = np.array(look_at)
    if isinstance(up_dir, (list, tuple)):
        up_dir = np.array(up_dir)

    # Calculate the forward, right, and up vectors
    forward_dir = look_at - location
    forward_norm = np.linalg.norm(forward_dir)
    if forward_norm > 1e-8:
        forward_dir /= forward_norm
    else:
        forward_dir = np.array([0, 0, -1])

    right_dir = np.cross(forward_dir, up_dir)
    right_norm = np.linalg.norm(right_dir)
    if right_norm > 1e-8:
        right_dir /= right_norm
    else:
        right_dir = np.array([1, 0, 0])

    true_up_dir = np.cross(right_dir, forward_dir)
    true_up_dir /= np.linalg.norm(true_up_dir)

    # Create the rotation matrix
    R_c2w = np.array(
        [
            right_dir,
            true_up_dir,
            -forward_dir,
        ]
    ).T

    H_c2w = np.eye(4)
    H_c2w[:3, :3] = R_c2w
    H_c2w[:3, 3] = location

    # set H_c2w
    set_H_c2w(obj=obj, H_c2w=H_c2w)


def get_camera_intrinsics(
    camera: bpy.types.Object,
) -> Dict[str, T.Any]:
    """
    Get the 3x3 camera intrinsics

    Args:
        camera:
            blender camera object

    Returns:
        focal_length_mm:
        sensor_width_mm:
        sensor_height_mm:
        pixel_size_x_mm:
            width
        pixel_size_y_mm:
            height
        intrinsic:
            (3, 3) intrinsic matrix

    Ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    """
    bpy.context.view_layer.update()

    assert camera.type == "CAMERA"
    cam_data = camera.data

    # Get focal length (in millimeters)
    focal_length_mm = cam_data.lens

    # Get sensor width and height (in millimeters)
    sensor_width_mm = cam_data.sensor_width
    sensor_height_mm = cam_data.sensor_height

    # print(f'sensor_width_mm: {sensor_width_mm}, sensor_height_mm : {sensor_height_mm}')

    # Get resolution and calculate the principal point
    scene = bpy.context.scene
    render = scene.render
    resolution_x = render.resolution_x
    resolution_y = render.resolution_y
    resolution_percentage = render.resolution_percentage / 100.0
    resolution_x_px = resolution_x * resolution_percentage
    resolution_y_px = resolution_y * resolution_percentage

    # print(f'resolution_x_px: {resolution_x_px}, resolution_y_px : {resolution_y_px}')

    # Calculate the intrinsic matrix K
    # Get the camera shift values (percentage, 1 means shift +x by 1 fov)
    shift_x = cam_data.shift_x
    shift_y = cam_data.shift_y

    # note that blender shift is based on sensor_width (x direction) or sensor_height
    if cam_data.sensor_fit == "HORIZONTAL":
        shift_ref_px = resolution_x_px
    elif cam_data.sensor_fit == "VERTICAL":
        shift_ref_px = resolution_y_px
    else:
        # not sure what
        raise NotImplementedError

    principal_x_px = resolution_x_px * 0.5 - shift_x * shift_ref_px
    principal_y_px = resolution_y_px * 0.5 - shift_y * shift_ref_px

    # Calculate the pixel size
    pixel_size_x_mm = sensor_width_mm / resolution_x_px
    pixel_size_y_mm = sensor_height_mm / resolution_y_px

    # print(f'pixel_size_x_mm: {pixel_size_x_mm}, pixel_size_y_mm : {pixel_size_y_mm}')

    # Focal length in pixels
    focal_length_px_x = focal_length_mm / pixel_size_x_mm
    focal_length_px_y = focal_length_mm / pixel_size_y_mm

    # Intrinsic matrix K
    intrinsic = np.array(
        [
            [focal_length_px_x, 0, principal_x_px],
            [0, focal_length_px_y, principal_y_px],
            [0, 0, 1],
        ]
    )  # (3, 3)

    return dict(
        intrinsic=intrinsic,  # (3, 3)
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        pixel_size_x_mm=pixel_size_x_mm,
        pixel_size_y_mm=pixel_size_y_mm,
        width_px=resolution_x_px,
        height_px=resolution_y_px,
    )


def set_camera_intrinsics(
    camera: bpy.types.Object,
    intrinsic: np.ndarray,
    width_px: int = None,
    height_px: int = None,
    scale: float = None,
):
    """
    Set the camera (and rendering) intrinsics.

    Args:
        camera:
             blender camera object
        intrinsic:
            (3, 3) intrinsic matrix
        width_px:
            the rendering resolution in pixel. None: use the current setting.
        height_px:
            the rendering resolution in pixel. None: use the current setting.
        scale:
            the rendering scale. None: use the current setting.
    """
    assert camera.type == "CAMERA"
    assert intrinsic.shape == (3, 3)

    # set the rendering resolution
    set_render_resolution(
        width_px=width_px,
        height_px=height_px,
        scale=scale,
    )

    cam_data = camera.data

    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]

    # Get resolution and sensor size
    scene = bpy.context.scene
    render = scene.render
    resolution_x = render.resolution_x  # px
    resolution_y = render.resolution_y  # px
    resolution_percentage = render.resolution_percentage / 100.0
    resolution_x_px = resolution_x * resolution_percentage  # actual rendering resolution
    resolution_y_px = resolution_y * resolution_percentage

    # Calculate pixel size
    sensor_width_mm = cam_data.sensor_width
    sensor_height_mm = cam_data.sensor_height
    pixel_size_x_mm = sensor_width_mm / resolution_x_px
    pixel_size_y_mm = sensor_height_mm / resolution_y_px

    # Convert focal lengths from pixels to millimeters
    focal_length_mm_x = f_x * pixel_size_x_mm
    focal_length_mm_y = f_y * pixel_size_y_mm

    # blender only support isometric focal length
    assert np.isclose(focal_length_mm_x, focal_length_mm_y)

    # Set the focal length (average of x and y for simplicity)
    cam_data.lens = (focal_length_mm_x + focal_length_mm_y) / 2

    # Calculate shift values
    # note that blender shift is based on sensor_width (x direction) or sensor_height (y)
    if cam_data.sensor_fit == "HORIZONTAL":
        shift_ref_px = resolution_x_px
    elif cam_data.sensor_fit == "VERTICAL":
        shift_ref_px = resolution_y_px
    else:
        # not sure what
        raise NotImplementedError
    shift_x = (resolution_x_px * 0.5 - c_x) / shift_ref_px
    shift_y = (resolution_y_px * 0.5 - c_y) / shift_ref_px

    # Apply shift values
    cam_data.shift_x = shift_x
    cam_data.shift_y = shift_y

    bpy.context.view_layer.update()
    idict = get_camera_intrinsics(camera)
    assert np.allclose(idict["intrinsic"], intrinsic), f"{idict['intrinsic']}, {intrinsic}"


def save_camera_info(camera: bpy.types.Object, filename: str):
    """
    Save camera H_c2w, intrinsic, and other info to a json file.

    Args:
        camera:
        filename:
            output json filename
    """

    idict = get_camera_intrinsics(camera=camera)
    H_c2w_blender = get_H_c2w(obj=camera)
    intrinsic_blender = idict["intrinsic"]
    odict = blender_open3d_utils.convert_blender_camera_to_open3d(
        H_c2w=H_c2w_blender,
        intrinsic=intrinsic_blender,
        width_px=idict["width_px"],
        height_px=idict["height_px"],
    )
    cdict = dict(
        H_c2w_blender=H_c2w_blender.tolist(),  # (4, 4)
        intrinsic_blender=intrinsic_blender.tolist(),
        H_c2w_open3d=odict["H_c2w"].tolist(),  # (4, 4)
        intrinsic_open3d=odict["intrinsic"].tolist(),
        width_px=idict["width_px"],
        height_px=idict["height_px"],
        focal_length_mm=idict["focal_length_mm"],
        sensor_width_mm=idict["sensor_width_mm"],
        sensor_height_mm=idict["sensor_height_mm"],
        pixel_size_x_mm=idict["pixel_size_x_mm"],
        pixel_size_y_mm=idict["pixel_size_y_mm"],
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(cdict, f, indent=2)
    return cdict


def set_camera_fov(
    camera: bpy.types.Object,
    fov: float,
    mode: str = "horizontal",
):
    """
    Set the camera horizontal (or vertical) field of view in degree.

    Args:
        camera:
            blender camera object
        fov:
            horizontal (or vertical) field of view in degree
        mode:
            'horizontal'
            'vertical'
    """
    assert camera.type == "CAMERA"
    cam_data = camera.data

    # Store the original sensor fit
    original_sensor_fit = cam_data.sensor_fit

    if mode.lower() == "horizontal":
        # Set the sensor fit to 'HORIZONTAL'
        cam_data.sensor_fit = "HORIZONTAL"

        # Set the horizontal field of view in radians
        cam_data.angle = math.radians(fov)
    elif mode.lower() == "vertical":
        # Set the sensor fit to 'VERTICAL'
        cam_data.sensor_fit = "VERTICAL"

        # Set the vertical field of view in radians
        cam_data.angle_y = math.radians(fov)
    else:
        raise NotImplementedError

    # Restore the original sensor fit
    cam_data.sensor_fit = original_sensor_fit


def set_render_resolution(
    width_px: int,
    height_px: int,
    scale: float = 1,
):
    """
    Set the rendering resolution to
    Args:
        width_px:
        height_px:
        scale:
            scale the resolution, 1 means rendering the same resolution as set,
            2 means rendering at twice the resolution (ie, 2 * height_px, 2 * width_px).
    """
    scene = bpy.context.scene
    render = scene.render
    if width_px is not None:
        render.resolution_x = width_px
    if height_px is not None:
        render.resolution_y = height_px
    if scale is not None:
        render.resolution_percentage = scale * 100

    # make sure pixel is square
    width_px = render.resolution_x
    height_px = render.resolution_y
    scene.camera.data.sensor_fit = "HORIZONTAL"
    sensor_width = 36.0  # mm scene.camera.data.sensor_width
    sensor_height = sensor_width * height_px / width_px
    scene.camera.data.sensor_width = sensor_width
    scene.camera.data.sensor_height = sensor_height


def setup_blender_cycles(
    resolution_x: int = None,
    resolution_y: int = None,
    # rgb_file_format: str = 'PNG',  # "OPEN_EXR"
    # rgb_bit_depth: int = 8,
    # depth_file_format: str = 'OPEN_EXR',
    # depth_bit_depth: int = 32,
    # normal_file_format: str = 'OPEN_EXR',  # "OPEN_EXR"
    # normal_bit_depth: int = 32,
    # albedo_file_format: str = 'PNG',  # "OPEN_EXR"
    # albedo_bit_depth: int = 8,
    # obj_id_file_format: str = 'PNG',  # "OPEN_EXR"
    # obj_id_bit_depth: int = 8,  # max number of objs = 2 ** bit_depth
    samples: int = 128,
    diffuse_bounces: int = 1,
    glossy_bounces: int = 1,
    transparent_max_bounces: int = 1,
    transmission_bounces: int = 3,
    filter_width: float = 0.01,  # px
    device: Literal["GPU", "CPU"] = "CPU",
):
    """
    Setup blender to render rgb, depth, and normal.

    Args:
        # rgb_file_format:
        #     file format to save the rgb image
        #     'PNG', 'OPEN_EXR', 'HDR'
        # rgb_bit_depth:
        #     for PNG: 8, 16
        #     for OPEN_EXR: 16, 32
        # depth_file_format:
        #     file format to save the rgb image
        #     'PNG', 'OPEN_EXR', 'HDR'
        # depth_bit_depth:
        #     for PNG: 8, 16
        #     for OPEN_EXR: 16, 32
        # normal_file_format:
        #     file format to save the rgb image
        #     'PNG', 'OPEN_EXR', 'HDR'
        # normal_bit_depth:
        #     for PNG: 8, 16
        #     for OPEN_EXR: 16, 32
        # albedo_file_format:
        #     file format to save the rgb image
        #     'PNG', 'OPEN_EXR', 'HDR'
        # albedo_bit_depth:
        #     for PNG: 8, 16
        #     for OPEN_EXR: 16, 32
        samples:
            int, number of samples per pixel
        diffuse_bounces:
            int, number of diffuse bounces
        glossy_bounces:
            int, number of glossy bounces
        transparent_max_bounces:
            int, number of transparent_max_bounces
        transmission_bounces:
            int, number of transmission bounces
        filter_width:
            float, anti-aliasing pixel fileter width

    Returns:
        rgb_file_output:  rgba openexr, raw, no gamma correction
        srgb_file_output:  rgba png, gamma corrected rgb
        depth_file_output:  bw, openexr, abs(z_c)
        normal_file_output: rgb openexr, surface normal in the world coordinate
        albedo_file_output: rgba openexr
        obj_id_file_output:
    """

    # if using png, blender performs gamma correction
    # use openexr if want to save the raw value
    rgb_file_format = "OPEN_EXR"
    rgb_bit_depth = 32
    srgb_file_format = "PNG"
    srgb_bit_depth = 8
    depth_file_format = "OPEN_EXR"
    depth_bit_depth = 32
    normal_file_format = "OPEN_EXR"
    normal_bit_depth = 32
    albedo_file_format = "OPEN_EXR"
    albedo_bit_depth = 32
    obj_id_file_format = "OPEN_EXR"
    obj_id_bit_depth = 32

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = "CYCLES"
    render.image_settings.file_format = rgb_file_format
    render.image_settings.color_mode = "RGBA"
    render.image_settings.color_depth = str(rgb_bit_depth)
    set_render_resolution(
        width_px=resolution_x,
        height_px=resolution_y,
        scale=1,
    )

    # Set cycles settings
    scene.cycles.samples = samples  # number of samples per pixel
    scene.cycles.diffuse_bounces = diffuse_bounces
    scene.cycles.glossy_bounces = glossy_bounces
    scene.cycles.transparent_max_bounces = transparent_max_bounces
    scene.cycles.transmission_bounces = transmission_bounces
    scene.cycles.film_exposure = 1
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = filter_width  # px
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    if platform.system() == "Darwin":
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "METAL"
        print(f"set cycles to use METAL")
    else:
        if (device is not None) and (device == "GPU"):
            try:
                scene.cycles.device = "GPU"
                bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"
                print(f"set cycles to use CUDA")
            except:
                print(f"failed to set to CUDA")

    bpy.context.view_layer.update()
    print(f"render engine: {bpy.context.scene.render.engine}")
    print(f"cycles device: {scene.cycles.device}")
    print(f"cycles uses {bpy.context.preferences.addons['cycles'].preferences.compute_device_type}")

    # world nodes
    world = bpy.context.scene.world
    world.use_nodes = True

    # scene nodes
    scene.use_nodes = True
    for view_layer in scene.view_layers:
        view_layer.use_pass_z = True
        view_layer.use_pass_normal = True
        view_layer.use_pass_diffuse_color = True
        view_layer.use_pass_object_index = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new("CompositorNodeRLayers")

    # Create rgb output nodes (raw/hdr)
    if rgb_file_format is not None and rgb_file_format.lower() != "none":
        rgb_file_output = nodes.new(type="CompositorNodeOutputFile")
        rgb_file_output.label = "RGB Output"
        rgb_file_output.base_path = ""  # not used
        rgb_file_output.file_slots[0].use_node_format = True
        rgb_file_output.format.file_format = rgb_file_format
        rgb_file_output.format.color_mode = "RGBA"
        rgb_file_output.format.color_depth = str(rgb_bit_depth)
        links.new(render_layers.outputs["Image"], rgb_file_output.inputs[0])
    else:
        rgb_file_output = None

    # Create srgb output nodes (gamma corrected)
    if srgb_file_format is not None and srgb_file_format.lower() != "none":
        srgb_file_output = nodes.new(type="CompositorNodeOutputFile")
        srgb_file_output.label = "sRGB Output"
        srgb_file_output.base_path = ""  # not used
        srgb_file_output.file_slots[0].use_node_format = True
        srgb_file_output.format.file_format = srgb_file_format
        srgb_file_output.format.color_mode = "RGBA"
        srgb_file_output.format.color_depth = str(srgb_bit_depth)
        links.new(render_layers.outputs["Image"], srgb_file_output.inputs[0])
    else:
        srgb_file_output = None

    # Create depth output nodes
    if depth_file_format is not None and depth_file_format.lower() != "none":
        depth_file_output = nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = "Depth Output"
        depth_file_output.base_path = ""  # not used
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = depth_file_format
        # depth_file_output.format.color_mode = "BW"
        depth_file_output.format.color_depth = str(depth_bit_depth)
        links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
    else:
        depth_file_output = None

    # Create normal output nodes
    if normal_file_format is not None and normal_file_format.lower() != "none":
        scale_node = nodes.new(type="CompositorNodeMixRGB")
        scale_node.blend_type = "MULTIPLY"
        # scale_node.use_alpha = True
        scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

        bias_node = nodes.new(type="CompositorNodeMixRGB")
        bias_node.blend_type = "ADD"
        # bias_node.use_alpha = True
        bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(scale_node.outputs[0], bias_node.inputs[1])

        normal_file_output = nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = "Normal Output"
        normal_file_output.base_path = ""  # not used
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = normal_file_format
        normal_file_output.format.color_mode = "RGB"
        normal_file_output.format.color_depth = str(normal_bit_depth)
        links.new(bias_node.outputs[0], normal_file_output.inputs[0])
    else:
        normal_file_output = None

    # Create albedo output nodes
    if albedo_file_format is not None and albedo_file_format.lower() != "none":
        alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
        links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
        links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

        albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
        albedo_file_output.label = "Albedo Output"
        albedo_file_output.base_path = ""  # not used
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = albedo_file_format
        albedo_file_output.format.color_mode = "RGBA"
        albedo_file_output.format.color_depth = str(albedo_bit_depth)
        links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])
    else:
        albedo_file_output = None

    # Create id map output nodes
    if obj_id_file_format is not None and obj_id_file_format.lower() != "none":
        id_file_output = nodes.new(type="CompositorNodeOutputFile")
        id_file_output.label = "ID Output"
        id_file_output.base_path = ""  # not used
        id_file_output.file_slots[0].use_node_format = True
        id_file_output.format.file_format = obj_id_file_format
        id_file_output.format.color_depth = str(obj_id_bit_depth)

        if obj_id_file_format == "OPEN_EXR":
            links.new(render_layers.outputs["IndexOB"], id_file_output.inputs[0])
        else:
            id_file_output.format.color_mode = "BW"
            divide_node = nodes.new(type="CompositorNodeMath")
            divide_node.operation = "DIVIDE"
            divide_node.use_clamp = False
            divide_node.inputs[1].default_value = 2 ** int(obj_id_bit_depth)
            links.new(render_layers.outputs["IndexOB"], divide_node.inputs[0])
            links.new(divide_node.outputs[0], id_file_output.inputs[0])
    else:
        id_file_output = None

    return dict(
        rgb_file_output=rgb_file_output,
        srgb_file_output=srgb_file_output,
        depth_file_output=depth_file_output,
        normal_file_output=normal_file_output,
        albedo_file_output=albedo_file_output,
        obj_id_file_output=id_file_output,
    )


def render(out_dir: str, output_handle_dict: T.Dict[str, T.Any], filename_prefix: str, frame=0, frame_start=0):
    """
    Render an image.

    Args:
        out_dir:
            dir to save all the outputs
        output_handle_dict:
            rgb_file_output:
            depth_file_output:
            normal_file_output:
            albedo_file_output:
            obj_id_file_output:
        filename_prefix:
            the prefix of the filename {out_dir}/{filename_prefix}_rgb.png

    Returns:
        rgb_filename:
            rgba exr file, no tone mapping or gamma correction
        srgb_filename:
            rgb png, with tone mapping / gamma correction
        depth_filename:
            exr 1channel abs(z_c),
        normal_filename:
            exr surface normal xyz_w in the world coordinate
        albedo_filename:
            exr
        obj_id_filename:
            exr
        camera_filename:
            json, 'H_c2w', intrinsic, width_px, height_px
    """

    # set the output filenames
    bpy.context.scene.frame_set(frame + frame_start)
    render_filename_dict = dict()
    filename_dict = dict()
    for key in ["rgb", "srgb", "depth", "normal", "albedo", "obj_id"]:
        h = output_handle_dict.get(f"{key}_file_output", None)
        if h is not None:
            assert h.type == "OUTPUT_FILE"
            if os.path.isabs(out_dir):
                h.base_path = "/"  # NOTE: this is important to ensure that we use absolute path.
            ext = get_file_extension(h)
            h.file_slots[0].path = os.path.join(out_dir, f"{filename_prefix}_{key}")
            render_filename_dict[key] = f"{h.file_slots[0].path}{frame + frame_start:04d}{ext}"
            filename_dict[key] = os.path.join(out_dir, f"{filename_prefix}_{key}{ext}")

    # render
    bpy.ops.render.render(write_still=True)
    # the rendered filename would be os.path.join(out_dir, f'{filename_prefix}_{key}0000.png')
    # move and rename to the original desired filenames
    for key in render_filename_dict:
        assert key in filename_dict, f"{key} not found in {filename_dict}"
        assert os.path.exists(render_filename_dict[key]), f"{render_filename_dict[key]} not found"
        shutil.move(render_filename_dict[key], filename_dict[key])

    # save the camera
    cam = bpy.context.scene.camera
    cam_filename = os.path.join(out_dir, f"{filename_prefix}_camera.json")
    save_camera_info(camera=cam, filename=cam_filename)
    filename_dict["camera"] = cam_filename

    return filename_dict


def get_file_extension(node) -> str:
    assert node.type == "OUTPUT_FILE"
    if node.format.file_format == "PNG":
        return ".png"
    elif node.format.file_format == "OPEN_EXR":
        return ".exr"
    else:
        raise NotImplementedError


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self,
        object_path: T.Optional[str],
        scene: bpy.types.Scene,
        bdata: bpy.types.BlendData,
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = image_filepaths | material_filepaths | linked_libraries_filepaths
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = get_bbox(obj=None)
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += len(shape_keys.key_blocks) - 1  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        if self.object_path is not None:
            return os.path.getsize(self.object_path)
        else:
            return -1

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            # "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }


def read_json_config(
    filename: str,
) -> T.Dict[str, T.Any]:
    """
    Read the json config file of scene, caemera, and lighting.

    Args:
        filename:

    Returns:
        mesh_dicts:
        camera_dicts:
        light_dicts:
    """

    if filename.endswith("json"):
        with open(filename, "r") as f:
            config = json.load(f)
    elif filename.endswith("yaml") or filename.endswith("yml"):
        import yaml

        with open(filename, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise NotImplementedError

    # read meshes
    if "meshes" in config:
        mesh_dicts = read_mesh(config["meshes"])
    else:
        mesh_dicts = []

    # read cameras
    camera_dicts = read_camera(config["cameras"])

    # read lightings
    light_dicts = read_lighting(config["lighting"])

    # read point cloud
    if "point_clouds" in config:
        pcd_dicts = read_point_clouds(config["point_clouds"])
    else:
        pcd_dicts = []

    # read point cloud
    if "planes" in config:
        plane_dicts = read_planes(config["planes"])
    else:
        plane_dicts = []

    return dict(
        mesh_dicts=mesh_dicts,
        camera_dicts=camera_dicts,
        light_dicts=light_dicts,
        pcd_dicts=pcd_dicts,
        plane_dicts=plane_dicts,
    )


def read_mesh(mesh_config: T.List[T.Dict[str, T.Any]]) -> T.List[T.Dict[str, T.Any]]:
    """
    Parse the mesh config from the json

    Args:
        mesh_config:
            name:  # can be arbitrary
            filename:  # filename of mesh1
            H_c2w:  # (4, 4) rotation and translation (no scale)
            scale:  # (3xyz,) or null
            normalize_first: (bool)

    Returns:
        a list containing config for each mesh
        name:
        filename:
        H_c2w:  (4, 4)
        scale:  (3,)
        normalize_first: bool
    """
    mesh_dicts = []
    for i in range(len(mesh_config)):
        mdict = dict()
        mdict["name"] = mesh_config[i]["name"]
        mdict["filename"] = mesh_config[i]["filename"]
        mdict["H_c2w"] = np.array(mesh_config[i]["H_c2w"])
        assert mdict["H_c2w"].shape == (4, 4)
        mdict["scale"] = mesh_config[i]["scale"]
        if mdict["scale"] is None:
            mdict["scale"] = [1, 1, 1.0]
        mdict["scale"] = np.array(mdict["scale"])
        assert mdict["scale"].shape == (3,)
        mdict["normalize_first"] = mesh_config[i]["normalize_first"]
        mdict["cut_aabb_center"] = mesh_config[i].get("cut_aabb_center", None)
        mdict["cut_aabb_radius"] = mesh_config[i].get("cut_aabb_radius", None)
        mesh_dicts.append(mdict)

    return mesh_dicts


def read_point_clouds(pcd_config: T.List[T.Dict[str, T.Any]]) -> T.List[T.Dict[str, T.Any]]:
    """
    Parse the mesh config from the json

    Args:
        pcd_config:
            name:  # can be arbitrary
            filename:  # filename of npz containing
                'xyz_w': (n, 3xyz_w)
                'rgba': (n, 4rgba)
                'scale_xyz_w':  (n, 3)  optional, radius for each axis
                'R_c2w': (n, 3, 3)  optional, rotation matrix
            radius:
                sphere size for each point
            metallic: float
            roughness: float
            refractive_index: float

    Returns:
        a list containing config for each point cloud
        name:
        filename:
        metallic: float
        roughness: float
        refractive_index: float
    """
    mesh_dicts = []
    for i in range(len(pcd_config)):
        mdict = dict()
        mdict["name"] = pcd_config[i]["name"]
        mdict["filename"] = pcd_config[i]["filename"]
        mdict["radius"] = pcd_config[i].get("radius", 0.01)
        mdict["metallic"] = pcd_config[i].get("metallic", 0.0)
        mdict["roughness"] = pcd_config[i].get("roughness", 0.5)
        mdict["refractive_index"] = pcd_config[i].get("refractive_index", 1.5)
        mesh_dicts.append(mdict)

    return mesh_dicts


def read_planes(plane_config: T.List[T.Dict[str, T.Any]]) -> T.List[T.Dict[str, T.Any]]:
    """
    Parse the plane config from the json

    Args:
        plane_config:
            name:  # can be arbitrary
            H_c2w:  # (4, 4) rotation and translation (no scale)
            length_x:
                float, full width in meter
            length_y:
                float, full height in meter
            rgba:
                (4rgba,)  [0, 1]
            metallic: float
            roughness: float
            refractive_index: float

    Returns:
        a list containing config for each point cloud
        name:
        H_c2w:
        length_x:
        length_y:
        rgba:
        metallic: float
        roughness: float
        refractive_index: float
    """
    mesh_dicts = []
    for i in range(len(plane_config)):
        mdict = dict()
        mdict["name"] = plane_config[i]["name"]
        mdict["H_c2w"] = np.array(plane_config[i]["H_c2w"])
        assert mdict["H_c2w"].shape == (4, 4)
        mdict["length_x"] = plane_config[i].get("length_x", 1.0)
        mdict["length_y"] = plane_config[i].get("length_y", 1.0)
        mdict["rgba"] = plane_config[i].get("rgba", (0.5, 0.5, 0.5, 1))
        mdict["metallic"] = plane_config[i].get("metallic", 0.0)
        mdict["roughness"] = plane_config[i].get("roughness", 0.5)
        mdict["refractive_index"] = plane_config[i].get("refractive_index", 1.5)
        mesh_dicts.append(mdict)

    return mesh_dicts


def read_camera(camera_config: T.List[T.Dict[str, T.Any]]) -> T.List[T.Dict[str, T.Any]]:
    """
    Parse the camera config from the json

    Args:
        camera_config:
            list of dict
                intrinsic:  # (3, 3)
                H_c2w:  # (4, 4)
                width_px:  # int
                height_px:  # int

    Returns:
        a list containing config for each camera
            intrinsic:  (3, 3)
            H_c2w:  (4, 4)
            width_px:  int
            height_px:  int
    """
    camera_dicts = []
    for i in range(len(camera_config)):
        mdict = dict()
        mdict["H_c2w"] = np.array(camera_config[i]["H_c2w"])
        assert mdict["H_c2w"].shape == (4, 4)
        mdict["intrinsic"] = np.array(camera_config[i]["intrinsic"])
        assert mdict["intrinsic"].shape == (3, 3)
        mdict["width_px"] = int(camera_config[i]["width_px"])
        mdict["height_px"] = int(camera_config[i]["height_px"])
        camera_dicts.append(mdict)

    return camera_dicts


def read_lighting(light_config: T.List[T.Dict[str, T.Any]]) -> T.List[T.Dict[str, T.Any]]:
    """
    Parse the light config and create light config for individual light

    Args:
        light_config:
            name (str): Name of the light object.
            light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
            H_c2w: (4, 4)
            energy (float): Energy of the light.
            use_shadow (bool, optional): Whether to use shadows. Defaults to False.
            specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.
            size: float, optional for area light, full width along x axis
            size_y: float, optional for area light, full width along y axis. None: the same as size

    Returns:
        a list containing config for each light
            name (str): Name of the light object.
            light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
            H_c2w: (4, 4)
            energy (float): Energy of the light.
            use_shadow (bool, optional): Whether to use shadows. Defaults to False.
            specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.
            size: float, optional for area light, full width along x axis
            size_y: float, optional for area light, full width along y axis. None: the same as size
    """

    light_dicts = []
    for i in range(len(light_config)):
        mdict = dict()
        mdict["name"] = light_config[i]["name"]
        mdict["light_type"] = light_config[i]["light_type"]
        if mdict["light_type"] == "diffuse":
            mdict["color"] = light_config[i].get("color", [1.0, 1.0, 1.0, 1.0])
            mdict["strength"] = light_config[i].get("strength", 1.0)
        else:
            mdict["H_c2w"] = np.array(light_config[i]["H_c2w"])
            assert mdict["H_c2w"].shape == (4, 4)
            mdict["energy"] = float(light_config[i]["energy"])
            mdict["use_shadow"] = bool(light_config[i].get("use_shadow", False))
            mdict["specular_factor"] = float(light_config[i].get("specular_factor", 1.0))
            mdict["size"] = float(light_config[i].get("size", 1.0))
            mdict["size_y"] = float(light_config[i].get("size", None))
        light_dicts.append(mdict)

    return light_dicts


def get_scene_root_objects():
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_all_children(obj):
    """Get all children of an object."""
    children = []

    def recurse(parent):
        for child in parent.children:
            children.append(child)
            recurse(child)

    recurse(obj)
    return children


def select_animation(
    selected_objs: T.List[bpy.types.Object] = None,
    num_frames: int = -1,
    dynamic: bool = True,
    animation_number: int = 0,
):
    """
    Choose the action to an object, and drive the object with it.

    The function assumes the action name is "{animation_name}_{object_name}",
    which means it can be mapped to the object.
    Also note that multiple actions may need to be associated to different objects
    to create the animation.

    Args:
        num_frames:
        dynamic:
        animation_number:

    Returns:
        animation_names:
            list of str, (m,)
        ending_frame_list:
            list of int, (m,)  the end frame
    """
    if selected_objs is None:
        selected_objs = bpy.context.selected_objects

    # count animated frames
    animation_names = []
    ending_frame_list = dict()

    # all action is stored together. Usually artists use name to identify one action can be applied to one object
    for k in bpy.data.actions.keys():
        # eg: 'Armature.001|Flop_Object_5'

        print(f"checking action {k} with selected objects {selected_objs}")

        matched_obj_name = ""
        for obj in selected_objs:  # all selected objects contained in the hierarchy
            # if "_" + obj.name in k and len(obj.name) > len(matched_obj_name):
            if k.endswith(f"_{obj.name}") and len(obj.name) > len(matched_obj_name):  # max match is selected
                matched_obj_name = obj.name
        a_name = k.replace("_" + matched_obj_name, "")  # 'Armature.001|Flop'
        a = bpy.data.actions[k]  # action
        frame_start, frame_end = map(int, a.frame_range)
        print(f"action name: {a.name}: {frame_start}-{frame_end}")
        # action name: Armature.001|Flop_Object_5: 0-75
        # action name: Armature.001|T-Pose_Object_5: 0-0
        if a_name not in animation_names:
            animation_names.append(a_name)
            ending_frame_list[a_name] = frame_end
        else:
            ending_frame_list[a_name] = max(frame_end, ending_frame_list[a_name])

    selected_a_name = animation_names[animation_number]

    for obj in selected_objs:
        if obj.animation_data is not None:
            obj_a_name = selected_a_name + "_" + obj.name
            if obj_a_name in bpy.data.actions:
                print("Found ", obj_a_name)
                obj.animation_data.action = bpy.data.actions[obj_a_name]
            else:
                print("Miss ", obj_a_name)

    # if dynamic set the num_frames to the min of num_frames, total frames
    if dynamic and num_frames < 0:
        num_frames = ending_frame_list[selected_a_name]
    elif dynamic and num_frames > 0:
        num_frames = num_frames
    else:  # if not dynamic, just use 1 frame
        num_frames = 1

    return num_frames


def load_mesh(
    name: str,
    filename: str,
    H_c2w: np.ndarray,  # (4, 4)
    scale: np.ndarray,  # (3,)
    normalize_first: bool,
    cut_aabb_center: T.List[float] = None,
    cut_aabb_radius: T.List[float] = None,
    animation_number=0,
    dynamic=True,
    num_frames=-1,
) -> bpy.types.Object:
    """
    Load mesh and place it into the scene.

    Args:
        name:
            name of the object
        filename:
            filename of the mesh
        H_c2w:
            (4, 4)  from obj coordinate to the world coordinate
        scale:
            (3,)
        normalize_first:
            bool, whether to normalize first before the scale and H_c2w operation
        cut_aabb_center:
            (3,) center xyz_w of the cutting aabb. No cutting if None.
        cut_aabb_radius:
            (3,), radius (half width) for xyz.  No cutting if None.

    Returns:
        bpy.types.Object: The parent Blender object.
        Dict[str, np.ndarray]: information about normalization with the following key-values:
            normalize_matrix_for_scale:
                (4, 4), a matrix for re-scaling.
            normalize_matrix_wo_scale:
                (4, 4), a matrix for SE(3) transformation after re-scaling, i.e., normalize_matrix_for_scale.
            normalize_matrix_with_scale:
                (4, 4), the full matrix for normalization transformation, i.e.,
                xyz_normalized = normalize_matrix_for_scale @ xyz_raw,
                where xyz_raw of (4, N) is the homogeneous coordinates of 3D point.
                normalize_matrix_with_scale = normalize_matrix_wo_scale @ normalize_matrix_for_scale.
            normalize_matrix_wo_scale_rotmat_det:
                float, the determinant of normalize_matrix_wo_scale[:3, :3].
            normalize_matrix_wo_scale_so3_check_ortho:
                bool, true means normalize_matrix_wo_scale[:3, :3] is an orthogonal matrix.
            normalize_matrix_wo_scale_so3_check_det:
                bool, true means normalize_matrix_wo_scale[:3, :3] has a determinant of one.
            normalize_matrix_wo_scale_so3_check_pass:
                bool, true indicates that the normalize_matrix_wo_scale[:3, :3] is a rotation matrix.


    Note:
        normalize (to [-1,1] bbox) -> scale -> H_c2w
    """
    assert os.path.exists(filename), f"{filename} not exists"

    print(f"loading {filename} as {name}:\n  normalize_first: {normalize_first}\n  H_c2w: {H_c2w}\n  scale: {scale}\n")

    bpy.ops.object.select_all(action="DESELECT")
    load_object(object_path=filename)

    selected_objs = bpy.context.selected_objects
    print(f"selected_objs: {selected_objs}")

    # select all objects
    all_child_objs = []
    for obj in selected_objs:
        child_objs = get_all_children(obj)
        all_child_objs += child_objs

    # attach animation to objects and get the number of frames of the animation
    num_frames = select_animation(
        selected_objs=all_child_objs,
        num_frames=num_frames,
        animation_number=animation_number,
        dynamic=dynamic,
    )
    print(f"num_frames: {num_frames}")

    def scene_bbox(obj=None, ignore_matrix=False):
        """Returns the bounding box of the scene.

        Taken from Shap-E rendering script
        (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

        Args:
            single_obj (Optional[bpy.types.Object], optional): If not None, only computes
                the bounding box for the given object. Defaults to None.
            ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
                to False.

        Raises:
            RuntimeError: If there are no objects in the scene.

        Returns:
            Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
        """
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False

        for i in range(num_frames):
            bpy.context.scene.frame_set(i)
            # bpy.context.scene.frame_set(i * args.downsample)
            for obj in get_scene_meshes() if obj is None else [obj]:
                found = True
                for coord in obj.bound_box:
                    coord = Vector(coord)
                    if not ignore_matrix:
                        coord = obj.matrix_world @ coord
                    bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                    bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")

        return Vector(bbox_min), Vector(bbox_max)

    # get only the mesh
    selected_mesh_objs = list(
        {get_root_parent(obj) for obj in all_child_objs if isinstance(obj.data, (bpy.types.Mesh,))}
    )

    # find all root objects
    # imported_objects = [obj for obj in get_scene_root_objects()]
    imported_objects = [obj for obj in selected_mesh_objs if not obj.parent]

    print(f"imported_objects: {imported_objects}")

    # # make sure object is imported
    # imported_objects = [obj for obj in bpy.context.selected_objects]

    normalize_matrix_for_scale = np.eye(4)
    normalize_matrix_wo_scale = np.eye(4)
    normalize_matrix_with_scale = np.eye(4)
    normalize_matrix_wo_scale_so3_check_pass = True
    normalize_matrix_wo_scale_so3_det = None
    normalize_matrix_wo_scale_so3_check_ortho = None
    normalize_matrix_wo_scale_so3_check_det = None

    # Example: rename, set position, and set rotation for the first imported object
    if imported_objects:
        # if len(imported_objects) > 1:

        # Create an empty object to be used as a parent for all root objects
        # we will apply normalization to this object
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)
        # Parent all root objects to the empty object
        for obj in imported_objects:
            if obj != parent_empty:
                obj.parent = parent_empty

        obj = parent_empty
        # print(f'obj name: {obj.name}, bbox: {np.array(obj.bound_box)}')

        # normalize to [-1,1] bbox
        if normalize_first:
            # NOTE: we must create a clean parent node to ensure this parent has identity transformations!!!
            # The reason is that we later will apply any transformation to this clean parent node.
            # If the root node has any non-identity transformations, it will mess up transformations and cause discrepancy.

            # normalize using only meshes
            # bbox_min, bbox_max = scene_bbox(obj=obj)  # in world coordinate
            bbox_min, bbox_max = get_bbox_of_sequence(end_frame_idx=num_frames, obj=obj)  # in world coordinate

            print(f"before normalization: ")
            print(f"  bbox_min: {bbox_min}")
            print(f"  bbox_max: {bbox_max}")
            print(f"{obj.matrix_world=}")
            print(f"{obj.scale=}")

            # first scale
            normalize_scale = 1.999 / max(max(bbox_max - bbox_min), 1e-9)  # [-1, 1]
            # obj.scale = obj.scale * normalize_scale
            obj.scale = obj.scale * normalize_scale
            bpy.context.view_layer.update()

            normalize_matrix_for_scale[:3, :3] = np.diag(np.array(obj.scale))

            # then translate only meshes
            # bbox_min, bbox_max = scene_bbox(obj=obj)  # in world coordinate
            bbox_min, bbox_max = get_bbox_of_sequence(end_frame_idx=num_frames, obj=obj)  # in world coordinate
            # print(f'bbox_min: {bbox_min}')
            # print(f'bbox_max: {bbox_max}')
            normalize_c2w_trans_offset_after_scale = -1 * (bbox_min + bbox_max) / 2
            # obj.matrix_world.translation += normalize_c2w_trans_offset_after_scale
            obj.matrix_world.translation += normalize_c2w_trans_offset_after_scale
            bpy.context.view_layer.update()

            print(f"after normalization: ")
            print(f"  bbox_min: {bbox_min}")
            print(f"  bbox_max: {bbox_max}")
            print(f"{obj.matrix_world=}")
            print(f"{obj.scale=}")

            normalize_matrix_with_scale = np.array(obj.matrix_world)

            # T_full = T_after_scale @ T_scale
            # -> T_after_scale = T_full @ T_scale^{-1}
            normalize_matrix_wo_scale = normalize_matrix_with_scale @ np.linalg.inv(normalize_matrix_for_scale)

            # check the rotation part is SO(3)
            rot_mat_prod = normalize_matrix_wo_scale[:3, :3] @ normalize_matrix_wo_scale[:3, :3].T
            normalize_matrix_wo_scale_so3_check_ortho = np.allclose(rot_mat_prod, np.eye(3))
            normalize_matrix_wo_scale_so3_det = np.linalg.det(normalize_matrix_wo_scale[:3, :3])
            normalize_matrix_wo_scale_so3_check_det = np.allclose(normalize_matrix_wo_scale_so3_det, 1)

            normalize_matrix_wo_scale_so3_check_pass = (
                normalize_matrix_wo_scale_so3_check_ortho and normalize_matrix_wo_scale_so3_check_det
            )

            # # debug
            # print(f'after translation')
            # bbox_min, bbox_max = get_bbox(obj=obj)  # in world coordinate
            # print(f'bbox_min: {bbox_min}')
            # print(f'bbox_max: {bbox_max}')
            # # end debug

            # print(f'obj.scale: {obj.scale}')
            # print(f'obj.matrix_world: {obj.matrix_world}')
            bpy.data.objects["Camera"].parent = None

            # # manipulate parent from now on
            # obj = parent

        # Rename the object
        # NOTE: this is important, after setting obj to tha parent,
        # we avoid manually multiplying the c2w with the the original object's c2w.
        # We leave this to Blender.
        parent = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(parent)
        # parent the obj to the new parent
        assert obj.parent is None
        obj.parent = parent
        obj = parent

        bbox_min, bbox_max = get_bbox_of_sequence(end_frame_idx=num_frames, obj=obj)  # in world coordinate
        print(f"  final_bbox_min: {bbox_min}")
        print(f"  final_bbox_max: {bbox_max}")
        print(f"{imported_objects[0].scale=}")

        # set the scale and H_c2w
        set_H_c2w(
            obj=obj,
            H_c2w=H_c2w,
            scale=scale,
        )

        # cut outside aabb
        if cut_aabb_center is not None and cut_aabb_radius is not None:
            cut_outside_aabb_open(
                obj=obj,
                aabb_center=cut_aabb_center,
                aabb_radius=cut_aabb_radius,
            )

    else:
        print(f"Failed to import {filename}")
        obj = None

    return obj, {
        "normalize_matrix_for_scale": normalize_matrix_for_scale,
        "normalize_matrix_wo_scale": normalize_matrix_wo_scale,
        "normalize_matrix_with_scale": normalize_matrix_with_scale,
        "normalize_matrix_wo_scale_rotmat_det": normalize_matrix_wo_scale_so3_det,
        "normalize_matrix_wo_scale_so3_check_ortho": normalize_matrix_wo_scale_so3_check_ortho,
        "normalize_matrix_wo_scale_so3_check_det": normalize_matrix_wo_scale_so3_check_det,
        "normalize_matrix_wo_scale_so3_check_pass": normalize_matrix_wo_scale_so3_check_pass,
        "num_frames": num_frames,
    }


def load_plane(
    name: str,
    H_c2w: np.ndarray,  # (4, 4)
    length_x: float,
    length_y: float,
    rgba: np.ndarray,  # (4,)
    metallic: float,
    roughness: float,
    refractive_index: float,
) -> bpy.types.Object:
    """
    Load mesh and place it into the scene.

    Args:
        name:
            name of the object
        H_c2w:
            (4, 4)  from obj coordinate to the world coordinate
        length_x:
            float, full width in x direction (before rotation)
        length_y:
            float, full width in y direction (before rotation)
        rgba:
            (4,) rgba [0, 1], base color of bsdf
        metallic:
            float, metallic of the psdf material
        roughness:
            float, roughness of the psdf material
        refractive_index:
            float, refractive_index of the psdf material

    Note:
        normalize (to [-1,1] bbox) -> scale -> H_c2w
    """

    print(f"creating plane {name}:\n  H_c2w: {H_c2w}\n  length_x: {length_x}\n  length_y: {length_y}\n  rgba: {rgba}\n")

    bpy.ops.object.select_all(action="DESELECT")

    # create a square plane lying on xy plane with full width = 1
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    obj = bpy.context.object

    # Rename the object
    obj.name = name

    scale = np.array([length_x, length_y, 1])

    # set the scale and H_c2w
    set_H_c2w(
        obj=obj,
        H_c2w=H_c2w,
        scale=scale,
    )

    # Create a new material
    material = bpy.data.materials.new(name=f"Material_{name}")
    material.use_nodes = True  # Enable nodes to access the Principled BSDF shader

    # Get the node tree of the material
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")  # Access the Principled BSDF shader node

    # Set the color, roughness, and metallic properties
    bsdf.inputs["Base Color"].default_value = rgba  # Set color (RGBA)
    bsdf.inputs["Roughness"].default_value = roughness  # Set roughness (0.0 - 1.0)
    bsdf.inputs["Metallic"].default_value = metallic  # Set metallic (0.0 - 1.0)
    bsdf.inputs["IOR"].default_value = refractive_index  # Set metallic (0.0 - 1.0)

    # Assign the material to the plane
    if obj.data.materials:
        obj.data.materials[0] = material  # Replace existing material
    else:
        obj.data.materials.append(material)  # Add new material if none exists

    return obj


def load_point_cloud(
    name: str,
    filename: str,
    radius: float,
    metallic: float,
    roughness: float,
    refractive_index: float,
    coating_weight: float = 1.0,
) -> bpy.types.Object:
    """
    Load mesh and place it into the scene.

    Args:
        name:
            name of the object
        filename:
            filename of npz containing
                'xyz_w': (n, 3xyz_w)
                'rgba': (n, 4rgba)
                'scale_xyz_w':  (n, 3)  optional, radius for each axis
                'R_c2w': (n, 3, 3)  optional, rotation matrix
        metallic:
            float, metallic of the psdf material
        roughness:
            float, roughness of the psdf material
        refractive_index:
            float, refractive_index of the psdf material

    """
    assert os.path.exists(filename), f"{filename} not exists"

    print(f"loading {filename} as {name}:\n")
    bpy.ops.object.select_all(action="DESELECT")

    # Hide scene updates for performance
    bpy.context.view_layer.active_layer_collection.hide_viewport = True

    # load npz
    pcd_data = np.load(filename, allow_pickle=True)
    xyz_w = pcd_data["xyz_w"]  # (n, 3)
    rgba = pcd_data["rgba"]  # (n, 4)
    scale_xyz_w = pcd_data.get("scale_xyz_w", None)  # (n, 3) or None
    R_c2w = pcd_data.get("R_c2w", None)  # (n, 3, 3) or None
    normal_w = pcd_data.get("normal_w", None)  # (n, 3) or None
    n = xyz_w.shape[0]
    assert rgba.shape[0] == n
    if rgba.shape[1] == 3:
        rgba = np.concatenate([rgba, np.ones((n, 1))], axis=1)  # (n, 4)

    # create shared bsdf material
    material = bpy.data.materials.new(name="Shared_Material")
    material.use_nodes = True

    # Access the material's node tree and get nodes
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Add new nodes
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    principled_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    object_info_node = nodes.new(type="ShaderNodeObjectInfo")

    principled_bsdf.inputs[1].default_value = metallic
    principled_bsdf.inputs[2].default_value = roughness
    principled_bsdf.inputs[3].default_value = refractive_index
    principled_bsdf.inputs[18].default_value = coating_weight

    # Link the nodes: attribute (object color) -> base color of the BSDF shader
    links.new(object_info_node.outputs["Color"], principled_bsdf.inputs["Base Color"])
    links.new(principled_bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # Create a single sphere mesh to reuse for each instance
    # bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1)  # we will scale individually
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1)  # we will scale individually
    sphere_mesh = bpy.context.object
    sphere_mesh.name = f"Base_Sphere_{name}"

    # create a collection for spheres
    sphere_collection = bpy.data.collections.new(f"Sphere_Collection_{name}")
    bpy.context.scene.collection.children.link(sphere_collection)

    # create each sphere
    print_freq = 1000
    for i in range(n):
        if i % print_freq == 0:
            print(f"\rLoading point cloud: {i} / {n}", end="", flush=True)

        # Duplicate the base sphere and move it to the position
        new_sphere = sphere_mesh.copy()
        # new_sphere.data = sphere_mesh.data.copy()
        new_sphere.data = sphere_mesh.data  # instance sharing
        new_sphere.name = f"point_{i}"

        # new_sphere.location = xyz_w[i]

        # scale -> rotate -> move
        if scale_xyz_w is None or scale_xyz_w.ndim == 0:
            assert radius is not None
            scale_factors = (radius, radius, radius)
        else:
            scale_factors = (radius * scale_xyz_w[i]).tolist()
        scaling_matrix = Matrix.Diagonal(scale_factors).to_4x4()

        # rotate
        if R_c2w is not None and R_c2w.ndim > 0:
            rotation_translation_matrix = Matrix(
                [
                    R_c2w[i, 0].tolist() + [xyz_w[i][0]],
                    R_c2w[i, 1].tolist() + [xyz_w[i][1]],
                    R_c2w[i, 2].tolist() + [xyz_w[i][2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        else:
            rotation_translation_matrix = Matrix(
                [
                    [1.0, 0.0, 0.0, xyz_w[i][0]],
                    [0.0, 1.0, 0.0, xyz_w[i][1]],
                    [0.0, 0.0, 1.0, xyz_w[i][2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        combined_matrix = rotation_translation_matrix @ scaling_matrix
        new_sphere.matrix_world = combined_matrix

        # Assign the shared material
        if len(new_sphere.data.materials) == 0:
            new_sphere.data.materials.append(material)
        else:
            new_sphere.data.materials[0] = material

        # Set color
        new_sphere.color = rgba[i]

        # save normal to attribute
        if normal_w is not None and normal_w.ndim > 0:
            new_sphere["normal_w"] = normal_w[i]  # (3,)

        # add to collection
        sphere_collection.objects.link(new_sphere)

    # Delete the original sphere template
    bpy.data.objects.remove(sphere_mesh)

    # Unhide the scene updates
    bpy.context.view_layer.active_layer_collection.hide_viewport = False

    return sphere_collection


def load_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    H_c2w: np.ndarray,
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
    size: float = 1.0,
    size_y: float = None,
) -> bpy.types.Object:
    """
    Load light and put it in the scene.

    name (str):
        Name of the light object.
    light_type (Literal["POINT", "SUN", "SPOT", "AREA"]):
        Type of the light.
    H_c2w:
        (4, 4).  Note that the original sun/spot/area points to -z
    energy (float):
        Energy of the light.
        See https://docs.blender.org/manual/en/latest/render/lights/light_object.html#power-of-lights
    use_shadow (bool, optional):
        Whether to use shadows. Defaults to False.
    specular_factor (float, optional):
        Specular factor of the light. Defaults to 1.0.
    size:
        size (full width) for area light along x axis
    size_y:
        size (full width) for area light along y axis. None: the same as size

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    set_H_c2w(obj=light_object, H_c2w=H_c2w)
    if light_type == "AREA":
        light_data.size = size
        if size_y is not None:
            light_data.size_y = size_y
    return light_object


def load_camera(
    H_c2w: np.ndarray,
    intrinsic: np.ndarray,
    width_px: int,
    height_px: int,
) -> bpy.types.Object:
    """
    setup the camera

    H_c2w:
        (4, 4).  Note that the original camera looks at -z  (x to right of image, y to top of image, z to far)
    intrinsic:
        (3, 3)  camera intrinsic matrix of the given width_px and height_px
    width_px:
        horizontal (x) resolution
    height_px:
        vertical (y) resolution

    Returns:
        bpy.types.Object

    Note:
        we assume the scene contains only one camera object
    """

    print(f"loading camera: \n  H_c2w: {H_c2w}\n  intrinsic: {intrinsic}\n  width: {width_px}\n  height: {height_px}\n")

    scene = bpy.context.scene
    cam = scene.objects["Camera"]

    set_render_resolution(width_px=width_px, height_px=height_px, scale=1)
    set_H_c2w(obj=cam, H_c2w=H_c2w)
    set_camera_intrinsics(camera=cam, intrinsic=intrinsic, width_px=width_px, height_px=height_px)

    return cam


def render_json(
    filename: str,
    out_dir: str,
    save_blend_file_only: bool = False,
    debug: bool = False,
    device: Literal["GPU", "CPU"] = "CPU",
    normalized_mesh_fname: str = "blender_normalized_mesh.ply",
    normalization_info_fname: str = "config_after_blender_normalization.json",
    dynamic: bool = False,
    num_frames=-1,
    animation_number=0,
    frame_start=0,
):
    """
    Construct a scene using the json config file, render, and
    save the resulted images in out_dir.
    """
    print(f"DYNAMIC {dynamic}")
    os.makedirs(out_dir, exist_ok=True)
    # save config
    config_filename = os.path.join(out_dir, "config.json")
    shutil.copy(src=filename, dst=config_filename)

    config_dict = read_json_config(
        filename=filename,
    )  # mesh_dicts, camera_dicts, light_dicts

    config_dict_after_blender = copy.deepcopy(config_dict)

    mesh_dicts = config_dict["mesh_dicts"]
    camera_dicts = config_dict["camera_dicts"]
    light_dicts = config_dict["light_dicts"]
    pcd_dicts = config_dict["pcd_dicts"]
    plane_dicts = config_dict["plane_dicts"]

    # remove everything from the scene
    reset_scene(
        remove_light=True,
        remove_camera=True,
    )

    # add a camera to the scene "Cemera"
    reset_cameras()

    # reset world
    reset_world()

    def _update_normalization_info(dict_ori, dict_normalization):
        for k in dict_normalization:
            assert k not in dict_ori, f"{k=}, {list(dict_normalization.keys())=}"
        dict_ori.update(dict_normalization)

    # Load individual mesh
    for i in range(len(mesh_dicts)):
        _, tmp_normalized_info = load_mesh(
            **mesh_dicts[i], animation_number=animation_number, num_frames=num_frames, dynamic=dynamic
        )
        _update_normalization_info(config_dict_after_blender["mesh_dicts"][i], tmp_normalized_info)
    num_frames = tmp_normalized_info["num_frames"]

    # debug
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=filename)
    # end debug

    # Load individual point cloud
    for i in range(len(pcd_dicts)):
        load_point_cloud(**pcd_dicts[i])

    # Load individual plane
    for i in range(len(plane_dicts)):
        load_plane(**plane_dicts[i])

    scene = bpy.context.scene
    cam = scene.objects["Camera"]

    # Load individual light
    print(f"adding lights: {light_dicts}", flush=True)
    for i in range(len(light_dicts)):
        if light_dicts[i]["light_type"] == "diffuse":
            print(f"adding a diffuse environment map with strength {light_dicts[i]['strength']}")
            # add a background environment map
            world = bpy.context.scene.world
            world.use_nodes = True
            node_tree = world.node_tree
            nodes = node_tree.nodes
            background_node = nodes.new(type="ShaderNodeBackground")
            background_node.name = "Background"
            background_node.inputs["Color"].default_value = light_dicts[i]["color"]  # (4,)
            background_node.inputs["Strength"].default_value = light_dicts[i]["strength"]

            active_output_node = get_active_world_output_node()
            if active_output_node.inputs["Surface"].is_linked:
                link = active_output_node.inputs["Surface"].links[0]
                node_tree.links.remove(link)

            # Create a new link
            node_tree.links.new(background_node.outputs["Background"], active_output_node.inputs["Surface"])
        else:
            load_light(**light_dicts[i])

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    try:
        metadata_extractor = MetadataExtractor(object_path=None, scene=scene, bdata=bpy.data)
        metadata = metadata_extractor.get_metadata()

        # replace texture missing texture image with random color
        missing_textures = delete_missing_textures()
        metadata["missing_textures"] = missing_textures

        # save metadata
        metadata_path = os.path.join(out_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)
    except Exception as e:
        print(f"saving metadata failed with {e}", flush=True)

    # setup blender to use cycles and set up cycles
    hdict = setup_blender_cycles(device=device)

    if save_blend_file_only:
        # save the blender file
        filename = os.path.join(out_dir, "scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=filename)
        return

    # save normalized mesh
    out_dir = pathlib.Path(out_dir)

    for frame in range(num_frames):
        # render the images
        for i in range(len(camera_dicts)):
            # set camera
            cam = load_camera(**camera_dicts[i])

            # render
            render(
                out_dir=out_dir,
                output_handle_dict=hdict,
                filename_prefix=f"{frame:04d}_{i:04d}" if dynamic else f"{i:04d}",
                frame=frame,
                frame_start=frame_start,  # here it starts with frame start!
            )  # file will be saved at {out_dir}/{filename_prefix}_srgb.png

        filepath = (
            str(out_dir / f"{frame:04d}_{normalized_mesh_fname}") if dynamic else str(out_dir / normalized_mesh_fname)
        )

        # obj = bpy.context.active_object
        # # Step 1: Sanitize all MESH objects
        # for obj in bpy.data.objects:
        #     if obj.type == 'MESH':
        #         clean_mesh(obj)
        #         sanitize_vertex_group_names(obj)
        bpy.ops.wm.ply_export(filepath=filepath, export_normals=True, ascii_format=True)

    # save normalized info
    normalization_f = out_dir / normalization_info_fname
    with open(normalization_f, "w") as f:
        json.dump(config_dict_after_blender, f, indent=2, cls=json_utils.PureNumpyJsonEncoder)

    if debug:
        # save the blender file
        filename = os.path.join(out_dir, "scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=filename)


class tmpclass:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path to the json file that configures the scene, light, and camera.",
    )
    parser.add_argument(
        "--save_blend_file_only",
        type=int,
        default=0,
        help="whether to just create the blend file without rendering",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="whether to debug",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=-1,
        help="whether to debug",
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="whether to debug",
    )
    parser.add_argument(
        "--animation_number",
        type=int,
        default=0,
        help="whether to debug",
    )
    parser.add_argument(
        "--dynamic",
        type=bool,
        default=False,
        help="whether to render a dynamic scene",
    )
    parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU"], help="which device to use")
    parser.add_argument(
        "--normalized_mesh_fname",
        type=str,
        default="blender_normalized_mesh.ply",
        help="Filename for saving the normalized mesh",
    )
    parser.add_argument(
        "--normalization_info_fname",
        type=str,
        default="config_after_blender_normalization.json",
        help="Filename for saving the normalization information",
    )

    # Example command using gpu 0:
    """
    CUDA_VISIBLE_DEVICES=0 blender --background --python blender_utils.py -- --filename 'xxxx.json' --out_dir 'out'
    """

    try:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        args = parser.parse_args(argv)
    except:
        # debug
        workdir = str(pathlib.Path(__file__).absolute().parent.parent / "data/blender_data_debug")
        args = tmpclass()
        args.filename = os.path.join(workdir, "example_config_bunny.json")
        args.out_dir = os.path.join(workdir, "outputs")
        args.save_blend_file_only = False
        args.debug = False
        args.device = "CPU"
        args.normalized_mesh_fname = "blender_normalized_mesh.ply"
        args.normalization_info_fname = "config_after_blender_normalization.json"

    render_json(
        filename=args.filename,
        out_dir=args.out_dir,
        save_blend_file_only=bool(args.save_blend_file_only),
        debug=bool(args.debug),
        device=args.device.upper(),
        normalized_mesh_fname=args.normalized_mesh_fname,
        normalization_info_fname=args.normalization_info_fname,
        dynamic=args.dynamic,
        num_frames=args.num_frames,
        animation_number=args.animation_number,
        frame_start=args.frame_start,
    )


# This is a hacky way to separate scripts for 1) running in terminal; and 2) running in a console.
# Debugging line-by-line locally requires avoding the "__main__" structure.
RUN_IN_TERMINAL = True

if RUN_IN_TERMINAL:
    if __name__ == "__main__":
        main()
else:
    main()
