import json
import os
import pathlib
import platform
import time

import numpy as np
import numpy.typing as nptyping
import trimesh


def get_blender_exe(
    version: str = "4.2.0",
    root_dir: str = "/mnt",
    download: bool = True,
    use_s3: bool = False,
) -> str:
    """
    get blender executable location (download if not exists).

    Args:
        version:
            blender version
        download:
            whether to download if not exists

    Returns:
        location of the blender executable
    """

    a, b, c = version.split(".")
    os_name = platform.system()

    if os_name == "Darwin":
        blender_exe = "/Applications/Blender.app/Contents/MacOS/Blender"
        assert os.path.exists(blender_exe)
        return blender_exe
    elif os_name == "Linux":
        blender_dir = os.path.join(root_dir, f"blender-{version}-linux-x64")
        blender_exe = os.path.join(blender_dir, "blender")
        if os.path.exists(blender_exe):
            return blender_exe

        if download:
            url = f"https://download.blender.org/release/Blender{a}.{b}/blender-{version}-linux-x64.tar.xz"
            cmd = f"cd {root_dir} && wget {url} && tar -xf {os.path.basename(url)} && rm {os.path.basename(url)}"

            # NOTE: this is important to avoid throttle on S3 with large scale jobs.
            # Essentially, we randomly pause the request for downloading from S3 if the previous requst fails.
            to_download = True
            attempt = 0
            while to_download:
                try:
                    attempt += 1

                    if os.path.exists(os.path.join(root_dir, os.path.basename(url))):
                        # It seems like there are situations that we connect to url
                        # but the download stopped in the middle of the process.
                        # In such cases, there will be a tar file in root_dir even though the downloading is not completed.
                        # For next round of downloading, wget will name the downloaded file XXX.tar.xz.1
                        # Therefore, we need to delete the incomplete file
                        os.remove(os.path.join(root_dir, os.path.basename(url)))

                    os.system(cmd)
                    assert os.path.exists(blender_exe)
                    to_download = False
                except Exception:
                    BASE_DELAY = 2  # seconds
                    MAX_DELAY = 60  # cap delay to avoid runaway

                    delay = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
                    jitter = np.random.uniform(0, 1)
                    wait_time = delay + jitter
                    print(f"Attempt {attempt} downloading Blender from server): retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
        else:
            raise NotImplementedError

        assert os.path.exists(blender_exe)
        return blender_exe

    elif os_name == "Windows":
        raise NotImplementedError
    else:
        raise NotImplementedError


def get_blender_utils_path():
    return os.path.normpath(os.path.join(__file__, "../blender_utils.py"))


def get_blender_utils_v2_path():
    return os.path.normpath(os.path.join(__file__, "../blender_utils_v2.py"))


def get_blender_utils_v3_path():
    return os.path.normpath(os.path.join(__file__, "../blender_utils_v3.py"))


def transform_xyz(transform_mat: nptyping.NDArray, xyz: nptyping.NDArray):
    """Transform a set of 3D points."""
    assert (xyz.ndim == 2) and (xyz.shape[1] == 3), f"{xyz.shape=}"  # (N, 3)
    xyz_w = np.pad(xyz, ((0, 0), (0, 1)), constant_values=1)
    xyz_w_after = (transform_mat @ xyz_w.T).T  # (N, 4)
    return xyz_w_after[:, :3] / xyz_w_after[:, 3:]


def blender_to_yup():
    """Transform to Y-up.

    Ref:
    - https://github.com/mikedh/trimesh/issues/1938#issuecomment-1596271890
    - https://github.com/KhronosGroup/glTF-Blender-IO/blob/5c52c313bcadb4703eb34ec6d5b51d1e47c60089/addons/io_scene_gltf2/blender/com/gltf2_blender_math.py#L59-L66
    """
    return np.array(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def normalize_mesh_with_saved_normalize_matrix_from_blender(
    *, raw_mesh_f: str | pathlib.Path, save_dir: str | pathlib.Path, normalize_matrix_f: str | pathlib.Path
):
    """This function manually normalize a mesh with the normalize matrix saved from blender_utils.py.

    Args:
        raw_mesh_f:
            the original before-normalized mesh file
        save_dir:
            directory for saving the manually normalized mesh
        normalize_matrix_f:
            the normalization_info json file saved by blender_rendering/blender_utils.py,
            which contains the normalization matrix used to transform the orignal mesh.
    """
    raw_mesh_f = pathlib.Path(raw_mesh_f)
    normalize_matrix_f = pathlib.Path(normalize_matrix_f)
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    assert normalize_matrix_f.suffix == ".json", f"{normalize_matrix_f}"
    with open(normalize_matrix_f, "r") as f:
        normalize_matrix_info = json.load(f)
    assert len(normalize_matrix_info["mesh_dicts"]) == 1, f'{len(normalize_matrix_info["mesh_dicts"])=}'

    normalize_matrix = np.array(normalize_matrix_info["mesh_dicts"][0]["normalize_matrix_with_scale"])

    flag_gltf = raw_mesh_f.suffix in [".glb", ".gltf"]

    mesh_raw = trimesh.load(raw_mesh_f, process=False)
    if flag_gltf:
        geo_list = list(mesh_raw.geometry.values())
        mesh_raw = geo_list[0]
    verts_raw = np.array(mesh_raw.vertices)

    if flag_gltf:
        # When importing gltf file, Blender add an ad-hoc transformation
        # while trimesh sticks to the original coordinates in the raw mesh file.
        # Thus, we mimic the Blender's behaviour since we use the saved normalizing matrix from Blender.
        verts_raw = transform_xyz(blender_to_yup().T, verts_raw)

    verts_normalized = transform_xyz(normalize_matrix, verts_raw)
    mesh_normalized = trimesh.Trimesh(vertices=verts_normalized, faces=mesh_raw.faces, process=False)

    save_f = save_dir / f"{raw_mesh_f.stem}_normalized.ply"
    with open(save_f, "wb") as f:
        f.write(trimesh.exchange.ply.export_ply(mesh_normalized, encoding="ascii"))

    return save_f
