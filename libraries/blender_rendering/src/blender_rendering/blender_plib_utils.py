#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements the connection between blender and plib.


import glob
import json
import os
import typing as T

import imageio.v3 as iio
import numpy as np
import tqdm

import torch

from plibs import exr_utils, structures


def read_blender_raw_results(
    result_dir: str,
    from_idx: int = 0,
    to_idx: int = None,
    fields: T.List[str] = ("rgb", "srgb", "depth", "normal", "camera", "obj_id"),
    verbose: bool = False,
) -> T.List[T.Dict[str, T.Any]]:
    """
    Read the blender rendering results.

    Args:
        result_dir:
            where the rendering are saved
        from_idx:
            starting frame idx
        to_idx:
            ending frame idx (excluding)
        fields:
            name of the field of read

    Returns:
        a list of number of frames, each is a dict containing the fields

        rgb:
            (h, w, 4rgba) hdr, without tonemapping. RGB is straight.
        srgb:
            (h, w, 4rgba) lhr, tonemapped, [0, 1].
            RGB is straight.
        depth:
            (h, w, 1)  z_c
        normal:
            (h, w, 4)  normal in the world coordinate
        obj_id:
            (h, w, 1) float
        cam_info:
            H_c2w_blender:
                (4, 4)
            intrinsic_blender:
                (3, 3)
            H_c2w_open3d:
                (4, 4)
            intrinsic_open3d:
                (3, 3)
            width_px:
            height_px:
            focal_length_mm:
                float
            sensor_width_mm:
                float
            sensor_height_mm:
                float
            pixel_size_x_mm:
                float
            pixel_size_y_mm:
                float

    """

    assert os.path.exists(result_dir), f"{result_dir} not exists"

    # figure out total number of frames
    fns = glob.glob(os.path.join(result_dir, f"*_srgb.png"))
    total = len(fns)
    if total == 0:
        fns = glob.glob(os.path.join(result_dir, f"*_rgb.exr"))
        total = len(fns)
    assert total > 0, f"{result_dir=}"

    if to_idx is None:
        to_idx = total
    if from_idx < 0:
        from_idx = from_idx + total
    if to_idx < 0:
        to_idx = to_idx + total

    all_results = []
    # for ii in range(2): # range(H_c2w_o3d.shape[0]):
    for ii in tqdm.tqdm(range(from_idx, to_idx), desc="read_blender_raw_results", disable=not verbose):
        # exr
        ddict = dict()
        for key in ["rgb", "depth", "obj_id"]:
            if key in fields:
                filename = os.path.join(result_dir, f"{ii:04d}_{key}.exr")
                if os.path.exists(filename):
                    arr = exr_utils.read_exr(filename)  # (h, w, c)
                    if (key == "depth") and (arr.shape[-1] > 1):
                        # due to discrepancies in different versions of Blender,
                        # depth in OPEN_EXR can have 3 duplicated channels, e.g., Blender 3.0.
                        # Thus we choose only one of them to ease our processing later.
                        depth_arr_ch_diff = np.sum(np.abs(arr - arr[..., :1]))
                        assert depth_arr_ch_diff == 0, f"{filename=}, {depth_arr_ch_diff=}"
                        arr = arr[..., :1]
                    ddict[key] = arr

        if "normal" in fields:
            key = "normal"
            filename = os.path.join(result_dir, f"{ii:04d}_{key}.exr")
            if os.path.exists(filename):
                arr = exr_utils.read_exr(filename)  # (h, w, c)
                ddict[key] = arr * 2 - 1

        # srgb
        if "srgb" in fields:
            filename = os.path.join(result_dir, f"{ii:04d}_srgb.png")
            if os.path.exists(filename):
                arr = iio.imread(filename)
                if arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255
                elif arr.dtype == np.uint16:
                    arr = arr.astype(np.float32) / 65535
                else:
                    raise NotImplementedError
                ddict["srgb"] = arr

        # camera
        if "camera" in fields:
            filename = os.path.join(result_dir, f"{ii:04d}_camera.json")
            with open(filename, "r") as f:
                cam_info = json.load(f)
            ddict["cam_info"] = cam_info
            for nn in ["H_c2w_blender", "intrinsic_blender", "H_c2w_open3d", "intrinsic_open3d"]:
                ddict["cam_info"][nn] = np.array(ddict["cam_info"][nn])

        all_results.append(ddict)

    return all_results


def read_blender_raw_results_dynamic(
    result_dir: str,
    from_idx: int = 0,
    to_idx: int = None,
    from_bidx: int = 0,
    to_bidx: int = None,
    fields: T.List[str] = ("rgb", "srgb", "depth", "normal", "camera", "obj_id"),
) -> T.Tuple[T.List[T.Dict[str, T.Any]], int]:
    """
    Read the blender rendering results.

    Args:
        result_dir:
            where the rendering are saved
        from_idx:
            starting frame idx
        to_idx:
            ending frame idx (excluding)
        fields:
            name of the field of read

    Returns:
        a list of (number of frames * num of views,), each is a dict containing the fields

        rgb:
            (h, w, 4rgba) hdr, without tonemapping
        srgb:
            (h, w, 4rgba) lhr, tonemapped, [0, 1]
        depth:
            (h, w, 1)  z_c
        normal:
            (h, w, 4)  normal in the world coordinate
        obj_id:
            (h, w, 1) float
        cam_info:
            H_c2w_blender:
                (4, 4)
            intrinsic_blender:
                (3, 3)
            H_c2w_open3d:
                (4, 4)
            intrinsic_open3d:
                (3, 3)
            width_px:
            height_px:
            focal_length_mm:
                float
            sensor_width_mm:
                float
            sensor_height_mm:
                float
            pixel_size_x_mm:
                float
            pixel_size_y_mm:
                float

    """
    assert os.path.exists(result_dir), f"{result_dir} not exists"
    # figure out total number of frames
    fns = glob.glob(os.path.join(result_dir, f"*_0000_srgb.png"))
    total_frames = len(fns)

    # figure out total number of views
    fns = glob.glob(os.path.join(result_dir, f"0000_*_srgb.png"))
    total_views = len(fns)

    if total_frames == 0:
        # figure out total number of frames
        fns = glob.glob(os.path.join(result_dir, f"*_0000_rgb.exr"))
        total_frames = len(fns)

        # figure out total number of views
        fns = glob.glob(os.path.join(result_dir, f"0000_*_rgb.exr"))
        total_views = len(fns)

    assert total_frames > 0

    if to_idx is None:
        to_idx = total_views
    if from_idx < 0:
        from_idx = from_idx + total_views
    if to_idx < 0:
        to_idx = to_idx + total_views

    if to_bidx is None:
        to_bidx = total_frames

    total_frames = to_bidx - from_bidx

    all_results = []
    # {frame_number}_{view_number}
    # for ii in range(2): # range(H_c2w_o3d.shape[0]):
    for jj in range(from_bidx, to_bidx):
        for ii in range(from_idx, to_idx):
            # exr
            ddict = dict()
            for key in ["rgb", "depth", "obj_id"]:
                if key in fields:
                    filename = os.path.join(result_dir, f"{jj:04d}_{ii:04d}_{key}.exr")
                    if os.path.exists(filename):
                        arr = exr_utils.read_exr(filename)  # (h, w, c)
                        ddict[key] = arr

            if "normal" in fields:
                key = "normal"
                filename = os.path.join(result_dir, f"{jj:04d}_{ii:04d}_{key}.exr")
                if os.path.exists(filename):
                    arr = exr_utils.read_exr(filename)  # (h, w, c)
                    ddict[key] = arr * 2 - 1

            # srgb
            if "srgb" in fields:
                filename = os.path.join(result_dir, f"{jj:04d}_{ii:04d}_srgb.png")
                if os.path.exists(filename):
                    arr = iio.imread(filename)
                    if arr.dtype == np.uint8:
                        arr = arr.astype(np.float32) / 255
                    elif arr.dtype == np.uint16:
                        arr = arr.astype(np.float32) / 65535
                    else:
                        raise NotImplementedError
                    ddict["srgb"] = arr

            # camera
            if "camera" in fields:
                filename = os.path.join(result_dir, f"{jj:04d}_{ii:04d}_camera.json")
                with open(filename, "r") as f:
                    cam_info = json.load(f)
                ddict["cam_info"] = cam_info
                for nn in ["H_c2w_blender", "intrinsic_blender", "H_c2w_open3d", "intrinsic_open3d"]:
                    ddict["cam_info"][nn] = np.array(ddict["cam_info"][nn])

            all_results.append(ddict)

    return all_results, total_frames


def read_blender_results_to_rgbd(
    result_dir: str,
    from_idx: int = 0,
    to_idx: int = None,
    from_bidx: int = 0,
    to_bidx: int = None,
    use_srgb: bool = True,
    flag_save_space: bool = False,
    dynamic: bool = None,
    th_alpha: float = 0.5,
    min_depth: float = 0.0,
    max_depth: float = 1.0e4,
    fps: float = 24,
) -> structures.RGBDImage:
    """
    Read the blender rendering results to RGBDImage structure.

    Args:
        result_dir:
        from_idx:
        to_idx:
        use_srgb:
            whether to use srgb (gamma corrected) instead of rgb ()
        flag_save_space:
            if true, we will not load infos like obj_ids that occupies memory.
        th_alpha:
            a pixel / ray is considered hit if alpha > th_alpha

        dynamic:
            if True, assume {frame_idx:04d}_{view_idx:04d}_srgb.png
            else, assume {view_idx:04d}_rgb.png
            if None, auto detect.

    Returns:
        rgbd:
            (b=1, q, h, w)
    """

    if dynamic is None:
        _test_filenames = glob.glob(os.path.join(result_dir, "0000_0000_*.png"))
        dynamic = len(_test_filenames) > 0

    if not dynamic:
        # load raw results
        all_results = read_blender_raw_results(
            result_dir=result_dir,
            from_idx=from_idx,
            to_idx=to_idx,
        )
        num_frames = 1
        include_timestamp = False
    else:
        # load raw results
        all_results, num_frames = read_blender_raw_results_dynamic(
            result_dir=result_dir,
            from_idx=from_idx,
            to_idx=to_idx,
            from_bidx=from_bidx,
            to_bidx=to_bidx,
        )
        include_timestamp = True

    # create rgbd image from all_results
    rgbds = []
    for ii in range(len(all_results)):
        ddict = all_results[ii]  # rgb: (h, w, 4rgba) [0, 1] stright

        # for key in ddict:
        #     if isinstance(ddict[key], (np.ndarray, torch.Tensor)):
        #         print(f'{key}: {ddict[key].shape}')

        other_maps = dict()
        # hit_map = torch.from_numpy(ddict["rgb"][:, :, 3])  # (h, w)  [0, 1]
        other_maps["alpha"] = torch.from_numpy(ddict["srgb"])[:, :, 3:4].clone().float()  # (h, w, 1) [0, 1]
        hit_map = torch.from_numpy(ddict["srgb"][:, :, 3]) > th_alpha  # (h, w)  bool
        h, w = hit_map.shape
        if use_srgb:
            # we should not multiply here (otherwise we double multiply alpha when compositing with background)
            rgb = torch.from_numpy(ddict["srgb"][:, :, :3]).float()  # * other_maps['alpha']  # straight
            if not flag_save_space and ddict.get("rgb", None) is not None:
                other_maps["rgb_hdr"] = torch.from_numpy(ddict["rgb"]).float()  # * other_maps['alpha']  # straight
        else:
            assert ddict.get("rgb", None) is not None
            # we should not multiply here (otherwise we double multiply alpha when compositing with background)
            rgb = torch.from_numpy(ddict["rgb"][:, :, :3]).float()  # * other_maps['alpha']  # straight
            if not flag_save_space and ddict.get("srgb", None) is not None:
                other_maps["rgb_ldr"] = torch.from_numpy(ddict["srgb"]).float()  # * other_maps['alpha']  # straight
        normal_w = torch.from_numpy(ddict["normal"][:, :, :3]).float()  # (h, w, 3)
        depth = torch.from_numpy(ddict["depth"]).float()  # (h, w, 1)

        # refine hit_map to remove nan/inf
        valid_normal = normal_w.isfinite().all(dim=-1)  # (h, w)
        # print(f'valid_normal: {valid_normal.shape} {valid_normal.float().mean()}')
        valid_depth = torch.logical_and(
            depth.squeeze(-1).isfinite(),  # (h, w)
            torch.logical_and(depth >= min_depth, depth < max_depth).squeeze(-1),  # (h, w)
        )  # (h, w)
        # print(f'valid_depth: {valid_depth.shape} {valid_depth.float().mean()}')
        valid_rgb = rgb.isfinite().all(dim=-1)  # (h, w)
        # print(f'valid_rgb: {valid_rgb.shape} {valid_rgb.float().mean()}')
        valid_masks_finite = [valid_normal, valid_depth, valid_rgb]
        for key in other_maps:
            assert other_maps[key].ndim == 3
            v = other_maps[key].isfinite().all(dim=-1)  # (h, w)
            # print(f'valid {key}: {v.shape} {v.float().mean()}')
            valid_masks_finite.append(v)
        vmask_finite = valid_masks_finite[0].clone()  # (h, w)
        for v in valid_masks_finite[1:]:
            assert v.shape == (h, w)
            vmask_finite = torch.logical_and(vmask_finite, v)  # (h, w)
        # Note: since we set valid_depth with a hard threshold, we may still see
        # aliasing (sawtooth) pattern in the hit_map
        hit_map = hit_map.masked_fill(~vmask_finite, 0)  # (h, w)
        # hit_map = torch.logical_and(hit_map, v)  # (h, w)
        # print(f'final valid: {hit_map.shape} {hit_map.float().mean()}')
        assert hit_map.shape == (h, w), f"{hit_map.shape=}"
        assert rgb.shape == (h, w, 3), f"{rgb.shape=}"
        assert normal_w.shape == (h, w, 3), f"{normal_w.shape=}"
        assert depth.shape == (h, w, 1), f"{depth.shape=}"

        # replace non finite values to default values
        rgb = rgb.masked_fill(~valid_rgb.unsqueeze(-1), 0)  # (h, w, 3)
        normal_w = normal_w.masked_fill(~valid_normal.unsqueeze(-1), 1)  # (h, w, 3)
        normal_w = torch.nn.functional.normalize(normal_w, dim=-1)  # (h, w, 3)
        depth = depth.masked_fill(~valid_depth.unsqueeze(-1), structures.INF)  # (h, w)
        for key in other_maps:
            assert other_maps[key].size(0) == h
            assert other_maps[key].size(1) == w
            assert other_maps[key].ndim == 3
            # if key != "alpha":
            #     other_maps[key] = other_maps[key].masked_fill(non_finite.unsqueeze(-1), 0)

        if not flag_save_space and ddict.get("obj_id", None) is not None:
            other_maps["obj_id"] = torch.from_numpy(ddict["obj_id"]).float()  # (h, w, 1)
            # other_maps["obj_id"] = other_maps["obj_id"].masked_fill(non_finite.unsqueeze(-1), -1)
            assert other_maps["obj_id"].shape == (h, w, 1)

        assert (depth > -1e-9).all()

        # create camera
        cam_info = ddict["cam_info"]
        camera = structures.Camera(
            H_c2w=torch.from_numpy(cam_info["H_c2w_open3d"]).float().reshape(1, 1, 4, 4),
            intrinsic=torch.from_numpy(cam_info["intrinsic_open3d"]).float().reshape(1, 1, 3, 3),
            width_px=cam_info["width_px"],
            height_px=cam_info["height_px"],
            timestamp=torch.tensor([ii / fps])[None] if include_timestamp else None,
        )

        # create rgbd
        for key in other_maps:
            other_maps[key] = other_maps[key].reshape(1, 1, h, w, other_maps[key].size(-1))

        rgbd = structures.RGBDImage(
            rgb=rgb.reshape(1, 1, h, w, 3),
            depth=depth.reshape(1, 1, h, w),
            normal_w=normal_w.reshape(1, 1, h, w, 3),
            hit_map=hit_map.reshape(1, 1, h, w),
            camera=camera,
            other_maps=other_maps,
        )

        rgbds.append(rgbd)

    num_views = len(rgbds) // num_frames
    rgbd_views = []
    for i in range(num_frames):
        rgbd_views.append(structures.RGBDImage.cat(rgbds[i * num_views : (i + 1) * num_views], dim=1))

    if len(rgbd_views) > 1:
        rgbd = structures.RGBDImage.cat(rgbd_views, dim=0)  # (b=num_frames, q=num_views, h, w)
    else:
        rgbd = rgbd_views[0]  # (b=1, q=num_views, h, w)
    return rgbd
