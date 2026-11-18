#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import hashlib
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.cameras import PseudoCamera
from utils.mvs_init import (
    load_mvs_point_cloud,
    resolve_mvs_init_strategy,
)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.source_path = args.source_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        requested_init_mode = getattr(args, "init_mode", "mvs_anywhere")
        init_mode = requested_init_mode
        if init_mode not in ("colmap_dense", "mvs_anywhere"):
            raise ValueError("init_mode must be 'colmap_dense' or 'mvs_anywhere'")
        complement_mode = getattr(args, "mvs_complement_mode", "none")
        if not self.loaded_iter:
            # Reject retired/incompatible configurations even when an input
            # format has an explicit baseline-initialization exception.
            for removed_option in ("mvs_geometry_correction", "mvs_surface_init"):
                if getattr(args, removed_option, "none") != "none":
                    raise ValueError(f"{removed_option} was removed; use direct MVS or MVS complement initialization")
            if complement_mode not in ("none", "coverage", "concat", "balanced"):
                raise ValueError("mvs_complement_mode must be none, coverage, concat, or balanced")
            if complement_mode != "none" and init_mode != "mvs_anywhere":
                raise ValueError("MVS complement initialization requires init_mode=mvs_anywhere")
        random_pcd = bool(getattr(args, "rand_pcd", False))
        blender_random_override = False
        is_blender_scene = False
        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        print(args.source_path)
        if args.source_path.find('replica_few') != -1:
            print("Found replica path, assuming Replica data set!")
            scene_info = sceneLoadTypeCallbacks["Replica"](args.source_path, args.images, args.eval, args.n_views, args.rand_pcd)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            is_blender_scene = True
            # Explicit main-method protocol: Blender keeps DropGaussian's
            # random initializer, not a learned-MVS replacement. Force fresh
            # in-memory points even if a cached points3d.ply already exists.
            # mode=none remains the deliberately requested direct-MVS ablation.
            blender_random_override = (
                init_mode == "mvs_anywhere"
                and complement_mode in ("coverage", "concat", "balanced")
            )
            if blender_random_override:
                init_mode = "colmap_dense"
                random_pcd = True
                print("Blender complement protocol: original DropGaussian random initialization; "
                      "MVS loading and complement selection are disabled. Density control is unchanged.")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval,
                random_pcd,
                args.n_views,
                getattr(args, "dataset_name", "auto"),
            )
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.eval,
                args.n_views,
                dataset_name=getattr(args, "dataset_name", "auto"),
            )
        else:
            assert False, "Could not recognize scene type!"

        base_digest = hashlib.sha256()
        for values in (
            scene_info.point_cloud.points,
            scene_info.point_cloud.colors,
        ):
            array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
            base_digest.update(str(array.shape).encode("ascii"))
            base_digest.update(array.tobytes())
        self.base_initialization_sha256 = base_digest.hexdigest()
        self.train_camera_names = [
            str(camera.image_name) for camera in scene_info.train_cameras
        ]
        self.test_camera_names = [
            str(camera.image_name) for camera in scene_info.test_cameras
        ]

        self.effective_rand_pcd = random_pcd
        self.initialization_metadata = {
            "mode": init_mode,
            "path": str(scene_info.ply_path),
            "base_initialization_sha256": self.base_initialization_sha256,
        }
        if init_mode == "colmap_dense":
            base_count = int(len(scene_info.point_cloud.points))
            base_source = getattr(scene_info, "point_cloud_source", "unknown")
            blender_random = is_blender_scene and base_source == "random"
            self.initialization_metadata.update({
                "requested_mode": requested_init_mode,
                "requested_complement_mode": complement_mode,
                "effective_complement_mode": "none",
                "fusion_policy": ("blender_random_dropgaussian" if blender_random
                                  else "baseline_point_cloud"),
                "base_point_cloud_source": base_source,
                "base_point_count": base_count,
                "selected_base_point_count": base_count,
                "mvs_point_count": 0,
                "fused_point_count": base_count,
                "selected_point_count": base_count,
                "mvs_cache_used": False,
                "blender_random_override": blender_random_override,
            })
            if blender_random:
                # The dataset PLY may be stale or nonexistent: the actual
                # initialization artifact belongs to this run, not that path.
                self.initialization_metadata.update({
                    "path": os.path.join(self.model_path, "input.ply"),
                    "random_cloud_generated_in_memory": random_pcd,
                })
                print(f"Blender initialization: {base_count} original random points; 0 MVS points.")
        if not self.loaded_iter and init_mode == "mvs_anywhere":
            strategy = getattr(args, "mvs_init_strategy", "replace")
            resolved_strategy = resolve_mvs_init_strategy(strategy)
            base_point_count = int(scene_info.point_cloud.points.shape[0])
            base_pcd = scene_info.point_cloud
            base_ply_path = str(scene_info.ply_path)
            base_source = getattr(scene_info, "point_cloud_source", "unknown")
            base_is_geometry = (
                base_source in ("colmap_dense", "colmap_sparse", "reconstruction")
                and base_point_count > 0
            )
            complement_budget = getattr(args, "mvs_complement_max_points", 100000)
            total_budget = getattr(args, "mvs_complement_total_points", 0)
            load_budget = args.mvs_init_max_points
            if complement_mode != "none":
                for value, label, minimum in (
                    (complement_budget, "mvs_complement_max_points", 1),
                    (total_budget, "mvs_complement_total_points", 0),
                    (getattr(args, "mvs_complement_candidate_points", 500000),
                     "mvs_complement_candidate_points", 1),
                ):
                    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                        raise ValueError(f"{label} must be an integer >= {minimum}")
                if base_is_geometry:
                    load_budget = complement_budget
                    if complement_mode in ("coverage", "balanced"):
                        load_budget = max(load_budget, getattr(args, "mvs_complement_candidate_points", 500000))
                else:
                    # No random/unknown base cloud is promoted to a geometric
                    # anchor. Use exactly the direct loader/selection at this
                    # budget, rather than re-sampling a larger candidate pool.
                    load_budget = min(complement_budget, total_budget) if total_budget else complement_budget
            # Explicit direct-MVS comparisons still include Blender. A missing
            # or incompatible cache must fail rather than silently falling back.
            mvs_pcd, mvs_path, mvs_metadata = load_mvs_point_cloud(
                args.source_path,
                n_views=args.n_views,
                train_camera_names=self.train_camera_names,
                explicit_path=args.mvs_init_path,
                max_points=load_budget,
            )
            complement_metadata = None
            if complement_mode != "none":
                from utils.mvs_complement import build_complement_cloud
                if base_is_geometry and not mvs_metadata.get("camera_provenance_verified", False):
                    raise ValueError("MVS complement requires a training-view-verified MVS cache")
                complement_cameras = scene_info.train_cameras
                calibration_metadata = {"source": "provided_camera_info"}
                if base_is_geometry and complement_mode in ("coverage", "balanced"):
                    camera_bin = os.path.join(args.source_path, "sparse", "0", "cameras.bin")
                    camera_txt = os.path.join(args.source_path, "sparse", "0", "cameras.txt")
                    if os.path.isfile(camera_bin) or os.path.isfile(camera_txt):
                        from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text
                        from utils.train_view_controls import geometry_cameras_with_intrinsics
                        intrinsics = (read_intrinsics_binary(camera_bin) if os.path.isfile(camera_bin)
                                      else read_intrinsics_text(camera_txt))
                        # Correct only the copied cameras used for point
                        # selection. Training/rendering calibration is unchanged.
                        complement_cameras, calibration_metadata = geometry_cameras_with_intrinsics(
                            scene_info.train_cameras, intrinsics,
                        )
                candidate_sha256 = mvs_metadata["selected_sha256"]
                mvs_pcd, complement_metadata, complement_arrays = build_complement_cloud(
                    base_pcd, mvs_pcd, complement_cameras,
                    mode=complement_mode, mvs_budget=complement_budget,
                    total_budget=total_budget,
                    depth_cache_path=os.path.join(os.path.dirname(str(mvs_path)), "mvs_depths_masks.npz"),
                    base_is_geometry=base_is_geometry,
                    coverage_pixels=getattr(args, "mvs_complement_coverage_pixels", 2.0),
                    depth_relative_scale=getattr(args, "mvs_complement_depth_scale", 0.02),
                    direct_pcd=mvs_pcd if not base_is_geometry else None,
                )
                complement_metadata.update({
                    "base_point_cloud_source": base_source,
                    "base_path": base_ply_path,
                    "base_provenance": "inherited_baseline_asset_not_train_view_verified",
                    "base_camera_provenance_verified": False,
                    "selection_camera_names": self.train_camera_names,
                    "mvs_camera_provenance_verified": bool(mvs_metadata.get("camera_provenance_verified", False)),
                    "calibration": calibration_metadata,
                    "candidate_sha256": candidate_sha256,
                })
                digest = hashlib.sha256()
                digest.update(np.asarray(mvs_pcd.points, dtype=np.float32).tobytes())
                digest.update(np.asarray(mvs_pcd.colors, dtype=np.float32).tobytes())
                mvs_metadata["candidate_selected_sha256"] = candidate_sha256
                mvs_metadata["selected_sha256"] = digest.hexdigest()
                complement_metadata["output_sha256"] = digest.hexdigest()
                np.savez_compressed(os.path.join(self.model_path, "complement_initialization.npz"),
                                    **complement_arrays)
                with open(os.path.join(self.model_path, "complement_initialization.json"), "w") as stream:
                    json.dump(complement_metadata, stream, indent=2, sort_keys=True)
                print(f"MVS complement {complement_mode}: {base_point_count} input base points; "
                      f"{len(mvs_pcd.points)} output points. Base asset provenance is inherited, not verified.")
            scene_info = scene_info._replace(
                point_cloud=mvs_pcd, ply_path=str(mvs_path)
            )
            mvs_metadata.update(
                {
                    "requested_strategy": strategy,
                    "resolved_strategy": resolved_strategy,
                    "fusion_policy": "direct_mvs_replace",
                    "configured_max_points": int(args.mvs_init_max_points),
                    "base_point_count": base_point_count,
                    "mvs_point_count": int(mvs_pcd.points.shape[0]),
                    "fused_point_count": int(mvs_pcd.points.shape[0]),
                }
            )
            if complement_metadata is not None:
                mvs_metadata["fusion_policy"] = (
                    "direct_mvs_no_reconstructed_base"
                    if complement_metadata.get("fallback") else "baseline_geometry_mvs_complement"
                )
                mvs_metadata["complement_initialization"] = complement_metadata
                # This is a combined cloud, not an MVS-only replacement.
                mvs_metadata["candidate_load_max_points"] = load_budget
                mvs_metadata["candidate_selection"] = mvs_metadata.get("selection")
                mvs_metadata["selection"] = (
                    "unchanged_direct_mvs" if complement_metadata.get("fallback")
                    else f"base_plus_{complement_mode}_mvs"
                )
                mvs_metadata["selected_point_count"] = int(mvs_pcd.points.shape[0])
                mvs_metadata["mvs_point_count"] = complement_metadata["selected_mvs_count"]
                mvs_metadata["selected_base_point_count"] = complement_metadata["selected_base_count"]
                mvs_metadata["base_camera_provenance_verified"] = False
            self.initialization_metadata = {
                "mode": init_mode,
                "base_initialization_sha256": self.base_initialization_sha256,
                **mvs_metadata,
            }

        if not self.loaded_iter:
            input_path = os.path.join(self.model_path, "input.ply")
            if init_mode == "mvs_anywhere" or random_pcd:
                input_points = np.asarray(scene_info.point_cloud.points)
                input_colors = np.asarray(scene_info.point_cloud.colors)
                storePly(
                    input_path,
                    input_points,
                    input_colors * 255.0,
                )
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(input_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud,
                self.cameras_extent,
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]
