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
    build_adaptive_mvs_prior,
    fuse_residual_mvs_point_cloud,
    load_mvs_point_cloud,
    resolve_mvs_fusion_budget,
    resolve_mvs_init_point_budget,
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

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        print(args.source_path)
        if args.source_path.find('replica_few') != -1:
            print("Found replica path, assuming Replica data set!")
            scene_info = sceneLoadTypeCallbacks["Replica"](args.source_path, args.images, args.eval, args.n_views, args.rand_pcd)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval,
                args.rand_pcd,
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

        init_mode = getattr(args, "init_mode", "colmap_dense")
        if init_mode not in ("colmap_dense", "mvs_anywhere"):
            raise ValueError("init_mode must be 'colmap_dense' or 'mvs_anywhere'")
        self.initialization_metadata = {
            "mode": init_mode,
            "path": str(scene_info.ply_path),
            "base_initialization_sha256": self.base_initialization_sha256,
        }
        adaptive_prior = None
        if not self.loaded_iter and init_mode == "mvs_anywhere":
            strategy = getattr(args, "mvs_init_strategy", "auto")
            resolved_strategy = resolve_mvs_init_strategy(strategy)
            base_point_count = int(scene_info.point_cloud.points.shape[0])
            try:
                effective_max_points = resolve_mvs_init_point_budget(
                    strategy,
                    resolved_strategy,
                    args.mvs_init_max_points,
                )
                residual_budget = min(
                    int(args.mvs_init_max_points),
                    int(args.mvs_init_dense_residual_points),
                )
                # The adaptive scorer and sampler must operate on the same
                # resource-bounded population that can actually be admitted.
                # Otherwise a larger, randomly thinned candidate cache changes
                # the spatial distribution even when the final point budget is
                # unchanged, confounding reliability with reservoir sampling.
                if resolved_strategy == "adaptive":
                    effective_max_points = min(
                        effective_max_points, residual_budget
                    )
                mvs_pcd, mvs_path, mvs_metadata = load_mvs_point_cloud(
                    args.source_path,
                    n_views=args.n_views,
                    train_camera_names=[
                        str(camera.image_name)
                        for camera in scene_info.train_cameras
                    ],
                    explicit_path=args.mvs_init_path,
                    max_points=effective_max_points,
                )
                mvs_metadata["configured_max_points"] = int(
                    args.mvs_init_max_points
                )
                if resolved_strategy == "adaptive":
                    adaptive_prior = build_adaptive_mvs_prior(
                        scene_info.point_cloud,
                        mvs_pcd,
                        scene_info.train_cameras,
                        cache_path=mvs_path,
                        max_residual_points=residual_budget,
                    )
                    mvs_metadata.update(adaptive_prior.metadata)
                    mvs_metadata.update(
                        {
                            "base_point_count": base_point_count,
                            "fused_point_count": (
                                base_point_count + adaptive_prior.point_count
                            ),
                        }
                    )
                elif resolved_strategy == "replace":
                    scene_info = scene_info._replace(
                        point_cloud=mvs_pcd, ply_path=str(mvs_path)
                    )
                    mvs_metadata.update(
                        {
                            "fusion_policy": "explicit_replace_ablation",
                            "base_point_count": base_point_count,
                            "mvs_point_count": int(mvs_pcd.points.shape[0]),
                            "fused_point_count": int(mvs_pcd.points.shape[0]),
                        }
                    )
                else:
                    total_max_points = resolve_mvs_fusion_budget(
                        strategy,
                        args.mvs_init_total_max_points,
                        base_point_count=base_point_count,
                        dense_base_threshold=args.mvs_init_dense_base_threshold,
                        dense_residual_points=args.mvs_init_dense_residual_points,
                    )
                    fused_pcd, fusion_metadata = fuse_residual_mvs_point_cloud(
                        scene_info.point_cloud,
                        mvs_pcd,
                        total_max_points=total_max_points,
                    )
                    scene_info = scene_info._replace(
                        point_cloud=fused_pcd, ply_path=str(mvs_path)
                    )
                    mvs_metadata.update(fusion_metadata)
            except FileNotFoundError as error:
                if resolved_strategy != "adaptive":
                    raise
                mvs_path = scene_info.ply_path
                mvs_metadata = {
                    "path": str(mvs_path),
                    "fusion_policy": "adaptive_missing_prior_baseline",
                    "base_point_count": base_point_count,
                    "mvs_point_count": 0,
                    "fused_point_count": base_point_count,
                    "global_reliability": 0.0,
                    "load_error": str(error),
                    "camera_provenance_verified": None,
                }
            mvs_metadata.update(
                {
                    "requested_strategy": strategy,
                    "resolved_strategy": resolved_strategy,
                }
            )
            self.initialization_metadata = {
                "mode": init_mode,
                "base_initialization_sha256": self.base_initialization_sha256,
                **mvs_metadata,
            }

        if not self.loaded_iter:
            input_path = os.path.join(self.model_path, "input.ply")
            if init_mode == "mvs_anywhere" or getattr(args, "rand_pcd", False):
                input_points = np.asarray(scene_info.point_cloud.points)
                input_colors = np.asarray(scene_info.point_cloud.colors)
                if adaptive_prior is not None and adaptive_prior.point_count:
                    input_points = np.concatenate(
                        (input_points, adaptive_prior.point_cloud.points), axis=0
                    )
                    input_colors = np.concatenate(
                        (input_colors, adaptive_prior.point_cloud.colors), axis=0
                    )
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
                adaptive_prior=adaptive_prior,
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
