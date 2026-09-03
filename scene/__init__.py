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
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.cameras import PseudoCamera
from utils.mvs_init import (
    fuse_residual_mvs_point_cloud,
    load_mvs_point_cloud,
    resolve_mvs_fusion_budget,
    resolve_mvs_init_point_budget,
    resolve_mvs_init_strategy,
    should_use_dtu_mvs_prior,
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

        init_mode = getattr(args, "init_mode", "colmap_dense")
        if init_mode not in ("colmap_dense", "mvs_anywhere"):
            raise ValueError("init_mode must be 'colmap_dense' or 'mvs_anywhere'")
        self.initialization_metadata = {
            "mode": init_mode,
            "path": str(scene_info.ply_path),
        }
        if not self.loaded_iter and init_mode == "mvs_anywhere":
            strategy = getattr(args, "mvs_init_strategy", "auto")
            is_synthetic = os.path.exists(
                os.path.join(args.source_path, "transforms_train.json")
            )
            resolved_strategy = resolve_mvs_init_strategy(
                strategy,
                n_views=args.n_views,
                is_synthetic=is_synthetic,
                dataset_name=getattr(args, "dataset_name", "auto"),
            )
            base_point_count = int(scene_info.point_cloud.points.shape[0])
            dataset_name = str(getattr(args, "dataset_name", "auto")).lower()
            dtu_gate_enabled = strategy == "auto" and dataset_name == "dtu"
            dtu_gate_pre_rejected = dtu_gate_enabled and (
                int(args.n_views) > 3
                or base_point_count >= int(args.mvs_init_dtu_max_base_points)
            )
            if dtu_gate_pre_rejected:
                resolved_strategy = "passthrough"
            skip_mvs = resolved_strategy == "passthrough"
            if skip_mvs:
                mvs_pcd = scene_info.point_cloud
                mvs_path = scene_info.ply_path
                mvs_metadata = {
                    "path": str(mvs_path),
                    "fusion_policy": (
                        "dtu_quality_gate_passthrough"
                        if dtu_gate_pre_rejected
                        else (
                            "synthetic_base_passthrough"
                            if is_synthetic
                            else "dense_base_passthrough"
                        )
                    ),
                    "base_point_count": base_point_count,
                    "mvs_point_count": 0,
                    "fused_point_count": base_point_count,
                    "dense_base_threshold": int(
                        args.mvs_init_dense_base_threshold
                    ),
                    "camera_provenance_verified": None,
                }
                if dtu_gate_pre_rejected:
                    mvs_metadata["quality_gate"] = {
                        "enabled": True,
                        "use_mvs_prior": False,
                        "max_base_points": int(
                            args.mvs_init_dtu_max_base_points
                        ),
                        "min_mvs_source_points": int(
                            args.mvs_init_dtu_min_source_points
                        ),
                    }
            else:
                effective_max_points = resolve_mvs_init_point_budget(
                    strategy,
                    resolved_strategy,
                    args.mvs_init_max_points,
                    dataset_name=getattr(args, "dataset_name", "auto"),
                    dense_base=(
                        base_point_count >= args.mvs_init_dense_base_threshold
                    ),
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
                use_mvs_prior = should_use_dtu_mvs_prior(
                    strategy,
                    dataset_name=dataset_name,
                    n_views=args.n_views,
                    base_point_count=base_point_count,
                    mvs_source_point_count=mvs_metadata["source_point_count"],
                    max_base_points=args.mvs_init_dtu_max_base_points,
                    min_mvs_source_points=args.mvs_init_dtu_min_source_points,
                )
                mvs_metadata["quality_gate"] = {
                    "enabled": (
                        strategy == "auto" and dataset_name == "dtu"
                    ),
                    "use_mvs_prior": use_mvs_prior,
                    "max_base_points": int(args.mvs_init_dtu_max_base_points),
                    "min_mvs_source_points": int(
                        args.mvs_init_dtu_min_source_points
                    ),
                }
                if not use_mvs_prior:
                    resolved_strategy = "passthrough"
                    mvs_pcd = scene_info.point_cloud
                    mvs_metadata.update(
                        {
                            "fusion_policy": "dtu_quality_gate_passthrough",
                            "base_point_count": base_point_count,
                            "mvs_point_count": 0,
                            "fused_point_count": base_point_count,
                        }
                    )
                if resolved_strategy == "augment":
                    total_max_points = resolve_mvs_fusion_budget(
                        strategy,
                        args.mvs_init_total_max_points,
                        base_point_count=base_point_count,
                        dense_base_threshold=args.mvs_init_dense_base_threshold,
                        dense_residual_points=args.mvs_init_dense_residual_points,
                        dataset_name=getattr(args, "dataset_name", "auto"),
                    )
                    mvs_pcd, fusion_metadata = fuse_residual_mvs_point_cloud(
                        scene_info.point_cloud,
                        mvs_pcd,
                        total_max_points=total_max_points,
                    )
                    mvs_metadata.update(fusion_metadata)
            mvs_metadata.update(
                {
                    "requested_strategy": strategy,
                    "resolved_strategy": resolved_strategy,
                }
            )
            scene_info = scene_info._replace(
                point_cloud=mvs_pcd, ply_path=str(mvs_path)
            )
            self.initialization_metadata = {
                "mode": init_mode,
                **mvs_metadata,
            }

        if not self.loaded_iter:
            input_path = os.path.join(self.model_path, "input.ply")
            if init_mode == "mvs_anywhere":
                storePly(
                    input_path,
                    scene_info.point_cloud.points,
                    scene_info.point_cloud.colors * 255.0,
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
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)           

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
