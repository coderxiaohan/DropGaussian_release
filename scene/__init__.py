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

        init_mode = getattr(args, "init_mode", "mvs_anywhere")
        if init_mode not in ("colmap_dense", "mvs_anywhere"):
            raise ValueError("init_mode must be 'colmap_dense' or 'mvs_anywhere'")
        correction = getattr(args, "mvs_geometry_correction", "none")
        if not self.loaded_iter:
            if correction not in ("none", "similarity", "tps", "tps_consistent", "tps_local"):
                raise ValueError("mvs_geometry_correction must be none, similarity, tps, tps_consistent, or tps_local")
            if correction != "none" and init_mode != "mvs_anywhere":
                raise ValueError("MVS geometry correction requires init_mode=mvs_anywhere")
        self.initialization_metadata = {
            "mode": init_mode,
            "path": str(scene_info.ply_path),
            "base_initialization_sha256": self.base_initialization_sha256,
        }
        if not self.loaded_iter and init_mode == "mvs_anywhere":
            strategy = getattr(args, "mvs_init_strategy", "replace")
            resolved_strategy = resolve_mvs_init_strategy(strategy)
            base_point_count = int(scene_info.point_cloud.points.shape[0])
            # Direct MVS replacement is shared by every dataset. A missing or
            # incompatible cache must fail instead of silently running baseline.
            mvs_pcd, mvs_path, mvs_metadata = load_mvs_point_cloud(
                args.source_path,
                n_views=args.n_views,
                train_camera_names=self.train_camera_names,
                explicit_path=args.mvs_init_path,
                max_points=args.mvs_init_max_points,
            )
            if correction != "none":
                # Initialization-only: no change to dropout, loss, optimization,
                # Gaussian attribute rules, or any VCD/VCP operation.
                from utils.mvs_geometry import correct_mvs_geometry
                if not mvs_metadata.get("camera_provenance_verified", False):
                    raise ValueError("geometry correction requires a training-view-verified MVS cache")
                geometry_cameras = scene_info.train_cameras
                calibration_metadata = {"source": "provided_camera_info"}
                camera_bin = os.path.join(args.source_path, "sparse", "0", "cameras.bin")
                camera_txt = os.path.join(args.source_path, "sparse", "0", "cameras.txt")
                if os.path.isfile(camera_bin) or os.path.isfile(camera_txt):
                    from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text
                    from utils.train_view_controls import geometry_cameras_with_intrinsics
                    intrinsics = (read_intrinsics_binary(camera_bin) if os.path.isfile(camera_bin)
                                  else read_intrinsics_text(camera_txt))
                    # The legacy reader misinterprets SIMPLE_* parameter order
                    # in CameraInfo.K. Parse the model correctly only for this
                    # new geometry step; renderer/training camera data is untouched.
                    geometry_cameras, calibration_metadata = geometry_cameras_with_intrinsics(
                        scene_info.train_cameras, intrinsics,
                    )
                mvs_pcd, geometry_metadata, controls = correct_mvs_geometry(
                    mvs_pcd, geometry_cameras, mode=correction,
                    max_controls=getattr(args, "mvs_geometry_max_controls", 256),
                    smoothing=getattr(args, "mvs_geometry_smoothing", 0.001),
                    max_image_size=getattr(args, "mvs_geometry_max_image_size", 1600),
                    max_features=getattr(args, "mvs_geometry_max_features", 8000),
                    ratio=getattr(args, "mvs_geometry_ratio", 0.75),
                    reprojection_px=getattr(args, "mvs_geometry_reprojection_px", 2.0),
                    min_angle_deg=getattr(args, "mvs_geometry_min_angle_deg", 1.0),
                    match_radius_px=getattr(args, "mvs_geometry_match_radius_px", 3.0),
                    local_neighbors=getattr(args, "mvs_geometry_local_neighbors", 64),
                )
                geometry_metadata["calibration"] = calibration_metadata
                np.savez_compressed(os.path.join(self.model_path, "geometry_controls.npz"), **controls)
                with open(os.path.join(self.model_path, "geometry_correction.json"), "w") as stream:
                    json.dump(geometry_metadata, stream, indent=2, sort_keys=True)
                mvs_metadata["geometry_correction"] = geometry_metadata
                mvs_metadata["uncorrected_selected_sha256"] = mvs_metadata["selected_sha256"]
                mvs_metadata["selected_sha256"] = geometry_metadata["output_sha256"]
                print(f"MVS geometry {correction}: {geometry_metadata['selected_controls']} controls, "
                      f"median displacement {geometry_metadata['displacement_median']:.6g}")
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
