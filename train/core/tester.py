import os
import cv2
import torch
import tqdm
from loguru import logger
import numpy as np
from . import constants
from multi_person_tracker import MPT
from torchvision.transforms import Normalize
from glob import glob
from train.utils.train_utils import load_pretrained_model
from train.utils.vibe_image_utils import get_single_image_crop_demo
from train.utils.vis_utils import draw_skeleton

from . import config
from .config import update_hparams
from ..models.hmr import HMR
from ..models.head.smplx_cam_head import SMPLXCamHead
from ..utils.renderer_cam import render_image_group
from ..utils.renderer_pyrd import Renderer
from ..utils.image_utils import crop
import io
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R

import time
from torch.utils.data import DataLoader
from multi_person_tracker.data import ImageFolder


class MTPUpdate(MPT):

    def most_confident_detect(self, image_folder, output_file=None):
        image_dataset = ImageFolder(image_folder)

        dataloader = DataLoader(
            image_dataset, batch_size=self.batch_size, num_workers=0
        )
        start = time.time()
        detections = []
        for batch in dataloader:
            batch = batch.to(self.device)

            predictions = self.detector(batch)
            if not len(predictions[0]["boxes"]):
                detections.append([])
                continue
            # Finding the index of the box with the highest score
            max_index = torch.argmax(predictions[0]["scores"]).item()
            # Selecting the box with the highest score
            det = predictions[0]["boxes"][max_index]
            detections.append([det])

        runtime = time.time() - start
        fps = len(dataloader.dataset) / runtime

        if self.display:
            self.display_detection_results(image_folder, detections, output_file)
        detections = self.prepare_output_detections(detections)
        return detections


class Tester:
    def __init__(self, args):
        self.args = args
        self.model_cfg = update_hparams(args.cfg)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.normalize_img = Normalize(
            mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD
        )
        self.bboxes_dict = {}

        self.model = self._build_model()
        self.smplx_cam_head = SMPLXCamHead(img_res=self.model_cfg.DATASET.IMG_RES).to(
            self.device
        )
        self._load_pretrained_model()
        self.model.eval()

        ## @@@ added items
        self.smplx2smpl = pickle.load(open(config.SMPLX2SMPL, "rb"))
        self.smplx2smpl = torch.tensor(
            self.smplx2smpl["matrix"][None], dtype=torch.float32
        ).to(self.device)
        self.J_regressor = (
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M))
            .float()
            .to(self.device)
        )
        self.joint_mapper_h36m = constants.H36M_TO_J17  ###@@@@@ make it to 14 from 17

    def _build_model(self):
        self.hparams = self.model_cfg
        model = HMR(
            backbone=self.hparams.MODEL.BACKBONE,
            img_res=self.hparams.DATASET.IMG_RES,
            pretrained_ckpt=self.hparams.TRAINING.PRETRAINED_CKPT,
            hparams=self.hparams,
        ).to(self.device)
        return model

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        logger.info(f"Loading pretrained model from {self.args.ckpt}")
        ckpt = torch.load(self.args.ckpt)["state_dict"]
        load_pretrained_model(
            self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True
        )
        logger.info(f'Loaded pretrained weights from "{self.args.ckpt}"')

    def run_detector(self, all_image_folder):
        # run multi object tracker
        mot = MTPUpdate(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=False,
            detector_type=self.args.detector,
            output_format="dict",
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = []
        for fold_id, image_folder in enumerate(all_image_folder):
            bboxes.append(
                mot.most_confident_detect(image_folder)
            )  # See below comment for most_confident_detect

        return bboxes

    """ Copy paste inside library multi_person_tracker of mtp.py for 
    
    def most_confident_detect(self, image_folder, output_file=None):
        image_dataset = ImageFolder(image_folder)

        dataloader = DataLoader(image_dataset, batch_size=self.batch_size, num_workers=0)
        start = time.time()
        # print('Running Multi-Person-Tracker')
        detections = []
        for batch in dataloader:
            batch = batch.to(self.device)

            predictions = self.detector(batch)
            if not len(predictions[0]['boxes']):
                detections.append([])
                continue
            # Finding the index of the box with the highest score
            max_index = torch.argmax(predictions[0]['scores']).item()
            # Selecting the box with the highest score
            det = predictions[0]['boxes'][max_index]
            detections.append([det])

        runtime = time.time() - start
        fps = len(dataloader.dataset) / runtime
        # print(f'Finished. Detection + Tracking FPS {fps:.2f}')


        if self.display:
            self.display_detection_results(image_folder, detections, output_file)
        detections = self.prepare_output_detections(detections)
        return detections
    """

    def load_yolov5_bboxes(self, all_bbox_folder):
        # run multi object tracker
        for fold_id, bbox_folder in enumerate(all_bbox_folder):
            for bbox_file in os.listdir(bbox_folder):
                bbox = np.loadtxt(os.path.join(bbox_folder, bbox_file))
                fname = os.path.join(
                    "/".join(bbox_folder.split("/")[-3:-1]),
                    bbox_file.replace(".txt", ".png"),
                )
                self.bboxes_dict[fname] = bbox

    @torch.no_grad()
    def run_on_image_folder(
        self, all_image_folder, detections, output_folder, visualize_proj=True
    ):
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")
            ]
            image_file_names = sorted(image_file_names)
            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                # breakpoint()
                dets = detections[fold_idx][img_idx]
                if len(dets) < 1:
                    img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                    basename = img_fname.split("/")[-1]
                    filename = basename + "pred_%s.jpg" % "bedlam"
                    front_view_path = os.path.join(output_folder, filename)
                    logger.info(f"Writing output files to {output_folder}")
                    cv2.imwrite(front_view_path, img[:, :, ::-1])
                    continue

                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(
                    len(dets),
                    3,
                    self.model_cfg.DATASET.IMG_RES,
                    self.model_cfg.DATASET.IMG_RES,
                    device=self.device,
                    dtype=torch.float,
                )

                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []

                for det_idx, det in enumerate(dets):
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.0)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(
                        img,
                        bbox_center[-1],
                        bbox_scale[-1],
                        [
                            self.model_cfg.DATASET.IMG_RES,
                            self.model_cfg.DATASET.IMG_RES,
                        ],
                    )
                    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
                hmr_output = self.model(
                    inp_images,
                    bbox_center=bbox_center,
                    bbox_scale=bbox_scale,
                    img_w=img_w,
                    img_h=img_h,
                )
                focal_length = (img_w * img_w + img_h * img_h) ** 0.5
                pred_vertices_array = (
                    (hmr_output["vertices"] + hmr_output["pred_cam_t"].unsqueeze(1))
                    .detach()
                    .cpu()
                    .numpy()
                )
                renderer = Renderer(
                    focal_length=focal_length[0],
                    img_w=img_w[0],
                    img_h=img_h[0],
                    faces=self.smplx_cam_head.smplx.faces,
                    same_mesh_color=False,
                )
                front_view = renderer.render_front_view(
                    pred_vertices_array, bg_img_rgb=img.copy()
                )

                # save rendering results
                basename = img_fname.split("/")[-1]
                filename = basename + "pred_%s.jpg" % "bedlam"
                filename_orig = basename + "orig_%s.jpg" % "bedlam"
                front_view_path = os.path.join(output_folder, filename)
                orig_path = os.path.join(output_folder, filename_orig)
                logger.info(f"Writing output files to {output_folder}")
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                # cv2.imwrite(orig_path, img[:, :, ::-1])
                renderer.delete()

    ##@@@@@@@@
    def rotation_matrix_to_axis_angle(self, rot_matrices, size=24):
        """
        Convert a batch of rotation matrices to axis-angle representations.
        Args:
            rot_matrices (numpy.ndarray): An array of shape (24, 3, 3) containing rotation matrices.
        Returns:
            numpy.ndarray: An array of shape (72, 3) containing axis-angle representations.
        """
        axis_angles = np.zeros(
            (size, 3)
        )  # Initialize the array for axis-angle representations
        for i in range(size):
            # Convert each rotation matrix to an axis-angle representation
            rotation = R.from_matrix(rot_matrices[i])
            axis_angle = rotation.as_rotvec()
            axis_angles[i] = axis_angle
        return axis_angles.flatten()

    def run_on_image_folder_without_detection(self, all_image_folder, detections):
        # for fold_idx, image_folder in enumerate(all_image_folder):
        #     image_file_names = [
        #         os.path.join(image_folder, x)
        #         for x in os.listdir(image_folder)
        #         if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        #     ]
        #     image_file_names = (sorted(image_file_names))
        pred_pose = []
        pred_shape = []
        pred_cam = []
        for img_idx in tqdm.tqdm(range(len(all_image_folder))):
            img = all_image_folder[img_idx]
            dets = detections[img_idx]
            # if len(dets) < 1:
            #     img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            #     basename = img_fname.split('/')[-1]
            #     filename = basename + "pred_%s.jpg" % 'bedlam'
            #     front_view_path = os.path.join(output_folder, filename)
            #     logger.info(f'Writing output files to {output_folder}')
            #     cv2.imwrite(front_view_path, img[:, :, ::-1])
            #     continue

            # img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            orig_height, orig_width = img.shape[:2]
            inp_images = torch.zeros(
                len(dets),
                3,
                self.model_cfg.DATASET.IMG_RES,
                self.model_cfg.DATASET.IMG_RES,
                device=self.device,
                dtype=torch.float,
            )

            batch_size = inp_images.shape[0]
            bbox_scale = []
            bbox_center = []

            for det_idx, det in enumerate(dets):
                bbox = det
                bbox_scale.append(bbox[2] / 200.0)
                bbox_center.append([bbox[0], bbox[1]])
                rgb_img = crop(
                    img.cpu().numpy(),
                    bbox_center[-1],
                    bbox_scale[-1],
                    [self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES],
                )
                rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
                rgb_img = torch.from_numpy(rgb_img)
                norm_img = self.normalize_img(rgb_img)
                inp_images[det_idx] = norm_img.float().to(self.device)

            bbox_center = torch.tensor(bbox_center).cuda().float()
            bbox_scale = torch.tensor(bbox_scale).cuda().float()
            img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
            img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
            hmr_output = self.model(
                inp_images,
                bbox_center=bbox_center,
                bbox_scale=bbox_scale,
                img_w=img_w,
                img_h=img_h,
            )
            # breakpoint()
            pred_pose.append(
                self.rotation_matrix_to_axis_angle(
                    hmr_output["pred_pose"].detach().cpu().numpy().squeeze(0), size=22
                )[None, :]
            )
            pred_shape.append(hmr_output["pred_shape"].detach().cpu().numpy())
            pred_cam.append(hmr_output["pred_cam"].detach().cpu().numpy())

        pred_pose = np.concatenate(pred_pose, 0)
        pred_shape = np.concatenate(pred_shape, 0)
        pred_cam = np.concatenate(pred_cam, 0)

        return pred_pose, pred_shape, pred_cam

    @torch.no_grad()
    def run_on_hbw_folder(
        self,
        all_image_folder,
        detections,
        output_folder,
        data_split="test",
        visualize_proj=True,
    ):
        img_names = []
        verts = []
        image_file_names = []
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")
            ]
            image_file_names = sorted(image_file_names)
            print(image_folder, len(image_file_names))

            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):
                if detections:
                    dets = detections[fold_idx][img_idx]
                    if len(dets) < 1:
                        img_names.append(
                            "/".join(img_fname.split("/")[-4:]).replace(
                                data_split + "_small_resolution", data_split
                            )
                        )
                        template_verts = (
                            self.smplx_cam_head.smplx()
                            .vertices[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        verts.append(template_verts)
                        continue
                else:
                    match_fname = "/".join(img_fname.split("/")[-3:])
                    if match_fname not in self.bboxes_dict.keys():
                        img_names.append(
                            "/".join(img_fname.split("/")[-4:]).replace(
                                data_split + "_small_resolution", data_split
                            )
                        )
                        template_verts = (
                            self.smplx_cam_head.smplx()
                            .vertices[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        verts.append(template_verts)
                        continue
                    dets = self.bboxes_dict[match_fname]
                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(
                    1,
                    3,
                    self.model_cfg.DATASET.IMG_RES,
                    self.model_cfg.DATASET.IMG_RES,
                    device=self.device,
                    dtype=torch.float,
                )
                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []
                if len(dets.shape) == 1:
                    dets = np.expand_dims(dets, 0)
                for det_idx, det in enumerate(dets):
                    if det_idx >= 1:
                        break
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.0)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(
                        img,
                        bbox_center[-1],
                        bbox_scale[-1],
                        [
                            self.model_cfg.DATASET.IMG_RES,
                            self.model_cfg.DATASET.IMG_RES,
                        ],
                    )
                    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()

                hmr_output = self.model(
                    inp_images,
                    bbox_center=bbox_center,
                    bbox_scale=bbox_scale,
                    img_w=img_w,
                    img_h=img_h,
                )
                img_names.append(
                    "/".join(img_fname.split("/")[-4:]).replace(
                        data_split + "_small_resolution", data_split
                    )
                )
                template_verts = (
                    self.smplx_cam_head.smplx(
                        betas=hmr_output["pred_shape"], pose2rot=False
                    )
                    .vertices[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                verts.append(template_verts)
                if visualize_proj:
                    focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                    pred_vertices_array = (
                        (hmr_output["vertices"][0] + hmr_output["pred_cam_t"])
                        .unsqueeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    renderer = Renderer(
                        focal_length=focal_length,
                        img_w=img_w,
                        img_h=img_h,
                        faces=self.smplx_cam_head.smplx.faces,
                        same_mesh_color=False,
                    )
                    front_view = renderer.render_front_view(
                        pred_vertices_array, bg_img_rgb=img.copy()
                    )

                    # save rendering results
                    basename = (
                        img_fname.split("/")[-3]
                        + "_"
                        + img_fname.split("/")[-2]
                        + "_"
                        + img_fname.split("/")[-1]
                    )
                    filename = basename + "pred_%s.jpg" % "bedlam"
                    filename_orig = basename + "orig_%s.jpg" % "bedlam"
                    front_view_path = os.path.join(output_folder, filename)
                    orig_path = os.path.join(output_folder, filename_orig)
                    logger.info(f"Writing output files to {output_folder}")
                    cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                    cv2.imwrite(orig_path, img[:, :, ::-1])
                    renderer.delete()
        np.savez(
            os.path.join(output_folder, data_split + "_hbw_prediction.npz"),
            image_name=img_names,
            v_shaped=verts,
        )

    def run_on_dataframe(self, dataframe_path, output_folder, visualize_proj=True):
        dataframe = np.load(dataframe_path)
        centers = dataframe["center"]
        scales = dataframe["scale"]
        image = dataframe["image"]
        for ind, center in tqdm.tqdm(enumerate(centers)):
            center = centers[ind]
            scale = scales[ind]
            img = image[ind]
            orig_height, orig_width = img.shape[:2]
            rgb_img = crop(
                img,
                center,
                scale,
                [self.hparams.DATASET.IMG_RES, self.hparams.DATASET.IMG_RES],
            )

            rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
            rgb_img = torch.from_numpy(rgb_img).float().cuda()
            rgb_img = self.normalize_img(rgb_img)

            img_h = torch.tensor(orig_height).repeat(1).cuda().float()
            img_w = torch.tensor(orig_width).repeat(1).cuda().float()
            center = torch.tensor(center).cuda().float()
            scale = torch.tensor(scale).cuda().float()

            hmr_output = self.model(
                rgb_img.unsqueeze(0),
                bbox_center=center.unsqueeze(0),
                bbox_scale=scale.unsqueeze(0),
                img_w=img_w,
                img_h=img_h,
            )
            # Need to convert SMPL-X meshes to SMPL using conversion tool before calculating error
            import trimesh

            mesh = trimesh.Trimesh(
                vertices=hmr_output["vertices"][0].detach().cpu().numpy(),
                faces=self.smplx_cam_head.smplx.faces,
            )
            output_mesh_path = os.path.join(output_folder, str(ind) + ".obj")
            mesh.export(output_mesh_path)

            if visualize_proj:
                focal_length = (img_w * img_w + img_h * img_h) ** 0.5

                pred_vertices_array = (
                    (hmr_output["vertices"][0] + hmr_output["pred_cam_t"])
                    .unsqueeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                renderer = Renderer(
                    focal_length=focal_length,
                    img_w=img_w,
                    img_h=img_h,
                    faces=self.smplx_cam_head.smplx.faces,
                    same_mesh_color=False,
                )
                front_view = renderer.render_front_view(
                    pred_vertices_array, bg_img_rgb=img.copy()
                )

                # save rendering results
                basename = str(ind)
                filename = basename + "pred_%s.jpg" % "bedlam"
                filename_orig = basename + "orig_%s.jpg" % "bedlam"
                front_view_path = os.path.join(output_folder, filename)
                orig_path = os.path.join(output_folder, filename_orig)
                logger.info(f"Writing output files to {output_folder}")
                cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                cv2.imwrite(orig_path, img[:, :, ::-1])
                renderer.delete()

    def interpolate_matrices(self, A, B, num_points):
        # returns points including start A and end B,
        # numpoints if 1 then result is [A,i1,B]
        # Generating interpolation points
        interp_points = np.linspace(0, 1, num_points + 2)

        # Perform linear interpolation
        interpolated_matrices = []
        for i in interp_points:
            interpolated_matrix = A * (1 - i) + B * i
            interpolated_matrices.append(interpolated_matrix)

        return interpolated_matrices

    def interpolate_list(self, lst):

        if not lst:
            return None

        start = 0
        while start < len(lst):

            if len(lst[start]) > 0:
                start += 1

            else:
                intp_start = start - 1
                while len(lst[start]) == 0:
                    start += 1

                intp_end = start
                ip_mats = self.interpolate_matrices(
                    lst[intp_start], lst[intp_end], intp_end - intp_start - 1
                )
                for cnt, i in enumerate(range(intp_start, intp_end + 1)):
                    lst[i] = ip_mats[cnt]
                start += 1
        return lst

    def extrapolate_list(self, lst):
        start_len, end_len = len(lst[0]), len(lst[-1])
        total_size = sum([len(i) for i in lst])

        if not total_size:
            return None

        # pad intial empty values
        if not start_len:
            start = 0
            while start < len(lst):
                start += 1
                if len(lst[start]) > 0:
                    break
            for i in range(start + 1):
                lst[i] = lst[start]

        # pad ending empty values
        if not end_len:
            start = len(lst) - 1
            while start > 0:
                start -= 1
                if len(lst[start]) > 0:
                    break
            for i in range(start, len(lst)):
                lst[i] = lst[start]
        return lst

    def fill_missing_values(self, data, interpolated_vals):
        if not interpolated_vals:
            print(
                "There was no detection, interpolation failed for, ", list(data.keys())
            )
            return None
        for cnt, (k, v) in enumerate(data.items()):
            if not len(v):
                data[k] = interpolated_vals[cnt]
        return data

    @torch.no_grad()
    def save_pseudo_labels(
        self,
        all_image_folder,
        detections,
        output_folder,
        visualize_proj=True,
        torch_save=True,
    ):
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")
            ]
            image_file_names = sorted(image_file_names)

            empty_det_count = sum(
                [1 for array in detections[fold_idx] if len(array) == 0]
            )
            logger.info(
                f"YOLO Failed for {empty_det_count} out of {len(detections[fold_idx])} samples"
            )

            # if empty_det_count > 0:
            #     ## START INTERPOLATING bboxes for which detections failed.
            #     det = self.linear_interpolation(detections[fold_idx])

            # store_vertices = []
            # store_cam_t = []
            store_vertices = {}
            store_cam = {}
            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):

                dets = detections[fold_idx][img_idx]
                if len(dets) < 1:
                    store_vertices[img_fname] = []
                    store_cam[img_fname] = []
                    continue

                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(
                    len(dets),
                    3,
                    self.model_cfg.DATASET.IMG_RES,
                    self.model_cfg.DATASET.IMG_RES,
                    device=self.device,
                    dtype=torch.float,
                )

                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []
                for det_idx, det in enumerate(dets):
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.0)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(
                        img,
                        bbox_center[-1],
                        bbox_scale[-1],
                        [
                            self.model_cfg.DATASET.IMG_RES,
                            self.model_cfg.DATASET.IMG_RES,
                        ],
                    )
                    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
                hmr_output = self.model(
                    inp_images,
                    bbox_center=bbox_center,
                    bbox_scale=bbox_scale,
                    img_w=img_w,
                    img_h=img_h,
                )
                store_vertices[img_fname] = hmr_output["vertices"].detach().cpu()
                store_cam[img_fname] = (
                    hmr_output["pred_cam_t"].detach().cpu().unsqueeze(1)
                )
                # save rendering results
                # basename = img_fname.split('/')[-1]
                # filename = basename[:-4] + ".pth"
                # front_view_path = os.path.join(output_folder, filename)
                # torch.save(store_vertices, front_view_path)

        # DO extrapolation and interpolations

        ### For pose
        result_vertices = self.interpolate_list(
            self.extrapolate_list(list(store_vertices.values()))
        )
        store_vertices = self.fill_missing_values(store_vertices, result_vertices)

        ### For camera
        result_cam = self.interpolate_list(
            self.extrapolate_list(list(store_cam.values()))
        )
        store_cam = self.fill_missing_values(store_cam, result_cam)

        for img_fname, vert_x in tqdm.tqdm(store_vertices.items()):

            basename = img_fname.split("/")[-1]
            filename = basename[:-4] + ".pth"
            front_view_path = os.path.join(output_folder, filename)
            save_dict = {
                "vertices": vert_x,
                "pred_cam_t": store_cam[img_fname],
                "img_name": img_fname,
            }
            if torch_save:
                torch.save(save_dict, front_view_path)

            if visualize_proj:
                batch_size = 1
                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)

                orig_height, orig_width = img.shape[:2]
                focal_length = (img_w * img_w + img_h * img_h) ** 0.5
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()

                pred_vertices_array = (
                    (save_dict["vertices"] + save_dict["pred_cam_t"])
                    .detach()
                    .cpu()
                    .numpy()
                )
                renderer = Renderer(
                    focal_length=focal_length[0],
                    img_w=img_w[0],
                    img_h=img_h[0],
                    faces=self.smplx_cam_head.smplx.faces,
                    same_mesh_color=False,
                )
                front_view = renderer.render_front_view(
                    pred_vertices_array, bg_img_rgb=img.copy()
                )

                # save rendering results

                # basename = img_fname.split('/')[-3]+'_'+img_fname.split('/')[-2]+'_'+img_fname.split('/')[-1]
                # filename = basename + "pred_%s.jpg" % 'bedlam'
                # filename_orig = basename + "orig_%s.jpg" % 'bedlam'
                # front_view_path = os.path.join(output_folder, filename)
                # orig_path = os.path.join(output_folder, filename_orig)
                # front_view_path = img_fname.replace()
                render_path = front_view_path.replace("pth", "jpg").replace(
                    "bedlam_pl", "render_im"
                )
                os.makedirs(os.path.dirname(render_path), exist_ok=True)
                cv2.imwrite(render_path, front_view[:, :, ::-1])
                renderer.delete()

    @torch.no_grad()
    def save_keypoint_vis(
        self,
        all_image_folder,
        detections,
        output_folder,
        store_3d_kps=False,
        store_2d_kps=True,
        visualize_proj=True,
    ):

        def perspective_projection(points, rotation, translation, cam_intrinsics):
            K = cam_intrinsics
            points = torch.einsum("bij,bkj->bki", rotation, points)
            points = points + translation.unsqueeze(1)
            projected_points = points / points[:, :, -1].unsqueeze(-1)
            projected_points = torch.einsum("bij,bkj->bki", K, projected_points.float())
            return projected_points[:, :, :-1]

        def vis(kp_2d, img_fname, save_img_pth, save_2d=True):
            """
            Visualise 2d keypoints on image
            """
            # y = torch.load(load_pth) # Saved using HMRoutput in
            # kp_2d = y['joints2d'].squeeze().cpu()
            image = plt.imread(img_fname)
            image = draw_skeleton(image, kp_2d, unnormalize=False, res=224)
            if save_2d:
                plt.imsave(save_img_pth, image)
            # print('saved vis at:',save_img_pth )

        save_2d_kp = {}
        for fold_idx, image_folder in enumerate(all_image_folder):
            image_file_names = [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")
            ]
            image_file_names = sorted(image_file_names)
            for img_idx, img_fname in tqdm.tqdm(enumerate(image_file_names)):

                dets = detections[fold_idx][img_idx]
                if len(dets) < 1:
                    img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                    basename = img_fname.split("/")[-1]
                    filename = basename + "pred_%s.jpg" % "bedlam"
                    filename_orig = basename + "orig_%s.jpg" % "bedlam"
                    front_view_path = os.path.join(output_folder, filename)
                    orig_path = os.path.join(output_folder, filename)
                    logger.info(f"Writing output files to {output_folder}")
                    # cv2.imwrite(front_view_path, front_view[:, :, ::-1])
                    cv2.imwrite(orig_path, img[:, :, ::-1])
                    continue

                # print('check', img_idx, img_fname)
                img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
                orig_height, orig_width = img.shape[:2]
                inp_images = torch.zeros(
                    len(dets),
                    3,
                    self.model_cfg.DATASET.IMG_RES,
                    self.model_cfg.DATASET.IMG_RES,
                    device=self.device,
                    dtype=torch.float,
                )

                batch_size = inp_images.shape[0]
                bbox_scale = []
                bbox_center = []

                for det_idx, det in enumerate(dets):
                    bbox = det
                    bbox_scale.append(bbox[2] / 200.0)
                    bbox_center.append([bbox[0], bbox[1]])
                    rgb_img = crop(
                        img,
                        bbox_center[-1],
                        bbox_scale[-1],
                        [
                            self.model_cfg.DATASET.IMG_RES,
                            self.model_cfg.DATASET.IMG_RES,
                        ],
                    )
                    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
                    rgb_img = torch.from_numpy(rgb_img)
                    norm_img = self.normalize_img(rgb_img)
                    inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_center = torch.tensor(bbox_center).cuda().float()
                bbox_scale = torch.tensor(bbox_scale).cuda().float()
                img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
                img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
                focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
                hmr_output = self.model(
                    inp_images,
                    bbox_center=bbox_center,
                    bbox_scale=bbox_scale,
                    img_w=img_w,
                    img_h=img_h,
                )

                focal_length = (img_w * img_w + img_h * img_h) ** 0.5
                pred_vertices_array = (
                    (hmr_output["vertices"] + hmr_output["pred_cam_t"].unsqueeze(1))
                    .detach()
                    .cpu()
                    .numpy()
                )

                ## @@@@@@@@ changes for 2d joints
                pred_cam_vertices = hmr_output["vertices"]
                breakpoint()
                pred_cam_vertices = torch.matmul(
                    self.smplx2smpl.repeat(batch_size, 1, 1).cuda(), pred_cam_vertices
                )
                J_regressor_batch_smpl = self.J_regressor[None, :].expand(
                    batch_size, -1, -1
                )
                pred_keypoints_3d = torch.matmul(
                    J_regressor_batch_smpl, pred_cam_vertices
                )
                pred_keypoints_3d = pred_keypoints_3d[:, self.joint_mapper_h36m, :]
                # pred_keypoints_3d = pred_keypoints_3d - ((pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)

                cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
                cam_intrinsics[:, 0, 0] = focal_length  # [:, 0]
                cam_intrinsics[:, 1, 1] = focal_length  # [:, 1]
                cam_intrinsics[:, 0, 2] = img_w / 2.0
                cam_intrinsics[:, 1, 2] = img_h / 2.0

                joints2d = perspective_projection(
                    pred_keypoints_3d,
                    rotation=torch.eye(3, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1),
                    translation=hmr_output["pred_cam_t"],
                    cam_intrinsics=cam_intrinsics,
                )
                single_id = (bbox_scale == max(bbox_scale)).nonzero().item()
                joints2d = joints2d[single_id, :, :].unsqueeze(0)

                renderer = Renderer(
                    focal_length=focal_length[0],
                    img_w=img_w[0],
                    img_h=img_h[0],
                    faces=self.smplx_cam_head.smplx.faces,
                    same_mesh_color=False,
                )
                front_view = renderer.render_front_view(
                    pred_vertices_array, bg_img_rgb=img.copy()
                )

                # breakpoint()
                # save rendering results
                basename = img_fname.split("/")[-1]
                filename = basename + "pred_%s.jpg" % "bedlam"
                filename_orig = basename + "orig_%s.jpg" % "bedlam"
                filename_2d = basename + "pred2d_%s.jpg" % "bedlam"
                filename_3d = basename + "pred3d_%s.jpg" % "bedlam"
                front_view_path = os.path.join(output_folder, filename)
                front_view_path_2d = os.path.join(output_folder, filename_2d)
                front_view_path_3d = os.path.join(output_folder, filename_3d)
                orig_path = os.path.join(output_folder, filename_orig)
                logger.info(f"Writing output files to {output_folder}")
                vis(
                    joints2d.squeeze().cpu(),
                    img_fname,
                    front_view_path_2d,
                    save_2d=store_2d_kps,
                )

                save_2d_kp[img_fname] = joints2d.squeeze().cpu()

                if store_3d_kps:

                    def plot3d_keypoints(j3d1, final_save_path):

                        def get_img_from_fig(fig, dpi=120):
                            buf = io.BytesIO()
                            fig.savefig(
                                buf,
                                format="png",
                                dpi=dpi,
                                bbox_inches="tight",
                                pad_inches=0,
                            )
                            buf.seek(0)
                            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                            buf.close()
                            img = cv2.imdecode(img_arr, 1)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                            return img

                        joint_pairs = [
                            [0, 1],
                            [1, 2],
                            [3, 4],
                            [4, 5],
                            [6, 7],
                            [7, 8],
                            [8, 2],
                            [8, 9],
                            [9, 3],
                            [2, 3],
                            [8, 12],
                            [9, 10],
                            [12, 9],
                            [10, 11],
                            [12, 13],
                        ]

                        color_gt = "blue"
                        plt.tight_layout()

                        fig = plt.figure(figsize=(10, 10))
                        ax = plt.axes(projection="3d")
                        ax.set_xlim(-512, 0)
                        ax.set_ylim(-256, 256)
                        ax.set_zlim(-512, 0)
                        ax.view_init(elev=12.0, azim=80)
                        plt.tick_params(
                            left=False,
                            right=False,
                            labelleft=False,
                            labelbottom=False,
                            bottom=False,
                        )
                        if len(j3d1.shape) > 3:
                            j3d1 = j3d1[0]

                        for i in range(len(joint_pairs)):
                            limb = joint_pairs[i]
                            xs1, ys1, zs1 = [
                                np.array([j3d1[limb[0], j], j3d1[limb[1], j]])
                                for j in range(3)
                            ]
                            ax.plot(
                                -xs1,
                                -zs1,
                                -ys1,
                                color=color_gt,
                                lw=3,
                                marker="o",
                                markerfacecolor="w",
                                markersize=3,
                                markeredgewidth=2,
                                alpha=1.0,
                            )

                        frame_vis = get_img_from_fig(fig)
                        cv2.imwrite(final_save_path, frame_vis)

                    def pixel2world_vis_motion(motion, dim=2, is_tensor=False):
                        #     pose: (17,2,N)
                        N = motion.shape[-1]
                        if dim == 2:
                            offset = np.ones([2, N]).astype(np.float32)
                        else:
                            offset = np.ones([3, N]).astype(np.float32)
                            offset[2, :] = 0
                        if is_tensor:
                            offset = torch.tensor(offset)
                        return (motion + offset) * 512 / 2

                    pixel2world_3d = pixel2world_vis_motion(
                        pred_keypoints_3d.cpu().squeeze().unsqueeze(-1).numpy(), dim=3
                    )
                    plot3d_keypoints(pixel2world_3d, front_view_path_3d)

                renderer.delete()

    @torch.no_grad()
    def save_2dkeypoint_briar(self, all_image_folder, detections):

        def perspective_projection(points, rotation, translation, cam_intrinsics):
            K = cam_intrinsics
            points = torch.einsum("bij,bkj->bki", rotation, points)
            points = points + translation.unsqueeze(1)
            projected_points = points / points[:, :, -1].unsqueeze(-1)
            projected_points = torch.einsum("bij,bkj->bki", K, projected_points.float())
            return projected_points[:, :, :-1]

        def vis(kp_2d, img_fname, save_img_pth, save_2d=True):
            """
            Visualise 2d keypoints on image
            """
            # y = torch.load(load_pth) # Saved using HMRoutput in
            # kp_2d = y['joints2d'].squeeze().cpu()
            image = plt.imread(img_fname)
            image = draw_skeleton(image, kp_2d, unnormalize=False, res=224)
            if save_2d:
                plt.imsave(save_img_pth, image)
            # print('saved vis at:',save_img_pth )

        save_2d_kp = {}
        # for fold_idx, image_folder in enumerate(all_image_folder):
        #     image_file_names = [
        #         os.path.join(image_folder, x)
        #         for x in os.listdir(image_folder)
        #         if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        #     ]
        #     image_file_names = (sorted(image_file_names))
        for img_idx, img in tqdm.tqdm(enumerate(all_image_folder)):

            dets = detections[img_idx]

            # print('check', img_idx, img_fname)
            # img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
            orig_height, orig_width = img.shape[:2]
            inp_images = torch.zeros(
                len(dets),
                3,
                self.model_cfg.DATASET.IMG_RES,
                self.model_cfg.DATASET.IMG_RES,
                device=self.device,
                dtype=torch.float,
            )

            batch_size = inp_images.shape[0]
            bbox_scale = []
            bbox_center = []

            for det_idx, det in enumerate(dets):
                bbox = det
                bbox_scale.append(bbox[2] / 200.0)
                bbox_center.append([bbox[0], bbox[1]])
                rgb_img = crop(
                    img,
                    bbox_center[-1],
                    bbox_scale[-1],
                    [self.model_cfg.DATASET.IMG_RES, self.model_cfg.DATASET.IMG_RES],
                )
                rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
                rgb_img = torch.from_numpy(rgb_img)
                norm_img = self.normalize_img(rgb_img)
                inp_images[det_idx] = norm_img.float().to(self.device)

            bbox_center = torch.tensor(bbox_center).cuda().float()
            bbox_scale = torch.tensor(bbox_scale).cuda().float()
            img_h = torch.tensor(orig_height).repeat(batch_size).cuda().float()
            img_w = torch.tensor(orig_width).repeat(batch_size).cuda().float()
            # focal_length = ((img_w * img_w + img_h * img_h) ** 0.5).cuda().float()
            hmr_output = self.model(
                inp_images,
                bbox_center=bbox_center,
                bbox_scale=bbox_scale,
                img_w=img_w,
                img_h=img_h,
            )

            focal_length = (img_w * img_w + img_h * img_h) ** 0.5
            pred_vertices_array = (
                (hmr_output["vertices"] + hmr_output["pred_cam_t"].unsqueeze(1))
                .detach()
                .cpu()
                .numpy()
            )

            ## @@@@@@@@ changes for 2d joints
            pred_cam_vertices = hmr_output["vertices"]
            # pred_cam_vertices = torch.matmul(self.smplx2smpl.repeat(batch_size, 1, 1).cuda(), pred_cam_vertices)
            J_regressor_batch_smpl = self.J_regressor[None, :].expand(
                batch_size, -1, -1
            )
            pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)
            # breakpoint()
            pred_keypoints_3d = pred_keypoints_3d[:, self.joint_mapper_h36m, :]
            # pred_keypoints_3d = pred_keypoints_3d - ((pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)

            cam_intrinsics = torch.eye(3).repeat(batch_size, 1, 1).cuda().float()
            cam_intrinsics[:, 0, 0] = focal_length  # [:, 0]
            cam_intrinsics[:, 1, 1] = focal_length  # [:, 1]
            cam_intrinsics[:, 0, 2] = img_w / 2.0
            cam_intrinsics[:, 1, 2] = img_h / 2.0

            joints2d = perspective_projection(
                pred_keypoints_3d,
                rotation=torch.eye(3, device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1),
                translation=hmr_output["pred_cam_t"],
                cam_intrinsics=cam_intrinsics,
            )
            single_id = (bbox_scale == max(bbox_scale)).nonzero().item()
            joints2d = joints2d[single_id, :, :].unsqueeze(0)

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.imshow(img)
            for idx, (x, y) in enumerate(joints2d.squeeze(0).cpu().numpy()):
                ax.plot(x, y, "ro")
                ax.text(x + 5, y, str(idx), color="white", fontsize=6)
            plt.savefig(
                "/data/AmitRoyChowdhury/Yash/GRAIL_HOME/briar-eval-offline/check_2d_kp2_bedlam_lsp.jpg"
            )
            # breakpoint()

            renderer = Renderer(
                focal_length=focal_length[0],
                img_w=img_w[0],
                img_h=img_h[0],
                faces=self.smplx_cam_head.smplx.faces,
                same_mesh_color=False,
            )
            front_view = renderer.render_front_view(
                pred_vertices_array, bg_img_rgb=img.copy()
            )

            cv2.imwrite(
                "/data/AmitRoyChowdhury/Yash/GRAIL_HOME/briar-eval-offline/check_2d_kp2_bedlam_lsp_mesh.jpg",
                front_view[:, :, ::-1],
            )
            breakpoint()
            # save rendering results
            basename = img_fname.split("/")[-1]
            filename = basename + "pred_%s.jpg" % "bedlam"
            filename_orig = basename + "orig_%s.jpg" % "bedlam"
            filename_2d = basename + "pred2d_%s.jpg" % "bedlam"
            filename_3d = basename + "pred3d_%s.jpg" % "bedlam"
            front_view_path = os.path.join(output_folder, filename)
            front_view_path_2d = os.path.join(output_folder, filename_2d)
            front_view_path_3d = os.path.join(output_folder, filename_3d)
            orig_path = os.path.join(output_folder, filename_orig)
            logger.info(f"Writing output files to {output_folder}")
            vis(
                joints2d.squeeze().cpu(),
                img_fname,
                front_view_path_2d,
                save_2d=store_2d_kps,
            )

            save_2d_kp[img_fname] = joints2d.squeeze().cpu()
