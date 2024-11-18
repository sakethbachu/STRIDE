import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data
from tqdm import tqdm

batch_size = 1
smplx2smpl = pickle.load(open("./data/utils/smplx2smpl.pkl", "rb"))
smplx2smpl_update = torch.tensor(smplx2smpl["matrix"][None], dtype=torch.float32).cuda()
J_regressor = (
    torch.from_numpy(np.load("./data/utils/J_regressor_h36m.npy")).float().cuda()
)
J_regressor_batch_smpl = J_regressor[None, :].expand(batch_size, -1, -1)
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
joint_mapper_gt = J24_TO_J17[:14]


def convert_vertices_to_3dkp_and_gt(file, cliff=None):
    ## get the gt_kp
    # gt_keypoints_3d = torch.tensor(torch.load(file)['joints']).unsqueeze(0).cuda()
    # # gt_keypoints_3d = gt_keypoints_3d[:, self.joint_mapper_gt, :-1]
    # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
    # gt_keypoints_3d = gt_keypoints_3d - ((gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2).unsqueeze(1)

    ## get the pl kp
    # pl_file = file.replace('gt_h36m', pl_path)
    # print(file)
    pl_data = torch.load(file)
    img_name = pl_data["img_name"]
    # print(img_name, pl_data['vertices'].shape)
    pred_cam_vertices = pl_data["vertices"][0].unsqueeze(0).cuda()

    # print('checking shape', pred_cam_vertices.shape)
    if not cliff:
        pred_cam_vertices = torch.matmul(
            smplx2smpl_update.repeat(batch_size, 1, 1), pred_cam_vertices
        )
    # # Get 14 predicted joints from the mesh
    pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices)

    # gt_keypoints_3d = pred_keypoints_3d
    return pred_keypoints_3d.cpu(), img_name


# input: pseudo_label path, save visualization (if path save else None; default None), save stride(Nx17x3)


def read_bedlam_psuedo_labels(pl_path, stride, cliff=None):
    batched_dataset = []
    action_dirs = [pl_path]
    pkl_id = 0
    for action_dir in tqdm(action_dirs):
        # print(action_dir)
        # breakpoint()
        pth_files = sorted(glob.glob(os.path.join(action_dir, "*.pth")))
        print("length of sequence: ", len(pth_files))
        i = 0
        global_id = 0
        tmp_batch = []
        # tmp_gt_batch = []
        tmp_img_name = []
        for id, pth_id in enumerate(range(len(pth_files))):
            i += 1
            global_id += 1
            # print('pth_file id:', pth_files[pth_id])
            if i <= stride:
                # print(i)
                temp_file = pth_files[
                    pth_id
                ]  # os.path.join(action_dir, pth_files[pth_id])
                temp_kp_3d, img_name = convert_vertices_to_3dkp_and_gt(
                    temp_file, cliff=cliff
                )
                # print('shape of 3d key-points: ', temp_kp_3d.shape, i)
                # breakpoint()
                tmp_batch.append(temp_kp_3d)
                # tmp_gt_batch.append(gt_kp)
                tmp_img_name.append(img_name)

            if i == stride or global_id == len(pth_files):
                # print('global_id', global_id)
                i = 0
                tmp_batch = torch.cat(tmp_batch, 0)
                # tmp_gt_batch = torch.cat(tmp_gt_batch, 0)
                # batched_dataset.append(tmp_batch.unsqueeze(0))
                # breakpoint()
                # print('shape of tmp_batch', tmp_batch.shape)
                data_dict = {
                    "data_input": tmp_batch,
                    "batch_img_name": tmp_img_name,
                }
                # with open(os.path.join(save_path, "%08d.pkl" % pkl_id), "wb") as myprofile:
                #     pickle.dump(data_dict, myprofile)
                tmp_batch = []
                # tmp_gt_batch = []
                tmp_img_name = []
                pkl_id += 1
                batched_dataset.append(data_dict)
                # exit(0)

    # batched_dataset = torch.cat(batched_dataset,0).numpy()
    return batched_dataset


class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split):  # data_split: train/test
        np.random.seed(0)
        # self.data_root = args.data_root
        # self.subset_list = subset_list
        # self.data_split = data_split
        file_list_all = []
        # for subset in self.subset_list:
        #     data_path = os.path.join(self.data_root, subset, 'train')  #@@@@@@@ data_split
        #     motion_list = sorted(os.listdir(data_path))
        #     for i in motion_list:
        #         file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError


class MotionDataset3D_infer(MotionDataset):
    def __init__(self, args, data_split, pl_path, cliff=None):
        super(MotionDataset3D_infer, self).__init__(
            args,
            data_split,
            pl_path,
        )
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d
        self.data_split = data_split
        self.file_list = read_bedlam_psuedo_labels(pl_path, stride=243, cliff=cliff)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.file_list)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        motion_file = self.file_list[index]
        # motion_file = read_pkl(file_path)

        if self.data_split == "train":
            motion_3d = motion_file["data_input"]
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:, :, :2] = motion_3d[:, :, :2]
                motion_2d[:, :, 2] = 1  # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:  # Have 2D detection
                motion_2d = motion_file["data_input"]  # .numpy()
                if (
                    self.flip and random.random() > 0.5
                ):  # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError("Training illegal.")

            return (
                torch.FloatTensor(motion_2d),
                torch.FloatTensor(motion_3d),
                motion_file["batch_img_name"],
            )

        elif self.data_split == "test":
            # if self.use_h36m:
            #     ground_truth = motion_file["ground_truth"]
            # else:
            #     ground_truth = motion_file["ground_truth"]
            # ground_truth = ground_truth.to(torch.float).numpy()
            # ground_truth = torch.from_numpy(self.aug.augment3D(ground_truth))

            motion_2d = motion_file["data_input"]  # .numpy()
            if self.gt_2d:
                motion_2d[:, :, :2] = motion_3d[:, :, :2]
                motion_2d[:, :, 2] = 1
            return (
                torch.FloatTensor(motion_2d),
                torch.FloatTensor(motion_2d),
                motion_file["batch_img_name"],
            )
        else:
            raise ValueError("Data split unknown.")
