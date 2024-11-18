import os
import numpy as np
import argparse
import errno
import math
import pickle

# import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random

# import prettytable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d_stride_infer import MotionDataset3D_infer
from lib.data.augmentation import Augmenter2D
from lib.model.loss import *
from lib.utils.vismo import render_and_save, render_and_save_new, camera_to_world

from lib.utils.ocmotion_camera import extrinsic_camera
import wandb

J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
joint_mapper_gt = J24_TO_J17[:14]

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
joint_mapper_h36m = H36M_TO_J17[:14]

human36m_dict = {}
ocmotion_dict = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="stride/configs/pose3d/MB_ft_h36m.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint/pose3d/ro",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        default="checkpoint/",
        type=str,
        metavar="PATH",
        help="pretrained checkpoint directory",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        metavar="FILENAME",
        help="checkpoint to resume (file name)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        default="",
        type=str,
        metavar="FILENAME",
        help="checkpoint to evaluate (file name)",
    )
    parser.add_argument(
        "-ms",
        "--selection",
        default="latest_epoch.bin",
        type=str,
        metavar="FILENAME",
        help="checkpoint to finetune (file name)",
    )
    parser.add_argument("-sd", "--seed", default=0, type=int, help="random seed")
    parser.add_argument("-w", "--wandb", default="", type=str, help="setting wandb")
    opts = parser.parse_args()
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print("Saving checkpoint to", chk_path)
    torch.save(
        {
            "epoch": epoch + 1,
            "lr": lr,
            "optimizer": optimizer.state_dict(),
            "model_pos": model_pos.state_dict(),
            "min_loss": min_loss,
        },
        chk_path,
    )


def evaluate(args, model_pos, test_loader):
    # file_list_all = torch.load('/data/AmitRoyChowdhury/Yash/MotionBERT/data/yg_data/'+
    #                args.eval_pth_dir.split('/')[-1] + '.pth')
    # dict_file_list = {fn['batch_img_name']: fn['ground_truth'] for fn in file_list_all}

    # print(np.unique([l.split('/')[2] for l in list(dict_file_list.keys())]))
    # breakpoint()

    results_pred = []
    resuls_img_name = []
    model_pos.eval()
    with torch.no_grad():
        for batch_input, ground_truth, img_name in test_loader:
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                ground_truth = ground_truth.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                batch_gt[:, 0, 0, 2] = 0

            # if args.gt_2d:
            #     predicted_3d_pos[...,:2] = batch_input[...,:2]

            # convert to 14 keypoints
            predicted_3d_pos = predicted_3d_pos.reshape(-1, 17, 3)
            # predicted_3d_pos = predicted_3d_pos[:, joint_mapper_h36m, :] ##@@change after visualization
            predicted_3d_pos = predicted_3d_pos - (
                (predicted_3d_pos[:, 2, :] + predicted_3d_pos[:, 3, :]) / 2
            ).unsqueeze(1)
            ## code to visulize
            vis_pred = predicted_3d_pos.unsqueeze(0).cpu().numpy()
            results_pred.append(vis_pred)
            resuls_img_name += img_name
    results_pred = np.hstack(results_pred)
    results_pred = np.concatenate(results_pred)
    save_dict = {"keypoints": results_pred, "img_name": resuls_img_name}

    if args.do_vis:
        print("Saving Vis at:", args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(save_dict, args.save_dir + "pred_keypoints.pth")
        all_pred = [results_pred]
        loss = []  # [val_mpjpe, val_pampjpe, val_mpjpe_pf, val_pampjpe_pf]
        render_and_save_new(
            all_pred,
            resuls_img_name,
            loss,
            "%s/%s" % (args.save_dir, "visual_pred"),
            keep_imgs=False,
            fps=None,
        )


def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    model_pos.train()
    for idx, (batch_input, batch_gt, img_name) in enumerate(train_loader):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(
                    batch_input[:, :, :, 2:]
                )  # For 2D data, weight/confidence is at the last channel
            if args.rootrel:
                batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
            else:
                batch_gt[:, :, :, 2] = (
                    batch_gt[:, :, :, 2] - batch_gt[:, 0:1, 0:1, 2]
                )  # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(
                    batch_input, noise=(args.noise and has_gt), mask=args.mask
                )
        # Predict 3D poses

        predicted_3d_pos = model_pos(batch_input)  # (N, T, 17, 3)
        optimizer.zero_grad()
        if has_3d:
            ## accel loss
            # accel = np.mean(compute_accel(pred_j3ds)) * m2mm
            # accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm

            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            # return predicted_3d_pos, batch_gt, loss_3d_pos
            # breakpoint()
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            loss_total = (
                args.lambda_3d_pos * loss_3d_pos
                + args.lambda_scale * loss_3d_scale
                + args.lambda_3d_velocity * loss_3d_velocity
                + args.lambda_lv * loss_lv
                + args.lambda_lg * loss_lg
                + args.lambda_a * loss_a
                + args.lambda_av * loss_av
            )
            losses["3d_pos"].update(loss_3d_pos.item(), batch_size)
            losses["3d_scale"].update(loss_3d_scale.item(), batch_size)
            losses["3d_velocity"].update(loss_3d_velocity.item(), batch_size)
            losses["lv"].update(loss_lv.item(), batch_size)
            losses["lg"].update(loss_lg.item(), batch_size)
            losses["angle"].update(loss_a.item(), batch_size)
            losses["angle_velocity"].update(loss_av.item(), batch_size)
            losses["total"].update(loss_total.item(), batch_size)
            # print('3d_pos', '3d_scale', '3d_vel', 'lv', loss_3d_pos, loss_3d_scale, loss_3d_velocity, loss_lv)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses["2d_proj"].update(loss_2d_proj.item(), batch_size)
            losses["total"].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()


def train_with_config(args, opts):
    print(args)
    if opts.checkpoint:
        try:
            os.makedirs(opts.checkpoint)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError(
                    "Unable to create checkpoint directory:", opts.checkpoint
                )

    print("Loading dataset...")
    trainloader_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 1,
        #   'pin_memory': True,
        #   'prefetch_factor': 4,
        #   'persistent_workers': True
    }

    testloader_params = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 1,
        #   'pin_memory': True,
        #   'prefetch_factor': 4,
        #   'persistent_workers': True
    }
    pl_path = args.pl_path
    train_dataset = MotionDataset3D_infer(
        args, "train", pl_path, cliff=opts.use_og_cliff
    )
    test_dataset = MotionDataset3D_infer(args, "test", pl_path, cliff=opts.use_og_cliff)
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader_3d = DataLoader(test_dataset, **testloader_params)
    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)

    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print("INFO: Trainable parameter count:", model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    running_all_mpjpe = []
    running_all_pampjpe = []
    running_all_accl = []
    # dict_save={}
    ### training over dataset
    # for idx, ((batch_input, batch_gt, img_name), (batch_input_test, batch_gt_test, img_name_test)) in enumerate(zip(train_loader_3d_whole, test_loader_whole)):
    start_infer_time = time()
    # print(img_name[0], img_name[-1])
    # train_loader_3d = [[batch_input, batch_gt, img_name]]
    # test_loader_3d = [[batch_input_test, batch_gt_test, img_name_test]]

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print("Loading checkpoint", chk_filename)
            checkpoint = torch.load(
                chk_filename, map_location=lambda storage, loc: storage
            )
            model_backbone.load_state_dict(checkpoint["model_pos"], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print("Loading checkpoint", chk_filename)
            checkpoint = torch.load(
                chk_filename, map_location=lambda storage, loc: storage
            )
            model_backbone.load_state_dict(checkpoint["model_pos"], strict=True)
            model_pos = model_backbone
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print("Loading checkpoint", chk_filename)
            checkpoint = torch.load(
                chk_filename, map_location=lambda storage, loc: storage
            )
            model_backbone.load_state_dict(checkpoint["model_pos"], strict=True)
            model_pos = model_backbone

    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model_pos.parameters()),
            lr=lr,
            weight_decay=args.weight_decay,
        )
        lr_decay = args.lr_decay
        st = 0
        # if args.train_2d:
        #     print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d), len(instav_loader_2d) + len(posetrack_loader_2d)))
        # else:
        #     print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        if opts.resume:
            st = checkpoint["epoch"]
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                print(
                    "WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized."
                )
            lr = checkpoint["lr"]
            if "min_loss" in checkpoint and checkpoint["min_loss"] is not None:
                min_loss = checkpoint["min_loss"]

        args.mask = args.mask_ratio > 0 and args.mask_T_ratio > 0
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)

        # Training
        for epoch in range(st, args.epochs):
            # print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses["3d_pos"] = AverageMeter()
            losses["3d_scale"] = AverageMeter()
            losses["2d_proj"] = AverageMeter()
            losses["lg"] = AverageMeter()
            losses["lv"] = AverageMeter()
            losses["total"] = AverageMeter()
            losses["3d_velocity"] = AverageMeter()
            losses["angle"] = AverageMeter()
            losses["angle_velocity"] = AverageMeter()
            N = 0

            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(
                    args,
                    model_pos,
                    posetrack_loader_2d,
                    losses,
                    optimizer,
                    has_3d=False,
                    has_gt=True,
                )
                train_epoch(
                    args,
                    model_pos,
                    instav_loader_2d,
                    losses,
                    optimizer,
                    has_3d=False,
                    has_gt=False,
                )
            train_epoch(
                args,
                model_pos,
                train_loader_3d,
                losses,
                optimizer,
                has_3d=True,
                has_gt=True,
            )

            elapsed = (time() - start_time) / 60

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay

            # Save checkpoints
            if opts.checkpoint:
                chk_path = os.path.join(opts.checkpoint, "epoch_{}.bin".format(epoch))
                chk_path_latest = os.path.join(opts.checkpoint, "latest_epoch.bin")
                chk_path_best = os.path.join(
                    opts.checkpoint, "best_epoch.bin".format(epoch)
                )

                save_checkpoint(
                    chk_path_latest, epoch, lr, optimizer, model_pos, min_loss
                )
                if (epoch + 1) % args.checkpoint_frequency == 0:
                    save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
                if e1 < min_loss:
                    min_loss = e1
                    save_checkpoint(
                        chk_path_best, epoch, lr, optimizer, model_pos, min_loss
                    )

        evaluate(args, model_pos, test_loader_3d)
        print("time taken", time() - start_infer_time)
    # torch.save(ocmotion_dict, '/data/AmitRoyChowdhury/Yash/MotionBERT/visualize/ocmotion_stride_poseformer_gt_preds.pth')

    # exit(0)
    # _, _, _, e1, e2, e3 = evaluate(args, model_pos, test_loader_3d, idx)
    # running_all_mpjpe.append(e1)
    # running_all_pampjpe.append(e2)
    # running_all_accl.append(e3)
    # avg_mpjpe = sum(running_all_mpjpe)/len(running_all_mpjpe)
    # avg_pampjpe = sum(running_all_pampjpe)/len(running_all_pampjpe)
    # avg_accl = sum(running_all_accl)/len(running_all_accl)

    # print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f e3 %f avg_e1 %f avg_e2 %f avg_e3 %f' % (
    #     idx+1,
    #     elapsed,
    #     lr,
    #     losses['3d_pos'].avg,
    #     e1, e2, e3, avg_mpjpe, avg_pampjpe, avg_accl))

    # wandb.log({
    #     # 'epoch': epoch+1,
    # #     # 'train_loss/loss_3d_pos': losses['3d_pos'].avg,
    # #     # 'train_loss/loss_2d_proj': losses['2d_proj'].avg,
    # #     # 'train_loss/loss_3d_scale': losses['3d_scale'].avg,
    # #     # 'train_loss/loss_3d_velocity': losses['3d_velocity'].avg,
    # #     # 'train_loss/loss_lv': losses['lv'].avg,
    # #     # 'train_loss/loss_lg': losses['lg'].avg,
    # #     # 'train_loss/loss_a': losses['angle'].avg,
    # #     # 'train_loss/loss_av': losses['angle_velocity'].avg,
    # #     # 'train_loss/loss_total': losses['total'].avg,
    # #     # 'test_loss/mpjpe_loss': e1,
    # #     # 'test_loss/pa-mpjpe_loss': e2,
    #     'idx': idx,
    #     'mpjpe': e1,
    #     'pampjpe': e2,
    #     'accl': e3,
    #     'running_mpjpe': avg_mpjpe,
    #     'running_pampjpe': avg_pampjpe,
    #     'running_accl': avg_accl
    # })


# '''

# if opts.evaluate:
#     e1, e2 = evaluate(args, model_pos, test_loader)

if __name__ == "__main__":
    opts = parse_args()
    print(opts)
    mode = "online" if len(opts.wandb) else "disabled"
    print("wandb mode ", mode, opts.wandb)
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    wandb.init(
        # set the wandb project where this run will be logged
        project="3d_consistent_pose",
        config=args,
        mode=mode,
        name=opts.wandb,
    )

    train_with_config(args, opts)
