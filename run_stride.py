import os
import sys

sys.path.append("stride")
import argparse
from loguru import logger
from glob import glob
from train.core.tester import Tester
import warnings, time
from stride.stride_inference import train_with_config, get_config, set_random_seed
import shutil
import time

warnings.filterwarnings("ignore")

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.append("")


def main(args):
    input_image_folder = args.image_folder
    output_path = args.output_folder
    bedlam_out_path = os.path.join(output_path, "bedlam_pl")
    os.makedirs(bedlam_out_path, exist_ok=True)

    logger.add(
        os.path.join(output_path, "demo.log"),
        level="INFO",
        colorize=False,
    )
    logger.info(f"Demo options: \n {args}")
    start_time = time.time()
    # if not os.path.exists(output_path):
    tester = Tester(args)

    all_image_folder = [input_image_folder]
    detections = tester.run_detector(all_image_folder)
    # tester.run_on_image_folder(all_image_folder, detections, output_path, visualize_proj=True)
    tester.save_pseudo_labels(
        all_image_folder,
        detections,
        bedlam_out_path,
        visualize_proj=True,
        torch_save=True,
    )
    del tester.model

    ### stride module begins to train and visualize
    # stride_opts = parse_args()
    # print(stride_opts)
    # breakpoint()

    set_random_seed(args.seed)
    for img_folder in all_image_folder:
        set_random_seed(args.seed)
        stride_args = get_config(args.config)
        stride_args.pl_path = (
            bedlam_out_path  # os.path.join(bedlam_out_path,img_folder.split('/')[-1])
        )
        stride_args.save_dir = os.path.join(output_path, "stride")
        train_with_config(stride_args, args)
        print("Total Time: ", time.time() - start_time)
        # shutil.rmtree(bedlam_out_path)


def visualise_results(folder_data, output_path):

    os.makedirs(output_path, exist_ok=True)

    logger.add(
        os.path.join(output_path, "demo.log"),
        level="INFO",
        colorize=False,
    )
    logger.info(f"Started Visualization: {folder_data}")
    tester = Tester(args)

    tester.load_from_data(folder_data, output_path)

    logger.info(f"Saved Visualization at: {output_path}")
    del tester.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/demo_bedlam_cliff.yaml",
        help="config file that defines model hyperparams",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="data/ckpt/bedlam_cliff.ckpt",
        help="checkpoint path",
    )

    parser.add_argument(
        "--image_folder", type=str, default="demo_images", help="input image folder"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="demo_images/results",
        help="output folder to write results",
    )

    parser.add_argument(
        "--tracker_batch_size",
        type=int,
        default=1,
        help="batch size of object detector used for bbox tracking",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="visualize the 3d body projection on image",
    )

    parser.add_argument(
        "--detector",
        type=str,
        default="yolo",
        choices=["yolo", "maskrcnn"],
        help="object detector to be used for bbox tracking",
    )

    parser.add_argument(
        "--yolo_img_size",
        type=int,
        default=416,
        help="input image size for yolo detector",
    )

    ## args for stride
    parser.add_argument(
        "--config",
        type=str,
        default="stride/configs/pose3d/MB_ft_h36m.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        default="stride/checkpoint/",
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
    parser.add_argument(
        "-cl", "--use_og_cliff", default=0, type=int, help="random seed"
    )

    args = parser.parse_args()
    main(args)
