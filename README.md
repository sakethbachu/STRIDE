<div align="center">
    
# [STRIDE: Single-video based Temporally Continuous Occlusion-Robust 3D Pose Estimation](https://arxiv.org/abs/2312.16221)
[Rohit Lal](https://rohitlal.net/), [Saketh Bachu](https://sakethbachu.github.io/), [Yash Garg](https://www.linkedin.com/in/yash-garg-881b73137/), [Arindam Dutta](https://www.linkedin.com/in/arindam-dutta-a07451292/), [Calvin-Khang Ta](https://www.linkedin.com/in/calvin-khang-ta/), [Dripta S. Raychaudhuri](https://driptarc.github.io/), [Hannah Dela Cruz](https://www.linkedin.com/in/hannah-dela-cruz-4a973725a/), [M. Salman Asif](https://intra.ece.ucr.edu/~sasif/), [Amit K. Roy-Chowdhury](https://vcg.ece.ucr.edu/amit)

[![Project](https://img.shields.io/badge/Project-Page-blue)](https://sites.google.com/ucr.edu/stride/home) 
<a href="https://wacv2025.thecvf.com/"><img alt="WACV" src="https://img.shields.io/badge/2025-WACV-9d4edd"></a> <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![arXiv](https://img.shields.io/badge/arXiv-2312.16221-b31b1b.svg)](https://arxiv.org/abs/2312.16221) 

https://github.com/user-attachments/assets/06067279-5849-4e15-b084-48765995741c

</div>

### Description

Accurately estimating 3D human poses under severe occlusions is crucial for tasks like action recognition, gait analysis, and AR/VR. Current models struggle with heavy occlusions due to limited temporal context or prolonged occlusions across frames. To address this, we introduce STRIDE (Single-video TempoRally contInuous occlusion-robust 3D Pose Estimation), a novel Test-Time Training (TTT) approach that refines noisy initial pose estimates into accurate, temporally coherent predictions. STRIDE is model-agnostic and enhances robustness and temporal consistency using any off-the-shelf 3D pose estimator. Experiments on challenging datasets show STRIDE significantly outperforms single-image and video-based methods, especially under substantial occlusions.

### Installation

If you need to run just the demo, please follow the following steps:
- Step 1. Register on [SMPL-X](https://smpl-x.is.tue.mpg.de/) website.
- Step 2. Register on [MANO](https://mano.is.tue.mpg.de/) website.
- Step 3. Register on [BEDLAM](https://bedlam.is.tue.mpg.de/) website.
- Step 4. Run the following script to fetch demo data. The script will need the username and password created in above steps.


Create a virtual environment and install all the requirements using `environment.yml` (conda env) and `requirements.txt`

    conda env create -f environment.yml
    conda activate stride
    pip install -r requirements.txt
    bash fetch_demo_data.sh

### Checkpoints download

Download the below files and place them at the location `stride/checkpoint/latest_epoch.bin` 

    mkdir -p stride/checkpoint/
    gdown --id 1k3UxjfzfDSs8ts1Fff_fEgcXIDaZP-Ik
    mv latest_epoch.bin stride/checkpoint/latest_epoch.bin

    gdown --id 1OmaBCC3oBjii9Eewdhdgeo8VTgV3plcN
    unzip utils.zip
    rm utils.zip

If the above download fails, directly download from the [Google Drive link](https://drive.google.com/drive/folders/1ml4rT4jnfPDWFGLf34uTml24nzE1LcuV?usp=drive_link) and place it in the respective folders

### Run the STRIDE demo
Run the demo code for a sample video. 
```
sh scripts/demo_stride.sh
```

You may have conflicting shared libraries. Running `export LD_LIBRARY_PATH=""` before the above command may solve this issue.

# **BibTex**
```bibtex
@misc{lal2024stridesinglevideobasedtemporally,
      title={STRIDE: Single-video based Temporally Continuous Occlusion-Robust 3D Pose Estimation}, 
      author={Rohit Lal and Saketh Bachu and Yash Garg and Arindam Dutta and Calvin-Khang Ta and Dripta S. Raychaudhuri and Hannah Dela Cruz and M. Salman Asif and Amit K. Roy-Chowdhury},
      year={2024},
      eprint={2312.16221},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2312.16221}, 
}
