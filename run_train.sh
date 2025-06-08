#!/bin/bash

# Activate your conda environment if needed
# conda activate bfvr 
cd /mmlabworkspace_new/WorkSpaces/ngaptb/HumanActionMimic/STERRGAN/BFVR-STC

# Set the path to the config file
CONFIG_PATH="options/VQGAN_FaceVAR_stage1.yml"

# Run the training script
# python basicsr/train.py -opt options/VQGAN_FaceVAR_stage1.yml --launcher pytorch 

PYTHONPATH='./' CUDA_VISIBLE_DEVICES=0 nohup python basicsr/train.py -opt options/VQGAN_FaceVAR_stage1.yml > nohup/VQGAN_FaceVAR_stage1_v2.log
PYTHONPATH='./' CUDA_VISIBLE_DEVICES=0 nohup python basicsr/train.py -opt options/CodeFormer_stage2_bfvr_FaceVAR.yml > nohup/CodeFormer_stage2_bfvr_FaceVAR.log  


PYTHONPATH='./' CUDA_VISIBLE_DEVICES=0 nohup python basicsr/train.py -opt options/CodeFormer_stage2_bfvr_FaceVAR.yml > nohup/CodeFormer_stage2_bfvr_FaceVAR.log

PYTHONPATH='./' CUDA_VISIBLE_DEVICES=0 nohup python basicsr/train.py -opt options/CodeFormer_stage2_bfvr_FaceVAR.yml > nohup/CodeFormer_stage2_bfvr_FaceVAR_v3.log 



CUDA_VISIBLE_DEVICES=0 python scripts/infer_bfvr.py --input_path input.mp4 --output_base .
CUDA_VISIBLE_DEVICES=0 python scripts/infer_bfvr.py --input_path input2.mp4 --output_base .
