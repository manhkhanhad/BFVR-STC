import cv2
import math
import random
import numpy as np
import os.path as osp
from scipy.io import loadmat
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    normalize,
)
from basicsr.data import gaussian_kernels as gaussian_kernels
from basicsr.data.transforms import augment
from basicsr.data.data_util import paths_from_folder, brush_stroke_mask, random_ff_mask
from basicsr.utils import (
    FileClient,
    get_root_logger,
    imfrombytes,
    img2tensor,
    img2tensor_np,
)
from basicsr.utils.registry import DATASET_REGISTRY

from basicsr.utils.video_util import VideoReader, VideoWriter
import time
import random
import ffmpeg
import io
import av
import math
from PIL import Image
from io import BytesIO
import json
import glob

@DATASET_REGISTRY.register()
class FaceVarDataset(data.Dataset):
    def __init__(self, opt):
        super(FaceVarDataset, self).__init__()
        self.logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.lq_folder = opt["lq_folder"]
        self.gt_folder = opt["gt_folder"]
        with open(opt["split_file"], 'r') as f:
            self.file_names = f.readlines()
        self.gt_size = opt.get("gt_size", 256)
        self.in_size = opt.get("in_size", 256)
        assert self.gt_size >= self.in_size, "Wrong setting."
        self.n_frame = opt.get("n_frame", 8)

        self.component_path = opt.get("component_file", None)

        if self.component_path is not None:
            self.crop_components = True
            with open(self.component_path, 'r') as f:
                self.component_list = json.load(f)
            self.eye_enlarge_ratio = opt.get("eye_enlarge_ratio", 1.4)
            self.nose_enlarge_ratio = opt.get("nose_enlarge_ratio", 1.1)
            self.mouth_enlarge_ratio = opt.get("mouth_enlarge_ratio", 1.3)
        else:
            self.crop_components = False

        self.paths = []
        invalid_count = 0
        for file_name in self.file_names:
            file_name = file_name.strip()
            lq_path = osp.join(self.lq_folder, file_name)
            gt_path = osp.join(self.gt_folder, file_name)
            if not osp.exists(lq_path) or not osp.exists(gt_path):
                invalid_count += 1
                continue
            if len(glob.glob(osp.join(lq_path, '*'))) < self.n_frame or len(glob.glob(osp.join(gt_path, '*'))) < self.n_frame:
                invalid_count += 1
                continue
            self.paths.append((lq_path, gt_path))
        print(f"Invalid count: {invalid_count}")

    def get_component_locations(self, name, status):
        components_bbox = self.component_list[name]
        locations_gt = {}
        locations_in = {}
        for part in ["left_eye", "right_eye", "mouth"]:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if "eye" in part:
                half_len *= self.eye_enlarge_ratio
            elif part == "nose":
                half_len *= self.nose_enlarge_ratio
            elif part == "mouth":
                half_len *= self.mouth_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations_gt[part] = loc
            loc_in = loc / (self.gt_size // self.in_size)
            locations_in[part] = loc_in
        return locations_gt, locations_in
    
    def process_frame(self, frame):
        frame = cv2.resize(
            frame,
            (int(self.in_size), int(self.in_size)),
            interpolation=cv2.INTER_LINEAR,
        )
        frame = img2tensor_np(frame, bgr2rgb=True, float32=True)
        return frame

    def __getitem__(self, index):
        # load gt image
        lq_path, gt_path = self.paths[index]
        name = osp.basename(lq_path)
        n_frame = self.n_frame
        img_gt_list = []
        img_lq_list = []
        lq_sequences = sorted(glob.glob(osp.join(lq_path, '*')))
        frames_list = [osp.basename(frame_path) for frame_path in lq_sequences]
        n_all = len(frames_list)

        
        while True:
            try:
                frames_name = []
                start_idx = random.randint(0, n_all - n_frame)
                slected_frames = frames_list[start_idx : start_idx + n_frame]
                for frame_name in slected_frames:
                    lq_frame_path = osp.join(lq_path, frame_name)
                    gt_frame_path = osp.join(gt_path, frame_name)
                    if not osp.exists(lq_frame_path):
                        raise ValueError(f"LQ frame {lq_frame_path} not found")
                    if not osp.exists(gt_frame_path):
                        raise ValueError(f"GT frame {gt_frame_path} not found")
                    lq_frame = cv2.imread(lq_frame_path)
                    gt_frame = cv2.imread(gt_frame_path)
                    img_lq_list.append(self.process_frame(lq_frame))
                    img_gt_list.append(self.process_frame(gt_frame))
                break
            except:
                self.logger.error(f"Error processing {lq_frame_path}")
        
        img_lq_np = np.array(img_lq_list)
        img_gt_np = np.array(img_gt_list)

        img_lq_np = img_lq_np.astype(np.float32) / 255.0
        img_gt_np = img_gt_np.astype(np.float32) / 255.0

        img_lq = img_lq_np.transpose(3, 0, 1, 2)
        img_gt = img_gt_np.transpose(3, 0, 1, 2)
        slected_frames = "|".join(slected_frames)
        return_dict = {"in": img_lq, "gt": img_gt, "gt_path": gt_path, "frames_name": slected_frames}
        return return_dict

    def __len__(self):
        return len(self.paths)
    

@DATASET_REGISTRY.register()
class FaceVarDatasetTest(data.Dataset):
    def __init__(self, opt):
        super(FaceVarDatasetTest, self).__init__()
        self.logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.lq_folder = opt["lq_folder"]
        self.gt_folder = opt["gt_folder"]
        with open(opt["split_file"], 'r') as f:
            self.file_names = f.readlines()
        self.gt_size = opt.get("gt_size", 256)
        self.in_size = opt.get("in_size", 256)
        assert self.gt_size >= self.in_size, "Wrong setting."
        self.n_frame = opt.get("n_frame", 8)

        self.component_path = opt.get("component_file", None)

        if self.component_path is not None:
            self.crop_components = True
            with open(self.component_path, 'r') as f:
                self.component_list = json.load(f)
            self.eye_enlarge_ratio = opt.get("eye_enlarge_ratio", 1.4)
            self.nose_enlarge_ratio = opt.get("nose_enlarge_ratio", 1.1)
            self.mouth_enlarge_ratio = opt.get("mouth_enlarge_ratio", 1.3)
        else:
            self.crop_components = False

        self.paths = []
        self.gt_chunks = []
        self.lq_chunks = []
        invalid_count = 0
        for file_name in self.file_names:
            file_name = file_name.strip()
            lq_path = osp.join(self.lq_folder, file_name)
            gt_path = osp.join(self.gt_folder, file_name)
            if not osp.exists(lq_path) or not osp.exists(gt_path):
                invalid_count += 1
                continue
            if len(glob.glob(osp.join(lq_path, '*'))) < self.n_frame or len(glob.glob(osp.join(gt_path, '*'))) < self.n_frame:
                invalid_count += 1
                continue
            self.paths.append((lq_path, gt_path))
            all_lq_frames = sorted(glob.glob(osp.join(lq_path, '*')))
            all_gt_frames = sorted(glob.glob(osp.join(gt_path, '*')))
            for i in range(0, len(all_lq_frames), self.n_frame):
                self.lq_chunks.append(all_lq_frames[i:i+self.n_frame])
            for i in range(0, len(all_gt_frames), self.n_frame):
                self.gt_chunks.append(all_gt_frames[i:i+self.n_frame])
        print(f"Invalid count: {invalid_count}")

    def get_component_locations(self, name, status):
        components_bbox = self.component_list[name]
        locations_gt = {}
        locations_in = {}
        for part in ["left_eye", "right_eye", "mouth"]:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if "eye" in part:
                half_len *= self.eye_enlarge_ratio
            elif part == "nose":
                half_len *= self.nose_enlarge_ratio
            elif part == "mouth":
                half_len *= self.mouth_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations_gt[part] = loc
            loc_in = loc / (self.gt_size // self.in_size)
            locations_in[part] = loc_in
        return locations_gt, locations_in
    
    def process_frame(self, frame):
        frame = cv2.resize(
            frame,
            (int(self.in_size), int(self.in_size)),
            interpolation=cv2.INTER_LINEAR,
        )
        frame = img2tensor_np(frame, bgr2rgb=True, float32=True)
        return frame

    def __getitem__(self, index):
        # load gt image

        lq_chunk = self.lq_chunks[index]
        gt_chunk = self.gt_chunks[index]

        lq_path, gt_path = self.paths[index]
        name = osp.basename(lq_path)
        n_frame = self.n_frame
        img_gt_list = []
        img_lq_list = []
        frames_list = [osp.basename(frame_path) for frame_path in lq_chunk]
        for frame_name in frames_list:
            lq_frame_path = osp.join(lq_path, frame_name)
            gt_frame_path = osp.join(gt_path, frame_name)
            if not osp.exists(lq_frame_path):
                raise ValueError(f"LQ frame {lq_frame_path} not found")
            if not osp.exists(gt_frame_path):
                raise ValueError(f"GT frame {gt_frame_path} not found")
            lq_frame = cv2.imread(lq_frame_path)
            gt_frame = cv2.imread(gt_frame_path)
            img_lq_list.append(self.process_frame(lq_frame))
            img_gt_list.append(self.process_frame(gt_frame))
        
        img_lq_np = np.array(img_lq_list)
        img_gt_np = np.array(img_gt_list)

        img_lq_np = img_lq_np.astype(np.float32) / 255.0
        img_gt_np = img_gt_np.astype(np.float32) / 255.0

        img_lq = img_lq_np.transpose(3, 0, 1, 2)
        img_gt = img_gt_np.transpose(3, 0, 1, 2)
        slected_frames = "|".join(frames_list)
        return_dict = {"in": img_lq, "gt": img_gt, "gt_path": gt_path, "frames_name": slected_frames}
        return return_dict

    def __len__(self):
        return len(self.lq_chunks)

# if __name__ == "__main__":
#     opt = {
#         "lq_folder": "/mmlabworkspace_new/WorkSpaces/ngaptb/khanhnhm/khanhngo/VideoRestoration/VideoRestoration/dataset/TalkingHead/new_data/degradation/degraded_images",
#         "gt_folder": "/mmlabworkspace_new/WorkSpaces/ngaptb/khanhnhm/khanhngo/VideoRestoration/VideoRestoration/dataset/TalkingHead/new_data/degradation/images",
#         "split_file": "/mmlabworkspace_new/WorkSpaces/ngaptb/HumanActionMimic/STERRGAN/BFVR-STC/data/test_list.txt",
#         "component_file": "/mmlabworkspace_new/WorkSpaces/ngaptb/khanhnhm/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/data/test_video.json",
#     }
#     dataset = FaceVarDataset(opt)
#     print(len(dataset))
#     sample = dataset.__getitem__(0)
#     breakpoint()
