import numpy as np

from .onnxdet import inference_detector
from .onnxpose import inference_pose
from ..trt_utilities import Engine
import torch
import torch.nn.functional as F
import folder_paths
import os


class Wholebody:
    def __init__(self):
        self.engine = Engine(os.path.join(
            folder_paths.models_dir, "tensorrt", "dwpose", "yolox_l.engine"))
        self.engine.load()
        self.engine.activate()
        self.engine.allocate_buffers()

        self.engine2 = Engine(os.path.join(
            folder_paths.models_dir, "tensorrt", "dwpose", "dw-ll_ucoco_384.engine"))
        self.engine2.load()
        self.engine2.activate()
        self.engine2.allocate_buffers()

    def __call__(self, image_np_hwc, detect_threshold):
        cudaStream = torch.cuda.current_stream().cuda_stream

        det_result = inference_detector(
            engine=self.engine, cudaStream=cudaStream, image_np_hwc=image_np_hwc)
        keypoints, scores = inference_pose(
            engine=self.engine2, cudaStream=cudaStream, out_bbox=det_result, image_np_hwc=image_np_hwc)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > detect_threshold,
            keypoints_info[:, 6, 2:4] > detect_threshold).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]

        return keypoints, scores
