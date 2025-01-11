import numpy as np
import json
import torch.nn.functional as F
import torch
from comfy.utils import ProgressBar

from .dwpose import DWposeDetector


class OpenposeEstimatorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "show_body": ("BOOLEAN", {"default": True}),
                "detect_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "round": 0.00001,
                    "display": "number",
                    "lazy": True
                }),
                "resolution_x": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 12800,
                }),
                "pose_marker_size": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 100
                }),
                "face_marker_size": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 100
                }),
                "hand_marker_size": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 100
                }),
            }
        }
    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "POSE_JSON")
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    FUNCTION = "main"
    CATEGORY = "ultimate-openpose"

    def main(self, images, show_face, show_hands, show_body, detect_threshold, resolution_x, pose_marker_size, face_marker_size, hand_marker_size):

        pbar = ProgressBar(images.shape[0])
        dwpose = DWposeDetector()
        pose_frames = []
        json_frames = []

        for img in images:
            img_np_hwc = (img.cpu().numpy() * 255).astype(np.uint8)
            pose_img, openpose_json = dwpose(image_np_hwc=img_np_hwc, show_face=show_face,
                            show_hands=show_hands, show_body=show_body, detect_threshold=detect_threshold, resolution=resolution_x, pose_marker_size=pose_marker_size, face_marker_size=face_marker_size, hand_marker_size=hand_marker_size)
            pose_frames.append(pose_img)
            json_frames.append(openpose_json)
            pbar.update(1)

        pose_frames_np = np.array(pose_frames).astype(np.float32) / 255
        return (torch.from_numpy(pose_frames_np),json_frames, json.dumps(json_frames, indent = 4))
