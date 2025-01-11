# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from . import util
from .wholebody import Wholebody
def draw_pose(pose, H, W, pose_marker_size, face_marker_size, hand_marker_size):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset, pose_marker_size)

    canvas = util.draw_handpose(canvas, hands, hand_marker_size)

    canvas = util.draw_facepose(canvas, faces, face_marker_size)

    return canvas


class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, image_np_hwc, show_face, show_hands, show_body, detect_threshold, resolution, pose_marker_size, face_marker_size, hand_marker_size):
        image_np_hwc = image_np_hwc.copy()

        H, W, C = image_np_hwc.shape

        candidate, subset = self.pose_estimation(image_np_hwc, detect_threshold)
        org_candidate = candidate.copy()
        org_subset = subset.copy()

        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:,:18].copy()
        body = body.reshape(nums*18, locs)
        score = subset[:,:18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > detect_threshold:
                    score[i][j] = int(18*i+j)
                else:
                    score[i][j] = -1

        un_visible = subset < detect_threshold
        candidate[un_visible] = -1

        foot = candidate[:,18:24]

        faces = candidate[:,24:92]

        hands = candidate[:,92:113]
        hands = np.vstack([hands, candidate[:,113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        pose = dict(bodies=bodies if show_body else {'candidate':[], 'subset':[]}, faces=faces if show_face else [], hands=hands if show_hands else [])

        # format keypoints output for openopse editor use,
        # openpose composite body pose, face, left hand and right hand, total 4 parts
        # openpose face consists of 70 points in total, while dwpose only 68 points, padding the last 2 points

        openpose_json = []
        for i in range(len(score)):
            score = org_subset[:,:18][i].tolist()
            body_xy = org_candidate[:, :18][i].tolist()
            body_keypoints = [[body_xy[j][0],body_xy[j][1],1.0] if score[j] > detect_threshold else [0.0, 0.0, 0.0] for j in range(len(score))]
            body_keypoints = [item for point in body_keypoints for item in point]

            lhand_xy = org_candidate[:, 92:113][i].tolist()
            score = org_subset[:,92:113][i].tolist()
            left_hand = [[lhand_xy[j][0],lhand_xy[j][1],1.0] if score[j] > detect_threshold else [0.0, 0.0, 0.0] for j in range(len(score))]
            left_hand = [item for point in left_hand for item in point]

            rhand_xy = org_candidate[:, 113:134][i].tolist()
            score = org_subset[:,113:134][i].tolist()
            right_hand = [[rhand_xy[j][0],rhand_xy[j][1],1.0] if score[j] > detect_threshold else [0.0, 0.0, 0.0] for j in range(len(score))]
            right_hand = [item for point in right_hand for item in point]

            face_xy = org_candidate[:, 24:92][i].tolist()
            score = org_subset[:,24:92][i].tolist()
            face = [[face_xy[j][0],face_xy[j][1],1.0] if score[j] > detect_threshold else [0.0, 0.0, 0.0] for j in range(len(score))]
            face = [item for point in face for item in point]

            if face is not None:
                score = org_subset[:,:18][i].tolist()
                # left eye
                if score[14] > detect_threshold:
                    face.append(body_xy[14][0])
                    face.append(body_xy[14][1])
                    face.append(1.0)
                else:
                    face.append(0.0)
                    face.append(0.0)
                    face.append(0.0)
                # right eye
                if score[15] > detect_threshold:
                    face.append(body_xy[15][0])
                    face.append(body_xy[15][1])
                    face.append(1.0)
                else:
                    face.append(0.0)
                    face.append(0.0)
                    face.append(0.0)

            """ Encode the pose as a dict following openpose JSON output format:
            https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
            """
            openpose_json.append(dict(pose_keypoints_2d=body_keypoints if show_body else [], face_keypoints_2d=face if show_face else [], hand_left_keypoints_2d=left_hand if show_hands else [], hand_right_keypoints_2d=right_hand if show_hands else []))

        openpose_json = {
                        'people': openpose_json,
                        'canvas_height': H,
                        'canvas_width': W,
                        }

        W_scaled = resolution
        if resolution < 64:
            W_scaled = W
        H_scaled = int(H*(W_scaled*1.0/W))
        return draw_pose(pose, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size), openpose_json
