import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random
import torch.nn.functional as F
import torch.utils.data as data
import torch
import numpy as np
import cv2
import os

from utils.fh_utils import load_db_annotation, read_img_abs, read_img, read_mask_woclip, projectPoints
from utils.kinematics import mano_to_mpii, mpii_to_mano
from utils.heatmaputils import gen_heatmap_with_kp_bbox
from utils.augmentation import Augmentation
from utils.vis import cnt_area, base_transform
from utils.preprocessing import augmentation, augmentation_2d
from termcolor import cprint
from utils.text_utils import joints, mano_joints

class FreiHAND(data.Dataset):
    def __init__(
            self,
            mode='train',
            contrastive=False,
            root='/data/datasets/FreiHAND',
            inp_res=256,
            hm_res=64,
            sigma=2.0,
            color_aug=False,
            affine_aug=False
    ):
        super(FreiHAND, self).__init__()

        self.mode = mode
        self.root = root
        self.inp_res = inp_res
        self.hm_res = hm_res
        self.sigma = sigma
        self.contrastive = contrastive
        self.affine_aug_state = affine_aug
        self.color_aug_state = color_aug
        self.status = None

        if self.affine_aug_state and 'train' in self.mode:
            self.phase = 'train'
        elif self.affine_aug_state and 'contrastive' in self.mode:
            self.phase = 'train'
        elif 'eval' in self.mode:
            self.phase = 'eval'
        else:
            self.phase = 'train'
            self.status = True

        self.db_data_anno = tuple(
            load_db_annotation(
                base_path=self.root,
                set_name=self.phase
            )
        )
        # self.color_aug = Augmentation() if 'train' or 'contrastive' in self.mode else None
        if self.color_aug_state:
            self.color_aug = Augmentation()
        else:
            self.color_aug = None
        self.one_version_len = self.db_data_anno.__len__()
        if self.phase == 'train' and self.status == None:
            self.db_data_anno *= 4
        elif self.phase == 'train' and self.status:
            self.db_data_anno *= 1
        else:
            self.db_data_anno *= 1

        cprint(
            'Loaded FreiHand {} {} samples'.format(
                self.mode, str(len(self.db_data_anno))
            ), 'yellow'
        )

    def get_train_sample(self, id):

        img = read_img_abs(id, self.root, 'training')
        mask = read_mask_woclip(id % self.one_version_len, self.root, 'training')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5]
        w, h = bbox[2], bbox[3]
        bbox = [
            center[0] - 0.5 * max(w, h), center[1] - 0.5 * max(w, h),
            max(w, h), max(w, h)
        ]
        K, mano, joint_cam = self.db_data_anno[id]
        K, joint_cam, mano = np.array(K), np.array(joint_cam), np.array(mano)
        joint_img = projectPoints(joint_cam, K)
        princpt = K[0:2, 2].astype(np.float32)
        focal = np.array([K[0, 0], K[1, 1]], dtype=np.float32)

        ''' Augmentation '''
        roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(
            img,
            bbox,
            self.phase,
            exclude_flip=True,
            input_img_shape=(self.inp_res, self.inp_res),
            mask=mask,
            base_scale=1.3,
            scale_factor=0.2,
            rot_factor=90,
            shift_wh=[bbox[2], bbox[3]],
            gaussian_std=3
        )
        if self.color_aug is not None:
            roi = self.color_aug(roi)
        roi = base_transform(roi, self.inp_res, mean=0.5, std=0.5)

        roi = torch.from_numpy(roi).float()
        mask = torch.from_numpy(mask).float()
        bb2img_trans = torch.from_numpy(bb2img_trans).float()

        ''' joint_img augmentation '''
        joint_img, princpt = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
        joint_img = joint_img[:, :2] / self.inp_res

        ''' joint_3d and vertices augmentation '''
        rot = aug_param[0]
        rot_aug_mat = np.array(
            [
                [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                [0, 0, 1]
            ], dtype=np.float32
        )
        joint_cam = np.dot(rot_aug_mat, joint_cam.T).T
        ''' Generate Naive Text 3D Prompt '''
        texts = self.get_text(joint_cam, joints)

        ''' Intrinsic matrix augmentation '''
        focal = focal * roi.size(1) / (bbox[2] * aug_param[1])
        calib = np.eye(4)
        calib[0, 0] = focal[0]
        calib[1, 1] = focal[1]
        calib[:2, 2:3] = princpt[:, None]
        calib = torch.from_numpy(calib).float()

        ''' joint (vert) to relative joint (vert) '''
        d_max, d_min = np.max(joint_cam[:, -1]), np.min(joint_cam[:, -1])
        d_range = d_max - d_min
        joint_uvd = np.concatenate([joint_img, joint_cam[:, -1][:, None]], axis=-1)
        joint_img = torch.from_numpy(joint_img).float()

        root = joint_cam[0].copy()
        joint_cam -= root
        root = torch.from_numpy(root).float()
        joint_mano = mpii_to_mano(joint_cam)
        joint_cam = torch.from_numpy(joint_cam).float()
        joint_mano = torch.from_numpy(joint_mano).float()

        ''' generate gaussian hm and hm_veil '''
        img_ys, img_xs = [], []
        hm = np.zeros((21, self.hm_res, self.hm_res), dtype='float32')
        hm_veil = np.ones(21, dtype='float32')
        kp2d = joint_img.numpy() * self.inp_res
        for i in range(21):
            kp = ((kp2d[i] / self.inp_res) * self.hm_res).astype(np.int32)
            hm[i], aval, img_y, img_x = gen_heatmap_with_kp_bbox(hm[i], kp, self.sigma)
            img_xs.append(img_x)
            img_ys.append(img_y)
            # hm[i], aval = gen_heatmap(hm[i], kp, self.sigma)
            hm_veil[i] *= aval

        hm = torch.from_numpy(hm).float()
        hm_veil = torch.from_numpy(hm_veil).float()

        img_ys = torch.from_numpy(np.array(img_ys)).float()
        img_xs = torch.from_numpy(np.array(img_xs)).float()

        return {
            'img': roi,
            'joint_img': joint_img, 'joint_cam': joint_cam,
            'joint_root': root, 'joint_uvd': joint_uvd,
            'joint_mano': joint_mano,
            'calib': calib, 'aug_param': aug_param, 'bb2img_trans': bb2img_trans,
            'hm': hm, 'mask': mask, 'hm_veil': hm_veil,
            'img_ys': img_ys, 'img_xs': img_xs,

            'text': texts
        }

    def get_contrastive_sample(self, id):

        img = read_img_abs(id, self.root, 'training')
        mask = read_mask_woclip(id % self.one_version_len, self.root, 'training')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5]
        w, h = bbox[2], bbox[3]
        bbox = [
            center[0] - 0.5 * max(w, h), center[1] - 0.5 * max(w, h),
            max(w, h), max(w, h)
        ]
        K, mano, joint_cam = self.db_data_anno[id]
        K, joint_cam, mano = np.array(K), np.array(joint_cam), np.array(mano)
        joint_img = projectPoints(joint_cam, K)
        princpt = K[0:2, 2].astype(np.float32)
        focal = np.array([K[0, 0], K[1, 1]], dtype=np.float32)

        ''' define lists and augmentation '''
        roi_list, mask_list, calib_list, aug_param_list, bb2img_trans_list = [], [], [], [], []
        root_list, vert_list, joint_cam_list, joint_img_list, joint_uvd_list, joint_mano_list = [], [], [], [], [], []
        joint_bone_list = []
        hm_list, hm_veil_list, img_ys_list, img_xs_list = [], [], [], []
        text_list, text_mano_list = [], []
        for _ in range(2):
            roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, roi_mask = augmentation(
                img.copy(),
                bbox,
                self.phase,
                exclude_flip=True,
                input_img_shape=(self.inp_res, self.inp_res),
                mask=mask.copy(),
                base_scale=1.3,
                scale_factor=0.2,
                rot_factor=90,
                shift_wh=[bbox[2], bbox[3]],
                gaussian_std=3
            )
            if self.color_aug is not None:
                roi = self.color_aug(roi)
            roi = base_transform(roi, self.inp_res, mean=0.5, std=0.5)
            roi = torch.from_numpy(roi).float()
            roi_mask = torch.from_numpy(roi_mask).float()
            bb2img_trans = torch.from_numpy(bb2img_trans).float()
            aug_param = torch.from_numpy(aug_param).float()

            ''' joint_img augmentation '''
            joint_img_, princpt_ = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
            joint_img_ = torch.from_numpy(joint_img_[:, :2]).float() / self.inp_res

            ''' joint_3d and vert augmentation '''
            rot = aug_param[0].item()
            rot_aug_mat = np.array([
                [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                [0, 0, 1]], dtype=np.float32
            )
            joint_cam_ = np.dot(rot_aug_mat, joint_cam.T).T
            text = self.get_text(joint_cam_, joints=joints)
            root = joint_cam_[0].copy()
            joint_cam_ -= root
            joint_mano = mpii_to_mano(joint_cam_)
            text_mano = self.get_text(joint_mano, joints=mano_joints)

            d_max, d_min = np.max(joint_cam_[:, -1]), np.min(joint_cam_[:, -1])
            d_range = d_max - d_min
            joint_d = (joint_cam_[:, -1] - d_min) / (d_range * 1.2)
            joint_d = torch.from_numpy(joint_d).float()
            # joint_uvd = torch.cat([joint_img_, joint_d[:, None]], dim=-1)
            joint_uvd = torch.cat([
                joint_img_,
                torch.from_numpy(joint_cam_[:, -1][:, None]).float()
            ], dim=-1)

            root = torch.from_numpy(root).float()
            joint_cam_np = joint_cam_ # numpy
            joint_cam_ = torch.from_numpy(joint_cam_np).float()
            joint_mano_np = joint_mano # numpy
            joint_mano = torch.from_numpy(joint_mano_np).float()

            ''' K augmentation '''
            focal_ = focal * roi.size(1) / (bbox[2]*aug_param[1])
            calib = np.eye(4)
            calib[0, 0] = focal_[0]
            calib[1, 1] = focal_[1]
            calib[:2, 2:3] = princpt_[:, None]
            calib = torch.from_numpy(calib).float()

            joint_bone = 0
            self.ref_bone_link = (0, 9)
            for jid, nextjid in zip(self.ref_bone_link[:-1], self.ref_bone_link[1:]):
                joint_bone += np.linalg.norm(joint_cam_np[nextjid] - joint_cam_np[jid])

            joint_bone = torch.from_numpy(np.asarray(joint_bone)).float()

            ''' generate gaussian hm and hm_veil '''
            img_ys, img_xs = [], []
            hm = np.zeros((21, self.hm_res, self.hm_res), dtype='float32')
            hm_veil = np.ones(21, dtype='float32')
            kp2d = joint_img_.numpy() * self.inp_res
            for i in range(21):
                kp = ((kp2d[i] / self.inp_res) * self.hm_res).astype(np.int32)
                hm[i], aval, img_y, img_x = gen_heatmap_with_kp_bbox(hm[i], kp, self.sigma)
                img_xs.append(img_x)
                img_ys.append(img_y)
                # hm[i], aval = gen_heatmap(hm[i], kp, self.sigma)
                hm_veil[i] *= aval

            hm = torch.from_numpy(hm).float()
            hm_veil = torch.from_numpy(hm_veil).float()

            img_ys = torch.from_numpy(np.array(img_ys)).float()
            img_xs = torch.from_numpy(np.array(img_xs)).float()

            roi_list.append(roi)
            mask_list.append(roi_mask.unsqueeze(0))
            calib_list.append(calib)
            root_list.append(root)
            joint_uvd_list.append(joint_uvd)
            joint_cam_list.append(joint_cam_)
            joint_bone_list.append(joint_bone.unsqueeze(0))
            joint_mano_list.append(joint_mano)
            joint_img_list.append(joint_img_)
            aug_param_list.append(aug_param)
            bb2img_trans_list.append(bb2img_trans)
            hm_list.append(hm)
            hm_veil_list.append(hm_veil)
            img_ys_list.append(img_ys)
            img_xs_list.append(img_xs)
            text_list.append(text)
            text_mano_list.append(text_mano)

        roi = torch.cat(roi_list, 0)
        mask = torch.cat(mask_list, 0)
        calib = torch.cat(calib_list, 0)
        root = torch.cat(root_list, -1)
        joint_uvd = torch.cat(joint_uvd_list, -1)
        joint_cam = torch.cat(joint_cam_list, -1)
        joint_mano = torch.cat(joint_mano_list, -1)
        joint_bones = torch.cat(joint_bone_list, 0)
        joint_img = torch.cat(joint_img_list, -1)
        aug_param = torch.cat(aug_param_list, 0)
        bb2img_trans = torch.cat(bb2img_trans_list, -1)
        hms = torch.cat(hm_list, 0)
        hm_veils = torch.cat(hm_veil_list, 0)
        img_ys = torch.cat(img_ys_list, 0)
        img_xs = torch.cat(img_xs_list, 0)

        return {
            'img': roi,
            'joint_img': joint_img, 'joint_cam': joint_cam, 'joint_mano': joint_mano, 'joint_bone': joint_bones,
            'joint_root': root, 'joint_uvd': joint_uvd,
            'calib': calib, 'aug_param': aug_param, 'bb2img_trans': bb2img_trans,
            'hm': hms, 'mask': mask, 'hm_veil': hm_veils,
            'img_ys': img_ys, 'img_xs': img_xs,

            'text': text_list, 'text_mano': text_mano_list
        }


    def get_text(self, joint_cam, joints):
        ids = list(range(21))
        new_ids, rest = random_split(ids, 21)

        joints_x = joint_cam[:, 0][new_ids]
        x_sort = list(np.argsort(joints_x))
        x_sort_original = [new_ids[i] for i in x_sort]
        left_right_text = generate_text_left_right(x_sort_original, joints)

        joints_y = joint_cam[:, 1][new_ids]
        y_sort = list(np.argsort(joints_y))
        y_sort_original = [new_ids[i] for i in y_sort]
        top_bottom_text = generate_text_top_bottom(y_sort_original, joints)

        joints_z = joint_cam[:, 2][new_ids]
        z_sort = list(np.argsort(joints_z))
        z_sort_original = [new_ids[i] for i in z_sort]
        near_far_text = generate_text_near_far(z_sort_original, joints)

        return {
            'left_right_text': left_right_text,
            'top_bottom_text': top_bottom_text,
            'near_far_text': near_far_text
        }

    # def get_text(self, joint_cam):
    #     new_ids_lr = np.argsort(joint_cam[:, 0])  # sort, from left to right
    #     new_ids_lr_, _ = random_split(list(new_ids_lr), 16)
    #     left_right_text = generate_text_left_right(new_ids_lr_)
    #
    #     new_ids_tb = np.argsort(joint_cam[:, 1])  # sort, from top to bottom
    #     new_ids_tb_, _ = random_split(list(new_ids_tb), 16)
    #     top_bottom_text = generate_text_top_bottom(new_ids_tb_)
    #
    #     new_ids_nf = np.argsort(joint_cam[:, 2])  # sort, from near to far
    #     new_ids_nf_, _ = random_split(list(new_ids_nf), 16)
    #     near_far_text = generate_text_near_far(new_ids_nf_)
    #
    #     return {
    #         'left_right_text': left_right_text,
    #         'top_bottom_text': top_bottom_text,
    #         'near_far_text': near_far_text
    #     }
    # def get_text(self, joint_cam):
    #     ids_ = [4, 8, 12, 16, 20]
    #
    #     new_ids_lr = []
    #     ids = np.argsort(joint_cam[:, 0])  # sort, from left to right
    #     for id in ids:
    #         if id not in ids_:
    #             new_ids_lr.append(id)
    #     left_right_text = generate_text_left_right(new_ids_lr)
    #
    #     new_ids_tb = []
    #     ids = np.argsort(joint_cam[:, 1])  # sort, from top to bottom
    #     for id in ids:
    #         if id not in ids_:
    #             new_ids_tb.append(id)
    #     top_bottom_text = generate_text_top_bottom(new_ids_tb)
    #
    #     new_ids_nf = []
    #     ids = np.argsort(joint_cam[:, 2])  # sort, from near to far
    #     for id in ids:
    #         if id not in ids_:
    #             new_ids_nf.append(id)
    #     near_far_text = generate_text_near_far(new_ids_nf)
    #
    #     return {
    #         'left_right_text': left_right_text,
    #         'top_bottom_text': top_bottom_text,
    #         'near_far_text': near_far_text
    #     }

    def __getitem__(self, item):
        if 'train' in self.mode:
            return self.get_train_sample(item)
        elif 'contrastive' in self.mode:
            return self.get_contrastive_sample(item)
        else:
            return self.get_train_sample(item)
            # raise Exception("No this mode, only support train, eval or contrastive")

    def __len__(self):
        return len(self.db_data_anno)

def random_split(items, size):
    sample = set(random.sample(items, size))
    a = [x for x in items if x in sample]
    b = [x for x in items if x not in sample]

    return a, b

def generate_text_near_far(ids, joints=joints):
    text = "Order are: "
    for id in ids[:-1]:
        text = text + joints[id]
        text = text + ', '
    text = text + ' and ' + joints[ids[-1]] + '.'
    text = text.replace(',  and', ' and')

    return text

def generate_text_top_bottom(ids, joints=joints):
    text = "Order are: "
    for id in ids[:-1]:
        text = text + joints[id]
        text = text + ', '
    text = text + ' and ' + joints[ids[-1]] + '.'
    text = text.replace(',  and', ' and')

    return text

def generate_text_left_right(ids, joints=joints):
    text = "Order are: "
    for id in ids[:-1]:
        text = text + joints[id]
        text = text + ', '
    text = text + ' and ' + joints[ids[-1]] + '.'
    text = text.replace(',  and', ' and')

    return text
