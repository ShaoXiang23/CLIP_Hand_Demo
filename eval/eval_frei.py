from utils.heatmaputils import accuracy_heatmap
from torch.utils.data import Dataset, DataLoader
from datasets.Frei import FreiHAND
from demo.model_inderence import CLIP_Hand_3D_PE
from utils.eval import EvalUtil, rigid_align
from utils import misc
from termcolor import colored
from progress.bar import Bar
import torch

scaler = torch.cuda.amp.GradScaler()

import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import time
import torch.backends.cudnn as cudnn
import clip
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

from utils.handutils import uvd2xyz, xyz2uvd

def validate(val_loader, model):
    evaluator = EvalUtil()
    model.eval()
    bar = Bar(colored('Eval', 'yellow'), max=len(val_loader), check_tty=False)

    with torch.no_grad():
        for i, zipper in enumerate(val_loader):
            img = zipper['img'].to(device, non_blocking=True)  # [B, 3, 256, 256]
            joint_root = zipper['joint_root'].to(device)
            joint_bone = zipper['joint_bone'][:, None].to(device)
            intr = zipper['calib'][:, :3, :3].to(device)

            joint_gt = zipper['joint_mano'].to(device) + joint_root[:, None, :] # [B, 21, 3]

            ''' forward '''
            results = model(img)

            ''' uvd2xyz '''
            pred_verts_uvd = results['pred_verts']
            pred_verts_xyz = uvd2xyz(pred_verts_uvd, joint_root, joint_bone, intr)
            pred_joints = torch.bmm(model.J_Reg[None, :, :].repeat(img.shape[0], 1, 1), pred_verts_xyz)

            bs = img.shape[0]
            pred_joints_aligned = np.zeros([bs, 21, 3])
            for i in range(bs):
                pred_joint = pred_joints[i].detach().cpu().numpy()
                pred_joint_aligned = rigid_align(pred_joint, joint_gt[i].detach().cpu().numpy())
                pred_joints_aligned[i] = pred_joint_aligned

            for targj, predj in zip(joint_gt * 1000., pred_joints_aligned * 1000.):
                evaluator.feed(targj, predj)

            pck20 = evaluator.get_pck_all(20)
            pck30 = evaluator.get_pck_all(30)
            pck40 = evaluator.get_pck_all(40)

            bar.suffix = (
                '({batch}/{size}) '
                'pck20avg: {pck20:.3f} | '
                'pck30avg: {pck30:.3f} | '
                'pck40avg: {pck40:.3f} | '
            ).format(
                batch=i + 1,
                size=len(val_loader),
                pck20=pck20,
                pck30=pck30,
                pck40=pck40,
            )
            bar.next()

        bar.finish()
        (
            epe_mean_all,
            epe_mean_joint,
            epe_median_all,
            auc_all,
            pck_curve_all,
            thresholds
        ) = evaluator.get_measures(
            0, 50, 50
        )

        print('3D EPE: {}'.format(epe_mean_all))
        print("3D AUC all: {}".format(auc_all))

    return auc_all


def evaluation(
        model_weight_path='',
        Vertx_dict_path='',
        face_path='',
        joints_num=21,
        hm_res=64,
        sigma=2.5,
        J_dim=512,
):
    ''' Network Definition '''
    model = CLIP_Hand_3D_PE(
        Vertx_dict_path=Vertx_dict_path,
        joints_num=joints_num,
        hm_res=hm_res,
        sigma=sigma,
        J_dim=J_dim
    ).to(device, non_blocking=True)

    ''' Load Model Pre-trained Weight '''
    model2_dict = model.state_dict()
    pretrained_model_path = model_weight_path
    pretrained_model = torch.load(pretrained_model_path)['state_dict']
    state_dict = {k: v for k, v in pretrained_model.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)

    ''' CLIP original model '''
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    bs = 32
    kwargs = {"pin_memory": True, "num_workers": 8}

    val_dataset = FreiHAND(mode='eval', inp_res=256, hm_res=64)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, drop_last=False, **kwargs)

    face = np.load(face_path)
    face = torch.from_numpy(face).long()

    ''' Evaluation on FreiHand test dataset '''
    best_acc = validate(val_loader, model)


if __name__ == '__main__':
    '''
    Results:
    3D EPE: 6.47, 3D AUC all: 0.871
    '''
    evaluation(
        model_weight_path='/your/path/to/CLIP_Hand_3D_PE_0604_44.pth.tar',
        Vertx_dict_path='/your/path/to/vertices.npy',
        face_path='/your/path/to/right_faces.npy',
    )






