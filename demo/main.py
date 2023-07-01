from models.model import CLIP_Hand_3D_PE, feature_match
from torch.utils.data import DataLoader
from datasets.Frei import FreiHAND

import torch
import clip
import random
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    ''' Please Modify the download weight path. '''
    WEIGHT_PATH = '../CLIP_HAND_3D_0402.pth.tar'
    DATASET_PATH = '../FreiHAND'
    BATCH_SIZE = 32

    model = CLIP_Hand_3D_PE().to(device)
    model2_dict = model.state_dict()
    pretrained_model = torch.load(WEIGHT_PATH)
    state_dict = {k: v for k, v in pretrained_model.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)
    model.eval()

    bs = BATCH_SIZE

    train_dataset = FreiHAND(
        root=DATASET_PATH,
        mode='train',
        affine_aug=False,
        color_aug=False,
        inp_res=256,
        hm_res=64,
        sigma=2.0
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True
    )

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    rand_id = random.randint(0, bs - 1)
    print("Selected Id is: ", rand_id)

    for i, metas in enumerate(train_loader):
        img = metas['img'].to(device, non_blocking=True)
        img0 = img[rand_id, :, :, :][None, :, :, :].repeat(bs, 1, 1, 1)
        mask = metas['mask'].to(device, non_blocking=True)

        ''' Clip Input '''
        text = metas['text']
        nf_text = clip.tokenize(text['near_far_text']).to(device)
        lr_text = clip.tokenize(text['left_right_text']).to(device)
        td_text = clip.tokenize(text['top_bottom_text']).to(device)

        with torch.no_grad():
            nf_features = clip_model.encode_text(nf_text).float()
            lr_features = clip_model.encode_text(lr_text).float()
            td_features = clip_model.encode_text(td_text).float()

            results = model(
                img0,
                {
                    'nf': nf_features,
                    'lr': lr_features,
                    'tb': td_features
                }
            )

            latent_x = results['latent_x'] / results['latent_x'].norm(dim=-1, keepdim=True)
            latent_y = results['latent_y'] / results['latent_y'].norm(dim=-1, keepdim=True)
            latent_z = results['latent_z'] / results['latent_z'].norm(dim=-1, keepdim=True)
            nf_features_ = results['nf_features'] / results['nf_features'].norm(dim=-1, keepdim=True)
            lr_features_ = results['lr_features'] / results['lr_features'].norm(dim=-1, keepdim=True)
            tb_features_ = results['tb_features'] / results['tb_features'].norm(dim=-1, keepdim=True)

            probs = feature_match(
                latent_x, latent_y, latent_z,
                nf_features_, lr_features_, tb_features_,
                model.logit_scale_nf, model.logit_scale_lr, model.logit_scale_td
            )

            fig = plt.figure(figsize=(20, 10))
            ax1 = plt.subplot(2, 4, 1)
            ax2 = plt.subplot(2, 4, 2)
            ax3 = plt.subplot(2, 4, 3)
            ax4 = plt.subplot(2, 4, 4)

            ax5 = plt.subplot(2, 4, 5)
            ax6 = plt.subplot(2, 4, 6)
            ax7 = plt.subplot(2, 4, 7)
            ax8 = plt.subplot(2, 4, 8)

            from utils.vis import inv_base_tranmsform

            img = inv_base_tranmsform(img0[0].detach().cpu().numpy())
            ax1.imshow(img);
            ax1.set_title("RGB Image")
            ax2.imshow(probs['probs_x']);
            ax2.set_title("X Matrix")
            ax3.imshow(probs['probs_y'], cmap='jet');
            ax3.set_title("Y Matrix")
            ax4.imshow(probs['probs_z'], cmap='rainbow');
            ax4.set_title("Z Matrix")
            text_z = str(100 * probs['probs_z'][0][rand_id])[:5] + '%, ' + \
                     'From near to far, Joints ' + text['near_far_text'][rand_id]
            text_x = str(100 * probs['probs_x'][0][rand_id])[:5] + '%, ' + \
                     'From left to right, Joints ' + text['left_right_text'][rand_id]
            text_y = str(100 * probs['probs_y'][0][rand_id])[:5] + '%, ' + \
                     'From top to bottom, Joints ' + text['top_bottom_text'][rand_id]

            import textwrap

            text_x = textwrap.fill(text_x, width=30)
            text_y = textwrap.fill(text_y, width=30)
            text_z = textwrap.fill(text_z, width=30)

            ax5.text(0.5, 0.5,
                     "Batch size is: " + str(bs) + ", id is: " + str(rand_id),
                     ha='center', va='center', size=18)
            ax6.text(0.5, 0.5, text_x, ha='center', va='center', size=18)
            ax7.text(0.5, 0.5, text_y, ha='center', va='center', size=18)
            ax8.text(0.5, 0.5, text_z, ha='center', va='center', size=18)

            # print(probs['probs_x'][0])
            # print(probs['probs_y'][0])
            # print(probs['probs_z'][0])

            fig.tight_layout()
            plt.show()
            exit()
