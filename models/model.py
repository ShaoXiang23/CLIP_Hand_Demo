import random

import matplotlib.pyplot as plt

from models.resnet import ResNetBackbone
from models.posenet import PoseNet, make_conv_layers
from models.transformer import Transformer, AttentionPool2d
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)

class Pose2Feat(nn.Module):
    def __init__(self, joint_num=21, hm_res=64):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.hm_res = hm_res

        self.conv0 = make_conv_layers([
            64 + joint_num * self.hm_res, self.joint_num])
        self.conv1 = make_conv_layers([self.joint_num, self.joint_num])

    def forward(self, img_feat, joint_heatmap_3d):
        joint_heatmap_3d = joint_heatmap_3d.view(
            -1,
            self.joint_num * self.hm_res,
            self.hm_res,
            self.hm_res)
        feat = torch.cat((img_feat, joint_heatmap_3d), 1)
        feat = self.conv0(feat)
        feat = self.conv1(feat)

        return feat

def project(img_feat, joints, feat_size):
    v = joints[:, :, :2]
    v = v.unsqueeze(2)
    v = v / 0.5 - 1.0
    output = F.grid_sample(img_feat, v, align_corners=False)

    return output.squeeze(-1).permute(0, 2, 1)

class CLIP_Hand_3D_PE(nn.Module):
    def __init__(self,
                 Vertx_dict_path='/home/gsx/data/datasets/FreiHAND/vertices.npy',
                 joints_num=21,
                 hm_res=64,
                 sigma=2.5,
                 J_dim=512,
                 joints_lists=[98, 195, 389, 778],
                 joints_dims=[256, 128, 64, 32],
                 feat_sizes=[8, 16, 32, 64],
                 joints_feats=[2048, 1024, 512, 256],
                 Attn_depth=2,
                 n_heads=8):
        super(CLIP_Hand_3D_PE, self).__init__()

        ''' Config Parameters '''
        self.joints_num = joints_num
        self.hm_res = hm_res
        self.sigma = sigma
        self.J_dim = J_dim
        self.joints_lists = joints_lists
        self.joints_dims = joints_dims
        self.joints_feats = joints_feats
        self.feat_sizes = feat_sizes
        self.Attn_depth = Attn_depth
        self.n_heads = n_heads

        ''' Vertices Template '''
        Vertx_dict = np.load(Vertx_dict_path, allow_pickle=True).item()
        J_Reg = torch.from_numpy(Vertx_dict['J_Reg']).float()
        self.register_buffer('J_Reg', J_Reg)
        Vertx_98 = Vertx_dict['98'].float()
        self.register_buffer('Vertx_98', Vertx_98)
        Vertx_195 = Vertx_dict['195'].float()
        self.register_buffer('Vertx_195', Vertx_195)
        Vertx_389 = Vertx_dict['389'].float()
        self.register_buffer('Vertx_389', Vertx_389)
        Vertx_778 = Vertx_dict['778'].float()
        self.register_buffer('Vertx_778', Vertx_778)

        ''' Visual Encoder: ResNet50 '''
        self.Encoder = ResNetBackbone(resnet_type=50)
        # self.Encoder.init_weights()

        ''' PoseNet '''
        self.PoseNet = PoseNet(joint_num=self.joints_num, hm_res=self.hm_res)
        weights_init(self.PoseNet)

        ''' Pose2Feat '''
        self.Pose2Feat_ = Pose2Feat(joint_num=self.joints_num, hm_res=self.hm_res)
        weights_init(self.Pose2Feat_)

        self.Feat_down_ = nn.Sequential(
            nn.Linear(self.hm_res ** 2, self.J_dim),
            # nn.Linear(self.J_dim, self.joints_dims[0])
            # nn.Dropout(0.1) # 0528
        )
        # self.Feat_down_ = nn.Sequential(
        #     nn.Linear(self.hm_res**2, self.J_dim),
        #     nn.LayerNorm(self.J_dim),
        #     nn.GELU(),
        #     # nn.Linear(self.J_dim, self.joints_dims[0])
        # )
        weights_init(self.Feat_down_)

        ''' Mesh Regressor '''
        # pass

        # weights_init(self.MR_21_98_)
        # weights_init(self.MR_98_195)
        # weights_init(self.MR_195_389)
        # weights_init(self.MR_389_778)

        # self.Out_Layer = nn.Linear(32, 3)
        # weights_init(self.Out_Layer)

        ''' clip settings model '''
        self.latent_x_layer = nn.Linear(672, 512)
        self.latent_y_layer = nn.Linear(672, 512)
        self.latent_z_layer = nn.Linear(1344, 512)

        self.text_nf_encoder = Transformer(512, 2, 16)
        self.text_lr_encoder = Transformer(512, 2, 16)
        self.text_tb_encoder = Transformer(512, 2, 16)

        self.logit_scale_nf = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_lr = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_td = nn.Parameter(torch.log(torch.ones([]) * 100))

        self.logit_scale_xz = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_yz = nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_xy = nn.Parameter(torch.log(torch.ones([]) * 100))

    def make_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(self.hm_res).to(device)
        y = torch.arange(self.hm_res).to(device)
        z = torch.arange(self.hm_res).to(device)
        zz, yy, xx = torch.meshgrid(z, y, x)
        xx = xx[None, None, :, :, :].float()
        yy = yy[None, None, :, :, :].float()
        zz = zz[None, None, :, :, :].float()

        x = joint_coord_img[:, :, 0, None, None, None]
        y = joint_coord_img[:, :, 1, None, None, None]
        z = joint_coord_img[:, :, 2, None, None, None]

        heatmap = torch.exp(
            -(((xx - x) / self.sigma) ** 2) / 2 -
            (((yy - y) / self.sigma) ** 2) / 2 -
            (((zz - z) / self.sigma) ** 2) / 2
        )

        return heatmap

    def forward(self, x, text=None):
        if x.size(1) == 6:
            results = []
            for i in range(2):
                text_i = {
                    'nf': text['nf'][i],
                    'lr': text['lr'][i],
                    'tb': text['tb'][i],
                    # 'nf_mano': text['nf_mano'][i],
                    # 'lr_mano': text['lr_mano'][i],
                    # 'tb_mano': text['tb_mano'][i]
                }
                result = self.forward_(x[:, 3 * i:3 * i + 3], text_i)
                results.append(result)
        else:
            results = self.forward_(x, text)

        return results

    def forward_(self, x, text=None):
        '''
        shared_img_feat: shallow features F0: [B, 64, 64, 64]
        pose_img_feat: F4: [B, 2048, 8, 8]
        feats: F1: [B, 256, 64, 64], F2: [B, 512, 32, 32], F3: [B, 1024, 16, 16]
        feats128: [B, 64, 128, 128]
        '''
        bs = x.shape[0]
        shared_img_feat, pose_img_feat, feats, feats128 = self.Encoder(x)
        F1, F2, F3, F4 = feats[0], feats[1], feats[2], pose_img_feat

        '''
        input: [B, 2048, 8, 8] -> output: [B, 21, 3]
        heatmap_x / y / z: [B, 21, 32], [B, 21, 32], [B, 21, 64]
        '''
        joint_coord_img, heatmap_x, heatmap_y, heatmap_z = self.PoseNet(pose_img_feat)

        if text:
            pred_mask = None

        ''' [B, 21, 3] -> [B, 21, 64, 64, 64] '''
        with torch.no_grad():
            heatmap = self.make_gaussian_heatmap(joint_coord_img * self.hm_res)

        ''' [B, 21, 64, 64, 64] + [B, 64, 64, 64] -> [B, 21, 64, 64] '''
        F_J = self.Pose2Feat_(shared_img_feat, heatmap)

        ''' [B, 21, 64, 64] -> [B, 21, 512] '''
        F_J_21 = self.Feat_down_(F_J.reshape(bs, self.joints_num, -1))

        ''' PE Encodings '''
        PE_98 = self.Vertx_98[None, :, :].repeat(bs, 1, 1)
        PE_195 = self.Vertx_195[None, :, :].repeat(bs, 1, 1)
        PE_389 = self.Vertx_389[None, :, :].repeat(bs, 1, 1)
        PE_778 = self.Vertx_778[None, :, :].repeat(bs, 1, 1)

        ''' Mesh Regressor '''


        ''' CLIP Forward '''
        if text is not None:
            text_nf_features = self.text_nf_encoder(text['nf'])
            text_lr_features = self.text_lr_encoder(text['lr'])
            text_tb_features = self.text_tb_encoder(text['tb'])
            heatmap_x = heatmap_x.reshape(x.shape[0], -1)  # [B, 672]
            heatmap_y = heatmap_y.reshape(x.shape[0], -1)  # [B, 672]
            heatmap_z = heatmap_z.reshape(x.shape[0], -1)  # [B, 1344]
            latent_x = self.latent_x_layer(heatmap_x)  # [B, 512]
            latent_y = self.latent_y_layer(heatmap_y)  # [B, 512]
            latent_z = self.latent_z_layer(heatmap_z)  # [B, 512]
        else:
            text_nf_features = None
            text_lr_features = None
            text_tb_features = None
            latent_x = None
            latent_y = None
            latent_z = None

        return {
            'joint_img': joint_coord_img,

            'latent_x': latent_x,
            'latent_y': latent_y,
            'latent_z': latent_z,
            'nf_features': text_nf_features,
            'lr_features': text_lr_features,
            'tb_features': text_tb_features
        }

def speed_():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIP_Hand_3D_PE().to(device)
    x = torch.randn([1, 3, 256, 256]).to(device)
    from thop import profile
    from thop import clever_format

    macs, params = profile(model, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")

    print(macs, params)
    # exit()

    import time

    model.eval()
    with torch.no_grad():
        for i in range(10000):
            s = time.time()
            model(x)
            # exit()
            d = time.time()
            print("FPS is {}".format(1 / (d - s)))

def feature_match(
    latent_pose_x,
    latent_pose_y,
    latent_pose_z,
    near_far_features,
    left_right_features,
    top_down_features,
    logit_scale1=None,
    logit_scale2=None,
    logit_scale3=None
):
    logit_scale1 = torch.clamp(logit_scale1.exp(), min=1.0, max=100.0)
    logit_scale2 = torch.clamp(logit_scale2.exp(), min=1.0, max=100.0)
    logit_scale3 = torch.clamp(logit_scale3.exp(), min=1.0, max=100.0)

    logit_latent_nf = logit_scale1 * latent_pose_z @ near_far_features.T
    logit_latent_lr = logit_scale2 * latent_pose_x @ left_right_features.T
    logit_latent_td = logit_scale3 * latent_pose_y @ top_down_features.T

    probs_z = logit_latent_nf.softmax(dim=-1).cpu().numpy()
    probs_x = logit_latent_lr.softmax(dim=-1).cpu().numpy()
    probs_y = logit_latent_td.softmax(dim=-1).cpu().numpy()

    return {
        'probs_x': probs_x,
        'probs_y': probs_y,
        'probs_z': probs_z,
    }


if __name__ == '__main__':
    pass
