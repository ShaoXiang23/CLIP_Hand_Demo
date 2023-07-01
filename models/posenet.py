import torch.nn as nn
import torch
import torch.nn.functional as F

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class PoseNet(nn.Module):
    def __init__(self, joint_num=21, hm_res=64):
        super(PoseNet, self).__init__()

        self.joint_num = joint_num
        self.hm_res = hm_res

        self.deconv = make_deconv_layers([2048, 256, 256])
        self.conv_x = make_conv1d_layers(
            [256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers(
            [256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        # self.conv_z_1 = make_conv1d_layers([2048, 256 * output_hm_shape[0]], kernel=1, stride=1, padding=0)
        # self.conv_z_2 = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers(
            [2048, 21 * hm_res], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers(
            [self.joint_num, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)

        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat)

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)

        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)

        # z axis
        img_feat_z = img_feat.mean((2, 3))[:, :, None] # [B, 2048, 8, 8]
        # img_feat_z = self.conv_z_1(img_feat_z)
        # img_feat_z = img_feat_z.view(-1, 256, output_hm_shape[0])
        # heatmap_z = self.conv_z_2(img_feat_z)
        # coord_z = self.soft_argmax_1d(heatmap_z)
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1, self.joint_num, self.hm_res)
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)

        joint_coord = torch.cat((coord_x, coord_y, coord_z), 2)

        # return joint_coord
        return joint_coord, heatmap_x, heatmap_y, heatmap_z