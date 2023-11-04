import torch

def uvd2xyz(
        uvd,
        joint_root,
        joint_bone,
        intr=None,
        trans=None,
        scale=None,
        inp_res=256,
        mode='persp'
):
    bs = uvd.shape[0]
    if mode in ['persp', 'perspective']:
        if intr is None:
            raise Exception("No intr found in perspective")
        '''1. denormalized uvd'''
        uv = uvd[:, :, :2] * inp_res  # 0~256
        # uv = uvd[:, :, :2]
        # depth = (uvd[:, :, 2] * cfg.DEPTH_RANGE) + cfg.DEPTH_MIN
        depth = uvd[:, :, 2]
        root_depth = joint_root[:, -1].unsqueeze(-1)  # (B, 1)
        z = depth * joint_bone.expand_as(uvd[:, :, 2]) + \
            root_depth.expand_as(uvd[:, :, 2])  # B x M

        '''2. uvd->xyz'''
        camparam = torch.zeros((bs, 4)).float().to(intr.device)  # (B, 4)
        camparam[:, 0] = intr[:, 0, 0]  # fx
        camparam[:, 1] = intr[:, 1, 1]  # fx
        camparam[:, 2] = intr[:, 0, 2]  # cx
        camparam[:, 3] = intr[:, 1, 2]  # cy
        camparam = camparam.unsqueeze(1).expand(-1, uvd.size(1), -1)  # B x M x 4
        xy = ((uv - camparam[:, :, 2:4]) / camparam[:, :, :2]) * \
             z.unsqueeze(-1).expand_as(uv)  # B x M x 2
        return torch.cat((xy, z.unsqueeze(-1)), -1)  # B x M x 3
    elif mode in ['ortho', 'orthogonal']:
        if trans is None or scale is None:
            raise Exception("No trans or scale found in orthorgnal")
        raise Exception("orth Unimplement !")
    else:
        raise Exception("Unkonwn mode type. should in ['persp', 'ortho']")


def xyz2uvd(
        xyz,
        joint_root,
        joint_bone,
        intr=None,
        trans=None,
        scale=None,
        inp_res=256,
        mode='persp'
):
    bs = xyz.shape[0]
    if mode in ['persp', 'perspective']:
        if intr is None:
            raise Exception("No intr found in perspective")
        z = xyz[:, :, 2]
        xy = xyz[:, :, :2]
        xy = xy / z.unsqueeze(-1).expand_as(xy)

        ''' 1. normalize depth : root_relative, scale_invariant '''
        root_depth = joint_root[:, -1].unsqueeze(-1)  # (B, 1)
        depth = (z - root_depth.expand_as(z)) / joint_bone.expand_as(z)

        '''2. xy->uv'''
        camparam = torch.zeros((bs, 4)).float().to(intr.device)  # (B, 4)
        camparam[:, 0] = intr[:, 0, 0]  # fx
        camparam[:, 1] = intr[:, 1, 1]  # fx
        camparam[:, 2] = intr[:, 0, 2]  # cx
        camparam[:, 3] = intr[:, 1, 2]  # cy
        camparam = camparam.unsqueeze(1).expand(-1, xyz.size(1), -1)  # B x M x 4
        uv = (xy * camparam[:, :, :2]) + camparam[:, :, 2:4]

        '''3. normalize uvd to 0~1'''
        uv = uv / inp_res
        # depth = (depth - cfg.DEPTH_MIN) / cfg.DEPTH_RANGE

        return torch.cat((uv, depth.unsqueeze(-1)), -1)
    elif mode in ['ortho', 'orthogonal']:
        if trans is None or scale is None:
            raise Exception("No trans or scale found in orthorgnal")
        raise Exception("orth Unimplement !")
    else:
        raise Exception("Unkonwn proj type. should in ['persp', 'ortho']")