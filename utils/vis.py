import numpy as np
import cv2

def plot3D(ax, pos, azim=0, elev=0):
    color_hand_joints = [[1.0, 0.0, 0.0],
                         [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                         [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                         [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                         [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                         [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1.0, 1.0, 1.0])
    position = np.copy(pos)
    uvd_pt = np.reshape(position, [21, 3])

    ax.dist = 8
    ax.grid(True)

    marker_sz = 20
    line_wd = 5

    for joint_ind in range(uvd_pt.shape[0]):
        ax.plot(uvd_pt[joint_ind:joint_ind + 1, 0], uvd_pt[joint_ind:joint_ind + 1, 1],
                uvd_pt[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(uvd_pt[[0, joint_ind], 0], uvd_pt[[0, joint_ind], 1], uvd_pt[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lw=line_wd)
        else:
            ax.plot(uvd_pt[[joint_ind - 1, joint_ind], 0], uvd_pt[[joint_ind - 1, joint_ind], 1],
                    uvd_pt[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    lw=line_wd)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    return ax

def inv_base_tranmsform(x, mean=0.5, std=0.5):
    x = x.transpose(1, 2, 0)
    image = (x * std + mean) * 255
    return image.astype(np.uint8)

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

def base_transform(img, size, mean=0.5, std=0.5):
    x = cv2.resize(img, (size, size)).astype(np.float32) / 255
    x -= mean
    x /= std
    x = x.transpose(2, 0, 1)

    return x