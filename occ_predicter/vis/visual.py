# from tools.mayaviOffScreen import mlab
import os, sys
import cv2, imageio

import mayavi.mlab as mlab
import numpy as np
import torch


# colors = np.array(
#     [
#         [0, 0, 0, 255],  # void, ignore 
#         [255, 120, 50, 255],  # barrier              orangey
#         [255, 192, 203, 255],  # bicycle              pink
#         [255, 255, 0, 255],  # bus                  yellow
#         [0, 150, 245, 255],  # car                  blue
#         [0, 255, 255, 255],  # construction_vehicle cyan
#         [200, 180, 0, 255],  # motorcycle           dark orange
#         [255, 0, 0, 255],  # pedestrian           red
#         [255, 240, 150, 255],  # traffic_cone         light yellow
#         [135, 60, 0, 255],  # trailer              brown
#         [160, 32, 240, 255],  # truck                purple
#         [255, 0, 255, 255],  # driveable_surface    dark pink    深粉色 可行驶表面
#         # [175,   0,  75, 255],       # other_flat           dark red
#         [139, 137, 137, 255],   # other_flat 
#         [75, 0, 75, 255],  # sidewalk             dard purple
#         [150, 240, 80, 255],  # terrain              light green
#         [230, 230, 250, 255],  # manmade              white  白色 人造
#         [0, 175, 0, 255],  # vegetation           green  绿色 植物     
#         [0, 255, 127, 255],  # ego car  17- free             dark cyan
#         [255, 99, 71, 255],
#         [0, 191, 255, 255]
#     ]
# ).astype(np.uint8)

#  rgb格式下的colors    255,   0,   0 绿色
colors = np.array(
    [
        [255,255,255, 255],  # 0 others red
        [255, 158, 0, 255],  # 1 'barrier'  orange
        [0, 0, 230, 255],    # 2 'bicycle'  Blue
        [47, 79, 79, 255],   # 3 'bus  Darkslategrey
        [220, 20, 60, 255],  # 4 'car' Crimson
        [255, 69, 0, 255],   # 5 'construction_vehicle  Orangered
        [255, 140, 0, 255],  # 6 'motorcycle' Darkorange
        [233, 150, 70, 255], # 7 'pedestrian  Darksalmon
        [255, 61, 99, 255],  # 8 'traffic_cone'  Red
        [112, 128, 144, 255],# 9 'trailer  Slategrey
        [222, 184, 135, 255],# 10'truck' Burlywood
        [0, 175, 0, 255],    # 11 driveable_surface'  Green  drive_surface  RGB
        [165, 42, 42, 255],  # 12 'other_flat'  nuTonomy green
        [0, 207, 191, 255],  # 13 'sidewalk'
        [75, 0, 75, 255], # 14 terrain'
        [255, 0, 0, 255], # 15 'manmade'  Red
        [0, 255, 0, 255],  # 16   vegetation Green
        [0, 0, 0, 255],  # 17 undefined  free   white  un-occupied

    ]).astype(np.uint8)

def voxel2pcd(voxel,voxel_size = 0.4):
    # pcd (N,4)   N*(x,y,z,class)
    x,y,z = voxel.shape[0],voxel.shape[1],voxel.shape[2]
    num = (voxel!=17).sum()
    num = x*y*z
    # voxel (X,Y,Z,class)   v[x,y,z]=class   example:(200,200,16)->(200*200*16,4)
    pcd = np.zeros((num,4),dtype=int)
    index = 0
    for i in range(x):
        for j in range(y):
            for k in range(z):
                # if voxel[i,j,k]!=17:
                pcd[index]=np.array([i,j,k,voxel[i,j,k]])
                index += 1

    return pcd


# rgb -> brg 格式下的color_map   0 1 2 -> 2 0 1
colors_grb = colors.copy()
colors_grb[:, [0, 1,2]] = colors_grb[:, [2,0, 1]]

#mlab.options.offscreen = True

voxel_size = 0.4
pc_range = [-40, -40,  -1, 40, 40, 5.4]

# visual_path = sys.argv[1]
# visual_path ='vis_npy/gt_seman_10.npy'

visual_path ='../occ_label.npy'
# visual_path ='../occ_pred.npy'

voxels = np.load(visual_path).astype(np.int32)

ignore_list = []


fov_voxels = voxel2pcd(voxels)


# fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
fov_voxels = fov_voxels[fov_voxels[..., 3] != 17]
fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
fov_voxels[:, 0] += pc_range[0]
fov_voxels[:, 1] += pc_range[1]
fov_voxels[:, 2] += pc_range[2]


figure = mlab.figure(size=(1440, 960), bgcolor=(0, 0, 0))

# figure = mlab.figure(size=(1440, 960), bgcolor=(1, 1, 1))
# pdb.set_trace()
plt_plot_fov = mlab.points3d(
    fov_voxels[:, 0],
    fov_voxels[:, 1],
    fov_voxels[:, 2],
    fov_voxels[:, 3],
    # colormap="viridis",
    colormap='brg',
    # scale_factor=voxel_size - 0.05*voxel_size,
    scale_factor=1,
    mode="cube",
    opacity=1.0,
    vmin=0,
    vmax=19,
)


plt_plot_fov.glyph.scale_mode = "scale_by_vector"
plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors_grb


visual_path ='../occ_pred.npy'
voxels2 = np.load(visual_path).astype(np.int32)

ignore_list = []


fov_voxels2 = voxel2pcd(voxels2)


fov_voxels2 = fov_voxels2[fov_voxels2[..., 3] > 0]
# fov_voxels2 = fov_voxels2[fov_voxels2[..., 3] != 17]
fov_voxels2[:, :3] = (fov_voxels2[:, :3] + 0.5) * voxel_size
fov_voxels2[:, 0] += pc_range[0]
fov_voxels2[:, 1] += pc_range[1]
fov_voxels2[:, 2] += pc_range[2]

figure2 = mlab.figure(size=(1440, 960), bgcolor=(0, 0, 0))
# figure2 = mlab.figure(size=(1440, 960), bgcolor=(1, 1, 1))

# pdb.set_trace()
plt_plot_fov = mlab.points3d(
    fov_voxels2[:, 0],
    fov_voxels2[:, 1],
    fov_voxels2[:, 2],
    fov_voxels2[:, 3],
    # colormap="viridis",
    colormap='brg',
    # scale_factor=voxel_size - 0.05*voxel_size,
    scale_factor=1,
    mode="cube",
    opacity=1.0,
    vmin=0,
    vmax=19,
)


plt_plot_fov.glyph.scale_mode = "scale_by_vector"
plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors_grb






# filename = os.path.basename(visual_path) 
# file_without_extension = os.path.splitext(filename)[0]

# mlab.savefig('3dOcc_vis/'+file_without_extension+'.png')
mlab.show()



