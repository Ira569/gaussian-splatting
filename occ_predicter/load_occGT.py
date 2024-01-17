# 读取某一帧的GT  200，200，16
# occ_gt_path = '../data/nuscenes/gts/scene-0061/0cd661df01aa40c3bb3a773ba86f753a/labels.npz'
#  scene-num/frame_token/labels.npz
import numpy as np

from nuscenes.nuscenes import NuScenes
version = "v1.0-mini"
dataroot = "../data/nuscenes"
nuscenes = NuScenes(version, dataroot, verbose=False)

sample = nuscenes.sample[0]
scene = nuscenes.get("scene",sample['scene_token'])
scene_name =scene['name']
sample_token = sample['token']
import os
occ_gt_path = os.path.join('../data/nuscenes/gts/',scene_name,sample_token,'labels.npz')

occ_labels = np.load(occ_gt_path)
semantics = occ_labels['semantics']
mask_lidar = occ_labels['mask_lidar']
mask_camera = occ_labels['mask_camera']

np.save('occ_label.npy',semantics)
np.save('mask_camera.npy',mask_camera)
print("load occ done!")
# results['voxel_semantics'] = semantics
# results['mask_lidar'] = mask_lidar
# results['mask_camera'] = mask_camera