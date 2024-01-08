# pip install nuscenes-devkit
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import numpy as np
import cv2 
import os


np.set_printoptions(precision=3, suppress=True)

version = "mini"
dataroot = "/data/mini"
nuscenes = NuScenes(version='v1.0-{}'.format(version), dataroot=dataroot, verbose=False)

cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
sample  = nuscenes.sample[0]

def to_matrix4x4_2(rotation, translation, inverse=True):
    output = np.eye(4)
    output[:3, :3] = rotation
    output[:3, 3]  = translation
    
    if inverse:
        output = np.linalg.inv(output)
    return output

def to_matrix4x4(m):
    output = np.eye(4)
    output[:3, :3] = m
    return output

# Lidar的点云是基于Lidar坐标系的
# 需要将lidar的坐标转换到ego，然后ego到global
lidar_sample_data = nuscenes.get('sample_data', sample['data']["LIDAR_TOP"])
lidar_file        = os.path.join(nuscenes.dataroot, lidar_sample_data["filename"])
lidar_pointcloud  = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)
ego_pose          = nuscenes.get('ego_pose', lidar_sample_data['ego_pose_token'])
ego_to_global     = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']), False)
lidar_sensor      = nuscenes.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
lidar_to_ego      = to_matrix4x4_2(Quaternion(lidar_sensor['rotation']).rotation_matrix, np.array(lidar_sensor['translation']), False)
lidar_to_global   = ego_to_global @ lidar_to_ego
lidar_points      = np.concatenate([lidar_pointcloud[:, :3], np.ones((len(lidar_pointcloud), 1))], axis=1)
lidar_points      = lidar_points @ lidar_to_global.T
# 1. annotation标注是global坐标系的
# 2. 由于时间戳的缘故，每个camera/lidar/radar都有ego_pose用于修正差异
for camera in cameras:
    camera_sample_data = nuscenes.get('sample_data', sample['data'][camera])
    image_file         = os.path.join(nuscenes.dataroot, camera_sample_data["filename"])
    ego_pose           = nuscenes.get('ego_pose', camera_sample_data['ego_pose_token'])
    global_to_ego      = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']))

    camera_sensor    = nuscenes.get('calibrated_sensor', camera_sample_data['calibrated_sensor_token'])
    camera_intrinsic = to_matrix4x4(camera_sensor['camera_intrinsic'])
    ego_to_camera    = to_matrix4x4_2(Quaternion(camera_sensor['rotation']).rotation_matrix, np.array(camera_sensor['translation']))

    global_to_image  = camera_intrinsic @ ego_to_camera @ global_to_ego
    image = cv2.imread(image_file)
    for annotation_token in sample['anns']:
        instance = nuscenes.get('sample_annotation', annotation_token)
        box = Box(instance['translation'], instance['size'], Quaternion(instance['rotation']))
        corners = np.ones((4, 8))
        corners[:3, :] = box.corners()
        corners = (global_to_image @ corners)[:3]
        corners[:2] /= corners[[2]]
        corners = corners.T.astype(int)
        
        ix, iy = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 1, 2, 3, 0, 5, 6, 7, 4]
        for p0, p1 in zip(corners[ix], corners[iy]):
            if p0[2] <= 0 or p1[2] <= 0: continue
            cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), (0, 255, 0), 2, 16)

    image_based_points = lidar_points @ global_to_image.T
    image_based_points[:, :2] /= image_based_points[:, [2]]
    show_points = image_based_points[image_based_points[:, 2] > 0][:, :2].astype(np.int32)
    for x, y in show_points:
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1, 16)

    cv2.imwrite(f"{camera}.jpg", image)