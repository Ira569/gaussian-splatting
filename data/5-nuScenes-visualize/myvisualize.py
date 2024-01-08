# pip install nuscenes-devkit
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import numpy as np
import cv2
import os

np.set_printoptions(precision=3, suppress=True)

# 构建nuScenes类
version = "v1.0-mini"
dataroot = "E:/NeRF_git/gaussian-splatting/data/nuscenes"
nuscenes = NuScenes(version, dataroot, verbose=False)

sample = nuscenes.sample[0]

# 获取lidar的数据
lidar_token = sample["data"]["LIDAR_TOP"]
lidar_sample_data = nuscenes.get('sample_data', lidar_token)
lidar_file = os.path.join(dataroot, lidar_sample_data["filename"])

# 加载点云数据，点云数据的坐标是相对于lidar的位置而言   x y z * * 
lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)

# 坐标系
# 1. 全局坐标系，global coordinate
#    - 可以简单的认为，车辆在t0时刻的位置认为是全局坐标系的原点
# 2. 车体坐标系，ego_pose,  ego coordinate
#    - 以车体为原点的坐标系
# 3. 传感器坐标系
#    - lidar的坐标系
#    - camera的坐标系
#    - radar的坐标系
#
# 标定calibater
# lidar的标定，获得的结果是：lidar相对于ego而言的位置(translation)，和旋转(rotation)
#                        - translation可以用3个float数字表示位置
#                             - 相对于ego而言的位置
#                        - rotation则是用4个float表示旋转，用的是四元数
# camera的标定，获得的结果是：camera相对于ego而言的位置(translation)，和旋转(rotation)
#                        - 相机内参 camera intrinsic(3d->2d平面)
#                        - 相机畸变参数（目前nuScenes数据集不考虑）
#                        - translation可以用3个float数字表示位置
#                             - 相对于ego而言的位置
#                        - rotation则是用4个float表示旋转，用的是四元数
#
# 不同传感器的频率不同，也就是捕获数据的时间不同
# lidar  捕获的timestamp是t0, t0 -> ego_pose0
# camera 捕获的timestamp是t1, t1 -> ego_pose1
# lidar_points -> camera
# lidar_points -> ego_pose0 -> global -> ego_pose1 -> camera -> intrinsic -> image

# lidar_points -> ego_pose0
# lidar_points是基于lidar coordinate而言的
# lidar_coordinate  lidar_pose(基于ego)
# lidar_points = lidar_pose @ lidar_points   # lidar -> ego

def get_matrix(calibrated_data, inverse=False):
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3, 3]  = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output

lidar_calibrated_data = nuscenes.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

# lidar_pose是基于ego而言的
# point = lidar_pose @ lidar_points.T  代表了 lidar -> ego 的过程
lidar_to_ego = get_matrix(lidar_calibrated_data)

#  ego_pose0 -> global
# ego_pose就是基于global坐标系的
# ego_pose @ ego_points.T     ego -> global
ego_pose0 = nuscenes.get("ego_pose", lidar_sample_data["ego_pose_token"])
ego_to_global0 = get_matrix(ego_pose0)
lidar_to_global = ego_to_global0 @ lidar_to_ego
# lidar_points -> N x 5(x, y, z, intensity, ringindex)
# x, y, z -> x, y, z, 1
hom_points = np.concatenate([lidar_points[:, :3], np.ones((len(lidar_points), 1))], axis=1)
global_points = hom_points @ lidar_to_global.T   # lidar_points -> global
cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

for cam in cameras:
    camera_token = sample["data"][cam]
    camera_data  = nuscenes.get("sample_data", camera_token)
    image_file = os.path.join(dataroot, camera_data["filename"])
    image = cv2.imread(image_file)
    
    camera_ego_pose = nuscenes.get("ego_pose", camera_data["ego_pose_token"])
    
    # ego pose本身就是基于global而言的
    global_to_ego = get_matrix(camera_ego_pose, True)  # global -> ego
    
    camera_calibrated = nuscenes.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
    ego_to_camera = get_matrix(camera_calibrated, True)  # ego -> camera
    camera_intrinsic = np.eye(4)
    camera_intrinsic[:3, :3] = camera_calibrated["camera_intrinsic"]
    
    # global -> camera_ego_pose -> image
    global_to_image = camera_intrinsic @ ego_to_camera @ global_to_ego
    
    for token in sample["anns"]:
        annotation = nuscenes.get("sample_annotation", token)
        box = Box(annotation['translation'], annotation['size'], Quaternion(annotation['rotation']))
        corners = box.corners().T  # 3x8
        global_corners = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
        image_based_corners = global_corners @ global_to_image.T
        image_based_corners[:, :2] /= image_based_corners[:, [2]]
        image_based_corners = image_based_corners.astype(np.int32)
        
        ix, iy = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 1, 2, 3, 0, 5, 6, 7, 4]
        for p0, p1 in zip(image_based_corners[ix], image_based_corners[iy]):
            if p0[2] <= 0 or p1[2] <= 0: continue
            # cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), (0, 255, 0), 2, 16)
            pass
            
    # 坐标变换了
    image_points = global_points @ global_to_image.T  # global -> image
    image_points[:, :2] /= image_points[:, [2]]
    
    # 过滤z<=0的点，因为它在图像平面的后面。形不成投影的
    # z > 0
    # for x, y in image_points[image_points[:, 2] > 0, :2].astype(int):
    #     cv2.circle(image, (x, y), 3, (255, 0, 0), -1, 16)
    
    cv2.imwrite(f"{cam}.jpg", image)


print ('done')