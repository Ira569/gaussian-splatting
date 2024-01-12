import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
#   nuscenes lidar (x,y,z,intensity,ring index)
# 来自lidar_top的点云数据 软件可以看到是在车前上方的
# lidar_file = r"nuscenes-6imgs/example.pcd.bin"

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import numpy as np
import cv2
import os

np.set_printoptions(precision=3, suppress=True)

# 构建nuScenes类
version = "v1.0-mini"
dataroot = "nuscenes"
nuscenes = NuScenes(version, dataroot, verbose=False)

sample = nuscenes.sample[0]

# 获取lidar的数据
lidar_token = sample["data"]["LIDAR_TOP"]
lidar_sample_data = nuscenes.get('sample_data', lidar_token)
lidar_file = os.path.join(dataroot, lidar_sample_data["filename"])


lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5) # (point_num,5)
lidar_points_xyz = lidar_points[:,:3]
#  将ladar_points 转到ego坐标系
lidar_calibrated_data = nuscenes.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
from nusc_demo.myvisualize import get_matrix
# lidar_pose是基于ego而言的
# point = lidar_pose @ lidar_points.T  代表了 lidar -> ego 的过程
lidar_to_ego = get_matrix(lidar_calibrated_data)

#  ego_pose0 -> global
# ego_pose就是基于global坐标系的
# ego_pose @ ego_points.T     ego -> global
# lidar_points -> N x 5(x, y, z, intensity, ringindex)
# x, y, z -> x, y, z, 1
hom_points = np.concatenate([lidar_points[:, :3], np.ones((len(lidar_points), 1))], axis=1)
ego_points = hom_points @ lidar_to_ego.T   # lidar_points -> ego
# TODO 在ego周围的x,y,z某范围内的点云直接删掉，不然会影响生成高斯可视化的效果
#  mask = ego_points

lidar_points_xyz = ego_points[:,:3]
mask_z_max = lidar_points_xyz[:,2]<1.7
mask_z_min = lidar_points_xyz[:,2]>-1
mask_x_max = lidar_points_xyz[:,0]<3
mask_x_min = lidar_points_xyz[:,0]>-0.5
mask_y_max = lidar_points_xyz[:,1]<0.8
mask_y_min = lidar_points_xyz[:,1]>-0.8
not_selected = mask_z_max & mask_z_min & mask_x_max & mask_x_min & mask_y_max & mask_y_min
selected = ~not_selected
selected_points = lidar_points_xyz[selected]

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(selected_points)

# 保存点云数据为PLY文件  但这还是lidar坐标系下的
o3d.io.write_point_cloud('./filtered_pc2ego_output.ply', point_cloud)
# ply格式：
# x：点的x坐标（浮点数# y：点的y坐标（浮点数# z：点的z坐标（浮点数）
# nx：法线向量的x分量（浮点数）# ny：法线向量的y分量（浮点数）# nz：法线向量的z分量（浮点数）
# red：点的红色分量（无符号字符# green：点的绿色分量（无符号字符）# blue：点的蓝色分量（无符号字符


# plydata = PlyData.read('nuscenes-6imgs/input_example.ply')
# vertices = plydata['vertex']
# x = vertices['x']
# y = vertices['y']
# z = vertices['z']
#
# points = np.column_stack((x, y, z))  # (6381,3)
#


# json 文件： id,img_name,width,height,position,rotation,fy,fx
# 其实就是内参矩阵 K 与 外参矩阵[R T]  K自带的数据集里有， RT好像也有需要转换一下，TODO 吃完饭在搞
# K = | fx   0   cx |
#     |  0  fy   cy |
#     |  0   0    1 |
# 其中，fx和fy表示焦距，cx和cy表示主点的像素坐标。最后一行通常为[0 0 1]，用于齐次坐标表示。
# cx cy 一般为宽和高的一半  fx fy一般差不多
# 只是不知道colmap的坐标系和 nuscenes的坐标系能不能统一
# 但是反正K是不会改变的， 而RT的大小与选择的世界坐标系有关(在ego 坐标系下的坐标？)
# nuscenes好像还有一个不会变的世界坐标系 就是出发点的坐标系，但是一般后面ego离世界坐标系有1000多m远，
# 离太远了应该也是没什么关系的，而且在实际预测的时候也只能输入当时的六个图片，也不可能考虑用后来的图片吧
# 感觉可以根据3dGaussion的代码来改，改之后进行3d占用预测的任务，
# 感觉尽量不用mmdet框架，不然真有点麻烦感觉