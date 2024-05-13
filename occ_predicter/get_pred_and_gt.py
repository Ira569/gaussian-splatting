import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import cv2
import os
version = "v1.0-mini"
dataroot = "../data/nuscenes"
nuscenes = NuScenes(version, dataroot, verbose=False)

def get_matrix(calibrated_data, inverse=False):
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3, 3]  = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output

x_list = []
iou_list = []
for i in range(200):
    sample = nuscenes.sample[i]

    scene = nuscenes.get("scene",sample['scene_token'])
    scene_name =scene['name']
    sample_token = sample['token']
    import os
    occ_gt_path = os.path.join('../data/nuscenes/gts/',scene_name,sample_token,'labels.npz')

    occ_labels = np.load(occ_gt_path)
    semantics = occ_labels['semantics']
    mask_lidar = occ_labels['mask_lidar']
    mask_camera = occ_labels['mask_camera']

    # np.save('occ_label.npy',semantics)
    # np.save('mask_camera.npy',mask_camera)
    # print("load occ done!")

    import open3d as o3d
    from plyfile import PlyData, PlyElement
    #   nuscenes lidar (x,y,z,intensity,ring index)
    # 来自lidar_top的点云数据 软件可以看到是在车前上方的
    # lidar_file = r"nuscenes-6imgs/example.pcd.bin"
    np.set_printoptions(precision=3, suppress=True)

    # 获取lidar的数据
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sample_data = nuscenes.get('sample_data', lidar_token)
    lidar_file = os.path.join(dataroot, lidar_sample_data["filename"])


    lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5) # (point_num,5)
    lidar_points_xyz = lidar_points[:,:3]
    #  将ladar_points 转到ego坐标系
    lidar_calibrated_data = nuscenes.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

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

    lidar_points_xyz = ego_points[:,:3]


    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(lidar_points_xyz)

    # 保存点云数据为PLY文件  但这还是lidar坐标系下的
    o3d.io.write_point_cloud('./pc2ego_output.ply', point_cloud)
    # print('done')






    from plyfile import PlyData
    # 需要初始点云标签的过滤，参考filter_points函数，过滤后IoU能到10.6
    # IoU= 10.624343287123343
    # 试一下只用初始点云啥结果
    plydata = PlyData.read('./pc2ego_output.ply')
    # IoU= 8.523155830441997

    vertices = plydata['vertex']
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    points = np.column_stack((x, y, z))  # (Num,3)
    #先筛选出在范围内的points
    mask_x_min = points[:, 0] >= 40
    mask_x_max = points[:, 0] <= -40
    mask_y_min = points[:, 1] >= 40
    mask_y_max = points[:, 1] <= -40
    mask_z_min = points[:, 2] >= 5.4
    mask_z_max = points[:, 2] <= -1
    pos_mask = mask_z_max | mask_z_min | mask_x_max | mask_x_min | mask_y_max | mask_y_min
    valid_points_mask = ~pos_mask

    points=points[valid_points_mask] #45w ->37w个点
    # 在对应格确定是否被占用
    pred = np.zeros((200,200,16))

    # todo 使用别的方法判断高斯是否占用 (按照体积大小？,)
    for p in points:
        cube_x = 2.5*p[0]+100 # int  -40~40 -> 0-199 -40 ->0  40 ->199
        cube_y = 2.5*p[1]+100  # int  -40~40 -> 0-199 80-200
        cube_z = 2.5*p[2]+2.5  # int  -1~5.4 -> 0-15 6.4-16
        cube_x,cube_y,cube_z=int(cube_x),int(cube_y),int(cube_z)
        pred[cube_x,cube_y,cube_z]=1
        # pass

    # np.save('occ_pred.npy',pred)

    from cal_IoU import cal_IoU
    # label = np.load('occ_label.npy')
    label = occ_labels['semantics']
    # pred = np.load('occ_pred.npy')
    # 1应该是free  0应该是 占用，才好用mIoU类

    label[label!=17] = 0
    label[label==17] = 1

    pred[pred==0]=2
    pred[pred==1]=0
    pred[pred==2]=1
    # emm,主要关注 标签应该是占用的，但是预测为未占用的，那就说明是点云重建不充分
    # 如果标签应该是未占用，但是预测为占用，那就是点云重建的太多了，在不应该重建的地方重建了
    # print('label occupied sum',200*200*16-label.sum())  # 30093
    # print('pred occupied sum',200*200*16-pred.sum())    # 38822

    # 调用函数 计算IoU
    IoU = cal_IoU(pred,label)
    # print('IoU=',IoU)
    x_list.append(i)
    iou_list.append(IoU)


import matplotlib.pyplot as plt
# 绘制折线图
plt.plot(x_list, iou_list, marker='o', linestyle='-')

# 添加标题和标签
plt.title('IoU vs Steps')
plt.xlabel('Steps')
plt.ylabel('IoU')

# 显示图形
plt.show()