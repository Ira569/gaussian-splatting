# pip install nuscenes-devkit
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import numpy as np
import cv2
import os



# 构建nuScenes类
version = "v1.0-mini"
dataroot = "nuscenes"
nuscenes = NuScenes(version, dataroot, verbose=False)
# TODO  按照判断取next是否为空的方式，获得该场景的所有sample,再进行训练.
sample = nuscenes.sample[0]
sample1 = nuscenes.sample[1]
sample2 = nuscenes.sample[2]
# sample1["timestamp"]-sample["timestamp"]
# sample["next"]
sample_next = nuscenes.get("sample",sample["next"])
sample_next_next = nuscenes.get("sample",sample_next["next"])
assert sample["next"]==sample_next["token"]
assert sample_next["next"]==sample_next_next["token"]


def get_matrix(calibrated_data, inverse=False):
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3, 3] = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output

cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT', ]

import json
nerf_like_json = [{"id": 1, "file_path": "./images/CAM_FRONT_LEFT",
                   "width": 1653, "height": 959,
                   "transform_matrix": [[-0.3456006895797708, 0.9382316025189897, 0.0167816386755018,1],
                                        [-0.9383334713320822, -0.3453464096120683, -0.016314225263176777,2],
                                        [-0.009511043048534566, -0.021384980773937943, 0.9997260738109348,3],
                                        [0,0,0,1]],
                   "fy": 1272.5979470598488, "fx": 1272.5979470598488}]
# gaussion_camera_json = [{"id": 0, "img_name": "CAM_BACK", "width": 1600, "height": 900,
#                          "position": [411.4449780367985, 1181.2631893914647, 0.0],
#                          "rotation": [[], [], []], "fy": 1, "fy": 1}]

json_list = []

id = 0
sample_list =[sample,sample_next,sample_next_next]
sample_list=[sample]
sample_id = 0
# 即使是在同一帧， 每个相机对应的time-stamp和ego_pose也都不一样
global2ego = None
cam = 'CAM_FRONT_LEFT'
camera_token = sample["data"][cam]
camera_data = nuscenes.get("sample_data", camera_token)
camera_ego_pose = nuscenes.get("ego_pose", camera_data["ego_pose_token"])
ego02global=get_matrix(camera_ego_pose)
# ego02global= np.linalg.inv(global2ego0)
for sample in sample_list:


    for cam in cameras:

        camera_token = sample["data"][cam]
        camera_data = nuscenes.get("sample_data", camera_token)
        image_file = os.path.join(dataroot, camera_data["filename"])
        image = cv2.imread(image_file)

        camera_ego_pose = nuscenes.get("ego_pose", camera_data["ego_pose_token"])

        # ego pose本身就是基于global而言的  ego_pose 本身就代表了 ego2global的矩阵
        global_to_ego = get_matrix(camera_ego_pose, True)  # global -> ego

        camera_calibrated = nuscenes.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
        ego_to_camera = get_matrix(camera_calibrated, True)  # ego -> camera
        camera_intrinsic = np.eye(4)
        camera_intrinsic[:3, :3] = camera_calibrated["camera_intrinsic"]
        # 我应该传c2w还是应该传w2c啊 应该传c2w
        # global -> camera_ego_pose -> image
        global_to_cam = ego_to_camera @ global_to_ego
        cam2global=np.linalg.inv(global_to_cam)
        ego02cam =  ego_to_camera@global_to_ego @ego02global
        cam2ego0=np.linalg.inv(ego02cam)
        global_to_image = camera_intrinsic @ ego_to_camera @ global_to_ego

        id = id + 1
        camera_dict = {"id": id, "file_path": image_file,
        "width": image.shape[1], "height": image.shape[0],
        "transform_matrix":cam2ego0.tolist(),
        "fy": camera_intrinsic[1][1], "fx": camera_intrinsic[0][0]}
        json_list.append(camera_dict)

                           # cv2.imwrite(f"{cam}.jpg", image)
    sample_id +=1

json_string = json.dumps(json_list)
with open("nusc_demo/transforms_nusc3imgs_train.json", "w") as json_file:
    json_file.write(json_string)

print("JSON 文件已生成！")

print('done')