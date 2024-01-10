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

def get_matrix(calibrated_data, inverse=False):
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3, 3] = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output

cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

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
for cam in cameras:

    camera_token = sample["data"][cam]
    camera_data = nuscenes.get("sample_data", camera_token)
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

    id = id + 1
    camera_data = {"id": id, "file_path": './images/'+cam,
                   "width": int(camera_intrinsic[0][2] * 2), "height": int(camera_intrinsic[1][2] * 2),
                   "transform_matrix":global_to_ego,
                   "fy": camera_intrinsic[1][1], "fx": camera_intrinsic[0][0]}
    json_list.append(camera_data)

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
    image_points[:, :2] /= image_points[:, [2]]  # x=x/z y=y/z z=z 1

    # 过滤z<=0的点，因为它在图像平面的后面。形不成投影的
    # z > 0
    # for x, y in image_points[image_points[:, 2] > 0, :2].astype(int):
    #     cv2.circle(image, (x, y), 3, (255, 0, 0), -1, 16)

    cv2.imwrite(f"{cam}.jpg", image)

json_string = json.dumps(json_list)
with open("json_list_exam.json", "w") as json_file:
    json_file.write(json_string)

print("JSON 文件已生成！")

print('done')