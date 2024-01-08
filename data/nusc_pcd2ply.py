import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
#   nuscenes lidar (x,y,z,intensity,ring index)
# 来自lidar_top的点云数据 软件可以看到是在车前上方的
lidar_file = r"nuscenes-6imgs/example.pcd.bin"
lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5) # (point_num,5)
lidar_points_xyz = lidar_points[:,:3]

# ply格式：
# x：点的x坐标（浮点数# y：点的y坐标（浮点数# z：点的z坐标（浮点数）
# nx：法线向量的x分量（浮点数）# ny：法线向量的y分量（浮点数）# nz：法线向量的z分量（浮点数）
# red：点的红色分量（无符号字符# green：点的绿色分量（无符号字符）# blue：点的蓝色分量（无符号字符


plydata = PlyData.read('nuscenes-6imgs/input_example.ply')
vertices = plydata['vertex']
x = vertices['x']
y = vertices['y']
z = vertices['z']

points = np.column_stack((x, y, z))  # (6381,3)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(lidar_points_xyz)

# 保存点云数据为PLY文件
o3d.io.write_point_cloud('nuscenes-6imgs/output.ply', point_cloud)

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