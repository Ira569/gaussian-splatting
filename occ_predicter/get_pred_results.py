# 通过点云位置，判断是否为占用。
# -40~40 -40~40 -1~5.4m ego坐标系 （sample的ego坐标系）
# 步骤参考
# 1.load ply文件 from output/6imgs-3000iter-center_filtered
# 2.在ego坐标系下，判断某位置是否有点存在，如果存在，则视为占用，如果不存在则视为空闲
# 3.不考虑语义，计算占用与空闲的预测准确的IoU 并且参考其他论文的IoU指标(参考什么论文还没确定）

# import

from plyfile import PlyData
plydata = PlyData.read('../output/6imgs-3000iter-center_filtered/point_cloud/iteration_3000/point_cloud.ply')
vertices = plydata['vertex']
x = vertices['x']
y = vertices['y']
z = vertices['z']
import numpy as np
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

for p in points:
    cube_x = 2.5*p[0]+100 # int  -40~40 -> 0-199 -40 ->0  40 ->199
    cube_y = 2.5*p[1]+100  # int  -40~40 -> 0-199 80-200
    cube_z = 2.5*p[2]+2.5  # int  -1~5.4 -> 0-15 6.4-16
    cube_x,cube_y,cube_z=int(cube_x),int(cube_y),int(cube_z)
    pred[cube_x,cube_y,cube_z]=1
    # pass

np.save('Occpred.npy',pred)
