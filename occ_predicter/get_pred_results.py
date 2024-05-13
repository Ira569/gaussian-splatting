# 通过点云位置，判断是否为占用。
# -40~40 -40~40 -1~5.4m ego坐标系 （sample的ego坐标系）
# 步骤参考
# 1.load ply文件 from output/6imgs-3000iter-center_filtered
# 2.在ego坐标系下，判断某位置是否有点存在，如果存在，则视为占用，如果不存在则视为空闲
# 3.不考虑语义，计算占用与空闲的预测准确的IoU 并且参考其他论文的IoU指标(参考什么论文还没确定）

# '../output/8a5815dc-c/point_cloud/iteration_2000/point_cloud.ply'
from plyfile import PlyData
# plydata = PlyData.read('../output/6imgs-3000iter-center_filtered/point_cloud/iteration_3000/point_cloud.ply')
plydata = PlyData.read('../output/48ef5f55-6/point_cloud/iteration_2000/point_cloud.ply')
#./output/7783c817-c 没有过滤的点云
# plydata = PlyData.read('../output/48ef5f55-6/point_cloud/iteration_2000/point_cloud.ply')
# 需要初始点云标签的过滤，参考filter_points函数，过滤后IoU能到10.6
# IoU= 10.624343287123343
# 试一下只用初始点云啥结果
plydata = PlyData.read('../data/nusc_demo/pc2ego_output.ply')
# IoU= 8.523155830441997

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

# todo 使用别的方法判断高斯是否占用 (按照体积大小？,)
for p in points:
    cube_x = 2.5*p[0]+100 # int  -40~40 -> 0-199 -40 ->0  40 ->199
    cube_y = 2.5*p[1]+100  # int  -40~40 -> 0-199 80-200
    cube_z = 2.5*p[2]+2.5  # int  -1~5.4 -> 0-15 6.4-16
    cube_x,cube_y,cube_z=int(cube_x),int(cube_y),int(cube_z)
    pred[cube_x,cube_y,cube_z]=1
    # pass

np.save('occ_pred.npy',pred)

from cal_IoU import cal_IoU
label = np.load('occ_label.npy')
# pred = np.load('occ_pred.npy')
# 1应该是free  0应该是 占用，才好用mIoU类

label[label!=17] = 0
label[label==17] = 1

pred[pred==0]=2
pred[pred==1]=0
pred[pred==2]=1
# emm,主要关注 标签应该是占用的，但是预测为未占用的，那就说明是点云重建不充分
# 如果标签应该是未占用，但是预测为占用，那就是点云重建的太多了，在不应该重建的地方重建了
print('label occupied sum',200*200*16-label.sum())  # 30093
print('pred occupied sum',200*200*16-pred.sum())    # 38822

# 调用函数 计算IoU
IoU = cal_IoU(pred,label)
print('IoU=',IoU)

