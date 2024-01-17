# 创建评估器
import numpy as np
from occ_metrics import *
evaluator = Metric_mIoU(            num_classes=3,
            use_lidar_mask=False,
            use_image_mask=False)
# 最后一个语义标签认为是未占据free，没计算IoU，计算的IoU就是严格按照定义算的，只是把百分数*100了
# 这么看的话，那才30%的IoU,感觉现在性能也没多准好像

# 假设有一批预测结果和真实标签
semantics_pred = np.array([0, 1, 2, 0, 1, 2])  # 预测的语义标签
semantics_gt = np.array([0, 1, 1, 0, 2, 2])  # 真实的语义标签
mask_lidar = np.array([True, True, False, True, False, True])  # 激光雷达的掩码
mask_camera = np.array([True, True, True, False, False, True])  # 图像的掩码

# 添加批次
evaluator.add_batch(semantics_pred, semantics_gt, mask_lidar, mask_camera)

# 计算并打印mIoU
evaluator.count_miou()

# 创建F-Score评估器
fscore_evaluator = Metric_FScore()

# 添加批次
fscore_evaluator.add_batch(semantics_pred, semantics_gt, mask_lidar, mask_camera)

# 计算并打印F-Score
fscore_evaluator.count_fscore()