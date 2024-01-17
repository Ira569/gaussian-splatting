import numpy as np
from occ_metrics import *
label = np.load('occ_label.npy')
pred = np.load('occ_pred.npy')

# 1应该是free 0应该是 占用，才好用mIoU类
label[label!=17] = 0
label[label==17] = 1

pred[pred==0]=2
pred[pred==1]=0
pred[pred==2]=1


print('label sum',label.sum())  # 30093
print('pred sum',pred.sum())    # 38822
print('todo ')

def cal_IoU(pred,label):
    ''' IoU = TP/(TP+FP+FN)
        True Positive,False Positive,False Negetive
        标签真预测真         标签假预测真        标签真预测假
        确定了算的没错，
        要先针对哪个类属于正确类，才能计算IoU,对于IoU,则认为被占据了是正确类，也就是0是正确类，1 free是错误类
    '''
    TP = ((label == 0) & (pred == 0)).sum()

    FP = ((label == 1) & (pred == 0)).sum()

    FN = ((label == 0) & (pred == 1)).sum()

    print('TP=', TP)
    print('FP=', FP)
    print('FN=', FN)
    IoU = TP / (TP + FP + FN)
    return IoU*100

IoU = cal_IoU(pred,label)
print('IoU=',IoU)
#IoU参考值


occ_eval_metrics = Metric_mIoU(
            num_classes=2,
            use_lidar_mask=False,
            use_image_mask=False)

occ_eval_metrics.add_batch(pred, label,mask_lidar=None,mask_camera=None)

# 确定了就是百分数
res = occ_eval_metrics.count_miou()
print(res)