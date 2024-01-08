from nuscenes.nuscenes import NuScenes

# 数据集根目录的路径
# data_root = '/path/to/nuscenes/dataset/'
data_root = r'CSstudy/gaussian-splatting/data/nuscenes'

# 创建NuScenes对象
nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

# 查看数据集中的场景数量
num_scenes = len(nusc.scene)
print(f"Number of scenes: {num_scenes}")

# 获取第一个场景的信息
scene = nusc.scene[0]
scene_token = scene['token']
scene_name = scene['name']
print(f"First scene: {scene_name} (token: {scene_token})")

# 获取第一个场景中的样本数量
sample_tokens = scene['nbr_samples']
print(f"Number of samples in the first scene: {sample_tokens}")

# 获取第一个样本的数据
sample = nusc.get('sample', sample_tokens[0])
sample_token = sample['token']
print(f"First sample token in the first scene: {sample_token}")

# 获取第一个样本的激光雷达数据
lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
print(f"Lidar data token: {lidar_data['token']}")