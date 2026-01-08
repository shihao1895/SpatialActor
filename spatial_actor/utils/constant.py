# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
IMAGE_SIZE = 128
VOXEL_SIZES = [100]  # 100x100x100 voxels
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}
CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
EPISODE_FOLDER = "episode%d"
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"  # the pkl file that contains language goals for each demonstration
DEMO_AUGMENTATION_EVERY_N = 10  # sample n-th frame in demo
ROTATION_RESOLUTION = 5  # degree increments per axis
NOISE_CFG = {
    'none': {'ratio': 0.0, 'std': 0.0},
    'light': {'ratio': 0.2, 'std': 0.05},
    'mid': {'ratio': 0.5, 'std': 0.1},
    'heavy': {'ratio': 0.8, 'std': 0.1},
}
