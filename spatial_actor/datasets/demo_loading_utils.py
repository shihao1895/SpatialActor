import os
import pickle
import logging
from typing import List
import numpy as np
from PIL import Image

from rlbench.demo import Demo
from rlbench.backend.observation import Observation
from rlbench.backend.utils import image_to_float_array
from rlbench import CameraConfig, ObservationConfig
from pyrep.const import RenderMode
from pyrep.objects import VisionSensor

# constants
EPISODE_FOLDER = 'episode%d'
CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_FORMAT = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'
DEPTH_SCALE = 2 ** 24 - 1
REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces',
               'task_low_dim_state', 'misc']


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo,
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum


def extract_obs(obs: Observation,
                cameras,
                t: int = 0,
                prev_action=None,
                channels_last: bool = False,
                episode_length: int = 10):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = np.array([
        obs.gripper_open,
        *obs.gripper_joint_positions])
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}

    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items() if type(v) == np.ndarray or type(v) == list}
    else:
        # add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
        obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
        obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

    # add timestep to low_dim_state
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['low_dim_state'] = np.concatenate(
        [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict


def create_obs_config(camera_names: List[str],
                       camera_resolution: List[int],
                       method_name: str):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


def get_stored_demo(data_path, index):
    episode_path = os.path.join(data_path, EPISODE_FOLDER % index)

    # low dim pickle file
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
        obs = pickle.load(f)

    # variation number
    with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
        obs.variation_number = pickle.load(f)

    num_steps = len(obs)
    for i in range(num_steps):
        obs[i].front_rgb = np.array(
            Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].left_shoulder_rgb = np.array(
            Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].right_shoulder_rgb = np.array(
            Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].wrist_rgb = np.array(
            Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))

        obs[i].front_depth = image_to_float_array(
            Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)),
            DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_FRONT)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_FRONT)]
        obs[i].front_depth = near + obs[i].front_depth * (far - near)

        obs[i].left_shoulder_depth = image_to_float_array(
            Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_LS)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_LS)]
        obs[i].left_shoulder_depth = near + obs[i].left_shoulder_depth * (far - near)

        obs[i].right_shoulder_depth = image_to_float_array(
            Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_RS)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_RS)]
        obs[i].right_shoulder_depth = near + obs[i].right_shoulder_depth * (far - near)

        obs[i].wrist_depth = image_to_float_array(
            Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)),
            DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_WRIST)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_WRIST)]
        obs[i].wrist_depth = near + obs[i].wrist_depth * (far - near)

        obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].front_depth,
                                                                                        obs[i].misc[
                                                                                            'front_camera_extrinsics'],
                                                                                        obs[i].misc[
                                                                                            'front_camera_intrinsics'])
        obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs[i].left_shoulder_depth,
            obs[i].misc['left_shoulder_camera_extrinsics'],
            obs[i].misc['left_shoulder_camera_intrinsics'])
        obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
            obs[i].right_shoulder_depth,
            obs[i].misc['right_shoulder_camera_extrinsics'],
            obs[i].misc['right_shoulder_camera_intrinsics'])
        obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].wrist_depth,
                                                                                        obs[i].misc[
                                                                                            'wrist_camera_extrinsics'],
                                                                                        obs[i].misc[
                                                                                            'wrist_camera_intrinsics'])

    return obs
