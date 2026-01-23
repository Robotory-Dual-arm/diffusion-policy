from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer_rel import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
    concatenate_normalizer
)
from diffusion_policy.common.pose_repr_util import compute_relative_pose
register_codecs()


class BimanualRobomimicReplayDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            pose_repr: dict={}, #차이
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer 
        # ===============================================차이============================
        self.pose_repr = pose_repr 
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'abs') 
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'abs') 

        # Rotation transformers for bimanual setup
        self.rot_quat2mat = RotationTransformer(
            from_rep='quaternion', to_rep='matrix')
        self.rot_aa2mat = RotationTransformer(
            from_rep='axis_angle', to_rep='matrix')
        self.rot_6d2mat = RotationTransformer(
            from_rep='rotation_6d', to_rep='matrix')
        
        # Setup rotation transformers for each key 
        self.rot_mat2target = dict()
        for key, attr in obs_shape_meta.items():
            if 'rotation_rep' in attr:
                self.rot_mat2target[key] = RotationTransformer(
                    from_rep='matrix', to_rep=attr['rotation_rep'])
        
        self.rot_mat2target['action'] = RotationTransformer(
            from_rep='matrix',
            to_rep=shape_meta['action']['rotation_rep'])
        # ===============================================차이============================

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            # if not self.temporally_independent_normalization:
            data_cache[key] = data_cache[key].reshape(B*T, D)

        # action - 18 dimensions for bimanual (9 per arm: 3 pos + 6 rot, no gripper)
        # Use relative data from data_cache instead of raw replay_buffer
        action_normalizers = list()
        # 2 arm action
        for arm_idx in range(2):  # 2 arms  
            start_idx = arm_idx * 9
            # Position (3 dims) - use range normalizer for relative data
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., start_idx:start_idx+3])
                )
            )
            # Rotation (6 dims)
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., start_idx+3:start_idx+9])
                )
            )
        
        # hand action
        action_normalizers.append(
            get_range_normalizer_from_stat(
                array_to_stats(data_cache['action'][..., 18:])
            )
        )
        
        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            # 두 팔간의 상대 위치 정보는 제외
            # if 'wrt' in key:
                # continue
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pose' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat') or 'quat' in key:
                # quaternion/rotation data
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif 'wrench' in key or 'force' in key or 'torque' in key:   # 힘, 토크
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                print("UNKNOWN KEY in get_normalizer", key)
                # Default to identity for unknown keys
                this_normalizer = get_identity_normalizer_from_stat(stat)

            normalizer[key] = this_normalizer

        print("action_normalizers scale:", normalizer['action'].params_dict['scale'].data)
        print("action_normalizers offset:", normalizer['action'].params_dict['offset'].data)
        print("--------------------------------")
        print("obs_normalizers scale:", normalizer['robot_pose_R'].params_dict['scale'].data)
        print("obs_normalizers offset:", normalizer['robot_pose_R'].params_dict['offset'].data)
        print("obs_normalizers scale:", normalizer['robot_pose_L'].params_dict['scale'].data)
        print("obs_normalizers offset:", normalizer['robot_pose_L'].params_dict['offset'].data)
        print("obs_normalizers scale:", normalizer['hand_pose_R'].params_dict['scale'].data)
        print("obs_normalizers offset:", normalizer['hand_pose_R'].params_dict['offset'].data)
        print("obs_normalizers scale:", normalizer['hand_pose_L'].params_dict['scale'].data)
        print("obs_normalizers offset:", normalizer['hand_pose_L'].params_dict['offset'].data)
        print("--------------------------------")
        print("obs_normalizers scale:", normalizer['robot_quat_R'].params_dict['scale'].data)
        print("obs_normalizers offset:", normalizer['robot_quat_R'].params_dict['offset'].data)
        print("obs_normalizers scale:", normalizer['robot_quat_L'].params_dict['scale'].data)
        print("obs_normalizers offset:", normalizer['robot_quat_L'].params_dict['offset'].data)
        print("--------------------------------")
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # Handle bimanual relative pose computation if needed
        if self.obs_pose_repr == 'relative' or self.action_pose_repr == 'relative':
            # For bimanual setup
            # Process left arm relative to right arm
            if 'robot_pose_L' in obs_dict and 'robot_quat_L' in obs_dict:
                current_pos_L = copy.copy(obs_dict['robot_pose_L'][-1])
                current_rot_mat_L = copy.copy(self.rot_quat2mat.forward(obs_dict['robot_quat_L'][-1]))
                
            if 'robot_pose_R' in obs_dict and 'robot_quat_R' in obs_dict:
                current_pos_R = copy.copy(obs_dict['robot_pose_R'][-1])
                current_rot_mat_R = copy.copy(self.rot_quat2mat.forward(obs_dict['robot_quat_R'][-1]))

            if 'hand_pose_L' in obs_dict:
                current_hand_pos_L = copy.copy(obs_dict['hand_pose_L'][-1])
            
            if 'hand_pose_R' in obs_dict:
                current_hand_pos_R = copy.copy(obs_dict['hand_pose_R'][-1])
                
            if self.obs_pose_repr == 'relative':
                # Process left arm
                obs_dict['robot_pose_L'], obs_dict['robot_quat_L'] = compute_relative_pose(
                    pos=obs_dict['robot_pose_L'],
                    rot=obs_dict['robot_quat_L'],
                    base_pos=current_pos_L,
                    base_rot_mat=current_rot_mat_L,
                    rot_transformer_to_mat=self.rot_quat2mat,
                    rot_transformer_to_target=self.rot_mat2target.get('robot_quat_L', self.rot_quat2mat)
                )
                obs_dict['robot_pose_L'] = obs_dict['robot_pose_L'].astype(np.float32)
                obs_dict['robot_quat_L'] = obs_dict['robot_quat_L'].astype(np.float32)
                
                # Process right arm
                obs_dict['robot_pose_R'], obs_dict['robot_quat_R'] = compute_relative_pose(
                    pos=obs_dict['robot_pose_R'],
                    rot=obs_dict['robot_quat_R'],
                    base_pos=current_pos_R,
                    base_rot_mat=current_rot_mat_R,
                    rot_transformer_to_mat=self.rot_quat2mat,
                    rot_transformer_to_target=self.rot_mat2target.get('robot_quat_R', self.rot_quat2mat)
                )
                obs_dict['robot_pose_R'] = obs_dict['robot_pose_R'].astype(np.float32)
                obs_dict['robot_quat_R'] = obs_dict['robot_quat_R'].astype(np.float32)

            
            # Process bimanual action (18 dims: left arm 9 + right arm 9) + hand
            if self.action_pose_repr == 'relative':
                # Left arm action (first 9 dims: 3 pos + 6 rot)
                left_action_pos, left_action_rot = compute_relative_pose(
                    pos=data['action'][..., :3],
                    rot=data['action'][..., 3:9],
                    base_pos=current_pos_L,
                    base_rot_mat=current_rot_mat_L,
                    rot_transformer_to_mat=self.rot_6d2mat,
                    rot_transformer_to_target=self.rot_mat2target['action']
                )
                left_action_pos = left_action_pos.astype(np.float32)
                left_action_rot = left_action_rot.astype(np.float32)
                
                # Right arm action (last 9 dims: 3 pos + 6 rot)
                right_action_pos, right_action_rot = compute_relative_pose(
                    pos=data['action'][..., 9:12],
                    rot=data['action'][..., 12:18],
                    base_pos=current_pos_R,
                    base_rot_mat=current_rot_mat_R,
                    rot_transformer_to_mat=self.rot_6d2mat,
                    rot_transformer_to_target=self.rot_mat2target['action']
                )
                right_action_pos = right_action_pos.astype(np.float32)
                right_action_rot = right_action_rot.astype(np.float32)


                hand_pos, hand_rot = compute_relative_pose(
                    pos=data['action'][..., 18:],
                    rot=[1,0,0,0,1,0],
                    base_pos=np.concatenate([current_hand_pos_L, current_hand_pos_R], axis=-1),
                    base_rot_mat=np.eye(3),
                    rot_transformer_to_mat=self.rot_6d2mat,
                    rot_transformer_to_target=self.rot_mat2target['action']
                )
                hand_pos = hand_pos.astype(np.float32)
                
                # Reconstruct action without gripper
                data['action'] = np.concatenate([
                    left_action_pos, left_action_rot,
                    right_action_pos, right_action_rot,
                    hand_pos
                ], axis=-1)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True
        elif raw_actions.shape[-1] == 18:
            # dual arm with 6D rotation, no gripper - already in correct format
            # No transformation needed since data is already in 6D rotation format
            actions = raw_actions.astype(np.float32)
            return actions

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
