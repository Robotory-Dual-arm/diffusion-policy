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
import scipy.spatial.transform as st

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
from diffusion_policy.common.pose_repr_util import compute_hand_relative_pose, convert_pose_mat_rep
register_codecs()
from diffusion_policy.model.common.pose_util import pose_to_mat, mat_to_pose10d, pose10d_to_mat


class BaeRobomimicReplayDataset(BaseImageDataset):
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
                        replay_buffer = _convert_robomimic_to_replay(   # hdf5 -> zarr
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
                        # 캐시 파일이 실제로 생성되었을 경우에만 삭제
                        if os.path.exists(cache_zarr_path):
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
                key_first_k[key] = n_obs_steps   # key_first_k[image0]=2, k[image1]=2, k[low_dim0]=2 ...

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
        
        # relative ===============================================================
        self.pose_repr = pose_repr 
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'abs') 
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'abs') 
        

        # Rotation transformers for bimanual setup
        self.quat2mat = RotationTransformer(from_rep='quaternion', to_rep='matrix')
        self.rot6d2mat = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')
        
        # Setup rotation transformers for each key 
        self.rot_mat2target = dict() 
        for key, attr in obs_shape_meta.items():
            if 'rotation_rep' in attr:
                self.rot_mat2target[key] = RotationTransformer(
                    from_rep='matrix', to_rep=attr['rotation_rep'])
        
        self.rot_mat2target['action'] = RotationTransformer(
            from_rep='matrix',
            to_rep=shape_meta['action']['rotation_rep'])
        

        self.use_left_arm = 'robot_pose_L' in self.lowdim_keys
        self.use_right_arm = 'robot_pose_R' in self.lowdim_keys
        self.use_left_hand = 'hand_pose_L' in self.lowdim_keys
        self.use_right_hand = 'hand_pose_R' in self.lowdim_keys
        


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
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'): # 여기서 __getitem__ 호출 -> sampler.sequence 뭐시기 실행
            # for key in self.lowdim_keys:
            #     print("[DEBUG] batch['obs'][key].shape:", batch['obs'][key].shape)
            # breakpoint()
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


        # Action에 normalizer
        # (왼팔), (오른팔), (양팔), (왼팔, 왼핸드), (오른팔, 오른핸드), (양팔, 양핸드) 구분

        # (왼팔) : robot_pose_L(3), robot_rot6d_L(6)
        # (오른팔) : robot_pose_R(3), robot_rot6d_R(6)

        # (양팔) : robot_pose_L(3), robot_rot6d_L(6), robot_pose_R(3), robot_rot6d_R(6)

        # (왼팔, 왼핸드) : robot_pose_L(3), robot_rot6d_L(6), hand_pose_L(?)
        # (오른팔, 오른핸드) : robot_pose_R(3), robot_rot6d_R(6), hand_pose_R(?)

        # (양팔, 양핸드) : robot_pose_L(3), robot_rot6d_L(6), robot_pose_R(3), 
        #                   robot_rot6d_R(6), hand_pose_L(?), hand_pose_R(?)
        
        action_normalizers = list()
        action_index = 0 

        if self.use_left_arm:
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., action_index:action_index+3])))
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., action_index+3:action_index+9])))
            action_index += 9

        if self.use_right_arm:
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., action_index:action_index+3])))
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., action_index+3:action_index+9])))
            action_index += 9

        if self.use_left_hand or self.use_right_hand:
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache['action'][..., action_index:])))
        
        normalizer['action'] = concatenate_normalizer(action_normalizers)


        # obs
        for key in self.lowdim_keys:
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

            print(f"obs normalize scale for {key}:", normalizer[key].params_dict['scale'].data)
            print(f"obs normalize offset for {key}:", normalizer[key].params_dict['offset'].data)
        
        print("action_normalizers scale:", normalizer['action'].params_dict['scale'].data)
        print("action_normalizers offset:", normalizer['action'].params_dict['offset'].data)
        
        
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


        # Obs & Action --> 'relative' or 'abs' 

        # Obs -> 'relative' or 'abs'
        # if self.use_left_arm:
        #     current_pose_L = copy.copy(obs_dict['robot_pose_L'][-1])
        #     current_rot_mat_L = copy.copy(self.quat2mat.forward(obs_dict['robot_quat_L'][-1]))
        #     if self.obs_pose_repr == 'relative':
        #         obs_dict['robot_pose_L'], obs_dict['robot_quat_L'] = compute_relative_pose(
        #             pos=obs_dict['robot_pose_L'],
        #             rot=obs_dict['robot_quat_L'],
        #             base_pos=current_pose_L,
        #             base_rot_mat=current_rot_mat_L,
        #             rot_transformer_to_mat=self.quat2mat,
        #             rot_transformer_to_target=self.rot_mat2target.get('robot_quat_L', self.quat2mat))
        #         obs_dict['robot_pose_L'] = obs_dict['robot_pose_L'].astype(np.float32)
        #         obs_dict['robot_quat_L'] = obs_dict['robot_quat_L'].astype(np.float32)
        # if self.use_right_arm:
        #     current_pose_R = copy.copy(obs_dict['robot_pose_R'][-1])
        #     current_rot_mat_R = copy.copy(self.quat2mat.forward(obs_dict['robot_quat_R'][-1]))
        #     if self.obs_pose_repr == 'relative':
        #         obs_dict['robot_pose_R'], obs_dict['robot_quat_R'] = compute_relative_pose(
        #                 pos=obs_dict['robot_pose_R'],
        #                 rot=obs_dict['robot_quat_R'],
        #                 base_pos=current_pose_R,
        #                 base_rot_mat=current_rot_mat_R,
        #                 rot_transformer_to_mat=self.quat2mat,
        #                 rot_transformer_to_target=self.rot_mat2target.get('robot_quat_R', self.quat2mat))
        #         obs_dict['robot_pose_R'] = obs_dict['robot_pose_R'].astype(np.float32)
        #         obs_dict['robot_quat_R'] = obs_dict['robot_quat_R'].astype(np.float32)


        if self.use_left_arm:
            obs_pose_mat_L = pose_to_mat(np.concatenate([
                obs_dict['robot_pose_L'],
                st.Rotation.from_quat(obs_dict['robot_quat_L']).as_rotvec()
            ], axis=-1)) 

            if self.obs_pose_repr == 'relative':
                obs_relative_pose_mat_L = convert_pose_mat_rep(
                    pose_mat=obs_pose_mat_L,
                    base_pose_mat=obs_pose_mat_L[-1],
                    pose_rep=self.obs_pose_repr,
                    backward=False)
                
                obs_relative_pose_L = mat_to_pose10d(obs_relative_pose_mat_L) # mat -> pos + rot6d (9)
                obs_dict['robot_pose_L'] = obs_relative_pose_L[..., :3].astype(np.float32)
                obs_dict['robot_quat_L'] = obs_relative_pose_L[..., 3:].astype(np.float32)

        if self.use_right_arm:
            obs_pose_mat_R = pose_to_mat(np.concatenate([
                obs_dict['robot_pose_R'],
                st.Rotation.from_quat(obs_dict['robot_quat_R']).as_rotvec()
            ], axis=-1))

            if self.obs_pose_repr == 'relative':
                obs_relative_pose_mat_R = convert_pose_mat_rep(
                    pose_mat=obs_pose_mat_R,
                    base_pose_mat=obs_pose_mat_R[-1],
                    pose_rep=self.obs_pose_repr,
                    backward=False)
                
                obs_relative_pose_R = mat_to_pose10d(obs_relative_pose_mat_R) # mat -> pos + rot6d (9)
                obs_dict['robot_pose_R'] = obs_relative_pose_R[..., :3].astype(np.float32)
                obs_dict['robot_quat_R'] = obs_relative_pose_R[..., 3:].astype(np.float32)

        if self.use_left_hand:
            if self.obs_pose_repr == 'relative':
                obs_dict['hand_pose_L'] = compute_hand_relative_pose(
                    pos=obs_dict['hand_pose_L'],
                    base_pos=obs_dict['hand_pose_L'][-1]).astype(np.float32)
            
        if self.use_right_hand:
            if self.obs_pose_repr == 'relative':
                obs_dict['hand_pose_R'] = compute_hand_relative_pose(
                    pos=obs_dict['hand_pose_R'],
                    base_pos=obs_dict['hand_pose_R'][-1]).astype(np.float32)


        # Action -> 'relative' or 'abs'
        if self.action_pose_repr == 'relative':
            relative_action_list = list()
            action_index = 0 

            # if self.use_left_arm:
            #     left_action_pos, left_action_rot = compute_relative_pose(
            #         pos=data['action'][..., action_index:action_index+3],
            #         rot=data['action'][..., action_index+3:action_index+9],
            #         base_pos=current_pose_L,
            #         base_rot_mat=current_rot_mat_L,
            #         rot_transformer_to_mat=self.rot6d2mat,
            #         rot_transformer_to_target=self.rot_mat2target['action'])
            #     action_index += 9
            #     relative_action_list.append(left_action_pos.astype(np.float32))
            #     relative_action_list.append(left_action_rot.astype(np.float32)) 
            # if self.use_right_arm:
            #     right_action_pos, right_action_rot = compute_relative_pose(
            #         pos=data['action'][..., action_index:action_index+3],
            #         rot=data['action'][..., action_index+3:action_index+9],
            #         base_pos=current_pose_R,
            #         base_rot_mat=current_rot_mat_R,
            #         rot_transformer_to_mat=self.rot6d2mat,
            #         rot_transformer_to_target=self.rot_mat2target['action'])
            #     action_index += 9
            #     relative_action_list.append(right_action_pos.astype(np.float32))
            #     relative_action_list.append(right_action_rot.astype(np.float32))

            if self.use_left_arm:
                action_relative_pose_mat_L = convert_pose_mat_rep(
                    pose_mat=pose10d_to_mat(data['action'][..., action_index:action_index+9]), # pos + rot6d -> mat
                    base_pose_mat=obs_pose_mat_L[-1],
                    pose_rep=self.action_pose_repr,
                    backward=False)
                action_relative_pose_L = mat_to_pose10d(action_relative_pose_mat_L) # mat -> pos + rot6d (9)
                relative_action_list.append(action_relative_pose_L.astype(np.float32))
                action_index += 9

            if self.use_right_arm:
                action_relative_pose_mat_R = convert_pose_mat_rep(
                    pose_mat=pose10d_to_mat(data['action'][..., action_index:action_index+9]), # pos + rot6d -> mat
                    base_pose_mat=obs_pose_mat_R[-1],
                    pose_rep=self.action_pose_repr,
                    backward=False)
                action_relative_pose_R = mat_to_pose10d(action_relative_pose_mat_R) # mat -> pos + rot6d (9)
                relative_action_list.append(action_relative_pose_R.astype(np.float32))
                action_index += 9

            hand_length = data['action'].shape[-1] - action_index
            if hand_length > 0:
                if self.use_left_hand:
                    if self.use_right_hand:
                        # both hand
                        action_relative_hand_L = compute_hand_relative_pose(
                            pos=data['action'][..., action_index:action_index + hand_length//2],
                            base_pos=obs_dict['hand_pose_L'][-1])
                        action_relative_hand_R = compute_hand_relative_pose(
                            pos=data['action'][..., action_index + hand_length//2:],
                            base_pos=obs_dict['hand_pose_R'][-1])
                        relative_action_list.append(action_relative_hand_L.astype(np.float32))
                        relative_action_list.append(action_relative_hand_R.astype(np.float32))
                    else:
                        # left hand
                        action_relative_hand_L = compute_hand_relative_pose(
                            pos=data['action'][..., action_index:],
                            base_pos=obs_dict['hand_pose_L'][-1])
                        relative_action_list.append(action_relative_hand_L.astype(np.float32))
                else:
                    # right hand
                    action_relative_hand_R = compute_hand_relative_pose(
                        pos=data['action'][..., action_index:],
                        base_pos=obs_dict['hand_pose_R'][-1])
                    relative_action_list.append(action_relative_hand_R.astype(np.float32))

            data['action'] = np.concatenate(relative_action_list, axis=-1)


        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy), # abs: quat(4) / relative: rot6d(6)
            'action': torch.from_numpy(data['action'].astype(np.float32)) # pos(3), rot6d(6), hand(?)
        }
        return torch_data

# Action이 axis_angle 일때 rotation_6d로 변환
def _convert_actions(raw_actions, abs_action, rotation_transformer):
    # actions = raw_actions
    # if abs_action:
        # is_dual_arm = False
        # if raw_actions.shape[-1] == 14:
        #     # dual arm
        #     raw_actions = raw_actions.reshape(-1,2,7)
        #     is_dual_arm = True
        # elif raw_actions.shape[-1] == 18:
        #     # dual arm with 6D rotation, no gripper - already in correct format
        #     # No transformation needed since data is already in 6D rotation format
        #     actions = raw_actions.astype(np.float32)
        #     return actions
        # pos = raw_actions[...,:3]
        # rot = raw_actions[...,3:6]
        # gripper = raw_actions[...,6:]
        # rot = rotation_transformer.forward(rot)
        # raw_actions = np.concatenate([
        #     pos, rot, gripper
        # ], axis=-1).astype(np.float32)
        # if is_dual_arm:
        #     raw_actions = raw_actions.reshape(-1,20)
        # actions = raw_actions
    actions = raw_actions.astype(np.float32)
    return actions

# hdf5 -> zarr
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
        # count total steps; 전체 스텝 일렬로 나열
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]   # 데이터 총 길이
        episode_starts = [0] + episode_ends[:-1]   # 각 에피소드별 시작점
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data; low_dim 별로 일렬로 나열하기
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
                # if 'quat' in key:
                #     assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['raw_shape'])
                # else:
                #     assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
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
