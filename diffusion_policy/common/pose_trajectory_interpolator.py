from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):  
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:6])
    end_rot = st.Rotation.from_rotvec(end_pose[3:6])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist

class PoseTrajectoryInterpolator:   
    def __init__(self, times: np.ndarray, poses: np.ndarray, action_type=None):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        self.dualarm_hand7 = False
        self.rightarm_hand6 = False

        if action_type == 'dualarm_hand7': # total dim = 26
            self.dualarm_hand7 = True # dualarm[posL(3), rotL(3), posR(3), rotR(3)] + hand[thumb(3), index(2), middle(2)]*2
        elif action_type == 'rightarm_hand6': # total dim = 12
            self.rightarm_hand6 = True # rightarm[posR(3), rotR(3)] + hand[thumb(3), index(1), middle(1), ring(1)]
        # action_type == None -> total dim = 12 (dualarm without hand)
        #                dualarm[posL(3), rotL(3), posR(3), rotR(3)]

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
            
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            if self.dualarm_hand7:
                assert len(poses[0]) == 26
                pos_L = poses[:,:3]
                rot_L = st.Rotation.from_rotvec(poses[:,3:6])
                pos_R = poses[:,6:9]
                rot_R = st.Rotation.from_rotvec(poses[:,9:12])
                hand_L = poses[:, 12:19]
                hand_R = poses[:, 19:26]

                self.pos_interp_L = si.interp1d(times, pos_L, axis=0, assume_sorted=True)
                self.rot_interp_L = st.Slerp(times, rot_L)
                self.pos_interp_R = si.interp1d(times, pos_R, axis=0, assume_sorted=True)
                self.rot_interp_R = st.Slerp(times, rot_R)
                self.hand_interp_L = si.interp1d(times, hand_L, axis=0, assume_sorted=True)
                self.hand_interp_R = si.interp1d(times, hand_R, axis=0, assume_sorted=True)
            
            elif self.rightarm_hand6:
                assert len(poses[0]) == 12
                pos_R = poses[:,:3]
                rot_R = st.Rotation.from_rotvec(poses[:,3:6])
                hand_R = poses[:,6:12]

                self.pos_interp_R = si.interp1d(times, pos_R, axis=0, assume_sorted=True)
                self.rot_interp_R = st.Slerp(times, rot_R)
                self.hand_interp_R = si.interp1d(times, hand_R, axis=0, assume_sorted=True)

            else:
                assert len(poses[0]) == 12
                pos_L = poses[:,:3]
                rot_L = st.Rotation.from_rotvec(poses[:,3:6])
                pos_R = poses[:,6:9]
                rot_R = st.Rotation.from_rotvec(poses[:,9:12])

                self.pos_interp_L = si.interp1d(times, pos_L, axis=0, assume_sorted=True)
                self.rot_interp_L = st.Slerp(times, rot_L)
                self.pos_interp_R = si.interp1d(times, pos_R, axis=0, assume_sorted=True)
                self.rot_interp_R = st.Slerp(times, rot_R)
            
            
    @property   # self.times
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp_R.x   # times
    
    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            
            if self.dualarm_hand7:
                poses = np.zeros((n, 26))
                poses[:,:3] = self.pos_interp_L.y
                poses[:,3:6] = self.rot_interp_L(self.times).as_rotvec()
                poses[:,6:9] = self.pos_interp_R.y
                poses[:,9:12] = self.rot_interp_R(self.times).as_rotvec()
                poses[:,12:19] = self.hand_interp_L.y
                poses[:,19:26] = self.hand_interp_R.y

            elif self.rightarm_hand6:
                poses = np.zeros((n, 12))
                poses[:,:3] = self.pos_interp_R.y
                poses[:,3:6] = self.rot_interp_R(self.times).as_rotvec()
                poses[:,6:12] = self.hand_interp_R.y

            else:
                poses = np.zeros((n, 12))
                poses[:,:3] = self.pos_interp_L.y
                poses[:,3:6] = self.rot_interp_L(self.times).as_rotvec()
                poses[:,6:9] = self.pos_interp_R.y
                poses[:,9:12] = self.rot_interp_R(self.times).as_rotvec()

            return poses


    def trim(self,   # start_t에서 end_t까지의 구간을 잘라냄
            start_t: float, end_t: float, action_type=None
            ) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses, action_type=action_type)
    
    # def drive_to_waypoint(self,   # 안씀
    #         pose, time, curr_time,
    #         max_pos_speed=np.inf, 
    #         max_rot_speed=np.inf
    #     ) -> "PoseTrajectoryInterpolator":
    #     assert(max_pos_speed > 0)
    #     assert(max_rot_speed > 0)
    #     time = max(time, curr_time)
        
    #     curr_pose = self(curr_time)
    #     pos_dist, rot_dist = pose_distance(curr_pose, pose)
    #     pos_min_duration = pos_dist / max_pos_speed
    #     rot_min_duration = rot_dist / max_rot_speed
    #     duration = time - curr_time
    #     duration = max(duration, max(pos_min_duration, rot_min_duration))
    #     assert duration >= 0
    #     last_waypoint_time = curr_time + duration

    #     # insert new pose
    #     trimmed_interp = self.trim(curr_time, curr_time)
    #     times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
    #     poses = np.append(trimmed_interp.poses, [pose], axis=0)

    #     # create new interpolator
    #     final_interp = PoseTrajectoryInterpolator(times, poses)
    #     return final_interp

    def schedule_waypoint(self,
            pose, time,   # target pose, target time
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None,
            action_type=None
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)
        
        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time, action_type=action_type)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses, action_type=action_type)
        return final_interp


    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        if self.dualarm_hand7:
            pose = np.zeros((len(t), 26))
        elif self.rightarm_hand6:
            pose = np.zeros((len(t), 12))
        else:
            pose = np.zeros((len(t), 12))

        if self.single_step:
            pose[:] = self._poses[0]   
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            if self.dualarm_hand7:
                pose[:,:3] = self.pos_interp_L(t)
                pose[:,3:6] = self.rot_interp_L(t).as_rotvec()
                pose[:,6:9] = self.pos_interp_R(t)
                pose[:,9:12] = self.rot_interp_R(t).as_rotvec()
                pose[:,12:19] = self.hand_interp_L(t)
                pose[:,19:26] = self.hand_interp_R(t)

            elif self.rightarm_hand6:
                pose[:,:3] = self.pos_interp_R(t)
                pose[:,3:6] = self.rot_interp_R(t).as_rotvec()
                pose[:,6:12] = self.hand_interp_R(t)
            
            else:
                pose[:,:3] = self.pos_interp_L(t)
                pose[:,3:6] = self.rot_interp_L(t).as_rotvec()
                pose[:,6:9] = self.pos_interp_R(t)
                pose[:,9:12] = self.rot_interp_R(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose   
