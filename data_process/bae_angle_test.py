import numpy as np
from scipy.spatial.transform import Rotation as R
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

def rot6d_to_euler(rx6d, degrees=True):
    """
    6D 회전 표현을 XYZ 오일러 각으로 변환

    Parameters:
        rx6d (array-like): shape (6,), rotation matrix의 앞 두 열을 flatten한 6D 벡터
        degrees (bool): True면 도 단위, False면 rad 단위 출력

    Returns:
        (rx, ry, rz): 오일러 각
    """
    rx6d = np.asarray(rx6d).reshape(6)
    r1 = rx6d[:3]
    r2 = rx6d[3:]
    print(np.linalg.norm(r1), np.linalg.norm(r2))
    # 두 벡터 정규직교화 (Gram-Schmidt)
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - np.dot(r1, r2) * r1
    print(r2)
    r2 = r2 / np.linalg.norm(r2)
    print(r2)
    r3 = np.cross(r1, r2)
    
    # 회전 행렬 구성
    rot_mat = np.stack([r1, r2, r3], axis=1)
    print(rot_mat)
    # scipy Rotation 객체로 변환 후 오일러 각 반환
    r = R.from_matrix(rot_mat)
    return r.as_euler('xyz', degrees=degrees)

# 예시
rx6d = [-2.42166683e-01,
   8.73556077e-01, -4.23396587e-01,  1.35202734e-02, -4.30794060e-01,
  -9.03296113e-01]  # 단위 행렬의 앞 두 열
rx, ry, rz = rot6d_to_euler(rx6d)
print(rx, ry, rz)


tf = RotationTransformer(from_rep='rotation_6d', to_rep='euler_angles', to_convention='XYZ')
a = tf.forward(np.array(rx6d))
print(a*180/np.pi)