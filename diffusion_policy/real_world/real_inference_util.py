from typing import Dict, Callable, Tuple
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform

def get_real_obs_dict(env_obs: Dict[str, np.ndarray], shape_meta: dict) -> Dict[str, np.ndarray]:
    import numpy as np
    print("[DEBUG] get_real_obs_dict 시작")
    print("[DEBUG] env_obs keys:", list(env_obs.keys()))
    obs_dict_np = {}
    obs_shape_meta = shape_meta['obs']
    print("[DEBUG] obs_shape_meta keys:", list(obs_shape_meta.keys()))

    for key, attr in obs_shape_meta.items():
        print(f"[DEBUG] key={key}, attr={attr}")
        type   = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        print(f"[DEBUG] type={type}, shape={shape}")

        # ---- 공통: 키 존재/유효성 체크 ----
        if key not in env_obs:
            raise RuntimeError(f"[ERROR] env_obs에 '{key}' 키가 없음. 현재 keys={list(env_obs.keys())}")
        print("[DEBUGDEBUG] val = env_obs[key]")
        val = env_obs[key]
        if val is None:
            raise RuntimeError(f"[ERROR] env_obs['{key}'] 값이 None임(프로듀서가 아직 채우지 않음).")

        # numpy 아니면 변환 (리스트/튜플 등)
        if not isinstance(val, np.ndarray):
            try:
                val = np.asarray(val)
                print(f"[DEBUG] '{key}' np.asarray 변환: shape={val.shape}, dtype={val.dtype}")
            except Exception as e:
                raise TypeError(f"[ERROR] env_obs['{key}'] 타입 {type(env_obs[key])} → np.asarray 실패: {e}")

        print("[DEBUGDBUG] val",  val)
        if type == 'rgb':
            this_imgs_in = val
            if this_imgs_in.ndim != 4:
                raise ValueError(f"[ERROR] '{key}'는 THWC(4D)여야 함, got {this_imgs_in.shape}")

            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape  # (C,H,W)
            print(f"[DEBUG] {key} this_imgs_in.shape={this_imgs_in.shape}, dtype={this_imgs_in.dtype}")
            print(f"[DEBUG] expected channels co={co}, output (Ho,Wo)=({ho},{wo})")

            assert ci == co, f"[ERROR] channel mismatch: ci={ci}, co={co}"

            out_imgs = this_imgs_in
            need_resize = (ho != hi) or (wo != wi)
            if need_resize or (this_imgs_in.dtype == np.uint8):
                print(f"[DEBUG] transform 적용 input_res=({wi},{hi}) → output_res=({wo},{ho})")
                tf = get_image_transform(
                    input_res=(wi, hi),
                    output_res=(wo, ho),
                    bgr_to_rgb=False
                )
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                print(f"[DEBUG] after transform out_imgs.shape={out_imgs.shape}, dtype={out_imgs.dtype}")

                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255.0
                    print("[DEBUG] uint8 → float32/255 변환")

            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)  # THWC -> TCHW
            print(f"[DEBUG] obs_dict_np[{key}].shape = {obs_dict_np[key].shape}")

        elif type == 'low_dim':
            print("[DEBUGDEBUG] val : ", val)
            this_data_in = val
            # 저수준 관측은 1D 또는 2D(TxD)일 수 있음
            print(f"[DEBUG] low_dim '{key}' shape={this_data_in.shape}, dtype={this_data_in.dtype}, ndim={this_data_in.ndim}")

            # 기대 shape 검사 (예: [3])
            if isinstance(shape, (list, tuple)) and len(shape) == 1:
                expected_dim = shape[0]
                if this_data_in.ndim == 1:
                    if this_data_in.shape[0] != expected_dim:
                        raise ValueError(f"[ERROR] '{key}' 길이 {this_data_in.shape[0]} != 기대 {expected_dim}")
                elif this_data_in.ndim == 2:
                    if this_data_in.shape[-1] != expected_dim:
                        raise ValueError(f"[ERROR] '{key}' 마지막 차원 {this_data_in.shape[-1]} != 기대 {expected_dim}")
                # 필요시 dtype 통일
                if this_data_in.dtype != np.float32 and this_data_in.dtype != np.float64:
                    this_data_in = this_data_in.astype(np.float32)
                    print(f"[DEBUG] '{key}' dtype → float32 변환")

            obs_dict_np[key] = this_data_in

        else:
            raise ValueError(f"[ERROR] 알 수 없는 type='{type}' for key '{key}'")

    print("[DEBUG] get_real_obs_dict 종료")
    return obs_dict_np



# obs에서 image의 해상도 출력 (width, height)
def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
