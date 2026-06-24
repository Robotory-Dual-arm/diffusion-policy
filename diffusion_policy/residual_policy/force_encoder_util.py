from omegaconf import OmegaConf

from diffusion_policy.model.force.force_encoder import CausalConvForceEncoder, GRUForceEncoder


def get_wrench_keys(shape_meta):
    return [
        key for key, attr in shape_meta["obs"].items()
        if attr.get("type", "low_dim") == "wrench"
    ]


def make_force_encoder(shape_meta, force_encoder_cfg):
    if force_encoder_cfg is None:
        return None, 0
    cfg = OmegaConf.to_container(force_encoder_cfg, resolve=True) if not isinstance(force_encoder_cfg, dict) else force_encoder_cfg
    input_dim = sum(shape_meta["obs"][key]["shape"][0] for key in get_wrench_keys(shape_meta))
    feature_dim = int(cfg["feature_dim"])
    model_name = cfg["model_name"]
    if model_name == "causalconv":
        return CausalConvForceEncoder(input_dim=input_dim, feature_dim=feature_dim), feature_dim
    if model_name == "gru":
        return GRUForceEncoder(input_dim=input_dim, feature_dim=feature_dim), feature_dim
    raise ValueError(f"Unsupported force encoder: {model_name}")
