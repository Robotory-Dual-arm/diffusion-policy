"""Small, checkpoint-friendly affine normalizers for residual RL.

The repository's BC normalizers use ``normalized = value * scale + offset``.
These classes keep exactly that convention while allowing the new base-action
condition (pose9 + hand7) to have its own statistics.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn


def _tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach().clone().flatten()


class _AffineField(nn.Module):
    def __init__(self, scale: Any, offset: Any):
        super().__init__()
        scale_tensor = _tensor(scale)
        offset_tensor = _tensor(offset)
        if scale_tensor.shape != offset_tensor.shape:
            raise ValueError(
                f"scale and offset must have the same shape, got "
                f"{tuple(scale_tensor.shape)} and {tuple(offset_tensor.shape)}"
            )
        if torch.any(scale_tensor == 0):
            raise ValueError("normalizer scale must be non-zero")
        self.register_buffer("scale", scale_tensor)
        self.register_buffer("offset", offset_tensor)

    def _reshape(self, parameter: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if parameter.numel() == 1:
            return parameter
        event_size = 1
        matched_dims = 0
        for size in reversed(value.shape[1:]):
            event_size *= int(size)
            matched_dims += 1
            if event_size == parameter.numel():
                return parameter.reshape((1,) * (value.ndim - matched_dims) + tuple(value.shape[-matched_dims:]))
            if event_size > parameter.numel():
                break
        if value.shape[-1] == parameter.numel():
            return parameter.reshape((1,) * (value.ndim - 1) + (parameter.numel(),))
        raise ValueError(
            f"Cannot broadcast {parameter.numel()} normalizer values over tensor "
            f"shape {tuple(value.shape)}"
        )

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        scale = self._reshape(self.scale, value)
        offset = self._reshape(self.offset, value)
        return value * scale + offset

    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        scale = self._reshape(self.scale, value)
        offset = self._reshape(self.offset, value)
        return (value - offset) / scale


class StructuredAffineNormalizer(nn.Module):
    """Per-field affine transforms with identity fallback.

    Common field names are the observation keys plus ``base_action`` and
    ``residual``. Missing fields intentionally pass through unchanged, which
    makes it possible to import only the statistics available in an older BC
    checkpoint.
    """

    def __init__(self, fields: Optional[Mapping[str, Mapping[str, Any]]] = None):
        super().__init__()
        modules = {}
        for key, params in (fields or {}).items():
            if "." in key:
                raise ValueError(f"Normalizer field names cannot contain '.': {key!r}")
            modules[key] = _AffineField(params["scale"], params["offset"])
        self.fields = nn.ModuleDict(modules)

    def has_field(self, key: str) -> bool:
        return key in self.fields

    def normalize_field(self, key: str, value: torch.Tensor) -> torch.Tensor:
        field = self.fields[key] if key in self.fields else None
        return value if field is None else field.normalize(value)

    def unnormalize_field(self, key: str, value: torch.Tensor) -> torch.Tensor:
        field = self.fields[key] if key in self.fields else None
        return value if field is None else field.unnormalize(value)

    def normalize_observation(
        self,
        observation: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: self.normalize_field(key, value)
            for key, value in observation.items()
        }

    @classmethod
    def from_linear_normalizer(
        cls,
        normalizer: nn.Module,
        field_map: Mapping[str, str],
    ) -> "StructuredAffineNormalizer":
        """Copy selected fields from the repository ``LinearNormalizer``.

        ``field_map`` maps the new name to the source name, for example
        ``{"residual": "action"}``.
        """
        params_dict = getattr(normalizer, "params_dict", None)
        if params_dict is None:
            raise TypeError("normalizer must expose params_dict")
        fields = {}
        for destination, source in field_map.items():
            if source not in params_dict:
                raise KeyError(f"Missing normalizer field {source!r}")
            params = params_dict[source]
            fields[destination] = {
                "scale": params["scale"].detach().cpu(),
                "offset": params["offset"].detach().cpu(),
            }
        return cls(fields)

    @classmethod
    def from_mean_std(
        cls,
        statistics: Mapping[str, Mapping[str, Any]],
        eps: float = 1e-6,
    ) -> "StructuredAffineNormalizer":
        fields = {}
        for key, stats in statistics.items():
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            safe_std = np.maximum(std, float(eps))
            fields[key] = {
                "scale": 1.0 / safe_std,
                "offset": -mean / safe_std,
            }
        return cls(fields)
