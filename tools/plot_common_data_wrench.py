#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DEMOS = ("demo_0", "demo_20", "demo_40", "demo_60", "demo_80")
FORCE_LABELS = ("x", "y", "z")
TORQUE_LABELS = ("rx", "ry", "rz")
COMPONENT_COLORS = ("red", "blue", "green")
WRENCH_CALIB_COUNT = 10
WRENCH_EMA_ALPHA = 0.03


def ema_filter(wrench: np.ndarray, alpha: float) -> np.ndarray:
    filtered = np.empty_like(wrench)
    filtered[0] = wrench[0]
    for index in range(1, len(wrench)):
        filtered[index] = alpha * wrench[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


def apply_controller_wrench_filter(
    time_s: np.ndarray,
    wrench: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offset = np.mean(wrench[:WRENCH_CALIB_COUNT], axis=0)
    corrected = wrench - offset
    corrected = corrected[WRENCH_CALIB_COUNT:]
    time_s = time_s[WRENCH_CALIB_COUNT:]
    filtered = ema_filter(corrected, WRENCH_EMA_ALPHA)
    return time_s, filtered


def load_demo(path: Path, demo: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        obs = f[f"data/{demo}/observations"]
        time_s = np.asarray(obs["timestamp_wrench"], dtype=float)
        wrench = np.asarray(obs["wrench_wrist_R"], dtype=float)

    time_s, wrench = apply_controller_wrench_filter(time_s, wrench)
    time_s = time_s - time_s[0]
    return time_s, wrench


def plot_demos(path: Path, demos: tuple[str, ...], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    force_axis, torque_axis = axes

    used_labels = set()
    for demo in demos:
        time_s, wrench = load_demo(path, demo)

        for component_index, (label, color) in enumerate(zip(FORCE_LABELS, COMPONENT_COLORS)):
            legend_label = label if label not in used_labels else None
            force_axis.plot(
                time_s,
                wrench[:, component_index],
                color=color,
                linewidth=1.0,
                alpha=0.55,
                label=legend_label,
            )
            used_labels.add(label)

        for component_index, (label, color) in enumerate(zip(TORQUE_LABELS, COMPONENT_COLORS), start=3):
            legend_label = label if label not in used_labels else None
            torque_axis.plot(
                time_s,
                wrench[:, component_index],
                color=color,
                linewidth=1.0,
                alpha=0.55,
                label=legend_label,
            )
            used_labels.add(label)

    demo_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.0, alpha=0.55, label=demo)
        for demo in demos
    ]

    force_axis.set_title("Force xyz")
    force_axis.set_ylabel("force (N)")
    torque_axis.set_title("Torque rx ry rz")
    torque_axis.set_ylabel("torque (Nm)")
    torque_axis.set_xlabel("time (s)")

    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        axis.grid(True, alpha=0.3)
        component_legend = axis.legend(loc="upper right", fontsize=9, ncol=3)
        axis.add_artist(component_legend)
        axis.legend(handles=demo_handles, loc="upper left", fontsize=8, ncol=len(demos))

    fig.suptitle(
        f"{path.stem}: wrist wrench, demos {', '.join(demos)}, "
        f"calib {WRENCH_CALIB_COUNT} samples + EMA alpha {WRENCH_EMA_ALPHA}"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "Downloads/common_data_height.hdf5")
    parser.add_argument("--demos", nargs="+", default=list(DEFAULT_DEMOS))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/wrench_downloads"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    demos = tuple(args.demos)
    demo_suffix = "_".join(demo.removeprefix("demo_") for demo in demos)
    output_path = args.output_dir / f"{args.input.stem}_wrench_demos_{demo_suffix}.png"
    plot_demos(args.input, demos, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
