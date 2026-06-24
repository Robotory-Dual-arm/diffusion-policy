#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


GROUPS = ("nowrench", "wrench")
FORCE_LABELS = ("x", "y", "z")
TORQUE_LABELS = ("rx", "ry", "rz")
COMPONENT_COLORS = ("red", "blue", "green")


def load_trial(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        time_s = np.asarray(f["wrist_ft/elapsed_s"])
        wrench = np.asarray(f["wrist_ft/wrench_wrist_R"])

    time_s = time_s - time_s[0]
    wrench = wrench - wrench[0]
    return time_s, wrench


def plot_group(files: list[Path], group_name: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    force_axis, torque_axis = axes

    used_labels = set()
    for trial_index, path in enumerate(files, start=1):
        time_s, wrench = load_trial(path)

        for component_index, (label, color) in enumerate(zip(FORCE_LABELS, COMPONENT_COLORS)):
            legend_label = label if label not in used_labels else None
            force_axis.plot(
                time_s,
                wrench[:, component_index],
                color=color,
                alpha=0.65,
                linewidth=1.1,
                label=legend_label,
            )
            used_labels.add(label)

        for component_index, (label, color) in enumerate(zip(TORQUE_LABELS, COMPONENT_COLORS), start=3):
            legend_label = label if label not in used_labels else None
            torque_axis.plot(
                time_s,
                wrench[:, component_index],
                color=color,
                alpha=0.65,
                linewidth=1.1,
                label=legend_label,
            )
            used_labels.add(label)

    force_axis.set_title("Force xyz")
    force_axis.set_ylabel("force (N)")
    torque_axis.set_title("Torque rx ry rz")
    torque_axis.set_ylabel("torque (Nm)")
    torque_axis.set_xlabel("time (s)")

    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=9, ncol=3)

    fig.suptitle(f"{group_name}: five trials, zeroed at start")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def files_for_group(input_dir: Path, group_name: str) -> list[Path]:
    files = [input_dir / f"{group_name}{index}.hdf5" for index in range(1, 6)]
    missing = [path for path in files if not path.exists()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing expected files: {missing_names}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/wrench_downloads"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for group_name in GROUPS:
        output_path = args.output_dir / f"{group_name}.png"
        plot_group(
            files_for_group(args.input_dir, group_name),
            group_name,
            output_path,
        )
        output_paths.append(output_path)

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
