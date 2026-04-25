#!/usr/bin/env python3
"""Inspect and plot ImplicitRDP pickle episodes.

The flip_v2 pickle files reference classes from
``ImplicitRDP.common.data_models``.  This script can still read them on a
machine that does not have ImplicitRDP installed by replacing those classes
with lightweight stubs during unpickling.
"""

import argparse
import csv
import gc
import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


DEFAULT_FIELDS = [
    "rightRobotTCP",
    "rightRobotTCPVel",
    "rightRobotTCPWrench",
    "rightRobotGripperState",
]
DEFAULT_IMAGE_FIELD = "rightWristCameraRGB"


class PickleStub:
    """Small replacement object for missing ImplicitRDP pydantic models."""

    def __setstate__(self, state):
        if not isinstance(state, dict):
            self.__dict__["_state"] = state
            return

        inner_dict = state.get("__dict__")
        if isinstance(inner_dict, dict):
            self.__dict__.update(inner_dict)
            for key, value in state.items():
                if key != "__dict__":
                    self.__dict__[key] = value
        else:
            self.__dict__.update(state)

    def __repr__(self):
        keys = [key for key in self.__dict__ if not key.startswith("__pydantic")]
        return f"<{self.__class__.__name__} keys={keys[:8]} len={len(keys)}>"


class StubUnpickler(pickle.Unpickler):
    """Unpickler that substitutes unavailable ImplicitRDP classes."""

    _class_cache = {}

    def find_class(self, module, name):
        if module.startswith("ImplicitRDP"):
            key = (module, name)
            if key not in self._class_cache:
                self._class_cache[key] = type(
                    name,
                    (PickleStub,),
                    {"__module__": module},
                )
            return self._class_cache[key]
        return super().find_class(module, name)


def natural_pkl_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def load_pickle(path):
    with path.open("rb") as file:
        return StubUnpickler(file).load()


def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def get_messages(data):
    """Return the sensorMessages list from common observed pickle layouts."""
    if isinstance(data, dict) and "sensor_message_list" in data:
        data = data["sensor_message_list"]

    messages = get_field(data, "sensorMessages")
    if messages is None and isinstance(data, list):
        messages = data

    if messages is None:
        raise KeyError("Could not find sensorMessages in pickle data.")
    return messages


def message_field_names(message):
    if isinstance(message, dict):
        keys = message.keys()
    else:
        keys = message.__dict__.keys()
    return [key for key in keys if not key.startswith("__pydantic")]


def field_value(message, field):
    if isinstance(message, dict):
        return message[field]
    return getattr(message, field)


def timestamp_array(messages):
    timestamps = np.array([float(field_value(msg, "timestamp")) for msg in messages])
    if timestamps.size == 0:
        return timestamps
    return timestamps - timestamps[0]


def collect_field(messages, field):
    values = [field_value(msg, field) for msg in messages]
    first = values[0]

    if isinstance(first, np.ndarray):
        if all(isinstance(value, np.ndarray) and value.shape == first.shape for value in values):
            return np.stack(values, axis=0)
        return np.array(values, dtype=object)

    if np.isscalar(first):
        return np.array(values)

    return np.array(values, dtype=object)


def short_sample(value, limit=6):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        flat = value.reshape(-1)[:limit].tolist()
        return repr(flat)
    return repr(value)


def safe_numeric_stats(array):
    if not isinstance(array, np.ndarray):
        return None
    if array.dtype == object or not np.issubdtype(array.dtype, np.number) or array.size == 0:
        return None
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def field_structure(messages, include_stats=True):
    fields = message_field_names(messages[0])
    rows = []
    for field in fields:
        first = field_value(messages[0], field)
        row = {
            "field": field,
            "type": type(first).__name__,
            "shape": "",
            "dtype": "",
            "sample": short_sample(first),
            "min": "",
            "max": "",
            "mean": "",
        }

        if isinstance(first, np.ndarray):
            row["shape"] = str(tuple(first.shape))
            row["dtype"] = str(first.dtype)
            should_scan_episode = include_stats and first.size <= 64
            stats = safe_numeric_stats(collect_field(messages, field)) if should_scan_episode else safe_numeric_stats(first)
            if stats:
                row["min"] = f"{stats['min']:.6g}"
                row["max"] = f"{stats['max']:.6g}"
                row["mean"] = f"{stats['mean']:.6g}"
        elif np.isscalar(first):
            stats = safe_numeric_stats(collect_field(messages, field)) if include_stats else None
            if stats:
                row["dtype"] = str(np.array(first).dtype)
                row["min"] = f"{stats['min']:.6g}"
                row["max"] = f"{stats['max']:.6g}"
                row["mean"] = f"{stats['mean']:.6g}"

        rows.append(row)
    return rows


def summarize_episode(path, detailed=False):
    data = load_pickle(path)
    messages = get_messages(data)
    if len(messages) == 0:
        raise ValueError(f"{path} has no messages.")

    timestamps_abs = np.array([float(field_value(msg, "timestamp")) for msg in messages])
    duration = float(timestamps_abs[-1] - timestamps_abs[0]) if len(timestamps_abs) > 1 else 0.0
    approx_hz = float((len(messages) - 1) / duration) if duration > 0 else 0.0
    summary = {
        "file": path.name,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "messages": len(messages),
        "start_timestamp": float(timestamps_abs[0]),
        "end_timestamp": float(timestamps_abs[-1]),
        "duration_s": duration,
        "approx_hz": approx_hz,
    }

    structure = field_structure(messages) if detailed else None

    del data
    return summary, structure


def print_structure(path, summary, structure):
    print(f"\nDetailed structure: {path.name}")
    print(
        f"  messages={summary['messages']} "
        f"duration={summary['duration_s']:.3f}s "
        f"approx_hz={summary['approx_hz']:.2f} "
        f"size={summary['size_mb']:.1f}MB"
    )
    print("  field | type | shape | dtype | min | max | sample")
    print("  " + "-" * 104)
    for row in structure:
        sample = row["sample"]
        if len(sample) > 45:
            sample = sample[:42] + "..."
        print(
            "  "
            f"{row['field']} | {row['type']} | {row['shape']} | {row['dtype']} | "
            f"{row['min']} | {row['max']} | {sample}"
        )


def write_summary_csv(summaries, output_dir):
    output_path = output_dir / "summary.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "size_mb",
                "messages",
                "start_timestamp",
                "end_timestamp",
                "duration_s",
                "approx_hz",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    return output_path


def write_structure_txt(path, summary, structure, output_dir):
    output_path = output_dir / f"{path.stem}_structure.txt"
    with output_path.open("w") as file:
        file.write(f"Detailed structure: {path.name}\n")
        file.write(
            f"messages={summary['messages']} "
            f"duration={summary['duration_s']:.6f}s "
            f"approx_hz={summary['approx_hz']:.6f} "
            f"size={summary['size_mb']:.3f}MB\n\n"
        )
        for row in structure:
            file.write(
                f"{row['field']}\n"
                f"  type: {row['type']}\n"
                f"  shape: {row['shape']}\n"
                f"  dtype: {row['dtype']}\n"
                f"  min/max/mean: {row['min']} / {row['max']} / {row['mean']}\n"
                f"  first sample: {row['sample']}\n\n"
            )
    return output_path


def prepare_matplotlib(show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_dataset_overview(summaries, output_dir, show=False):
    plt = prepare_matplotlib(show)

    files = [row["file"] for row in summaries]
    x = np.arange(len(files))
    durations = [row["duration_s"] for row in summaries]
    messages = [row["messages"] for row in summaries]
    hz = [row["approx_hz"] for row in summaries]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].bar(x, durations, color="#4C78A8")
    axes[0].set_ylabel("Duration (s)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, messages, color="#F58518")
    axes[1].set_ylabel("Messages")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, hz, color="#54A24B")
    axes[2].set_ylabel("Approx Hz")
    axes[2].set_xlabel("PKL file")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(files, rotation=90)

    fig.suptitle("flip_v2 dataset overview", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "dataset_overview.png"
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_timeseries(path, fields, output_dir, show=False):
    plt = prepare_matplotlib(show)

    data = load_pickle(path)
    messages = get_messages(data)
    time_s = timestamp_array(messages)
    available = set(message_field_names(messages[0]))
    selected_fields = [field for field in fields if field in available]
    if not selected_fields:
        raise ValueError(f"No requested fields are available in {path.name}: {fields}")

    fig, axes = plt.subplots(
        len(selected_fields),
        1,
        figsize=(14, 3.2 * len(selected_fields)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, field in zip(axes, selected_fields):
        arr = collect_field(messages, field)
        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            ax.text(0.5, 0.5, f"{field}: not numeric", ha="center", va="center")
            ax.set_axis_off()
            continue

        if arr.ndim == 1:
            ax.plot(time_s, arr, label=field)
        else:
            flat = arr.reshape(arr.shape[0], -1)
            if flat.shape[1] > 12:
                flat = flat[:, :12]
                suffix = " (first 12 dims)"
            else:
                suffix = ""
            for idx in range(flat.shape[1]):
                ax.plot(time_s, flat[:, idx], label=f"{field}[{idx}]", linewidth=1.2)
            ax.set_title(f"{field}{suffix}")
        ax.set_ylabel(field)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=min(6, max(1, arr.reshape(arr.shape[0], -1).shape[1] if arr.ndim > 1 else 1)))

    axes[-1].set_xlabel("Time from episode start (s)")
    fig.suptitle(f"{path.name} numeric time series", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / f"{path.stem}_timeseries.png"
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    del data
    return output_path


def plot_image_samples(path, image_field, output_dir, show=False):
    plt = prepare_matplotlib(show)

    data = load_pickle(path)
    messages = get_messages(data)
    available = set(message_field_names(messages[0]))
    if image_field not in available:
        raise ValueError(f"{image_field} is not available in {path.name}.")

    indices = [0, len(messages) // 2, len(messages) - 1]
    labels = ["first", "middle", "last"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, idx, label in zip(axes, indices, labels):
        image = field_value(messages[idx], image_field)
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            ax.text(0.5, 0.5, f"{image_field}: not image-like", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.imshow(image)
        ax.set_title(f"{label} frame ({idx})")
        ax.set_axis_off()

    fig.suptitle(f"{path.name} {image_field}", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / f"{path.stem}_{image_field}_samples.png"
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    del data
    return output_path


def resolve_plot_files(data_dir, pkl_paths, args):
    if args.plot_all:
        return pkl_paths

    resolved = []
    for name in args.plot_files:
        candidate = Path(name)
        if candidate.suffix != ".pkl":
            candidate = candidate.with_suffix(".pkl")
        if not candidate.is_absolute():
            candidate = data_dir / candidate.name
        if not candidate.exists():
            raise FileNotFoundError(f"Plot file not found: {candidate}")
        resolved.append(candidate)
    return sorted(set(resolved), key=natural_pkl_key)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect flip_v2 ImplicitRDP pkl files and save plots."
    )
    parser.add_argument("--data-dir", default="flip_v2", help="Directory containing pkl files.")
    parser.add_argument("--output-dir", default="flip_v2_check", help="Directory for CSV/txt/png outputs.")
    parser.add_argument(
        "--plot-files",
        nargs="+",
        default=["1"],
        help="Episode files to plot. You can pass 1 or 1.pkl. Ignored with --plot-all.",
    )
    parser.add_argument("--plot-all", action="store_true", help="Save per-episode plots for every pkl file.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Numeric fields to plot as time series.",
    )
    parser.add_argument("--image-field", default=DEFAULT_IMAGE_FIELD, help="Image field used for sample frames.")
    parser.add_argument("--no-images", action="store_true", help="Skip image sample plots.")
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows in addition to saving png files.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_paths = sorted(data_dir.glob("*.pkl"), key=natural_pkl_key)
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl files found under {data_dir}")

    print(f"Found {len(pkl_paths)} pkl files under {data_dir}")

    summaries = []
    structure_info = None
    structure_path = None
    for idx, path in enumerate(pkl_paths):
        detailed = idx == 0
        summary, structure = summarize_episode(path, detailed=detailed)
        summaries.append(summary)
        print(
            f"{path.name:>8}: messages={summary['messages']:>4} "
            f"duration={summary['duration_s']:>7.3f}s "
            f"hz={summary['approx_hz']:>5.2f} "
            f"size={summary['size_mb']:>6.1f}MB"
        )
        if detailed:
            structure_info = structure
            structure_path = path
            print_structure(path, summary, structure)
            structure_txt = write_structure_txt(path, summary, structure, output_dir)
            print(f"Saved structure text: {structure_txt}")
        gc.collect()

    summary_csv = write_summary_csv(summaries, output_dir)
    overview_png = plot_dataset_overview(summaries, output_dir, show=args.show)
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved overview plot: {overview_png}")

    plot_paths = resolve_plot_files(data_dir, pkl_paths, args)
    for path in plot_paths:
        timeseries_png = plot_timeseries(path, args.fields, output_dir, show=args.show)
        print(f"Saved time-series plot: {timeseries_png}")
        if not args.no_images:
            image_png = plot_image_samples(path, args.image_field, output_dir, show=args.show)
            print(f"Saved image sample plot: {image_png}")
        gc.collect()

    if structure_info is not None:
        print(f"\nFirst detailed file: {structure_path.name}")
    print(f"Done. Outputs are in: {output_dir}")


if __name__ == "__main__":
    main()
