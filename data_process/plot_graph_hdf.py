import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_timeseries(hdf_path='힘데이터/common_data.hdf5', demo_name="demo_0", group="observations"):
    # Load HDF5
    with h5py.File(hdf_path, "r") as f:
        base = f[f"data/{demo_name}/{group}"]

        # List all datasets
        datasets = [key for key in base.keys()]

        print(f"\nAvailable datasets in {demo_name}/{group}:")
        for d in datasets:
            print(f" - {d} (shape: {base[d].shape})")

        # Create plots
        for d in datasets:
            data = base[d][()]   # Load numpy array (T, D)
            T = data.shape[0]

            # Only plot non-image datasets (skip 4D image data)
            if data.ndim == 2:
                plt.figure(figsize=(12, 4))
                plt.title(f"{d} over time")
                plt.xlabel("time index (t)")
                plt.ylabel(d)

                for i in range(data.shape[1]):
                    plt.plot(data[:, i], label=f"{d}[{i}]")

                plt.legend()
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf_path", type=str)
    parser.add_argument("--demo", type=str, default="demo_0")
    args = parser.parse_args()

    plot_timeseries(args.hdf_path, demo_name=args.demo)