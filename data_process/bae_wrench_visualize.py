import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# ===== LPF 파라미터 =====
EMA_ALPHA = 0.05   # EMA 계수 (0~1, 작을수록 더 smooth)

def ema_filter(data, alpha):
    """Exponential Moving Average filter (각 열 독립 적용)"""
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

def plot_wrench(hdf5_file, demo_idx=0, target='wrist'):
    with h5py.File(hdf5_file, 'r') as f:
        obs = f[f'data/demo_{demo_idx}/observations']
        t = obs['timestamp_wrench'][:]
        
        sources = {
            'wrist': obs['wrench_wrist_R'][:],
            'thumb': obs['wrench_thumb_R'][:],
            'index': obs['wrench_index_R'][:],
            'middle': obs['wrench_middle_R'][:],
            'ring': obs['wrench_ring_R'][:],
            'baby': obs['wrench_baby_R'][:],
        }
    
    data = sources[target]
    data -= data[0]
    
    filtered = ema_filter(data, EMA_ALPHA)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    for i, label in enumerate(['Fx', 'Fy', 'Fz']):
        axes[0].plot(t, data[:, i], color=colors[i], alpha=0.3, linewidth=0.5)
        axes[0].plot(t, filtered[:, i], color=colors[i], label=f'{label} (EMA α={EMA_ALPHA})', linewidth=1.5)
    axes[0].set_ylabel('Force (N)')
    axes[0].set_title(f'{target.capitalize()} Forces')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for i, label in enumerate(['Tx', 'Ty', 'Tz']):
        axes[1].plot(t, data[:, 3+i], color=colors[i], alpha=0.3, linewidth=0.5)
        axes[1].plot(t, filtered[:, 3+i], color=colors[i], label=f'{label} (EMA α={EMA_ALPHA})', linewidth=1.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Torque (Nm)')
    axes[1].set_title(f'{target.capitalize()} Torques')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{target.capitalize()} Wrench - Demo {demo_idx}', fontsize=14)
    plt.tight_layout()
    plt.show()

TARGETS = ['wrist', 'thumb', 'index', 'middle', 'ring', 'baby']

if __name__ == "__main__":
    hdf5_file = os.path.expanduser("~/common_data.hdf5")
    target = 'wrist'
    
    if len(sys.argv) > 1:
        hdf5_file = sys.argv[1]
    if len(sys.argv) > 2:
        target = sys.argv[2]
    
    if target not in TARGETS:
        print(f"Error: target must be one of {TARGETS}")
        sys.exit(1)
    
    plot_wrench(hdf5_file, demo_idx=1, target=target)
