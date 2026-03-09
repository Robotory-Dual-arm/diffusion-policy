import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
import os

def plot_force_and_joint_data(hdf5_file, demo_idx):
    
    with h5py.File(hdf5_file, 'r') as f:
        demo = f[f'data/demo_{demo_idx}']
        obs = demo['observations']
        
        # Load robot data (20Hz)
        joint_R = obs['joint_R'][:]
        timestamp_robot = obs['timestamp_robot'][:]
        
        # Load wrench data (250Hz) - Right wrist
        wrench_wrist_R = obs['wrench_wrist_R'][:]
        timestamp_wrench = obs['timestamp_wrench'][:]
        
        # Load all finger wrenches (250Hz)
        wrench_thumb_R = obs['wrench_thumb_R'][:]
        wrench_index_R = obs['wrench_index_R'][:]
        wrench_middle_R = obs['wrench_middle_R'][:]
        wrench_ring_R = obs['wrench_ring_R'][:]
        wrench_baby_R = obs['wrench_baby_R'][:]
        
        wrench_hand_zeroset = True
        if wrench_hand_zeroset:
            wrench_thumb_R[:] -= wrench_thumb_R[0]
            wrench_index_R[:] -= wrench_index_R[0]
            wrench_middle_R[:] -= wrench_middle_R[0]
            wrench_ring_R[:] -= wrench_ring_R[0]
            wrench_baby_R[:] -= wrench_baby_R[0]
        
        print(f"Demo {demo_idx} loaded:")
        print(f"  Robot data: {len(joint_R)} samples at 20Hz")
        print(f"  Wrench data: {len(wrench_wrist_R)} samples at 250Hz")
        print(f"  Duration: {timestamp_robot[-1] - timestamp_robot[0]:.2f}s")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 14))
        gs = GridSpec(3, 1, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Joint positions (Right arm)
        ax1 = fig.add_subplot(gs[0, :])
        for i in range(6):
            ax1.plot(timestamp_robot, np.rad2deg(joint_R[:, i]), label=f'Joint {i+1}', alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Joint Angle (deg)')
        ax1.set_title(f'Right Arm Joint Positions (20Hz)')
        ax1.legend(ncol=6, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Wrist wrench - Forces
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(timestamp_wrench, wrench_wrist_R[:, 0], label='Fx', alpha=0.7)
        ax2.plot(timestamp_wrench, wrench_wrist_R[:, 1], label='Fy', alpha=0.7)
        ax2.plot(timestamp_wrench, wrench_wrist_R[:, 2], label='Fz', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force (N)')
        ax2.set_title('Right Wrist Forces (250Hz)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Finger Fz - forces magnitude (250Hz)
        ax3 = fig.add_subplot(gs[2, :])
        
        ax3.plot(timestamp_wrench, wrench_thumb_R[:, 2], label='Thumb', alpha=0.7)
        ax3.plot(timestamp_wrench, wrench_index_R[:, 2], label='Index', alpha=0.7)
        ax3.plot(timestamp_wrench, wrench_middle_R[:, 2], label='Middle', alpha=0.7)
        ax3.plot(timestamp_wrench, wrench_ring_R[:, 2], label='Ring', alpha=0.7)
        ax3.plot(timestamp_wrench, wrench_baby_R[:, 2], label='Baby', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Force Magnitude (N)')
        ax3.set_title('Right Hand Finger Forces (250Hz)')
        ax3.legend(ncol=1, loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        
        plt.suptitle(f'Force and Joint Data Visualization - Demo {demo_idx}', fontsize=16, y=0.995)
        plt.tight_layout()
        
        return fig


def plot_thumb_fz_spectrogram(demo_idx, fs=250):
    
    with h5py.File(hdf5_file, 'r') as f:
        demo = f[f'data/demo_{demo_idx}']
        obs = demo['observations']
        
        # Load wrench data (250Hz) - Right wrist
        timestamp_wrench = obs['timestamp_wrench'][:]
        
        # Load all finger wrenches (250Hz)
        wrench_thumb_R = obs['wrench_middle_R'][:]
     
        wrench_hand_zeroset = True
        if wrench_hand_zeroset:
            wrench_thumb_R[:] -= wrench_thumb_R[0]
            

    # Extract Fz component (index 2)
    thumb_fz = wrench_thumb_R[:, 2]
    
    # Create separate figure for spectrogram with custom layout
    fig_spec = plt.figure(figsize=(15, 10))
    # Create GridSpec with space for colorbar
    gs = GridSpec(2, 2, figure=fig_spec, hspace=0.3, wspace=0.05, 
                  width_ratios=[0.95, 0.05])
    
    # 1. Time domain plot
    ax1 = fig_spec.add_subplot(gs[0, 0])
    ax1.plot(timestamp_wrench, thumb_fz, linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force Fz (N)')
    ax1.set_title(f'Thumb Fz - Time Domain (250Hz) - Demo {demo_idx}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Spectrogram
    ax2 = fig_spec.add_subplot(gs[1, 0])
    
    # Compute spectrogram with padding to match original length
    nperseg = 256  # Window size
    noverlap = nperseg // 2  # 50% overlap
    
    # Add padding to the signal so spectrogram covers full time range
    pad_width = nperseg // 2
    thumb_fz_padded = np.pad(thumb_fz, (pad_width, pad_width), mode='edge')
    
    frequencies, times, Sxx = signal.spectrogram(
        thumb_fz_padded, 
        fs=fs, 
        nperseg=nperseg, 
        noverlap=noverlap,
        scaling='density'
    )
    
    # Adjust time axis to match actual timestamps
    time_offset = timestamp_wrench[0]
    actual_duration = timestamp_wrench[-1] - timestamp_wrench[0]
    
    # Compensate for padding: shift times back by pad_width/fs
    pad_time = pad_width / fs
    times_adjusted = times + time_offset - pad_time
    
    # Plot spectrogram
    pcm = ax2.pcolormesh(
        times_adjusted, 
        frequencies, 
        10 * np.log10(Sxx + 1e-10),  # Convert to dB scale, add small value to avoid log(0)
        shading='gouraud',
        cmap='viridis'
    )
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(f'Thumb Fz - Spectrogram - Demo {demo_idx}')
    ax2.set_ylim([0, 125])  # Show up to Nyquist frequency (125 Hz for 250 Hz sampling)
    
    # Match x-axis limits between both subplots
    ax1.set_xlim([timestamp_wrench[0], timestamp_wrench[-1]])
    ax2.set_xlim([timestamp_wrench[0], timestamp_wrench[-1]])
    
    # Add colorbar to separate axis
    cax = fig_spec.add_subplot(gs[1, 1])
    cbar = fig_spec.colorbar(pcm, cax=cax)
    cbar.set_label('Power Spectral Density (dB/Hz)')
    
    plt.suptitle(f'Thumb Fz Spectrogram Analysis - Demo {demo_idx}', fontsize=16, y=0.995)
    
    return fig_spec


if __name__ == "__main__":
    import sys
    
    hdf5_file = os.path.expanduser("~/common_data.hdf5")
    
    if len(sys.argv) > 1:
        hdf5_file = os.path.expanduser(sys.argv[1])
    
    # Plot main force and joint data
    fig_demo= plot_force_and_joint_data(hdf5_file, demo_idx=0)
    
    # Plot spectrogram in separate window
    # fig_spec = plot_thumb_fz_spectrogram(demo_idx=0)
    
    plt.show()
    