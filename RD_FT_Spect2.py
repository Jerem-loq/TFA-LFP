"""
Time-frequency analysis on LFP records.
Plot LFP record, Raster, Fourier Transform, and Spectrogram.
Compute the average amplitude of each LFP record.

Made by Jérémie Loquet 01/2025
"""

import os
import re
from glob import glob
from brian2 import *
import brian2.numpy_ as np
import scipy.signal as signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def read_data(path):
    """
    Read .npz files containing 1x5 arrays named as just below.
    Where mu_value is nothing but the evolving parameter for every simulation.
    :param path: pathway of the file
    :return: returns 5x1 arrays.
    """
    dataset_ = np.load(path)

    spike_time = dataset_['spike_times']
    neuron_indice = dataset_['neuron_indices']
    lfp_time = dataset_['LFP_time']
    lfp_value = dataset_['LFP_values']
    mu_value = dataset_['mu_value']

    if len(lfp_time) != len(lfp_value):
        raise ValueError(f"Mismatch in LFP time and value lengths: {len(lfp_time)} vs {len(lfp_value)}.")

    return spike_time, neuron_indice, lfp_time, lfp_value, mu_value


def get_psd(tv, xv):
    """
    Compute the  absolute Power spectrum density from a bin.
    :param tv: bin_n
    :param xv: bin_(n+1)
    :return: Array of sample frequencies over the bin, Array of segment times over the bin, the power spectrum density.
    """
    dt_ = tv[1] - tv[0]

    win_dt = .1
    nparseg = int(win_dt / dt_)
    noverlap = int(nparseg * 0.9)
    resolf = nparseg * 7

    sfv_, stv_, svv_ = signal.stft(
        xv, 1 / dt_,
        noverlap=noverlap,
        detrend='constant',
        nperseg=nparseg,
        nfft=resolf
    )

    stv_ = np.linspace(
        tv[0],
        tv[-1],
        stv_.size
    )
    psd_ = np.abs(svv_) ** 2

    return stv_, sfv_, psd_


def smooth_data(x, window_size):
    """
    Smooth the data (x) for a window length.
    :param x: data
    :param window_size: window length.
    :return: smoothed data
    """
    if window_size >= len(x):
        raise ValueError(f"Window size ({window_size}) must be smaller than the input array length ({len(x)}.")

    window = np.ones(window_size) / window_size
    smooth = np.convolve(x, window, mode='same')

    return smooth


def log_evo_wind_smo(x, y, window_min, window_max):
    """
    Smooth coupled data (x, y) for an evolving window length according to a log distribution.
    :param x: data that determine the window size (the less the value, the less the window size).
    :param y: actual data to smooth
    :param window_min: inferior borne for log distribution
    :param window_max: superior borne for log distribution
    :return: smoothed y data
    """
    x = np.clip(x, a_min=1e-10, a_max=None)

    log_x = np.log(x)
    log_x_min = np.min(log_x)
    log_x_max = np.max(log_x)

    window_sizes = window_min + (window_max - window_min) * (log_x - log_x_min) / (log_x_max - log_x_min)
    window_sizes = window_sizes.astype(int)

    smoothed_fft = np.zeros_like(y)
    for j, (x, mag) in enumerate(zip(x, y)):
        window_size = window_sizes[j]
        start = max(0, j - window_size // 2)
        end = min(len(y), j + window_size // 2 + 1)

        smoothed_fft[j] = np.mean(y[start:end])

    return smoothed_fft


def average_lfp_peak_amplitude(lfp_values, window_size=5):
    """
    Compute the average peak amplitude from lfp.
    :param lfp_values: data containing lfp values.
    :param window_size: on how many does the data must be averaged.
    :return: average value of lfp values.
    """
    peaks, _ = find_peaks(lfp_values)
    if len(peaks) == 0:
        return np.nan, np.nan  # Handle cases where no peaks are found

    peak_amplitudes_h = lfp_values[peaks]
    avg_amplitude_h = np.mean(peak_amplitudes_h)

    peak_amplitudes_std = [np.std(lfp_values[max(0, p - window_size): p + window_size]) for p in peaks]
    avg_amplitude_std = np.mean(peak_amplitudes_std)

    return avg_amplitude_h, avg_amplitude_std


def extract_number(file_name_):
    """
    Retrieve number from file name.
    :param file_name_: file name.
    :return: int
    """
    matches = re.findall(r"=(\d+)", os.path.basename(file_name_))
    valid_matches = [int(num) for num in matches]  # if int(num) >= 40]
    if valid_matches:
        return min(valid_matches)
    else:
        print("No number was found.")
        return float('inf')


if __name__ == '__main__':
    base_dir = r''  # File path to the data directory
    save_dir = r''  # File path to save directory
    subfolders = []  # list of data subfolders

    for subfolder in subfolders:
        os.makedirs(os.path.join(save_dir, subfolder), exist_ok=True)

    data = {}

    # Load all files dynamically in one number in the file name.
    for subfolder in subfolders:
        folder_path = os.path.join(base_dir, subfolder)
        file_paths = glob(os.path.join(folder_path, "*.npz"))

        if not file_paths:
            print(f"Warning: No files found in {subfolder}.")
            continue

        valid_files = [f for f in file_paths if extract_number(f) != float('inf')]

        # for valid_file in valid_files:
        #     x, y = extract_number(valid_file)
        #     if x != float('inf') and y != float('inf'):
        #         mus.append((x, y, valid_file))

        # Sort files numerically based on the number in the file name
        file_paths = sorted(valid_files, key=extract_number)
        # mus.sort(key=lambda item: item[0])

        data[subfolder] = []
        for file_path in file_paths:
            try:
                file_name = os.path.basename(file_path)
                print(f"Loading file: '{file_name}'.")

                spike_times, neuron_indices, LFP_times, LFP_values, mu_values = read_data(file_path)

                if len(LFP_times) != len(LFP_values):
                    print(
                        f"Warning: Length mismatch in {file_path} (LFP_times: {len(LFP_times)},"
                        f" LFP_values: {len(LFP_values)})."
                    )
                    continue

                delta = extract_number(file_path)
                data[subfolder].append([spike_times, neuron_indices, LFP_times, LFP_values, mu_values, delta])

            except Exception as e:
                print(f"Error reading {file_path}: {e}.")

    # Load all files dynamically if two number in the file name. Warning, Extract number function must be adapted.
    """for x, y, valid_file in mus:
        try:
            file_name = os.path.basename(valid_file)
            delta = y - x
            print(f"Loading file: '{file_name}' (µ1 = {x}, µ2 = {y}, Δµ = {z}).")

            spike_times, neuron_indices, LFP_times, LFP_values = read_data(valid_file)
            print(spike_times[:5])
            print(LFP_values[:5])
            
            if len(LFP_times) != len(LFP_values):
                print(
                    f"Warning: Length mismatch in {valid_file} (LFP_times: {len(LFP_times)},"
                    f" LFP_values: {len(LFP_values)})."
                    )
                continue

            data[subfolder].append([spike_times, neuron_indices, LFP_times, LFP_values, delta])

        except Exception as e:
            print(f"Error reading {valid_file}: {e}.")"""

    # Processing and Plotting
    for condition, datasets in data.items():
        if not datasets:
            print(f"No valid datasets for {condition}.")
            continue

        print(f"\nProcessing {len(datasets)} files from {condition}.\n")

        for idx, dataset in enumerate(datasets):
            z = dataset[5]

            avg_heigh_amp, avg_std_amp = average_lfp_peak_amplitude(dataset[3], window_size=5)
            avg_peak_amp = [avg_heigh_amp, avg_std_amp]
            print(f"[{condition} - File {idx + 1}] Average LFP peak amplitude from heigh: {avg_heigh_amp:.4f}, from "
                  f"std: {avg_std_amp:.4f}.")

            try:
                time_diff = np.diff(dataset[2])
                if np.any(time_diff <= 0):
                    raise ValueError("Invalid time differences in LFP_times (non-positive or zero).")

                fs = 1 / np.mean(time_diff)
            except Exception as e:
                print(f"Error computing sampling frequency for {condition} - File {idx + 1}: {e}.")
                continue

            normalized_lfp = dataset[3] / np.mean(dataset[3])

            count_samples = len(normalized_lfp)
            freqs = np.fft.rfftfreq(count_samples, d=1 / fs)
            fft_magnitude = np.abs(np.fft.rfft(normalized_lfp))
            fft_smoothed = log_evo_wind_smo(freqs, fft_magnitude, 5, 50)
            smoothed_LFP = smooth_data(normalized_lfp, window_size=10)

            if fft_magnitude.shape[0] != freqs.shape[0]:
                raise ValueError(
                    f"Mismatch in frequency and magnitude array shapes: {freqs.shape} vs {fft_magnitude.shape}"
                )

            argmax_idx = np.argmax(fft_smoothed[20:])
            max_val = np.max(fft_smoothed[20:])
            max_freq = freqs[argmax_idx]
            max_magnitude = np.percentile(fft_smoothed, 99)

            stv, sfv, psd = get_psd(dataset[2], dataset[3])
            psd /= psd.max()
            psd_log = np.log10(psd)

            neuron_indices = dataset[1]
            mu_values = dataset[4]
            firing_mu_values = mu_values[neuron_indices]

            sorted_indices = np.argsort(firing_mu_values)
            sorted_neuron_indices = neuron_indices[sorted_indices]
            sorted_mu_values = firing_mu_values[sorted_indices]

            print(
                f'Average peak amplitude by heigh:{avg_heigh_amp:.4f}.'
                f'\nAverage peak amplitude by std:\t{avg_std_amp:.4f}.'
                f'\nMaximum Amplitude Frequency:\t{max_freq:.4f} Hz.'
            )

            save_path_data = os.path.join(base_dir, condition, f"Amplitude_&_Freq_{condition}_Δµ={z}")
            np.savez(
                save_path_data,
                avg_heigh_amp=avg_heigh_amp,
                avg_std_amp=avg_std_amp,
                max_freq=max_freq

            )
            print(f"Data saved as 'Amplitude_&_Freq_{condition}_Δµ={z}.npz'.")

            print(f"[{condition} - File {idx + 1}] Processed successfully.")
            print(f"[{condition} - File {idx + 1}] Starting plotting...")

            subplot_labels = ['A', 'B', 'C', 'D']
            save_path = os.path.join(save_dir, condition, f"{condition}_File_{idx + 1}_Plots_Δµ={z}.png")

            mmin = -3
            mmax = 0
            cmap = 'jet'
            exps = [-3, -2, -1, 0]
            FIG = []

            fig, ax = plt.subplots(2, 2, figsize=(15, 9))
            fig.suptitle(f"{condition} - File {idx + 1} Plots for Δµ = {z}", fontsize=16)

            # Construct the Raster Plot
            # ax[0, 0].scatter(sorted_mu_values, sorted_neuron_indices, s=1)
            ax[0, 0].scatter(dataset[0], dataset[1], s=1)
            ax[0, 0].set_title('Raster Plot')
            ax[0, 0].set_xlabel('Time (sec)')
            # ax[0, 0].set_xlabel('µ values (mV)')
            ax[0, 0].set_ylabel('Neuron Index')

            # Construct the LFP plot
            ax[1, 0].plot(dataset[2], smoothed_LFP)
            ax[1, 0].set_title('LFP recording')
            ax[1, 0].set_xlabel('Time (sec)')
            ax[1, 0].set_ylabel('Amplitude')

            # Construct the Fourier Transform plot
            ax[0, 1].plot(freqs, fft_smoothed)
            ax[0, 1].set_title('Power Spectrum')
            ax[0, 1].set_xlabel('Frequency (Hz)')
            ax[0, 1].set_ylabel('Magnitude')
            ax[0, 1].set_xscale('log')
            ax[0, 1].set_xlim(15, 3000)
            if np.isnan(max_val) or np.isinf(max_val):
                ax[0, 1].set_ylim(0, 10)
            else:
                ax[0, 1].set_ylim(0, max_val + max_val * 0.1)

            # Construct the Spectrogram
            mesh = ax[1, 1].pcolormesh(stv, sfv, psd_log, cmap=cmap, shading='auto', vmin=mmin, vmax=mmax)
            cbar = fig.colorbar(mesh)
            cbar.set_ticks(exps)
            cbar.set_ticklabels([r"$10^{%i}$" % exp for exp in exps])

            ax[1, 1].set_ylim(0, 150)
            ax[1, 1].set_title(f'Spectrogram')
            ax[1, 1].set_xlabel("Time (sec)")
            ax[1, 1].set_ylabel("Frequency (Hz)")
            cbar.set_label("Power (dB) / max Power")

            # Plot and save the Figure according to sav_dir.
            FIG.append(fig)

            for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                ax[row, col].text(-0.05, 1.1, subplot_labels[i], transform=ax[row, col].transAxes,
                                  fontsize=16, va='top', ha='right')

            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[{condition} - File {idx + 1}] Plot done.")
            print(f"Figure saved as '{condition}_File_{idx + 1}_Plots_Δµ={z}.png'.\n")
            plt.close(fig)
