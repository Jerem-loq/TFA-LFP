"""
Plot Amplitude and Frequency against evolving parameter.
Amplitude plots can be log based 10 for better visualisation.

Made by Jérémie Loquet 01/25
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt


def read_data(base, subf):
    """
    Load amplitude (height & std) and frequency data from .npy files in 3 subfolders.
    :param base: Main folder location
    :param subf: Subfolder names
    :return: Dictionaries for amplitude (height, std) & frequency.
    """

    amp_height = {}
    amp_std = {}
    freq_data = {}

    # Loop through each subfolder and load files
    for subfolder in subf:
        folder_path = os.path.join(base, subfolder)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{subfolder}' does not exist!")
            continue

        all_files = os.listdir(folder_path)
        print(f"Files in {subfolder}: {all_files[:5]}...\n")

        npz_files = [f for f in all_files if f.endswith('.npz') and 'Amplitude' in f]
        print(f"Filtered .npz files in {subfolder}: {npz_files[:5]}.\n")

        files = sorted(npz_files, key=extract_number)  # Sort ensures order

        if not files:
            print(f"Warning: No valide 'Amplitude' .npz files found in '{subfolder}'!")
            continue

        print(f"Found {len(files)} files in {subfolder}:")
        print(files[:5], "...\n")

        amp_height[subfolder] = []
        amp_std[subfolder] = []
        freq_data[subfolder] = []
        mus_values_ = []

        for file in files:
            file_path = os.path.join(folder_path, file)
            print(f"Processing file: {file}")
            data = np.load(file_path)
            print(f"Keys in {file}: {list(data.keys())}\n")

            try:
                amp_h = float(data["avg_heigh_amp"])
                amp_s = float(data["avg_std_amp"])
                freq = float(data["max_freq"])
            except KeyError as e:
                print(f"Error: Missing expected key in '{file_path}': {e}.")
                continue

            mu_value = extract_number(file)
            if subfolder == 'While':
                mu_value = extract_number(file) + 40

            if mu_value != float('inf'):
                mus_values_.append(mu_value)
                amp_height[subfolder].append(amp_h)
                amp_std[subfolder].append(amp_s)
                freq_data[subfolder].append(freq)
            else:
                print(f"Skipping {file}, no valid Δµ found.")

        sorted_indices = np.argsort(mus_values_)
        mus_values_ = np.array(mus_values_)[sorted_indices].tolist()
        amp_height[subfolder] = (mus_values_, np.array(amp_height[subfolder])[sorted_indices].tolist())
        amp_std[subfolder] = (mus_values_, np.array(amp_std[subfolder])[sorted_indices].tolist())
        freq_data[subfolder] = (mus_values_, np.array(freq_data[subfolder])[sorted_indices].tolist())

    return amp_height, amp_std, freq_data


# Select this one whether there is one number in the file name.
def extract_number(file_name_):
    """
    Retrieve number from file name.
    :param file_name_: file name.
    :return: int
    """
    matches = re.findall(r"=(\d+)", os.path.basename(file_name_))
    valid_matches = [int(num) for num in matches]  # if int(num) >= 20] (If working with µ).
    if valid_matches:
        return min(valid_matches)
    else:
        print("No number was found.")
        return float('inf')


# This one returns a tuple if there are two numbers in the file name.
# def extract_number(file_name_):
#     file_name_ = os.path.basename(file_name_)
#
#     matches = re.search(r"µ=(\d+)_(\d+)", file_name_)
#
#     if matches:
#         print(f"Found number: {matches}")
#         x, y = map(int, matches.groups())
#         return x, y
#     else:
#         print(f"[{file_name_}] No valid 'µ=x_y' pattern was found.")
#         return float('inf'), float('inf')


def plot_data(data_dict, y_label, title, filename, log_scale=False):
    """
    Generic function to plot data with correctly aligned x-axis (µ values).
    :param data_dict: Dictionary of (x, y) values for each condition.
    :param y_label: Label for y-axis.
    :param title: Title of the plot.
    :param filename: Name to save the figure.
    :param log_scale: Whether to apply log_10 scale
    """

    plt.figure(figsize=(8, 5))

    for key, (x_, y_) in data_dict.items():
        y_plot = np.log10(y_) if log_scale else y_
        plt.plot(x_, y_plot, marker='o', linestyle='-', label=key)

    plt.xlabel("Δµ (mV)")
    plt.ylabel(y_label)
    plt.title(title)
    # plt.xlim(20, 84)
    plt.legend()
    plt.grid(True)

    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
        plt.tight_layout()
    except AttributeError:
        print("Window maximization not supported in this OS.")  # If this happens, look for the command of your and
        # change l137.

    plt.savefig(filename)
    print(f"Figure saved as '{filename}.png'.")
    plt.show()
    plt.close()


if __name__ == "__main__":
    base_dir = r''  # Path to data directory
    subfolders = []  # Data subfolders

    # Load data
    amps_height, amps_std, freqs = read_data(base_dir, subfolders)

    # Retrieve max and min values for amplitude and frequency.
    # This one is truncated at 84 but can be extended to the whole dataset.
    for condition in amps_height.keys():
        print(f"Condition: {condition}")
        amp_x, amp_y = amps_height[condition]
        freq_x, freq_y = freqs[condition]

        amp_y_filt = sorted([amp_y[i] for i in range(len(amp_x))
                              if 84 >= amp_x[i] >= 0 and amp_y[i] != 0 and not np.isnan(amp_y[i])])
        freq_y_filt = sorted([freq_y[i] for i in range(len(freq_x))
                              if 84 >= freq_x[i] >= 0 and freq_y[i] != 0 and not np.isnan(freq_y[i])])

        if amp_y_filt and freq_y_filt:
            min_amp, max_amp = amp_y_filt[0], amp_y_filt[-1]
            min_freq, max_freqs = freq_y_filt[0], freq_y_filt[-1]

        else:
            min_amp, max_amp = None, None
            min_freq, max_freqs = None, None

        print(f"Min Values: Amplitude = {min_amp}   Frequency = {min_freq}")
        print(f"Max Values: Amplitude = {max_amp}   Frequency = {max_freqs}")

    # Plot the data
    plot_data(amps_height, r"Amplitude $_{Height}$", "Amplitude Evolution", "Amplitude_Height_vs_Δµ")
    plot_data(amps_std, r"Amplitude $_{STD}$", "Amplitude Evolution", "Amplitude_STD_vs_Δµ")
    plot_data(freqs, r"Max Frequency (Hz)", "Max Frequency Evolution", "Frequency_vs_Δµ")
