import numpy as np


def calculate_snr(signal, noise):
    signal_power = np.sum(signal**2) / len(signal)
    noise_power = np.sum(noise**2) / len(noise)
    snr_value = 10 * np.log10(signal_power / noise_power)
    return snr_value
