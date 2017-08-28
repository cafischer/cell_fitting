import numpy as np


def get_phase_and_amplitude_fft(x, dt, window_len=None):
    """
    Computes the phase and amplitude of frequency components in x at every time step by Fourier transform.

    :param x: Input array.
    :type x: array
    :param dt: Time step in sec.
    :type dt: float
    :param window_len: Length of the sliding window used for the fft.
    :type window_len: int
    :return: Phase and amplitude with axes time vs frequency and corresponding frequencies.
    """

    if window_len is None:
        window_len = int(2 ** np.round(np.log(1 / dt) / np.log(2)))  # powers of 2 make fft faster
    taper = np.hanning(window_len)  # hanning window
    freqs = np.fft.fftfreq(window_len, d=dt)
    phases = np.zeros((len(x), len(freqs)))
    amp = np.zeros((len(x), len(freqs)))
    x_padded = np.concatenate((np.zeros(int(round(window_len / 2))), x, np.zeros(
        int(round(window_len / 2)))))  # now point is centered in the middle of the window

    for i, x_i in enumerate(x):
        # extract window around x_i and taper
        x_window = x_padded[i:i + window_len] * taper

        # fft
        x_fft = np.fft.fft(x_window)

        # extract the phase and amplitude
        phases[i, :] = np.angle(x_fft)
        amp[i, :] = np.abs(x_fft)

    return phases, amp, freqs