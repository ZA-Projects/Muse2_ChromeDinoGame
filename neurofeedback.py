# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers and Blink Detection
"""

import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import pyautogui  # Module to simulate key presses
import utils  # Utility functions for signal processing

# Handy little enum to make code more readable
class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

# Blink detection threshold (to be adjusted based on signal observations)
BLINK_THRESHOLD = 200

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    fs = int(info.nominal_srate())  # e.g., 256 Hz for Muse 2016

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    print('Press Ctrl-C in the console to break the while loop.')

    try:
        while True:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs)
            )

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            """ 3.2 DETECT BLINKS """
            # Calculate signal amplitude difference
            signal_diff = max(ch_data) - min(ch_data)

            # Debugging (Optional)
            print("Signal Difference:", signal_diff)

            # Check if signal difference exceeds the blink threshold
            if signal_diff > BLINK_THRESHOLD:
                pyautogui.press('space')
                print("Blink detected, Dino jumps!")

            """ 3.3 COMPUTE BAND POWERS """
            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True, filter_state=filter_state
            )

            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(
                band_buffer, np.asarray([band_powers])
            )

            # Compute the average band powers for all epochs in buffer
            smooth_band_powers = np.mean(band_buffer, axis=0)

            # Print band power metrics for reference
            print(
                f"Delta: {smooth_band_powers[Band.Delta]:.2f}, "
                f"Theta: {smooth_band_powers[Band.Theta]:.2f}, "
                f"Alpha: {smooth_band_powers[Band.Alpha]:.2f}, "
                f"Beta: {smooth_band_powers[Band.Beta]:.2f}"
            )

    except KeyboardInterrupt:
        print('Closing!')
