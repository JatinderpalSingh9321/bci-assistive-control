"""
BCI Assistive Control — Improved EEG Preprocessing
=====================================================
Improved preprocessing that preserves motor imagery signals.

Key changes from original EEGPreprocessor:
  - Removed ICA (too aggressive, removes motor signals)
  - Wider bandpass: 0.5–50 Hz (captures all motor-relevant bands)
  - Gentler artifact removal (5-sigma outliers only)
  - Common Average Reference (CAR) instead of ICA
  - Bad channel interpolation instead of rejection

Group No. 7 | 8th Semester Major Project
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

from src.utils import SAMPLING_RATE, setup_logger

logger = setup_logger("improved_preprocessing")


class ImprovedEEGPreprocessor:
    """
    Fixed EEG preprocessing that preserves motor imagery signals.

    Changes from previous:
    - Removed ICA (too aggressive)
    - Wider bandpass (0.5-50 Hz instead of 8-30 Hz)
    - Gentler artifact removal
    - Better channel interpolation

    Parameters
    ----------
    fs : int
        Sampling rate in Hz (default: 250).
    verbose : bool
        Whether to print progress.
    """

    def __init__(self, fs=SAMPLING_RATE, verbose=True):
        self.fs = fs
        self.verbose = verbose
        logger.info(f"ImprovedEEGPreprocessor initialized: fs={fs} Hz, "
                    f"bandpass=0.5-50 Hz, CAR referencing")

    def preprocess(self, epochs, baseline=None):
        """
        Preprocess EEG epochs with the improved pipeline.

        Args:
            epochs: (n_epochs, n_channels, n_timepoints) or (n_epochs, n_timepoints)
            baseline: Optional baseline for correction

        Returns:
            preprocessed: Same shape, cleaned
        """
        if self.verbose:
            logger.info(f"Starting improved preprocessing: {epochs.shape}")

        # Handle both single-channel (n_epochs, n_timepoints) and
        # multi-channel (n_epochs, n_channels, n_timepoints)
        single_channel = (epochs.ndim == 2)
        if single_channel:
            # Add channel dimension: (n_epochs, 1, n_timepoints)
            epochs = epochs[:, np.newaxis, :]

        preprocessed = []

        for i, epoch in enumerate(epochs):
            # 1. Remove extreme outliers (> 5 sigma only)
            cleaned = self._robust_outlier_removal(epoch, threshold_sigma=5)

            # 2. Notch filter (50 Hz power line noise)
            cleaned = self._notch_filter_50hz(cleaned)

            # 3. Bandpass (WIDER: 0.5-50 Hz instead of 8-30 Hz)
            cleaned = self._bandpass_filter(cleaned, l_freq=0.5, h_freq=50)

            # 4. Interpolate bad channels (instead of rejection)
            cleaned = self._interpolate_bad_channels(cleaned)

            # 5. Common average reference (gentle, better than ICA)
            if cleaned.shape[0] > 1:  # Only CAR for multi-channel
                cleaned = self._common_average_reference(cleaned)

            # 6. Baseline correction (minimal)
            if baseline is not None:
                cleaned = self._baseline_correct(cleaned, baseline)

            preprocessed.append(cleaned)

            if self.verbose and (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(epochs)} epochs")

        result = np.array(preprocessed)

        # Remove channel dimension if input was single-channel
        if single_channel:
            result = result[:, 0, :]

        if self.verbose:
            logger.info(f"Preprocessing complete: {result.shape}")

        return result

    def preprocess_single_channel(self, epochs):
        """
        Simplified preprocessing for single-channel data.
        Input shape: (n_epochs, n_timepoints)
        Output shape: (n_epochs, n_timepoints)
        """
        preprocessed = []

        for i, epoch in enumerate(epochs):
            # Work with 1D array directly
            cleaned = epoch.copy().astype(np.float64)

            # 1. Outlier removal
            mean = np.mean(cleaned)
            std = np.std(cleaned)
            outlier_mask = np.abs(cleaned - mean) > 5 * std
            if np.any(outlier_mask):
                x = np.arange(len(cleaned))
                good = ~outlier_mask
                if np.any(good):
                    cleaned[outlier_mask] = np.interp(
                        x[outlier_mask], x[good], cleaned[good]
                    )

            # 2. Notch filter 50 Hz
            b, a = signal.iirnotch(50, 30, fs=self.fs)
            if len(cleaned) > 3 * max(len(b), len(a)):
                cleaned = signal.filtfilt(b, a, cleaned)

            # 3. Wider bandpass 0.5-50 Hz
            sos = signal.butter(4, [0.5, min(49.9, self.fs/2 - 1)],
                                'bandpass', fs=self.fs, output='sos')
            if len(cleaned) > 15:
                cleaned = signal.sosfiltfilt(sos, cleaned)

            # 4. Baseline correction (subtract mean of first 10%)
            bl_samples = max(1, len(cleaned) // 10)
            cleaned = cleaned - np.mean(cleaned[:bl_samples])

            preprocessed.append(cleaned.astype(np.float32))

            if self.verbose and (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(epochs)} epochs")

        return np.array(preprocessed, dtype=np.float32)

    def _robust_outlier_removal(self, epoch, threshold_sigma=5):
        """
        Remove only EXTREME outliers (> 5 sigma).
        Don't remove normal variation!
        """
        epoch_clean = epoch.copy()

        for ch in range(epoch.shape[0]):
            mean = np.mean(epoch[ch])
            std = np.std(epoch[ch])

            # Only extreme values
            outlier_mask = np.abs(epoch[ch] - mean) > threshold_sigma * std

            if np.any(outlier_mask):
                # Interpolate over outliers
                x = np.arange(len(epoch[ch]))
                good_indices = ~outlier_mask

                if np.any(good_indices):
                    epoch_clean[ch][outlier_mask] = np.interp(
                        x[outlier_mask],
                        x[good_indices],
                        epoch[ch][good_indices]
                    )

        return epoch_clean

    def _notch_filter_50hz(self, epoch, freq=50, Q=30):
        """
        Notch filter for 50 Hz (power line noise).
        Use Q=30 for very narrow notch.
        """
        b, a = signal.iirnotch(freq, Q, fs=self.fs)
        epoch_filtered = np.zeros_like(epoch)

        for ch in range(epoch.shape[0]):
            if len(epoch[ch]) > 3 * max(len(b), len(a)):
                epoch_filtered[ch] = signal.filtfilt(b, a, epoch[ch])
            else:
                epoch_filtered[ch] = epoch[ch]

        return epoch_filtered

    def _bandpass_filter(self, epoch, l_freq=0.5, h_freq=50):
        """
        Bandpass filter: 0.5-50 Hz (WIDER than original 8-30 Hz).

        Why wider:
        - 0.5-4 Hz: Captures slow brain rhythms (delta)
        - 4-8 Hz: Theta band (important for cognitive states)
        - 8-12 Hz: Mu/alpha (motor cortex - primary MI signal!)
        - 12-30 Hz: Beta (motor planning)
        - 30-50 Hz: Gamma (decision making, fine motor)

        Motor imagery affects multiple bands, not just 8-30 Hz!
        """
        # Ensure h_freq doesn't exceed Nyquist
        h_freq = min(h_freq, self.fs / 2 - 1)
        sos = signal.butter(4, [l_freq, h_freq], 'bandpass',
                            fs=self.fs, output='sos')
        epoch_filtered = np.zeros_like(epoch)

        for ch in range(epoch.shape[0]):
            if len(epoch[ch]) > 15:
                epoch_filtered[ch] = signal.sosfiltfilt(sos, epoch[ch])
            else:
                epoch_filtered[ch] = epoch[ch]

        return epoch_filtered

    def _interpolate_bad_channels(self, epoch):
        """
        If a channel is noisy, interpolate from neighbors
        instead of rejecting the whole epoch.
        """
        if epoch.shape[0] <= 1:
            return epoch  # Can't interpolate with single channel

        epoch_clean = epoch.copy()

        # Find noisy channels (> 2x median STD)
        noise_levels = np.std(epoch, axis=1)
        median_noise = np.median(noise_levels)
        bad_ch_mask = noise_levels > 2 * median_noise

        bad_channels = np.where(bad_ch_mask)[0]

        if len(bad_channels) > 0 and len(bad_channels) < epoch.shape[0]:
            for ch in bad_channels:
                good_channels = np.where(~bad_ch_mask)[0]
                if len(good_channels) > 0:
                    epoch_clean[ch] = np.mean(epoch[good_channels], axis=0)

        return epoch_clean

    def _common_average_reference(self, epoch):
        """
        Common Average Reference (CAR).
        Subtract mean of all channels from each channel.
        Better than ICA for preserving local motor signals.
        """
        car = np.mean(epoch, axis=0, keepdims=True)
        return epoch - car

    def _baseline_correct(self, epoch, baseline):
        """
        Minimal baseline correction with z-score normalization.
        """
        epoch_corrected = epoch.copy()

        for ch in range(epoch.shape[0]):
            if isinstance(baseline, np.ndarray) and baseline.ndim >= 2:
                mean = np.mean(baseline[ch])
                std = np.std(baseline[ch])
            else:
                # Use first 10% of epoch as baseline
                bl_len = max(1, epoch.shape[1] // 10)
                mean = np.mean(epoch[ch, :bl_len])
                std = np.std(epoch[ch, :bl_len])

            epoch_corrected[ch] = (epoch[ch] - mean) / (std + 1e-10)

        return epoch_corrected


# USAGE:
if __name__ == "__main__":
    # Quick test with simulated data
    print("Testing ImprovedEEGPreprocessor...")

    # Single channel test (n_epochs, n_timepoints)
    test_epochs_1ch = np.random.randn(10, 500).astype(np.float32)
    preprocessor = ImprovedEEGPreprocessor(fs=250)
    result = preprocessor.preprocess_single_channel(test_epochs_1ch)
    print(f"  Single channel: {test_epochs_1ch.shape} → {result.shape}")

    # Multi-channel test (n_epochs, n_channels, n_timepoints)
    test_epochs_3ch = np.random.randn(10, 3, 500).astype(np.float32)
    result = preprocessor.preprocess(test_epochs_3ch)
    print(f"  Multi-channel:  {test_epochs_3ch.shape} → {result.shape}")

    print("✓ ImprovedEEGPreprocessor ready to use")
