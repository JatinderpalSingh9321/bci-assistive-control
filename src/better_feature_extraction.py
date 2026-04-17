"""
BCI Assistive Control — Better Feature Extraction
====================================================
Extracts 5 types of features for much better discrimination
compared to the original band-power-only approach.

Feature types:
  1. Band power (6 bands: delta, theta, alpha, beta-1, beta-2, gamma)
  2. Temporal statistics (mean, std, max, RMS, skew, kurtosis, energy, PTP)
  3. Frequency domain (peak freq, centroid, spectral entropy, spread)
  4. Time-frequency (wavelet energy across scales)
  5. Nonlinear (approximate entropy, sample entropy, Hurst exponent)

Total: ~22-100 features per epoch depending on channels
→ PCA reduces to 15-20 best dimensions

Group No. 7 | 8th Semester Major Project
"""

import numpy as np
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

from src.utils import SAMPLING_RATE, TARGET_SAMPLING_RATE, setup_logger

logger = setup_logger("better_features")


class BetterFeatureExtractor:
    """
    Extract 5 types of features for better discrimination.
    Total: ~22 raw features per channel → e.g., 22 for single-channel data

    Feature types:
    1. Band power (6 bands: delta to gamma)
    2. Temporal statistics (9 features)
    3. Frequency domain (4 features)
    4. Wavelet time-frequency (energy across scales - not used for short signals)
    5. Nonlinear (approximate entropy, sample entropy, Hurst exponent)

    Parameters
    ----------
    fs : int
        Sampling rate in Hz.
    """

    def __init__(self, fs=TARGET_SAMPLING_RATE):
        self.fs = fs
        logger.info(f"BetterFeatureExtractor initialized: fs={fs} Hz, "
                    f"6 bands + temporal + frequency + nonlinear features")

    def extract_features(self, epoch):
        """
        Extract all feature types from one epoch.

        Args:
            epoch: (n_channels, n_timepoints) or (n_timepoints,)

        Returns:
            features: (n_total_features,)
        """
        # Handle single-channel input
        if epoch.ndim == 1:
            epoch = epoch[np.newaxis, :]  # (1, n_timepoints)

        features = []

        # Type 1: Band power (most important for motor imagery)
        band_features = self._extract_band_power(epoch)
        features.append(band_features)

        # Type 2: Temporal features
        temporal_features = self._extract_temporal_features(epoch)
        features.append(temporal_features)

        # Type 3: Frequency domain features
        freq_features = self._extract_frequency_features(epoch)
        features.append(freq_features)

        # Type 4: Nonlinear features (entropy, complexity)
        nonlinear_features = self._extract_nonlinear_features(epoch)
        features.append(nonlinear_features)

        # Concatenate all
        all_features = np.concatenate(features)

        # Replace any NaN/Inf with 0
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

        return all_features

    def extract_batch(self, epochs):
        """
        Extract features from all epochs.

        Args:
            epochs: (n_epochs, n_timepoints) or (n_epochs, n_channels, n_timepoints)

        Returns:
            features: (n_epochs, n_features)
        """
        features = []

        for i, epoch in enumerate(epochs):
            feat = self.extract_features(epoch)
            features.append(feat)

            if (i + 1) % 100 == 0:
                logger.info(f"  Extracted {i + 1}/{len(epochs)} epochs")

        result = np.array(features)
        logger.info(f"Feature extraction complete: {result.shape[0]} epochs × "
                    f"{result.shape[1]} features")
        return result

    def _extract_band_power(self, epoch):
        """
        Power in 6 frequency bands.
        These are the MOST IMPORTANT features for motor imagery.
        """
        bands = {
            'delta': (0.5, 4),       # Slow waves
            'theta': (4, 8),         # Theta oscillations
            'alpha': (8, 12),        # Mu (motor cortex!)
            'lower_beta': (12, 20),  # Beta-1 (motor planning)
            'upper_beta': (20, 30),  # Beta-2
            'gamma': (30, min(50, self.fs / 2 - 1))  # High frequency
        }

        features = []

        for ch in range(epoch.shape[0]):
            # Welch PSD
            nperseg = min(256, epoch.shape[1])
            if nperseg < 4:
                features.extend([0.0] * len(bands))
                continue

            freqs, psd = signal.welch(epoch[ch], fs=self.fs, nperseg=nperseg)

            for band_name, (f_min, f_max) in bands.items():
                # Power in band
                mask = (freqs >= f_min) & (freqs <= f_max)
                if np.any(mask):
                    power = np.mean(psd[mask])
                else:
                    power = 1e-10

                # Log power (more meaningful than linear)
                log_power = np.log10(max(power, 1e-10))
                features.append(log_power)

        return np.array(features)

    def _extract_temporal_features(self, epoch):
        """
        Time domain statistics.
        9 features per channel.
        """
        features = []

        for ch in range(epoch.shape[0]):
            signal_data = epoch[ch]

            # Basic statistics
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            var_val = np.var(signal_data)
            max_val = np.max(np.abs(signal_data))
            rms_val = np.sqrt(np.mean(signal_data ** 2))

            # Higher-order moments
            skew_val = float(stats.skew(signal_data))
            kurt_val = float(stats.kurtosis(signal_data))

            # Signal energy
            energy = np.sum(signal_data ** 2) / len(signal_data)

            # Peak-to-peak
            ptp = np.ptp(signal_data)

            features.extend([mean_val, std_val, var_val, max_val, rms_val,
                             skew_val, kurt_val, energy, ptp])

        return np.array(features)

    def _extract_frequency_features(self, epoch):
        """
        Frequency domain features.
        4 features per channel.
        """
        features = []

        for ch in range(epoch.shape[0]):
            nperseg = min(256, epoch.shape[1])
            if nperseg < 4:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            freqs, psd = signal.welch(epoch[ch], fs=self.fs, nperseg=nperseg)

            # Peak frequency (most powerful frequency)
            peak_freq_idx = np.argmax(psd)
            peak_freq = freqs[peak_freq_idx]

            # Spectral centroid (center of mass in frequency domain)
            psd_sum = np.sum(psd) + 1e-10
            spec_centroid = np.sum(freqs * psd) / psd_sum

            # Spectral entropy (complexity)
            psd_norm = psd / psd_sum
            spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

            # Spectral spread
            spec_spread = np.sqrt(
                np.sum(((freqs - spec_centroid) ** 2) * psd) / psd_sum
            )

            features.extend([peak_freq, spec_centroid, spec_entropy, spec_spread])

        return np.array(features)

    def _extract_nonlinear_features(self, epoch):
        """
        Nonlinear complexity measures.
        3 features per channel.
        """
        features = []

        for ch in range(epoch.shape[0]):
            signal_data = epoch[ch]

            # Approximate entropy (simplified for speed)
            approx_ent = self._fast_approximate_entropy(signal_data)

            # Sample entropy (simplified)
            sample_ent = self._fast_sample_entropy(signal_data)

            # Hurst exponent (self-similarity)
            hurst = self._hurst_exponent(signal_data)

            features.extend([approx_ent, sample_ent, hurst])

        return np.array(features)

    @staticmethod
    def _fast_approximate_entropy(x, m=2, r_factor=0.2):
        """
        Fast approximate entropy using vectorized operations.
        Subsamples long signals for speed.
        """
        # Subsample long signals for speed
        if len(x) > 300:
            x = x[:300]

        r = r_factor * np.std(x)
        if r < 1e-10:
            return 0.0

        N = len(x)

        def _phi(m_val):
            templates = np.array([x[j:j + m_val] for j in range(N - m_val + 1)])
            n_templates = len(templates)
            if n_templates == 0:
                return 0.0

            counts = []
            for i in range(n_templates):
                dist = np.max(np.abs(templates - templates[i]), axis=1)
                count = np.sum(dist <= r) / n_templates
                counts.append(count)

            counts = np.array(counts)
            counts = counts[counts > 0]
            if len(counts) == 0:
                return 0.0
            return np.mean(np.log(counts))

        try:
            return abs(_phi(m + 1) - _phi(m))
        except Exception:
            return 0.0

    @staticmethod
    def _fast_sample_entropy(x, m=2, r_factor=0.2):
        """
        Fast sample entropy estimation.
        """
        if len(x) > 300:
            x = x[:300]

        r = r_factor * np.std(x)
        if r < 1e-10:
            return 0.0

        N = len(x)
        if N < m + 2:
            return 0.0

        try:
            templates_m = np.array([x[i:i + m] for i in range(N - m)])
            templates_m1 = np.array([x[i:i + m + 1] for i in range(N - m - 1)])

            # Count template matches for m
            n_m = len(templates_m)
            count_m = 0
            for i in range(n_m):
                dist = np.max(np.abs(templates_m - templates_m[i]), axis=1)
                count_m += np.sum(dist <= r) - 1  # Exclude self-match

            # Count template matches for m+1
            n_m1 = len(templates_m1)
            count_m1 = 0
            for i in range(n_m1):
                dist = np.max(np.abs(templates_m1 - templates_m1[i]), axis=1)
                count_m1 += np.sum(dist <= r) - 1

            if count_m == 0:
                return 0.0
            return -np.log((count_m1 + 1e-10) / (count_m + 1e-10))
        except Exception:
            return 0.0

    @staticmethod
    def _hurst_exponent(x, max_lag=None):
        """Hurst exponent (self-similarity measure)."""
        if max_lag is None:
            max_lag = min(len(x) // 2, 100)

        if max_lag <= 10:
            return 0.5  # Default

        lags = np.arange(10, max_lag)
        tau = []

        for lag in lags:
            diff = np.diff(x, lag)
            if len(diff) > 0:
                tau.append(np.sqrt(np.mean(diff ** 2)))
            else:
                tau.append(1e-10)

        tau = np.array(tau)
        tau[tau <= 0] = 1e-10

        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]  # Slope is Hurst exponent
        except Exception:
            return 0.5


# Feature names for reference
def get_feature_names(n_channels=1):
    """Get descriptive names for all features."""
    names = []
    bands = ['delta', 'theta', 'alpha', 'lower_beta', 'upper_beta', 'gamma']
    temporal = ['mean', 'std', 'var', 'max_abs', 'rms', 'skew', 'kurtosis', 'energy', 'ptp']
    freq = ['peak_freq', 'spec_centroid', 'spec_entropy', 'spec_spread']
    nonlinear = ['approx_entropy', 'sample_entropy', 'hurst']

    for ch in range(n_channels):
        prefix = f"ch{ch}_" if n_channels > 1 else ""
        names.extend([f"{prefix}bp_{b}" for b in bands])
        names.extend([f"{prefix}t_{t}" for t in temporal])
        names.extend([f"{prefix}f_{f}" for f in freq])
        names.extend([f"{prefix}nl_{n}" for n in nonlinear])

    return names


# USAGE:
if __name__ == "__main__":
    print("Testing BetterFeatureExtractor...")

    # Single-channel test
    test_epoch_1ch = np.random.randn(500)
    extractor = BetterFeatureExtractor(fs=125)
    features = extractor.extract_features(test_epoch_1ch)
    print(f"  Single channel features: {features.shape} ({len(features)} features)")

    # Batch test
    test_epochs = np.random.randn(20, 500)
    batch_features = extractor.extract_batch(test_epochs)
    print(f"  Batch features: {batch_features.shape}")

    # Print feature names
    names = get_feature_names(n_channels=1)
    print(f"  Feature names ({len(names)}):")
    for i, name in enumerate(names):
        print(f"    [{i:2d}] {name} = {features[i]:.4f}")

    print("✓ BetterFeatureExtractor ready to use")
