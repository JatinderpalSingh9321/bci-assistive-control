# SVM Accuracy Improvement Strategy
## Boost Your 34.5% SVM Baseline to 70%+ Without Recollecting Data

---

## 🚨 CURRENT SITUATION ANALYSIS

**Problem**: SVM accuracy is **34.5%** (barely better than random guessing = 50% for 2-class problem)

**Root Causes** (in order of likelihood):
1. **Poor feature extraction** (features not discriminative)
2. **Inadequate preprocessing** (artifacts, noise not removed)
3. **Wrong hyperparameters** (C, kernel, gamma not tuned)
4. **Data quality issues** (class imbalance, outliers)
5. **Feature scaling/normalization** (critical for SVM!)

**Good News**: We can improve this WITHOUT collecting more data by optimizing:
- ✅ Preprocessing pipeline
- ✅ Feature engineering
- ✅ Feature selection
- ✅ SVM hyperparameters
- ✅ Data augmentation techniques

---

## PHASE 1: DIAGNOSTIC ANALYSIS (Do This First!)

### Step 1.1: Analyze Current Features

Create `diagnose_features.py`:

```python
"""
Diagnose why features aren't working
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def analyze_feature_separability(X, y, feature_names=None):
    """
    Check if features can separate classes at all
    """
    
    print("\n" + "="*60)
    print("FEATURE SEPARABILITY ANALYSIS")
    print("="*60)
    
    # 1. Class balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass Distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # 2. Feature statistics
    print(f"\nFeature Statistics:")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Total samples: {X.shape[0]}")
    
    # 3. Fisher's Linear Discriminant Ratio (FLDR)
    # Higher FLDR = better class separability
    print(f"\nFisher Linear Discriminant Ratio per feature:")
    
    X0 = X[y == 0]  # Class 0 samples
    X1 = X[y == 1]  # Class 1 samples
    
    fldr_scores = []
    for i in range(X.shape[1]):
        mean0 = np.mean(X0[:, i])
        mean1 = np.mean(X1[:, i])
        var0 = np.var(X0[:, i])
        var1 = np.var(X1[:, i])
        
        # FLDR = (mean1 - mean0)^2 / (var0 + var1)
        fldr = (mean1 - mean0)**2 / (var0 + var1 + 1e-10)
        fldr_scores.append(fldr)
        
        fname = feature_names[i] if feature_names else f"Feat_{i}"
        print(f"  {fname:20} FLDR = {fldr:.4f}")
    
    fldr_scores = np.array(fldr_scores)
    print(f"\n  Average FLDR: {np.mean(fldr_scores):.4f}")
    print(f"  Median FLDR:  {np.median(fldr_scores):.4f}")
    
    if np.median(fldr_scores) < 0.1:
        print("  ⚠️  WARNING: Features have POOR separability!")
        print("  This is likely why accuracy is low.")
    elif np.median(fldr_scores) < 1.0:
        print("  ⚠️  Features have WEAK separability")
    else:
        print("  ✓ Features have reasonable separability")
    
    # 4. PCA visualization
    print(f"\nPCA Analysis (first 2 components):")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=50, alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA: Can classes be separated?')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig('feature_pca_analysis.png', dpi=150)
    print(f"  Saved to: feature_pca_analysis.png")
    
    # 5. Recommendation
    print(f"\nRECOMMENDATION:")
    if np.median(fldr_scores) < 0.1:
        print("  ❌ Current features are NOT discriminative")
        print("  → Need NEW feature extraction method")
        print("  → Try: CSP, FBCSP, Wavelet, or Spectral-Spatial features")
    else:
        print("  ✓ Features are OK, but may need tuning")
        print("  → Try: Feature selection, scaling, SVM hyperparameter tuning")
    
    return fldr_scores

# RUN THIS:
if __name__ == "__main__":
    # Load your actual data
    # X, y = load_your_training_data()
    # analyze_feature_separability(X, y)
    
    # Or use example data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
    analyze_feature_separability(X, y)
```

**Run it**:
```bash
python diagnose_features.py
```

**What to look for**:
- If median FLDR < 0.1 → Your features are trash, need new extraction
- If median FLDR > 1.0 → Your features are good, problem is SVM tuning
- If PCA shows classes overlapping → Preprocessing is removing important signals

---

## PHASE 2: FIX PREPROCESSING (Most Likely Culprit!)

### Problem: You're probably losing important signals!

### Step 2.1: Aggressive Artifact Removal

Current pipeline might be TOO aggressive. Update `preprocessing.py`:

```python
# preprocessing.py - IMPROVED VERSION

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

class ImprovedPreprocessor:
    """Better preprocessing that preserves signal quality"""
    
    def __init__(self, fs=250):
        self.fs = fs
    
    def preprocess(self, raw_eeg, verbose=False):
        """
        Improved pipeline:
        1. Remove extreme outliers (preserve 99% of data)
        2. Notch filter for 50/60 Hz only
        3. Bandpass 0.5-50 Hz (wider range)
        4. Interpolate bad channels
        5. NO ICA (too aggressive, removes motor imagery!)
        6. Baseline correct only
        """
        
        eeg = raw_eeg.copy()
        
        if verbose:
            print(f"Input shape: {eeg.shape}")
        
        # Step 1: Reject extreme outliers (beyond 5 sigma)
        eeg = self._robust_outlier_removal(eeg)
        
        # Step 2: Notch filter (50 Hz only, very narrow)
        eeg = self._notch_filter(eeg, freq=50, Q=30)
        
        # Step 3: Bandpass (0.5-50 Hz is much wider than 8-30!)
        eeg = self._bandpass_filter(eeg, l_freq=0.5, h_freq=50)
        
        # Step 4: Interpolate bad channels
        eeg = self._interpolate_bad_channels(eeg)
        
        # Step 5: Common average reference (gentler than ICA)
        eeg = self._common_average_ref(eeg)
        
        # Step 6: Baseline correction (minimal)
        eeg = self._minimal_baseline_correct(eeg)
        
        if verbose:
            print(f"Output shape: {eeg.shape}")
        
        return eeg
    
    def _robust_outlier_removal(self, eeg, threshold_sigma=5):
        """Remove only extreme outliers (5+ sigma)"""
        eeg_clean = eeg.copy()
        
        for ch in range(eeg.shape[0]):
            mean = np.mean(eeg[ch])
            std = np.std(eeg[ch])
            
            # Mark values > 5*sigma as outliers
            outlier_mask = np.abs(eeg[ch] - mean) > threshold_sigma * std
            
            if np.any(outlier_mask):
                # Linear interpolate over outliers
                x = np.arange(len(eeg[ch]))
                eeg_clean[ch][outlier_mask] = np.interp(
                    x[outlier_mask],
                    x[~outlier_mask],
                    eeg[ch][~outlier_mask]
                )
        
        return eeg_clean
    
    def _notch_filter(self, eeg, freq=50, Q=30):
        """Remove only 50 Hz (or 60 Hz), very narrow"""
        sos = signal.iirnotch(freq, Q, fs=self.fs, output='sos')
        eeg_filtered = np.zeros_like(eeg)
        
        for ch in range(eeg.shape[0]):
            eeg_filtered[ch] = signal.sosfilt(sos, eeg[ch])
        
        return eeg_filtered
    
    def _bandpass_filter(self, eeg, l_freq=0.5, h_freq=50):
        """
        Wider bandpass (0.5-50 Hz) preserves more signal
        Previously 8-30 Hz was TOO NARROW for motor imagery!
        """
        sos = signal.butter(4, [l_freq, h_freq], 'bandpass', 
                           fs=self.fs, output='sos')
        eeg_filtered = np.zeros_like(eeg)
        
        for ch in range(eeg.shape[0]):
            eeg_filtered[ch] = signal.sosfilt(sos, eeg[ch])
        
        return eeg_filtered
    
    def _interpolate_bad_channels(self, eeg):
        """Interpolate very noisy channels instead of rejecting"""
        eeg_clean = eeg.copy()
        
        # Find noisy channels (> 2x median noise)
        noise_levels = np.std(eeg, axis=1)
        median_noise = np.median(noise_levels)
        bad_chs = np.where(noise_levels > 2 * median_noise)[0]
        
        for ch in bad_chs:
            # Interpolate from neighboring channels
            neighbors = np.setdiff1d(np.arange(eeg.shape[0]), [ch])
            eeg_clean[ch] = np.mean(eeg[neighbors], axis=0)
        
        return eeg_clean
    
    def _common_average_ref(self, eeg):
        """Gentle common average reference"""
        car = np.mean(eeg, axis=0, keepdims=True)
        return eeg - car
    
    def _minimal_baseline_correct(self, eeg):
        """Minimal baseline (z-score per channel)"""
        eeg_corrected = eeg.copy()
        
        for ch in range(eeg.shape[0]):
            mean = np.mean(eeg[ch])
            std = np.std(eeg[ch])
            eeg_corrected[ch] = (eeg[ch] - mean) / (std + 1e-10)
        
        return eeg_corrected
```

### Step 2.2: Don't Use ICA!

❌ **ICA (Independent Component Analysis) often REMOVES motor imagery signals!**

ICA assumes components are statistically independent, but motor imagery is:
- Spatially correlated (multiple channels activate together)
- Not independent from background activity

✅ **Instead use**:
- Common Average Reference (CAR) ← Safer
- Surface Laplacian ← Better for motor cortex
- Spatial filtering (CSP) ← But that's a feature extraction, not preprocessing

---

## PHASE 3: BETTER FEATURE EXTRACTION (Critical!)

### Problem: Band power alone is weak

### Step 3.1: Use Multiple Feature Types

Create `better_features.py`:

```python
"""
Multiple complementary features for better discrimination
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import morlet2

class BetterFeatureExtractor:
    """Extract multiple feature types"""
    
    def __init__(self, fs=250):
        self.fs = fs
    
    def extract_all_features(self, epoch):
        """
        Extract ALL types of features:
        1. Band power (Welch)
        2. Temporal features (mean, variance, peak, slope)
        3. Frequency features (peak frequency, bandwidth)
        4. Time-frequency (wavelet energy)
        5. Non-linear (entropy, complexity)
        
        Total: ~100 features → PCA to 20
        """
        
        features = []
        
        # Type 1: Band power (most important)
        band_power = self._band_power_features(epoch)
        features.append(band_power)
        
        # Type 2: Temporal features
        temporal = self._temporal_features(epoch)
        features.append(temporal)
        
        # Type 3: Frequency domain
        freq_features = self._frequency_features(epoch)
        features.append(freq_features)
        
        # Type 4: Wavelet time-frequency
        wavelet = self._wavelet_features(epoch)
        features.append(wavelet)
        
        # Type 5: Non-linear dynamics
        nonlinear = self._nonlinear_features(epoch)
        features.append(nonlinear)
        
        # Concatenate all
        all_features = np.concatenate(features)
        
        return all_features
    
    def _band_power_features(self, epoch, bands=None):
        """
        Power in different frequency bands
        Motor imagery shows:
        - Mu (8-12 Hz): Decreases over motor cortex when imagining
        - Beta (12-30 Hz): Decreases during motor planning
        """
        
        if bands is None:
            bands = {
                'theta': (4, 8),
                'alpha': (8, 12),
                'lower_beta': (12, 20),
                'upper_beta': (20, 30),
                'gamma': (30, 50)
            }
        
        features = []
        
        for ch in range(epoch.shape[0]):
            freqs, psd = signal.welch(epoch[ch], fs=self.fs, nperseg=256)
            
            for band_name, (f_min, f_max) in bands.items():
                mask = (freqs >= f_min) & (freqs <= f_max)
                power = np.log(np.mean(psd[mask]) + 1e-10)
                features.append(power)
        
        return np.array(features)
    
    def _temporal_features(self, epoch):
        """Time domain statistics"""
        features = []
        
        for ch in range(epoch.shape[0]):
            # Basic stats
            mean = np.mean(epoch[ch])
            std = np.std(epoch[ch])
            variance = np.var(epoch[ch])
            max_val = np.max(np.abs(epoch[ch]))
            
            # Higher order moments
            skewness = stats.skew(epoch[ch])
            kurtosis = stats.kurtosis(epoch[ch])
            
            # Signal energy
            energy = np.sum(epoch[ch]**2)
            
            # Zero crossings
            zero_crossings = np.sum(np.diff(np.sign(epoch[ch])) != 0)
            
            # Peak to peak
            peak_to_peak = np.max(epoch[ch]) - np.min(epoch[ch])
            
            features.extend([mean, std, variance, max_val, skewness, 
                           kurtosis, energy, zero_crossings, peak_to_peak])
        
        return np.array(features)
    
    def _frequency_features(self, epoch):
        """Frequency domain features"""
        features = []
        
        for ch in range(epoch.shape[0]):
            freqs, psd = signal.welch(epoch[ch], fs=self.fs, nperseg=256)
            
            # Peak frequency
            peak_freq = freqs[np.argmax(psd)]
            
            # Spectral centroid (center of mass in frequency)
            spec_centroid = np.sum(freqs * psd) / np.sum(psd)
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            # Frequency spread
            freq_spread = np.sqrt(np.sum(((freqs - spec_centroid)**2) * psd) / np.sum(psd))
            
            features.extend([peak_freq, spec_centroid, spec_entropy, freq_spread])
        
        return np.array(features)
    
    def _wavelet_features(self, epoch):
        """Wavelet time-frequency energy"""
        features = []
        
        scales = np.arange(1, 128)  # 4-65 Hz range
        
        for ch in range(epoch.shape[0]):
            cwt = signal.morlet2(epoch[ch], M=scales, s=1.0)
            # Energy in each scale
            energy_per_scale = np.mean(np.abs(cwt)**2, axis=1)
            features.extend(energy_per_scale)
        
        return np.array(features)
    
    def _nonlinear_features(self, epoch):
        """Nonlinear signal properties"""
        features = []
        
        for ch in range(epoch.shape[0]):
            # Approximate entropy
            approx_entropy = self._approximate_entropy(epoch[ch], m=2, r=0.2*np.std(epoch[ch]))
            
            # Sample entropy
            sample_entropy = self._sample_entropy(epoch[ch], m=2, r=0.2*np.std(epoch[ch]))
            
            # Hurst exponent
            hurst = self._hurst_exponent(epoch[ch])
            
            features.extend([approx_entropy, sample_entropy, hurst])
        
        return np.array(features)
    
    def _approximate_entropy(self, x, m, r):
        """Approximate entropy (complexity measure)"""
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x_m = [tuple(x[j:j + m]) for j in range(len(x) - m + 1)]
            C = [len([1 for x_j in x_m if _maxdist(x_i, x_j) <= r]) / (len(x) - m + 1.0) 
                 for x_i in x_m]
            return (len(x) - m + 1.0)**(-1) * sum(np.log(C))
        
        return abs(_phi(m) - _phi(m + 1))
    
    def _sample_entropy(self, x, m, r):
        """Sample entropy"""
        N = len(x)
        templates = np.array([x[i:i + m] for i in range(N - m)])
        distances = np.sum((np.abs(templates[:, np.newaxis] - templates[np.newaxis, :]) <= r), axis=2)
        
        B = np.sum(np.sum(distances - 1)) / (2 * (N - m) * (N - m - 1))
        
        templates_m1 = np.array([x[i:i + m + 1] for i in range(N - m)])
        distances_m1 = np.sum((np.abs(templates_m1[:, np.newaxis] - templates_m1[np.newaxis, :]) <= r), axis=2)
        
        A = np.sum(np.sum(distances_m1 - 1)) / (2 * (N - m - 1) * (N - m - 2))
        
        return -np.log((A + 1e-10) / (B + 1e-10))
    
    def _hurst_exponent(self, x):
        """Hurst exponent (self-similarity measure)"""
        lags = range(10, min(len(x)//2, 100))
        tau = []
        
        for lag in lags:
            tau.append(np.sqrt(np.mean(np.diff(x, lag)**2)))
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]  # Slope is Hurst exponent
```

---

## PHASE 4: AGGRESSIVE SVM HYPERPARAMETER TUNING

### Step 4.1: Grid Search with Cross-Validation

```python
"""
Aggressive hyperparameter tuning for SVM
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def tune_svm_aggressively(X_train, y_train):
    """
    Find BEST SVM hyperparameters
    """
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
        'class_weight': [None, 'balanced']
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    
    # Grid search
    print("Running aggressive grid search (this takes ~1 hour)...")
    
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=2
    )
    
    grid.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    
    return grid.best_estimator_
```

### Step 4.2: Scale Features FIRST!

⚠️ **SVM is VERY sensitive to feature scaling!**

```python
from sklearn.preprocessing import StandardScaler

# MUST do this BEFORE training SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Then train SVM on X_scaled
```

---

## PHASE 5: FEATURE SELECTION (Remove Noise)

### Step 5.1: Keep only discriminative features

```python
"""
Remove features that don't help
"""

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC

def select_best_features(X_train, y_train, X_test, y_test, k=20):
    """
    Keep only K best features
    """
    
    # Method 1: ANOVA F-score
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    print(f"Selected feature indices: {selected_indices}")
    
    # Method 2: RFE (Recursive Feature Elimination)
    estimator = SVC(kernel='rbf', C=1)
    selector_rfe = RFE(estimator, n_features_to_select=k)
    X_train_selected_rfe = selector_rfe.fit_transform(X_train, y_train)
    X_test_selected_rfe = selector_rfe.transform(X_test)
    
    print(f"\nUsing SelectKBest - top {k} features:")
    print(f"  Train shape: {X_train_selected.shape}")
    print(f"  Test shape: {X_test_selected.shape}")
    
    return X_train_selected, X_test_selected
```

---

## PHASE 6: DATA AUGMENTATION (Synthetic Data)

### Step 6.1: Create synthetic samples

If you have LIMITED data, create synthetic training samples:

```python
"""
Generate synthetic training data without recollecting
"""

import numpy as np
from scipy.interpolate import interp1d

class SyntheticEEGGenerator:
    """
    Create synthetic EEG by:
    1. Mixing real epochs
    2. Adding controlled noise
    3. Time-warping
    4. Adding subject-specific patterns
    """
    
    @staticmethod
    def mixup_augmentation(X, y, alpha=0.2, n_new_samples=1000):
        """
        Mixup: X_new = lambda * X1 + (1 - lambda) * X2
        Creates smooth transitions between real samples
        """
        
        X_augmented = []
        y_augmented = []
        
        for _ in range(n_new_samples):
            # Pick two random samples from same class
            class_label = np.random.choice(np.unique(y))
            class_indices = np.where(y == class_label)[0]
            
            idx1, idx2 = np.random.choice(class_indices, 2, replace=False)
            
            # Mixup
            lambda_val = np.random.beta(alpha, alpha)
            x_new = lambda_val * X[idx1] + (1 - lambda_val) * X[idx2]
            
            X_augmented.append(x_new)
            y_augmented.append(class_label)
        
        return np.array(X_augmented), np.array(y_augmented)
    
    @staticmethod
    def jitter_augmentation(X, y, noise_std=0.01, n_copies=5):
        """
        Add small random noise to each sample
        """
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(n_copies):
            noise = np.random.normal(0, noise_std, X.shape)
            X_new = X + noise
            
            X_augmented.append(X_new)
            y_augmented.append(y)
        
        return np.concatenate(X_augmented), np.concatenate(y_augmented)
    
    @staticmethod
    def time_warp(X, y, warp_factor=0.1, n_warps=5):
        """
        Stretch/compress time axis (simulate different movement speeds)
        """
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(n_warps):
            X_warped = []
            
            for sample in X:
                # Random stretch/compress
                factor = 1 + np.random.uniform(-warp_factor, warp_factor)
                
                # Resample
                orig_indices = np.linspace(0, len(sample) - 1, len(sample))
                new_indices = np.linspace(0, len(sample) - 1, int(len(sample) * factor))
                new_indices = np.clip(new_indices, 0, len(sample) - 1)
                
                interp_func = interp1d(orig_indices, sample, kind='cubic')
                warped = interp_func(new_indices)
                
                # Pad/truncate to original length
                if len(warped) > len(sample):
                    warped = warped[:len(sample)]
                else:
                    warped = np.pad(warped, (0, len(sample) - len(warped)))
                
                X_warped.append(warped)
            
            X_augmented.append(np.array(X_warped))
            y_augmented.append(y)
        
        return np.concatenate(X_augmented), np.concatenate(y_augmented)
```

---

## 📋 IMPLEMENTATION CHECKLIST

Follow these steps IN ORDER:

### Week 1: Diagnosis & Preprocessing
- [ ] Run `diagnose_features.py` → Check feature separability
- [ ] Update preprocessing with improved version
- [ ] Remove ICA, use gentler CAR only
- [ ] Test with wider bandpass (0.5-50 Hz)
- [ ] Expected result: No major change yet, but cleaner signals

### Week 2: Feature Engineering
- [ ] Implement `BetterFeatureExtractor` with 5 feature types
- [ ] Extract 100+ features
- [ ] Use PCA to reduce to 20 features
- [ ] Expected result: Features should have better FLDR scores

### Week 3: SVM Tuning
- [ ] Implement aggressive grid search
- [ ] Test all kernel types (rbf, poly, sigmoid, linear)
- [ ] Test class weight balancing
- [ ] Use 5-fold cross-validation
- [ ] Expected result: SVM accuracy → **60-70%**

### Week 4: Feature Selection + Augmentation
- [ ] Use SelectKBest to keep top 15 features
- [ ] Try RFE as alternative
- [ ] Generate synthetic data with mixup & jitter
- [ ] Retrain with augmented data
- [ ] Expected result: SVM accuracy → **70-80%**

### Week 5: Ensemble & Fine-tuning
- [ ] Train multiple SVM models (different seeds)
- [ ] Combine with voting
- [ ] Try soft margins (probability calibration)

---

## 🎯 EXPECTED IMPROVEMENT TRAJECTORY

```
Current:     34.5%  ← Random-ish performance
↓
After preprocessing fix:    50-55%
↓
After better features:      60-70%
↓
After SVM tuning:           70-75%
↓
After augmentation:         75-85%
↓
Final (with ensemble):      80-87% ← Your goal!
```

---

## ⚡ QUICK WINS (Try These First!)

1. **Just scale features** (StandardScaler)
   - Often gives +5-10% improvement immediately

2. **Use RBF kernel instead of linear**
   - Better for complex, non-linear patterns

3. **Increase C parameter** (less regularization)
   - Default C=1 might be too strict
   - Try C=10, C=100

4. **Use balanced class weights**
   - `class_weight='balanced'` in SVC()

5. **Wider bandpass filter**
   - 0.5-50 Hz instead of 8-30 Hz
   - Preserves more motor imagery signal

---

## 📧 NEXT STEPS

1. **Run diagnostic script** → Tell me feature FLDR scores
2. **Apply preprocessing improvements** → Should preserve more signal
3. **Implement better features** → Use all 5 feature types
4. **Aggressive grid search** → Let me know best C and kernel
5. **Try augmentation** → Increase training data 10x

**Expected result: 70-85% SVM accuracy WITHOUT recollecting data!**

Are you ready to implement these improvements? Start with diagnostic analysis!
