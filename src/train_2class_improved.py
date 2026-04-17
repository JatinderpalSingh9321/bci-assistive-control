"""
Improved 2-class (BLINK vs WINK) training with better features.

Key insight: Blinks have LARGER amplitude and WIDER peaks than winks
since both eyelids move. Extract these discriminating features explicitly.
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle

SAMPLING_RATE = 250
CLASS_NAMES = ["BLINK", "WINK"]


def extract_eog_features_v2(sig_raw, fs=SAMPLING_RATE):
    """
    Extract features specifically designed to discriminate BLINK vs WINK.
    
    Blink characteristics (vs wink):
    - Higher peak amplitude (both eyes)
    - Wider peaks (slower, more deliberate)
    - More total energy
    - More symmetric waveform
    - Lower frequency content (slower movement)
    """
    sig = np.array(sig_raw, dtype=np.float64).flatten()
    
    # Notch filter 50 Hz
    b, a = scipy_signal.iirnotch(50.0, 30, fs)
    sig = scipy_signal.filtfilt(b, a, sig)
    
    # Baseline correction
    sig = sig - np.mean(sig)
    
    features = []
    abs_sig = np.abs(sig)
    
    # ── AMPLITUDE FEATURES ──
    # 1. Peak amplitude (BLINK > WINK)
    features.append(np.max(abs_sig))
    
    # 2. 95th percentile amplitude (robust peak)
    features.append(np.percentile(abs_sig, 95))
    
    # 3. 90th percentile
    features.append(np.percentile(abs_sig, 90))
    
    # 4. Mean amplitude
    features.append(np.mean(abs_sig))
    
    # 5. Median amplitude
    features.append(np.median(abs_sig))
    
    # 6. RMS
    features.append(np.sqrt(np.mean(sig ** 2)))
    
    # 7. Peak-to-RMS ratio (signal "peakiness")
    rms = np.sqrt(np.mean(sig ** 2))
    features.append(np.max(abs_sig) / (rms + 1e-10))
    
    # 8. Variance
    features.append(np.var(sig))
    
    # ── PEAK ANALYSIS ──
    threshold = np.mean(abs_sig) + 1.5 * np.std(sig)
    peaks, props = scipy_signal.find_peaks(
        abs_sig, height=threshold, distance=int(0.15 * fs)
    )
    
    # 9. Number of peaks
    features.append(len(peaks))
    
    # 10. Max peak height
    if len(peaks) > 0:
        features.append(np.max(props["peak_heights"]))
    else:
        features.append(0)
    
    # 11. Mean peak height
    if len(peaks) > 0:
        features.append(np.mean(props["peak_heights"]))
    else:
        features.append(0)
    
    # 12. Peak height std (consistency)
    if len(peaks) > 1:
        features.append(np.std(props["peak_heights"]))
    else:
        features.append(0)
    
    # 13-14. Peak widths (BLINK = wider)
    if len(peaks) > 0:
        widths = scipy_signal.peak_widths(abs_sig, peaks)[0]
        features.append(np.mean(widths))  # mean width
        features.append(np.max(widths))   # max width
    else:
        features.append(0)
        features.append(0)
    
    # 15. Total peak area (amplitude × width)
    if len(peaks) > 0:
        widths = scipy_signal.peak_widths(abs_sig, peaks)[0]
        areas = props["peak_heights"] * widths
        features.append(np.sum(areas))
    else:
        features.append(0)
    
    # ── ENERGY / POWER FEATURES ──
    # 16. Total energy
    features.append(np.sum(sig ** 2))
    
    # 17. Energy above threshold 
    above = abs_sig[abs_sig > threshold]
    features.append(np.sum(above ** 2) if len(above) > 0 else 0)
    
    # 18. Fraction of time above threshold
    features.append(len(above) / len(sig))
    
    # 19-20. Energy in first vs second half
    mid = len(sig) // 2
    e1 = np.sum(sig[:mid] ** 2)
    e2 = np.sum(sig[mid:] ** 2)
    features.append(e1 / (e1 + e2 + 1e-10))
    features.append(e1 / (e2 + 1e-10))
    
    # ── WAVEFORM SHAPE ──
    # 21. Kurtosis (peakedness)
    features.append(kurtosis(sig))
    
    # 22. Skewness
    features.append(skew(sig))
    
    # 23. Zero-crossing rate
    zcr = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)
    features.append(zcr)
    
    # 24. Signal range (peak-to-peak)
    features.append(np.ptp(sig))
    
    # 25. Positive/negative ratio
    pos = np.max(sig) if np.max(sig) > 0 else 1e-10
    neg = np.abs(np.min(sig)) if np.min(sig) < 0 else 1e-10
    features.append(pos / (neg + 1e-10))
    
    # ── FREQUENCY FEATURES ──
    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    
    # 26-29. Band power
    bands = [(0.5, 2, "DC"), (2, 5, "low"), (5, 15, "mid"), (15, 40, "high")]
    for fmin, fmax, name in bands:
        mask = (freqs >= fmin) & (freqs <= fmax)
        features.append(np.mean(psd[mask]) if np.any(mask) else 0)
    
    # 30. Low-to-high frequency ratio (BLINK = more low freq)
    low_power = np.mean(psd[(freqs >= 0.5) & (freqs <= 5)])
    high_power = np.mean(psd[(freqs >= 15) & (freqs <= 40)])
    features.append(low_power / (high_power + 1e-10))
    
    # 31. Dominant frequency
    features.append(freqs[np.argmax(psd)])
    
    # 32. Spectral centroid
    features.append(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))
    
    # ── TEMPORAL FEATURES ──
    # 33-35. Thirds energy distribution
    third = len(sig) // 3
    for i in range(3):
        e = np.sum(sig[i*third:(i+1)*third] ** 2)
        features.append(e / (np.sum(sig**2) + 1e-10))
    
    # 36. Hjorth mobility
    d1 = np.diff(sig)
    mobility = np.std(d1) / (np.std(sig) + 1e-10)
    features.append(mobility)
    
    # 37. Hjorth complexity
    d2 = np.diff(d1)
    mob_d1 = np.std(d2) / (np.std(d1) + 1e-10)
    features.append(mob_d1 / (mobility + 1e-10))
    
    # 38. Duration above threshold (in seconds)
    above_thresh = abs_sig > threshold
    if np.any(above_thresh):
        # Find contiguous regions above threshold
        changes = np.diff(above_thresh.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            if ends[0] < starts[0]:
                ends = ends[1:]
            min_len = min(len(starts), len(ends))
            durations = (ends[:min_len] - starts[:min_len]) / fs
            features.append(np.sum(durations))
        else:
            features.append(0)
    else:
        features.append(0)
    
    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

print("=" * 60)
print("  IMPROVED BLINK vs WINK TRAINING (Feature-Based)")
print("=" * 60)

all_data, all_labels = [], []
for sess in [1, 2, 3]:
    path = f"data/raw/subject_001/session_{sess:02d}_blink_wink.npz"
    d = np.load(path, allow_pickle=True)
    all_data.extend(d["data"])
    all_labels.extend(d["labels"])
    print(f"  Session {sess}: {len(d['labels'])} trials")

labels = np.array(all_labels)
labels_2class = np.where(labels == 0, 0, 1)  # BLINK=0, WINK=1
print(f"\n  BLINK: {np.sum(labels_2class==0)}, WINK: {np.sum(labels_2class==1)}")

# ──────────────────────────────────────────────
# EXTRACT FEATURES
# ──────────────────────────────────────────────

print("\n  Extracting features...")
X = []
valid_idx = []
for i, epoch in enumerate(all_data):
    try:
        feats = extract_eog_features_v2(epoch)
        if np.all(np.isfinite(feats)):
            X.append(feats)
            valid_idx.append(i)
    except Exception as e:
        print(f"  Warning: trial {i} failed: {e}")

X = np.array(X, dtype=np.float32)
y = labels_2class[valid_idx]
print(f"  Features: {X.shape} ({X.shape[1]} features × {X.shape[0]} trials)")

# ──────────────────────────────────────────────
# TRAIN MULTIPLE MODELS
# ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=10, gamma="scale", kernel="rbf", probability=True, random_state=42)),
    ]),
    "SVM (Linear)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1, kernel="linear", probability=True, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ]),
}

print("\n" + "-" * 60)
best_name, best_acc, best_model = "", 0, None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\n  {name}:")
    print(f"    Test:  {acc:.1%}")
    print(f"    CV:    {cv_mean:.1%} ± {cv_std:.1%}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    if cv_mean > best_acc:
        best_acc = cv_mean
        best_name = name
        best_model = model

# ──────────────────────────────────────────────
# SAVE BEST MODEL
# ──────────────────────────────────────────────

print("=" * 60)
print(f"  WINNER: {best_name} ({best_acc:.1%} CV)")
print("=" * 60)

# Save
model_path = "data/models/eog_2class_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "model": best_model,
        "class_names": CLASS_NAMES,
        "n_features": X.shape[1],
        "train_accuracy": float(accuracy_score(y_test, best_model.predict(X_test))),
        "cv_accuracy": float(best_acc),
        "feature_extractor": "extract_eog_features_v2",
        "source": "train_2class_improved.py",
    }, f)
print(f"  Model saved: {model_path}")

# Feature importance (if tree-based)
if "Forest" in best_name or "Boosting" in best_name:
    clf = best_model.named_steps["clf"]
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    print("\n  Top 10 important features:")
    feature_names = [
        "peak_amp", "p95_amp", "p90_amp", "mean_amp", "median_amp",
        "rms", "peak_to_rms", "variance", "n_peaks", "max_peak_h",
        "mean_peak_h", "peak_h_std", "mean_peak_w", "max_peak_w",
        "total_peak_area", "total_energy", "energy_above_thresh",
        "frac_above_thresh", "energy_ratio_1st", "energy_ratio_12",
        "kurtosis", "skewness", "zcr", "ptp", "pos_neg_ratio",
        "psd_DC", "psd_low", "psd_mid", "psd_high", "low_high_ratio",
        "dom_freq", "spectral_centroid", "e_third1", "e_third2", "e_third3",
        "hjorth_mob", "hjorth_comp", "dur_above_thresh"
    ]
    for rank, idx in enumerate(top_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"    {rank+1}. {name}: {importances[idx]:.3f}")
