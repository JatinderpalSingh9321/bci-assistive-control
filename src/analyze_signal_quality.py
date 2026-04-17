"""
EEG Signal Quality Analyzer
Diagnoses whether the raw EEG data contains discriminative
motor imagery patterns (LEFT vs RIGHT).
"""

import numpy as np
from scipy import signal as sp_signal
import json
from pathlib import Path

def analyze():
    print("=" * 60)
    print("  EEG DATA QUALITY ANALYSIS")
    print("=" * 60)

    for sid in [1, 2, 3, 4]:
        data_path = Path(f"data/raw/subject_{sid:03d}/session_01.npz")
        if not data_path.exists():
            continue

        raw = np.load(data_path, allow_pickle=True)
        trials = raw["data"]
        labels = raw["labels"]

        print(f"\nSubject {sid}:")
        print(f"  Trials: {len(trials)} (LEFT={sum(labels==0)}, RIGHT={sum(labels==1)})")

        # Separate by class
        left_trials = [np.array(t, dtype=np.float64) for t, l in zip(trials, labels) if l == 0]
        right_trials = [np.array(t, dtype=np.float64) for t, l in zip(trials, labels) if l == 1]

        # Mean amplitude stats
        left_means = [np.mean(t) for t in left_trials]
        right_means = [np.mean(t) for t in right_trials]
        left_stds = [np.std(t) for t in left_trials]
        right_stds = [np.std(t) for t in right_trials]

        print(f"  LEFT  mean: {np.mean(left_means):.3f} +/- {np.std(left_means):.3f}, std: {np.mean(left_stds):.3f}")
        print(f"  RIGHT mean: {np.mean(right_means):.3f} +/- {np.std(right_means):.3f}, std: {np.mean(right_stds):.3f}")

        # Check band power differences
        left_alpha, right_alpha = [], []
        left_beta, right_beta = [], []
        left_delta, right_delta = [], []
        left_theta, right_theta = [], []

        for t in left_trials[:20]:
            f, psd = sp_signal.welch(t, fs=250, nperseg=min(256, len(t)))
            left_delta.append(np.mean(psd[(f >= 0.5) & (f <= 4)]))
            left_theta.append(np.mean(psd[(f >= 4) & (f <= 8)]))
            left_alpha.append(np.mean(psd[(f >= 8) & (f <= 12)]))
            left_beta.append(np.mean(psd[(f >= 12) & (f <= 30)]))

        for t in right_trials[:20]:
            f, psd = sp_signal.welch(t, fs=250, nperseg=min(256, len(t)))
            right_delta.append(np.mean(psd[(f >= 0.5) & (f <= 4)]))
            right_theta.append(np.mean(psd[(f >= 4) & (f <= 8)]))
            right_alpha.append(np.mean(psd[(f >= 8) & (f <= 12)]))
            right_beta.append(np.mean(psd[(f >= 12) & (f <= 30)]))

        print(f"  Band Power Analysis (LEFT vs RIGHT):")
        print(f"    Delta  (0.5-4 Hz):  L={np.mean(left_delta):.4f}  R={np.mean(right_delta):.4f}  ratio={np.mean(left_delta)/max(np.mean(right_delta),1e-10):.3f}")
        print(f"    Theta  (4-8 Hz):    L={np.mean(left_theta):.4f}  R={np.mean(right_theta):.4f}  ratio={np.mean(left_theta)/max(np.mean(right_theta),1e-10):.3f}")
        print(f"    Alpha  (8-12 Hz):   L={np.mean(left_alpha):.4f}  R={np.mean(right_alpha):.4f}  ratio={np.mean(left_alpha)/max(np.mean(right_alpha),1e-10):.3f}")
        print(f"    Beta   (12-30 Hz):  L={np.mean(left_beta):.4f}  R={np.mean(right_beta):.4f}  ratio={np.mean(left_beta)/max(np.mean(right_beta),1e-10):.3f}")

        # T-test for significance
        from scipy.stats import ttest_ind
        t_alpha, p_alpha = ttest_ind(left_alpha, right_alpha)
        t_beta, p_beta = ttest_ind(left_beta, right_beta)
        print(f"  Statistical Test (t-test):")
        print(f"    Alpha band: t={t_alpha:.3f}, p={p_alpha:.4f} {'*** SIGNIFICANT' if p_alpha < 0.05 else '(not significant)'}")
        print(f"    Beta  band: t={t_beta:.3f},  p={p_beta:.4f} {'*** SIGNIFICANT' if p_beta < 0.05 else '(not significant)'}")

        # SNR
        trial_snrs = []
        for t in trials[:20]:
            t = np.array(t, dtype=np.float64)
            f, psd = sp_signal.welch(t, fs=250, nperseg=min(256, len(t)))
            sig = (f >= 8) & (f <= 30)
            noise = (f >= 50) & (f <= 60)
            if np.any(noise) and np.any(sig):
                snr = np.mean(psd[sig]) / max(np.mean(psd[noise]), 1e-10)
                trial_snrs.append(snr)
        if trial_snrs:
            snr_db = 10 * np.log10(np.mean(trial_snrs))
            print(f"  Signal/Noise (8-30 Hz vs 50-60 Hz): {np.mean(trial_snrs):.2f} ({snr_db:.1f} dB)")

        # Check metadata
        meta_path = Path(f"data/raw/subject_{sid:03d}/session_01_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            electrode = meta.get("electrode_placement", "unknown")
            snr_meta = meta.get("signal_snr_db", "N/A")
            notes = meta.get("notes", "none")
            print(f"  Electrode: {electrode}")
            print(f"  SNR from meta: {snr_meta} dB")
            if notes and notes != "none":
                print(f"  Notes: {notes}")

    # Diagnosis
    print()
    print("=" * 60)
    print("  DIAGNOSIS")
    print("=" * 60)
    print()
    print("  If LEFT/RIGHT band power ratios are close to 1.0 and")
    print("  p-values are > 0.05, the EEG signals are NOT differentiable")
    print("  between classes. Possible causes:")
    print()
    print("  1. Single electrode (C3/C4 bipolar) lacks spatial resolution")
    print("     -> Motor imagery needs BOTH C3 and C4 to detect lateralized ERD")
    print("  2. Subject wasn't performing motor imagery correctly")
    print("  3. Electrode contact was poor during recording")
    print("  4. Data was generated/simulated without class-specific patterns")
    print()
    print("  RECOMMENDATION: Check if data was collected with proper")
    print("  motor imagery paradigm and electrode placement.")


if __name__ == "__main__":
    analyze()
