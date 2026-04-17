"""
BCI Assistive Control -- Comprehensive Thesis Analysis Generator
=================================================================
Generates all analysis results, figures, and data needed for thesis.
Runs in one shot: signal quality, pipeline validation, comparisons.

Usage:
  python -m src.thesis_analysis

Group No. 7 | 8th Semester Major Project
"""

import numpy as np
import json
import time
from pathlib import Path
from scipy import signal as sp_signal
from scipy.stats import ttest_ind

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    RAW_DATA_DIR, PREPROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    SAMPLING_RATE, TARGET_SAMPLING_RATE, DOWNSAMPLE_FACTOR,
    CLASS_NAMES, EPOCH_SAMPLES,
    get_all_subjects, setup_logger, print_banner
)

logger = setup_logger("thesis_analysis")

# Make results directory for thesis figures
THESIS_DIR = RESULTS_DIR / "thesis_figures"
THESIS_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================== #
#  1. SIGNAL QUALITY ANALYSIS                                         #
# ================================================================== #

def analyze_signal_quality():
    """Full signal quality analysis across all subjects."""
    print("\n" + "=" * 60)
    print("  SECTION 1: SIGNAL QUALITY ANALYSIS")
    print("=" * 60)

    subjects = get_all_subjects()
    all_results = {}

    for sid in subjects:
        data_path = RAW_DATA_DIR / f"subject_{sid:03d}" / "session_01.npz"
        if not data_path.exists():
            continue

        raw = np.load(data_path, allow_pickle=True)
        trials = raw["data"]
        labels = raw["labels"]

        left = [np.array(t, dtype=np.float64) for t, l in zip(trials, labels) if l == 0]
        right = [np.array(t, dtype=np.float64) for t, l in zip(trials, labels) if l == 1]

        bands = {
            "Delta (0.5-4 Hz)": (0.5, 4),
            "Theta (4-8 Hz)": (4, 8),
            "Alpha/Mu (8-12 Hz)": (8, 12),
            "Beta (12-30 Hz)": (12, 30),
            "Gamma (30-50 Hz)": (30, 50),
        }

        band_results = {}
        for band_name, (lo, hi) in bands.items():
            left_powers = []
            right_powers = []
            for t in left:
                f, psd = sp_signal.welch(t, fs=SAMPLING_RATE, nperseg=min(256, len(t)))
                mask = (f >= lo) & (f <= hi)
                left_powers.append(np.mean(psd[mask]) if np.any(mask) else 0)
            for t in right:
                f, psd = sp_signal.welch(t, fs=SAMPLING_RATE, nperseg=min(256, len(t)))
                mask = (f >= lo) & (f <= hi)
                right_powers.append(np.mean(psd[mask]) if np.any(mask) else 0)

            t_stat, p_val = ttest_ind(left_powers, right_powers)
            ratio = np.mean(left_powers) / max(np.mean(right_powers), 1e-10)

            band_results[band_name] = {
                "left_mean": float(np.mean(left_powers)),
                "right_mean": float(np.mean(right_powers)),
                "ratio": float(ratio),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            }

        # SNR calculation
        snrs = []
        for t in trials:
            t = np.array(t, dtype=np.float64)
            f, psd = sp_signal.welch(t, fs=SAMPLING_RATE, nperseg=min(256, len(t)))
            sig = (f >= 8) & (f <= 30)
            noise = (f >= 50) & (f <= 60)
            if np.any(noise) and np.any(sig):
                s = np.mean(psd[sig])
                n = np.mean(psd[noise])
                if n > 0:
                    snrs.append(s / n)

        snr_linear = float(np.mean(snrs)) if snrs else 0
        snr_db = float(10 * np.log10(max(snr_linear, 1e-10)))

        all_results[sid] = {
            "n_trials": len(trials),
            "n_left": int(sum(labels == 0)),
            "n_right": int(sum(labels == 1)),
            "band_analysis": band_results,
            "snr_linear": snr_linear,
            "snr_db": snr_db,
            "signal_mean": float(np.mean([np.mean(np.array(t, dtype=np.float64)) for t in trials])),
            "signal_std": float(np.mean([np.std(np.array(t, dtype=np.float64)) for t in trials])),
        }

        print(f"\n  Subject {sid}: {len(trials)} trials, SNR={snr_db:.1f} dB")
        for bname, br in band_results.items():
            sig_mark = "***" if br["significant"] else "   "
            print(f"    {bname:25s} L/R={br['ratio']:.3f}  p={br['p_value']:.4f} {sig_mark}")

    # Save results as JSON for thesis
    results_path = THESIS_DIR / "signal_quality_analysis.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    return all_results


# ================================================================== #
#  2. GENERATE THESIS FIGURES                                         #
# ================================================================== #

def plot_signal_quality_figures(sq_results):
    """Generate all signal quality figures for thesis."""
    print("\n" + "=" * 60)
    print("  SECTION 2: GENERATING THESIS FIGURES")
    print("=" * 60)

    subjects = sorted(sq_results.keys())

    # --- Figure 1: SNR Comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    snr_values = [sq_results[s]["snr_db"] for s in subjects]
    colors = ["#e94560" if v < 3 else "#f5a623" if v < 10 else "#00d2ff" for v in snr_values]
    bars = ax.bar([f"S{s}" for s in subjects], snr_values, color=colors, edgecolor="white")
    ax.axhline(y=10, color="green", linestyle="--", alpha=0.7, label="Good SNR threshold (10 dB)")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Noise floor (0 dB)")
    ax.set_ylabel("SNR (dB)", fontsize=12)
    ax.set_title("Signal-to-Noise Ratio per Subject\n(8-30 Hz signal vs 50-60 Hz noise)", fontsize=13)
    ax.legend(fontsize=10)
    for bar, v in zip(bars, snr_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f} dB", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(min(snr_values) - 3, max(max(snr_values) + 3, 12))
    plt.tight_layout()
    path = THESIS_DIR / "fig_snr_per_subject.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 2: Band Power Left vs Right (grouped bar) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, sid in enumerate(subjects[:4]):
        ax = axes[idx]
        bands = sq_results[sid]["band_analysis"]
        names = list(bands.keys())
        short_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"][:len(names)]
        left_vals = [bands[n]["left_mean"] for n in names]
        right_vals = [bands[n]["right_mean"] for n in names]
        p_vals = [bands[n]["p_value"] for n in names]

        x = np.arange(len(short_names))
        w = 0.35
        ax.bar(x - w/2, left_vals, w, label="LEFT", color="#4a9eff", edgecolor="white")
        ax.bar(x + w/2, right_vals, w, label="RIGHT", color="#ff6b6b", edgecolor="white")

        for i, p in enumerate(p_vals):
            marker = "*" if p < 0.05 else "ns"
            max_v = max(left_vals[i], right_vals[i])
            ax.text(i, max_v * 1.05, f"p={p:.3f}\n({marker})",
                    ha="center", fontsize=8, style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=9)
        ax.set_ylabel("Mean Power", fontsize=10)
        ax.set_title(f"Subject {sid} (SNR: {sq_results[sid]['snr_db']:.1f} dB)", fontsize=11)
        ax.legend(fontsize=9)

    plt.suptitle("Band Power: LEFT vs RIGHT Motor Imagery\n(p-values from independent t-test, ns = not significant)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = THESIS_DIR / "fig_band_power_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 3: P-value Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 5))
    band_names_short = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    p_matrix = []
    for sid in subjects:
        bands = sq_results[sid]["band_analysis"]
        bnames = list(bands.keys())
        row = [bands[n]["p_value"] for n in bnames[:5]]
        p_matrix.append(row)

    p_matrix = np.array(p_matrix)
    sns.heatmap(p_matrix, annot=True, fmt=".3f", cmap="RdYlGn_r",
                xticklabels=band_names_short[:p_matrix.shape[1]],
                yticklabels=[f"Subject {s}" for s in subjects],
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={"label": "p-value"})
    ax.set_title("Statistical Significance of LEFT vs RIGHT Differences\n"
                 "(p < 0.05 = significant, green = bad/high p-value)", fontsize=13)
    ax.axhline(y=0, color="black", linewidth=2)
    plt.tight_layout()
    path = THESIS_DIR / "fig_pvalue_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 4: PSD Comparison (example subject) ---
    best_sid = subjects[0]
    data_path = RAW_DATA_DIR / f"subject_{best_sid:03d}" / "session_01.npz"
    raw = np.load(data_path, allow_pickle=True)
    trials = raw["data"]
    labels = raw["labels"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Average PSD for LEFT
    left_psds = []
    right_psds = []
    for t, l in zip(trials, labels):
        t = np.array(t, dtype=np.float64)
        f, psd = sp_signal.welch(t, fs=SAMPLING_RATE, nperseg=min(256, len(t)))
        if l == 0:
            left_psds.append(psd)
        else:
            right_psds.append(psd)

    # Truncate to same length
    min_len = min(min(len(p) for p in left_psds), min(len(p) for p in right_psds))
    left_avg = np.mean([p[:min_len] for p in left_psds], axis=0)
    right_avg = np.mean([p[:min_len] for p in right_psds], axis=0)
    f = f[:min_len]

    ax.semilogy(f, left_avg, color="#4a9eff", linewidth=2, label="LEFT imagery", alpha=0.9)
    ax.semilogy(f, right_avg, color="#ff6b6b", linewidth=2, label="RIGHT imagery", alpha=0.9)

    # Shade frequency bands
    ax.axvspan(8, 12, alpha=0.15, color="green", label="Alpha/Mu band (8-12 Hz)")
    ax.axvspan(12, 30, alpha=0.1, color="orange", label="Beta band (12-30 Hz)")

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density (log)", fontsize=12)
    ax.set_title(f"Average PSD: LEFT vs RIGHT Motor Imagery (Subject {best_sid})\n"
                 f"Note: Overlapping curves indicate no discriminative difference", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 60)
    plt.tight_layout()
    path = THESIS_DIR / "fig_psd_left_vs_right.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
#  3. ML PIPELINE VALIDATION                                          #
# ================================================================== #

def validate_ml_pipeline():
    """Validate pipeline on simulated + real data."""
    print("\n" + "=" * 60)
    print("  SECTION 3: ML PIPELINE VALIDATION")
    print("=" * 60)

    from src.improved_preprocessing import ImprovedEEGPreprocessor
    from src.better_feature_extraction import BetterFeatureExtractor
    from src.improved_svm_trainer import ImprovedSVMTrainer
    from src.feature_extraction import extract_features_batch
    from src.models import build_svm
    from sklearn.metrics import accuracy_score

    results = {}

    # --- 3A: Simulated data (should be ~100%) ---
    print("\n  3A: Simulated Data Validation")
    print("  " + "-" * 50)

    from src.acquisition import generate_simulated_dataset
    X_sim, y_sim, sids_sim = generate_simulated_dataset()

    # Old pipeline on simulated
    unique_subs = np.unique(sids_sim)
    old_preds, old_true = [], []
    for sub in unique_subs:
        te = sids_sim == sub
        tr = ~te
        feat_tr = extract_features_batch(X_sim[tr])
        feat_te = extract_features_batch(X_sim[te])
        model = build_svm()
        model.fit(feat_tr, y_sim[tr])
        old_preds.extend(model.predict(feat_te))
        old_true.extend(y_sim[te])
    sim_old_acc = accuracy_score(old_true, old_preds)

    # New pipeline on simulated
    preprocessor = ImprovedEEGPreprocessor(fs=SAMPLING_RATE, verbose=False)
    X_sim_clean = preprocessor.preprocess_single_channel(X_sim)
    extractor = BetterFeatureExtractor(fs=TARGET_SAMPLING_RATE)
    feat_sim = extractor.extract_batch(X_sim_clean)

    new_preds, new_true = [], []
    for sub in unique_subs:
        te = sids_sim == sub
        tr = ~te
        trainer = ImprovedSVMTrainer(verbose=False)
        trainer.train_simple(feat_sim[tr], y_sim[tr])
        preds, _ = trainer.predict(feat_sim[te])
        new_preds.extend(preds)
        new_true.extend(y_sim[te])
    sim_new_acc = accuracy_score(new_true, new_preds)

    print(f"    Old pipeline: {sim_old_acc:.1%}")
    print(f"    New pipeline: {sim_new_acc:.1%}")
    results["simulated_old"] = round(sim_old_acc, 4)
    results["simulated_new"] = round(sim_new_acc, 4)

    # --- 3B: Real data ---
    print("\n  3B: Real Data Validation")
    print("  " + "-" * 50)

    # Old pipeline on real preprocessed data
    pre_path = PREPROCESSED_DIR / "all_clean_epochs.npz"
    if pre_path.exists():
        loaded = np.load(pre_path)
        X_real, y_real, sids_real = loaded["X"], loaded["y"], loaded["subject_ids"]

        old_preds, old_true = [], []
        for sub in np.unique(sids_real):
            te = sids_real == sub
            tr = ~te
            feat_tr = extract_features_batch(X_real[tr])
            feat_te = extract_features_batch(X_real[te])
            model = build_svm()
            model.fit(feat_tr, y_real[tr])
            old_preds.extend(model.predict(feat_te))
            old_true.extend(y_real[te])
        real_old_acc = accuracy_score(old_true, old_preds)

        # New pipeline from raw
        subjects = get_all_subjects()
        all_trials, all_labels, all_sids = [], [], []
        for sid in subjects:
            dp = RAW_DATA_DIR / f"subject_{sid:03d}" / "session_01.npz"
            if not dp.exists():
                continue
            raw = np.load(dp, allow_pickle=True)
            for t, l in zip(raw["data"], raw["labels"]):
                all_trials.append(np.array(t, dtype=np.float64))
                all_labels.append(int(l))
                all_sids.append(sid)

        y_raw = np.array(all_labels, dtype=np.int32)
        sids_raw = np.array(all_sids, dtype=np.int32)

        # Re-preprocess from raw
        from src.train_improved import preprocess_raw_trials
        X_new_real = preprocess_raw_trials(all_trials)
        feat_real = extractor.extract_batch(X_new_real)

        new_preds, new_true = [], []
        for sub in np.unique(sids_raw):
            te = sids_raw == sub
            tr = ~te
            if sum(tr) < 10 or sum(te) < 2:
                continue
            trainer = ImprovedSVMTrainer(verbose=False)
            trainer.train_simple(feat_real[tr], y_raw[tr])
            preds, _ = trainer.predict(feat_real[te])
            new_preds.extend(preds)
            new_true.extend(y_raw[te])
        real_new_acc = accuracy_score(new_true, new_preds)

        print(f"    Old pipeline: {real_old_acc:.1%}")
        print(f"    New pipeline: {real_new_acc:.1%}")
        results["real_old"] = round(real_old_acc, 4)
        results["real_new"] = round(real_new_acc, 4)
    else:
        print("    No preprocessed data available")

    # Save results
    results_path = THESIS_DIR / "pipeline_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    return results


def plot_pipeline_comparison(val_results):
    """Generate pipeline comparison figure."""
    print("\n  Generating pipeline comparison figure...")

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = []
    old_vals = []
    new_vals = []

    if "simulated_old" in val_results:
        categories.append("Simulated Data\n(clean signals)")
        old_vals.append(val_results["simulated_old"] * 100)
        new_vals.append(val_results["simulated_new"] * 100)

    if "real_old" in val_results:
        categories.append("Real EEG Data\n(hardware-limited)")
        old_vals.append(val_results["real_old"] * 100)
        new_vals.append(val_results["real_new"] * 100)

    x = np.arange(len(categories))
    w = 0.3

    bars1 = ax.bar(x - w/2, old_vals, w, label="Old Pipeline\n(8-30 Hz, 5 features, default SVM)",
                   color="#ff6b6b", edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + w/2, new_vals, w, label="Improved Pipeline\n(0.5-50 Hz, 22 features, tuned SVM)",
                   color="#4ecdc4", edgecolor="white", linewidth=1.5)

    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5, label="Chance level (50%)")

    for bar, v in zip(bars1, old_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    for bar, v in zip(bars2, new_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("LOSO Accuracy (%)", fontsize=12)
    ax.set_title("ML Pipeline Performance: Old vs Improved\n"
                 "Both pipelines achieve chance-level on real data due to non-discriminative signals",
                 fontsize=13)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    path = THESIS_DIR / "fig_pipeline_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
#  4. HARDWARE DIAGNOSIS FIGURE                                        #
# ================================================================== #

def plot_hardware_diagnosis():
    """Generate hardware limitation explanation figure."""
    print("\n" + "=" * 60)
    print("  SECTION 4: HARDWARE DIAGNOSIS FIGURE")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Current setup (bipolar = signals cancel)
    ax1 = axes[0]
    t = np.linspace(0, 2, 500)

    # Simulated C3 signal (LEFT imagery = normal alpha at C3)
    c3_left = np.sin(2 * np.pi * 10 * t) * 1.0 + np.random.normal(0, 0.3, len(t))
    # Simulated C4 signal (LEFT imagery = suppressed alpha at C4 = ERD)
    c4_left = np.sin(2 * np.pi * 10 * t) * 0.3 + np.random.normal(0, 0.3, len(t))
    # Bipolar = C3 - C4
    bipolar_left = c3_left - c4_left

    c3_right = np.sin(2 * np.pi * 10 * t) * 0.3 + np.random.normal(0, 0.3, len(t))
    c4_right = np.sin(2 * np.pi * 10 * t) * 1.0 + np.random.normal(0, 0.3, len(t))
    bipolar_right = c3_right - c4_right

    ax1.plot(t[:200], bipolar_left[:200] + 3, "b-", linewidth=1, label="LEFT imagery")
    ax1.plot(t[:200], bipolar_right[:200], "r-", linewidth=1, label="RIGHT imagery")
    ax1.set_title("Current Setup: Single Bipolar (C3-C4)\nSignals look similar -- hard to classify", fontsize=12)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(fontsize=10)
    ax1.text(0.5, -0.15, "PROBLEM: Lateralized ERD differences cancel\n"
             "in bipolar derivation", transform=ax1.transAxes,
             ha="center", fontsize=10, style="italic", color="red")

    # Right: Ideal setup (independent channels)
    ax2 = axes[1]

    ax2.plot(t[:200], c3_left[:200] + 6, "b-", linewidth=1, label="LEFT: C3 (normal)")
    ax2.plot(t[:200], c4_left[:200] + 3, "b--", linewidth=1, label="LEFT: C4 (suppressed)")
    ax2.plot(t[:200], c3_right[:200], "r--", linewidth=1, label="RIGHT: C3 (suppressed)")
    ax2.plot(t[:200], c4_right[:200] - 3, "r-", linewidth=1, label="RIGHT: C4 (normal)")

    ax2.set_title("Ideal Setup: Independent C3 & C4\nClear lateralized differences visible", fontsize=12)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.text(0.5, -0.15, "SOLUTION: Separate channels preserve\n"
             "lateralized ERD/ERS patterns", transform=ax2.transAxes,
             ha="center", fontsize=10, style="italic", color="green")

    plt.suptitle("Hardware Setup: Why Single Bipolar Channel Limits Motor Imagery Classification",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = THESIS_DIR / "fig_hardware_diagnosis.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
#  5. IMPROVEMENTS SUMMARY TABLE                                       #
# ================================================================== #

def generate_improvements_summary():
    """Generate improvements summary table figure."""
    print("\n" + "=" * 60)
    print("  SECTION 5: IMPROVEMENTS SUMMARY TABLE")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    table_data = [
        ["Component", "Old Pipeline", "Improved Pipeline", "Impact"],
        ["Bandpass Filter", "8-30 Hz (narrow)", "0.5-50 Hz (wide)", "Captures all motor-relevant bands"],
        ["Artifact Removal", "ICA (aggressive)", "CAR + 5-sigma outliers", "Preserves motor imagery signals"],
        ["Features", "5 (band power only)", "22 (5 types)", "4x more discriminative information"],
        ["Feature Scaling", "None", "StandardScaler", "Critical for SVM performance"],
        ["Feature Selection", "None", "SelectKBest (top 20)", "Removes noise features"],
        ["Dim. Reduction", "None", "PCA (15 components)", "Optimal feature space"],
        ["SVM Tuning", "Default (C=1, rbf)", "GridSearchCV (100 combos)", "Finds optimal hyperparameters"],
        ["Data Augmentation", "Noise + shift", "Mixup + jitter + time-warp", "3 complementary strategies"],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(4):
            table[i, j].set_facecolor(color)

    ax.set_title("Table: ML Pipeline Improvements Summary", fontsize=14,
                 fontweight="bold", pad=20)
    plt.tight_layout()

    path = THESIS_DIR / "fig_improvements_table.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
#  6. ELECTRODE PLACEMENT DIAGRAM                                      #
# ================================================================== #

def plot_electrode_explanation():
    """Generate electrode placement explanation."""
    print("\n" + "=" * 60)
    print("  SECTION 6: ELECTRODE PLACEMENT FIGURE")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Head outline helper
    def draw_head(ax):
        circle = plt.Circle((0.5, 0.45), 0.35, fill=False, color="black", linewidth=2)
        ax.add_patch(circle)
        # Nose
        ax.plot([0.5, 0.47, 0.53, 0.5], [0.8, 0.82, 0.82, 0.8], "k-", linewidth=2)
        # Ears
        ax.plot([0.15, 0.12, 0.12, 0.15], [0.5, 0.48, 0.42, 0.4], "k-", linewidth=2)
        ax.plot([0.85, 0.88, 0.88, 0.85], [0.5, 0.48, 0.42, 0.4], "k-", linewidth=2)

    # Current setup
    draw_head(ax1)
    # C3 and C4 connected (bipolar)
    ax1.plot(0.35, 0.45, "ro", markersize=15, zorder=5)
    ax1.text(0.35, 0.38, "C3", ha="center", fontsize=11, fontweight="bold")
    ax1.plot(0.65, 0.45, "ro", markersize=15, zorder=5)
    ax1.text(0.65, 0.38, "C4", ha="center", fontsize=11, fontweight="bold")
    ax1.plot([0.35, 0.65], [0.45, 0.45], "r-", linewidth=3, alpha=0.5)
    ax1.text(0.5, 0.48, "BIPOLAR\n(single channel)", ha="center", fontsize=9, color="red")
    # Fpz
    ax1.plot(0.5, 0.65, "go", markersize=12, zorder=5)
    ax1.text(0.5, 0.58, "Fpz\n(ref)", ha="center", fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.95)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Current: Bipolar C3-C4\n(1 channel, differences cancel)", fontsize=12, color="red")

    # Ideal setup
    draw_head(ax2)
    ax2.plot(0.35, 0.45, "bo", markersize=15, zorder=5)
    ax2.text(0.35, 0.38, "C3", ha="center", fontsize=11, fontweight="bold")
    ax2.plot(0.65, 0.45, "bo", markersize=15, zorder=5)
    ax2.text(0.65, 0.38, "C4", ha="center", fontsize=11, fontweight="bold")
    # Earlobe references
    ax2.plot(0.14, 0.45, "gs", markersize=10, zorder=5)
    ax2.text(0.06, 0.45, "A1\n(ref)", ha="center", fontsize=8)
    ax2.plot(0.86, 0.45, "gs", markersize=10, zorder=5)
    ax2.text(0.94, 0.45, "A2\n(ref)", ha="center", fontsize=8)
    # Separate channels
    ax2.annotate("", xy=(0.14, 0.45), xytext=(0.35, 0.45),
                 arrowprops=dict(arrowstyle="<->", color="blue", lw=2))
    ax2.annotate("", xy=(0.86, 0.45), xytext=(0.65, 0.45),
                 arrowprops=dict(arrowstyle="<->", color="blue", lw=2))
    ax2.text(0.25, 0.5, "Ch1", ha="center", fontsize=9, color="blue")
    ax2.text(0.75, 0.5, "Ch2", ha="center", fontsize=9, color="blue")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.95)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Recommended: Independent C3 & C4\n(2 channels, lateralized ERD preserved)", fontsize=12, color="blue")

    plt.suptitle("Electrode Placement: Current vs Recommended Setup",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = THESIS_DIR / "fig_electrode_placement.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================== #
#  MAIN                                                                #
# ================================================================== #

def main():
    print_banner()
    print("  >> COMPREHENSIVE THESIS ANALYSIS GENERATOR")
    print()

    start = time.time()

    # 1. Signal quality analysis
    sq_results = analyze_signal_quality()

    # 2. Signal quality figures
    plot_signal_quality_figures(sq_results)

    # 3. ML pipeline validation
    val_results = validate_ml_pipeline()
    plot_pipeline_comparison(val_results)

    # 4. Hardware diagnosis
    plot_hardware_diagnosis()

    # 5. Improvements table
    generate_improvements_summary()

    # 6. Electrode placement
    plot_electrode_explanation()

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  ALL THESIS MATERIALS GENERATED")
    print("=" * 60)
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Location: {THESIS_DIR}")
    print()
    print("  Generated files:")
    for f in sorted(THESIS_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.0f} KB)")
    print()
    print("  Copy these directly into your thesis document!")


if __name__ == "__main__":
    main()
