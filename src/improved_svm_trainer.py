"""
BCI Assistive Control — Improved SVM Trainer
===============================================
Improved SVM training with:
  1. Proper feature scaling (CRITICAL for SVM!)
  2. Feature selection (keep top K discriminative features)
  3. Dimensionality reduction (PCA)
  4. Aggressive hyperparameter tuning (GridSearchCV)
  5. Data augmentation (mixup, jitter, time warp)
  6. Model saving & loading

Group No. 7 | 8th Semester Major Project
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import joblib
from pathlib import Path

from src.utils import MODELS_DIR, RESULTS_DIR, setup_logger

logger = setup_logger("improved_svm")


class ImprovedSVMTrainer:
    """
    Improved SVM training with:
    1. Proper feature scaling (CRITICAL!)
    2. Feature selection (keep top K)
    3. Dimensionality reduction (PCA)
    4. Aggressive hyperparameter tuning
    5. Model saving & loading

    Parameters
    ----------
    verbose : bool
        Whether to print progress.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.svm = None
        self.best_params_ = None
        self.best_cv_score_ = None

    def train_with_grid_search(self, X_train, y_train, cv_folds=5):
        """
        Aggressive grid search over SVM hyperparameters.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features).
        y_train : np.ndarray
            Training labels, shape (n_samples,).
        cv_folds : int
            Number of cross-validation folds.

        Returns
        -------
        float
            Best cross-validation accuracy.
        """
        logger.info("=" * 60)
        logger.info("AGGRESSIVE SVM HYPERPARAMETER GRID SEARCH")
        logger.info("=" * 60)

        n_samples, n_features = X_train.shape
        logger.info(f"Input: {n_samples} samples, {n_features} features")

        # Step 1: Scale features (CRITICAL!)
        logger.info("1. Scaling features (StandardScaler)...")
        X_scaled = self.scaler.fit_transform(X_train)

        # Step 2: Feature selection (keep top K features)
        k = min(20, n_features)  # Don't request more than available
        logger.info(f"2. Selecting top {k} features (ANOVA F-test)...")
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X_scaled, y_train)

        # Step 3: PCA dimensionality reduction
        n_components = min(15, X_selected.shape[1], n_samples - 1)
        logger.info(f"3. Reducing to {n_components} PCA components...")
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_selected)

        explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
        logger.info(f"   PCA explains {explained_var:.1f}% of variance")

        # Step 4: Grid search
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        }
        n_combos = len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma'])
        logger.info(f"4. Running grid search ({n_combos} combinations, "
                    f"{cv_folds}-fold CV)...")

        # Adjust CV folds if not enough samples
        actual_cv = min(cv_folds, min(np.bincount(y_train.astype(int))))
        actual_cv = max(2, actual_cv)

        grid = GridSearchCV(
            SVC(class_weight='balanced', probability=True, random_state=42),
            param_grid,
            cv=StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1 if self.verbose else 0,
            return_train_score=True
        )

        grid.fit(X_pca, y_train)

        # Step 5: Report results
        logger.info("=" * 60)
        logger.info("GRID SEARCH RESULTS")
        logger.info("=" * 60)

        self.best_params_ = grid.best_params_
        self.best_cv_score_ = grid.best_score_

        logger.info(f"Best parameters found:")
        for param, value in self.best_params_.items():
            logger.info(f"  {param:15} = {value}")

        logger.info(f"Best CV Accuracy: {grid.best_score_:.4f} "
                    f"({grid.best_score_ * 100:.2f}%)")

        # Show top 5 parameter combos
        results_df = _grid_results_to_sorted(grid)
        if results_df is not None:
            logger.info(f"\nTop 5 parameter combinations:")
            for i, (_, row) in enumerate(results_df.head(5).iterrows()):
                logger.info(f"  #{i+1}: accuracy={row['mean_test_score']:.4f} "
                           f"params={row['params']}")

        # Step 6: Train final model with best parameters
        logger.info("Training final model with best parameters...")
        self.svm = SVC(
            **self.best_params_,
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        self.svm.fit(X_pca, y_train)

        # Training accuracy
        train_acc = accuracy_score(y_train, self.svm.predict(X_pca))
        logger.info(f"Training accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")

        return grid.best_score_

    def train_simple(self, X_train, y_train):
        """
        Simple training with strong default parameters.
        Much faster than grid search.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        """
        logger.info("Training SVM with optimized default parameters...")

        n_samples, n_features = X_train.shape

        # Scale
        X_scaled = self.scaler.fit_transform(X_train)

        # Feature selection
        k = min(20, n_features)
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X_scaled, y_train)

        # PCA
        n_components = min(15, X_selected.shape[1], n_samples - 1)
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_selected)

        # Train with good defaults
        self.svm = SVC(
            C=100,                # Less regularization (default 1 is too conservative)
            kernel='rbf',         # RBF kernel usually best for EEG
            gamma='auto',         # Data-dependent gamma
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        self.svm.fit(X_pca, y_train)

        train_acc = accuracy_score(y_train, self.svm.predict(X_pca))
        logger.info(f"✓ SVM trained (simple mode): accuracy={train_acc:.4f}")

        self.best_params_ = {'C': 100, 'kernel': 'rbf', 'gamma': 'auto'}
        self.best_cv_score_ = train_acc

    def predict(self, X_test):
        """
        Make predictions on test data.
        Applies the same transformations (scale → select → PCA).

        Returns
        -------
        predictions : np.ndarray
        probabilities : np.ndarray
        """
        if self.svm is None:
            raise RuntimeError("Model not trained yet! Call train_simple() or "
                              "train_with_grid_search() first.")

        X_scaled = self.scaler.transform(X_test)
        X_selected = self.feature_selector.transform(X_scaled)
        X_pca = self.pca.transform(X_selected)

        predictions = self.svm.predict(X_pca)
        probabilities = self.svm.predict_proba(X_pca)

        return predictions, probabilities

    def evaluate(self, X_test, y_test):
        """
        Evaluate on test set with full metrics.

        Returns
        -------
        float
            Test accuracy.
        """
        predictions, probabilities = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        logger.info(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        logger.info(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"\n{cm}")
        logger.info(f"\nClassification Report:")
        report = classification_report(y_test, predictions,
                                       target_names=['LEFT', 'RIGHT'])
        logger.info(f"\n{report}")

        return accuracy

    def save(self, filepath=None):
        """Save model and all transformations to disk."""
        if filepath is None:
            filepath = MODELS_DIR / "improved_svm_model.pkl"

        joblib.dump({
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'svm': self.svm,
            'best_params': self.best_params_,
            'best_cv_score': self.best_cv_score_
        }, filepath)
        logger.info(f"✓ Improved SVM model saved to {filepath}")

    def load(self, filepath=None):
        """Load model from disk."""
        if filepath is None:
            filepath = MODELS_DIR / "improved_svm_model.pkl"

        components = joblib.load(filepath)
        self.scaler = components['scaler']
        self.feature_selector = components['feature_selector']
        self.pca = components['pca']
        self.svm = components['svm']
        self.best_params_ = components.get('best_params')
        self.best_cv_score_ = components.get('best_cv_score')
        logger.info(f"✓ Improved SVM model loaded from {filepath}")


# ──────────────────────────────────────────────
# DATA AUGMENTATION
# ──────────────────────────────────────────────

class SyntheticEEGGenerator:
    """
    Generate synthetic EEG data for augmentation.
    Useful when you have limited training samples.
    """

    @staticmethod
    def mixup_augmentation(X, y, n_new_samples=500, alpha=0.3):
        """
        Mixup augmentation: blend two real samples.

        Parameters
        ----------
        X : np.ndarray
            Features or raw data.
        y : np.ndarray
            Labels.
        n_new_samples : int
            Number of synthetic samples to generate.
        alpha : float
            Mixup parameter (0-1). Lower = closer to original.
        """
        X_new = []
        y_new = []

        for _ in range(n_new_samples):
            # Pick two random samples from same class
            cls = np.random.choice(np.unique(y))
            idx = np.where(y == cls)[0]

            if len(idx) < 2:
                continue

            i, j = np.random.choice(idx, 2, replace=False)

            # Blend with random lambda
            lam = np.random.beta(alpha, alpha)
            x_mixed = lam * X[i] + (1 - lam) * X[j]

            X_new.append(x_mixed)
            y_new.append(cls)

        X_aug = np.concatenate([X, np.array(X_new)], axis=0)
        y_aug = np.concatenate([y, np.array(y_new)], axis=0)

        logger.info(f"Mixup augmentation: {len(X)} → {len(X_aug)} samples")
        return X_aug, y_aug

    @staticmethod
    def jitter_augmentation(X, y, n_copies=3, noise_factor=0.05):
        """
        Jitter augmentation: add small Gaussian noise.

        Parameters
        ----------
        X : np.ndarray
            Features or raw data.
        y : np.ndarray
            Labels.
        n_copies : int
            Number of noisy copies per sample.
        noise_factor : float
            Noise standard deviation as fraction of data std.
        """
        X_new = [X]
        y_new = [y]

        noise_std = noise_factor * np.std(X, axis=0, keepdims=True)

        for _ in range(n_copies):
            noise = np.random.normal(0, 1, X.shape) * noise_std
            X_noisy = X + noise
            X_new.append(X_noisy)
            y_new.append(y.copy())

        X_aug = np.concatenate(X_new, axis=0)
        y_aug = np.concatenate(y_new, axis=0)

        logger.info(f"Jitter augmentation: {len(X)} → {len(X_aug)} samples "
                    f"({n_copies} copies)")
        return X_aug, y_aug

    @staticmethod
    def time_warp(X, y, n_warps=3, sigma=0.2):
        """
        Time warp augmentation: stretch/compress time axis.
        Only works on raw epoch data (not features).

        Parameters
        ----------
        X : np.ndarray
            Raw epochs (n_samples, n_timepoints).
        y : np.ndarray
            Labels.
        n_warps : int
            Number of warped copies per sample.
        sigma : float
            Amount of warping.
        """
        from scipy.interpolate import interp1d

        X_new = [X]
        y_new = [y]

        n_timepoints = X.shape[1] if X.ndim == 2 else X.shape[-1]

        for _ in range(n_warps):
            warped = np.zeros_like(X)

            for i in range(len(X)):
                # Create warp path
                original_steps = np.arange(n_timepoints)
                warp_amount = np.random.normal(0, sigma, 3)
                warp_knots = np.linspace(0, n_timepoints - 1, 3)
                warp_fn = interp1d(warp_knots, warp_knots + warp_amount,
                                   kind='linear', fill_value='extrapolate')
                warped_steps = warp_fn(original_steps)
                warped_steps = np.clip(warped_steps, 0, n_timepoints - 1)

                if X.ndim == 2:
                    f = interp1d(original_steps, X[i], kind='linear',
                                fill_value='extrapolate')
                    warped[i] = f(warped_steps)
                else:
                    for ch in range(X.shape[1]):
                        f = interp1d(original_steps, X[i, ch], kind='linear',
                                    fill_value='extrapolate')
                        warped[i, ch] = f(warped_steps)

            X_new.append(warped)
            y_new.append(y.copy())

        X_aug = np.concatenate(X_new, axis=0)
        y_aug = np.concatenate(y_new, axis=0)

        logger.info(f"Time warp augmentation: {len(X)} → {len(X_aug)} samples "
                    f"({n_warps} warps)")
        return X_aug, y_aug


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def _grid_results_to_sorted(grid):
    """Convert grid search results to a sorted DataFrame."""
    try:
        import pandas as pd
        results = pd.DataFrame(grid.cv_results_)
        return results.sort_values('mean_test_score', ascending=False)
    except ImportError:
        return None


# USAGE EXAMPLE:
if __name__ == "__main__":
    print("Testing ImprovedSVMTrainer...")

    # Generate fake data
    np.random.seed(42)
    n_samples = 100
    n_features = 22

    X = np.random.randn(n_samples, n_features)
    # Make feature 0 discriminative
    y = (X[:, 0] > 0).astype(int)

    # Train with simple mode (fast)
    trainer = ImprovedSVMTrainer()
    trainer.train_simple(X, y)

    # Evaluate on same data (just for testing)
    acc = trainer.evaluate(X, y)
    print(f"\nTest accuracy: {acc:.2%}")

    # Save and load
    trainer.save()
    trainer2 = ImprovedSVMTrainer()
    trainer2.load()

    print("\n✓ ImprovedSVMTrainer ready to use")
