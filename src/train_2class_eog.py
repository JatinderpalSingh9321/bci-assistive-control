"""Quick 2-class (BLINK vs WINK) training comparison."""
import numpy as np
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SAMPLING_RATE = 250
FIXED_LEN = 1750
CLASS_NAMES = ["BLINK", "WINK"]

# Load all 3 sessions
all_data, all_labels = [], []
for sess in [1, 2, 3]:
    path = f"data/raw/subject_001/session_{sess:02d}_blink_wink.npz"
    d = np.load(path, allow_pickle=True)
    all_data.extend(d["data"])
    all_labels.extend(d["labels"])
    print(f"Session {sess}: {len(d['labels'])} trials")

labels = np.array(all_labels)
# Remap: 0=BLINK stays 0, 1=WINK_LEFT->1, 2=WINK_RIGHT->1
labels_2class = np.where(labels == 0, 0, 1)
print(f"\n2-Class: BLINK={np.sum(labels_2class==0)}, WINK={np.sum(labels_2class==1)}")

# Preprocess
def preprocess(raw, fixed_len=FIXED_LEN):
    X = []
    for epoch in raw:
        sig = np.array(epoch, dtype=np.float64).flatten()
        b, a = scipy_signal.iirnotch(50.0, 30, SAMPLING_RATE)
        sig = scipy_signal.filtfilt(b, a, sig)
        sig = sig - np.mean(sig)
        std = np.std(sig)
        if std > 1e-10:
            sig = sig / std
        if len(sig) >= fixed_len:
            sig = sig[:fixed_len]
        else:
            sig = np.pad(sig, (0, fixed_len - len(sig)))
        X.append(sig)
    return np.array(X, dtype=np.float32)

X = preprocess(all_data)
print(f"Shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels_2class, test_size=0.2, stratify=labels_2class, random_state=42
)

print(f"Train: {len(y_train)} (BLINK={np.sum(y_train==0)}, WINK={np.sum(y_train==1)})")
print(f"Test:  {len(y_test)} (BLINK={np.sum(y_test==0)}, WINK={np.sum(y_test==1)})")

# ===== SVM =====
print("\n" + "=" * 50)
print("  SVM (2-class: BLINK vs WINK)")
print("=" * 50)

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(C=10, gamma="scale", kernel="rbf", probability=True, random_state=42)),
])
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm, X, labels_2class, cv=cv, scoring="accuracy")

print(f"Test Accuracy: {svm_acc:.1%}")
print(f"CV Accuracy:   {np.mean(cv_scores):.1%} +/- {np.std(cv_scores):.1%}")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===== CNN =====
print("\n" + "=" * 50)
print("  CNN (2-class: BLINK vs WINK)")
print("=" * 50)

from tensorflow import keras

X_cnn = X[:, :, np.newaxis]
X_tr, X_ts, y_tr, y_ts = train_test_split(
    X_cnn, labels_2class, test_size=0.2, stratify=labels_2class, random_state=42
)

model = keras.Sequential([
    keras.layers.Conv1D(16, 25, activation="relu", input_shape=(FIXED_LEN, 1), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(4),
    keras.layers.Dropout(0.2),

    keras.layers.Conv1D(32, 15, activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(4),
    keras.layers.Dropout(0.2),

    keras.layers.Conv1D(64, 7, activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(4),
    keras.layers.Dropout(0.3),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
    ],
)

y_cnn = (model.predict(X_ts, verbose=0) > 0.5).astype(int).flatten()
cnn_acc = accuracy_score(y_ts, y_cnn)

print(f"\nCNN Test Accuracy: {cnn_acc:.1%}")
print(f"Epochs trained: {len(history.history['loss'])}")
print(classification_report(y_ts, y_cnn, target_names=CLASS_NAMES))

# ===== SUMMARY =====
print("\n" + "=" * 50)
print("  SUMMARY")
print("=" * 50)
print(f"  SVM: {svm_acc:.1%} test, {np.mean(cv_scores):.1%} CV")
print(f"  CNN: {cnn_acc:.1%} test")
winner = "SVM" if svm_acc >= cnn_acc else "CNN"
print(f"  Winner: {winner}")
print("=" * 50)
