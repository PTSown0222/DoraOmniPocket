import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Constants and paths
TECHNIQUE_LABELS = [
    "straw_man", "appeal_to_fear", "fud", "bandwagon", "whataboutism",
    "loaded_language", "glittering_generalities", "euphoria",
    "cherry_picking", "cliche"
]

# Load logits from multiple models
val_logits = [
    np.load("/home/mykola/blacklab/datasets_unlp/logits/cp043/val_logits.npy"),
    np.load("/home/mykola/blacklab/datasets_unlp/logits/cp200ktype0/val_logits.npy"),
    np.load("/home/mykola/blacklab/datasets_unlp/logits/round2/val_logits.npy"),
]

test_logits = [
    np.load("/home/mykola/blacklab/datasets_unlp/logits/cp043/test_logits.npy"),
    np.load("/home/mykola/blacklab/datasets_unlp/logits/cp200ktype0/test_logits.npy"),
    np.load("/home/mykola/blacklab/datasets_unlp/logits/round2/test_logits.npy"),
]

# Concatenate logits horizontally to form features
X_val = np.concatenate(val_logits, axis=1)
X_test = np.concatenate(test_logits, axis=1)
y_val = np.load("/home/mykola/blacklab/datasets_unlp/labels.npy")  # Shape: (num_samples, num_labels)

# Ensure robustness with repeated cross-validation
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

final_preds_test = np.zeros((X_test.shape[0], len(TECHNIQUE_LABELS)))
f1_scores_cv = []

# Loop over labels
for i, label_name in enumerate(TECHNIQUE_LABELS):
    print(f"\n🔍 Training RF ensemble for label '{label_name}'")
    y_label = y_val[:, i]

    fold_best_thresholds = []
    f1_scores_per_fold = []
    rf_models = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(X_val, y_label)):
        X_train_fold, X_val_fold = X_val[train_idx], X_val[val_idx]
        y_train_fold, y_val_fold = y_label[train_idx], y_label[val_idx]

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        rf.fit(X_train_fold, y_train_fold)
        rf_models.append(rf)

        # Optimize threshold based on validation fold
        probs_val_fold = rf.predict_proba(X_val_fold)[:, 1]
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.linspace(0, 1, 101):
            preds_thr = (probs_val_fold >= thr).astype(int)
            f1 = f1_score(y_val_fold, preds_thr, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        print(f" Fold {fold+1} | best_thr={best_thr:.2f} | best_f1={best_f1:.4f}")
        f1_scores_per_fold.append(best_f1)
        fold_best_thresholds.append(best_thr)

    mean_f1_label = np.mean(f1_scores_per_fold)
    global_threshold = np.mean(fold_best_thresholds)
    print(f"Label '{label_name}' CV Mean F1: {mean_f1_label:.4f} | Global threshold: {global_threshold:.2f}")
    f1_scores_cv.append(mean_f1_label)

    # Averaging probabilities from all folds' models and applying global threshold
    probs_test_avg = np.mean([rf.predict_proba(X_test)[:, 1] for rf in rf_models], axis=0)
    final_preds_test[:, i] = (probs_test_avg >= global_threshold).astype(int)

# Overall CV metric
print(f"\n🧩 Overall Macro-F1 (repeated cross-validation): {np.mean(f1_scores_cv):.4f}")

# Prepare submission
submission_df = pd.read_csv("test.csv")[["id"]]
for idx, label in enumerate(TECHNIQUE_LABELS):
    submission_df[label] = final_preds_test[:, idx].astype(int)

submission_df.to_csv("submission_rf_stacked.csv", index=False)
print("✅ Submission saved as 'submission_rf_stacked_thres.csv'")