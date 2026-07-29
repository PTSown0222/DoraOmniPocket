import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Constants
TECHNIQUE_LABELS = [
    "straw_man", "appeal_to_fear", "fud", "bandwagon", "whataboutism",
    "loaded_language", "glittering_generalities", "euphoria", "cherry_picking", "cliche"
]

# 1. Load raw logits
val_logits_raw = [
    np.load("/home/mykola/blacklab/datasets_unlp/logits/cp043/val_logits.npy"),
    np.load("/home/mykola/blacklab/datasets_unlp/logits/cp200ktype0/val_logits.npy"),
    np.load("/home/mykola/blacklab/datasets_unlp/logits/round2/val_logits.npy"),
]
test_logits_raw = [
    np.load("/home/mykola/blacklab/datasets_unlp/logits/cp043/test_logits.npy"),
    np.load("/home/mykola/blacklab/datasets_unlp/logits/cp200ktype0/test_logits.npy"),
    np.load("/home/mykola/blacklab/datasets_unlp/logits/round2/test_logits.npy"),
]

y_val = np.load("/home/mykola/blacklab/datasets_unlp/labels.npy")

def transform_to_rank(logits_list):
    """Chuyển logits về Percentile Rank để cân bằng scale giữa các mô hình"""
    ranked_list = []
    for logit in logits_list:
        # Rank theo từng cột (từng sample/label)
        ranked = np.apply_along_axis(lambda x: rankdata(x) / len(x), axis=0, arr=logit)
        ranked_list.append(ranked)
    return ranked_list

def create_meta_features(logits_list):
    """Tạo Meta-features bao gồm Concatenation + Thống kê độ đồng thuận"""
    # Base concatenated logits
    concat_features = np.concatenate(logits_list, axis=1)
    
    # Statistical features across models
    stacked_array = np.stack(logits_list, axis=0) # Shape: (num_models, num_samples, num_features)
    mean_feat = np.mean(stacked_array, axis=0)
    std_feat = np.std(stacked_array, axis=0)
    max_feat = np.max(stacked_array, axis=0)
    min_feat = np.min(stacked_array, axis=0)
    
    return np.hstack([concat_features, mean_feat, std_feat, max_feat, min_feat])

# Biến đổi Rank & Tạo Features
val_logits_ranked = transform_to_rank(val_logits_raw)
test_logits_ranked = transform_to_rank(test_logits_raw)

X_val = create_meta_features(val_logits_ranked)
X_test = create_meta_features(test_logits_ranked)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
final_preds_test = np.zeros((X_test.shape[0], len(TECHNIQUE_LABELS)))
f1_scores_cv = []

for i, label_name in enumerate(TECHNIQUE_LABELS):
    print(f"\nTraining Meta-Learner cho label '{label_name}'")
    y_label = y_val[:, i]
    
    oof_probs = np.zeros(X_val.shape[0])
    oof_counts = np.zeros(X_val.shape[0])
    test_probs_folds = np.zeros(X_test.shape[0])
    
    # K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(rskf.split(X_val, y_label)):
        X_train_fold, X_val_fold = X_val[train_idx], X_val[val_idx]
        y_train_fold, y_val_fold = y_label[train_idx], y_label[val_idx]
        
        meta_model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        meta_model.fit(X_train_fold, y_train_fold)
        
        # Accumulate OOF Probabilities
        val_preds = meta_model.predict_proba(X_val_fold)[:, 1]
        oof_probs[val_idx] += val_preds
        oof_counts[val_idx] += 1
        
        # Predict on Test
        test_probs_folds += meta_model.predict_proba(X_test)[:, 1] / rskf.get_n_splits()

    # Tính OOF Probabilities trung bình
    oof_probs /= oof_counts
    
    # 2. Global OOF Threshold Optimization (Tránh leakage & nhiễu fold)
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.01, 0.99, 197):
        preds_thr = (oof_probs >= thr).astype(int)
        f1 = f1_score(y_label, preds_thr, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            
    print(f"Label '{label_name}' | Best OOF Threshold: {best_thr:.3f} | Best OOF F1: {best_f1:.4f}")
    f1_scores_cv.append(best_f1)
    
    # Áp dụng threshold tối ưu cho Test Set
    final_preds_test[:, i] = (test_probs_folds >= best_thr).astype(int)

print(f"\nOverall Macro-F1 (OOF Optimization): {np.mean(f1_scores_cv):.4f}")

# Save Submission
submission_df = pd.read_csv("test.csv")[["id"]]
for idx, label in enumerate(TECHNIQUE_LABELS):
    submission_df[label] = final_preds_test[:, idx].astype(int)

submission_df.to_csv("submission_improved_stacking.csv", index=False)
print("Submission saved as 'submission_improved_stacking.csv'")