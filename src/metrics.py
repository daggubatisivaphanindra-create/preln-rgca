"""
Statistical evaluation metrics: sample standard deviation (ddof=1), two-sided
Student-t 95% confidence intervals (df=9, t=2.2622), grouped lexical-frame
bootstrap confidence intervals, and Matthews Correlation Coefficient (MCC).
"""
import numpy as np
import scipy.stats as stats

def mean_sd_ci(vals):
    arr = np.array(vals, dtype=np.float64)
    n = len(arr)
    m = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=max(1, n - 1)))
    ci_half = float(t_crit * sd / np.sqrt(max(1, n)))
    return m, sd, m - ci_half, m + ci_half

def compute_grouped_bootstrap_ci(frame_ids_te, correct_vec, n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    u_frames = np.unique(frame_ids_te)
    frame_to_idx = {uf: np.where(frame_ids_te == uf)[0] for uf in u_frames}
    boot_means = []
    for _ in range(n_boot):
        sampled_f = rng.choice(u_frames, size=len(u_frames), replace=True)
        idx_list = np.concatenate([frame_to_idx[uf] for uf in sampled_f])
        boot_means.append(np.mean(correct_vec[idx_list]) * 100.0)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def compute_mcc(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
