import numpy as np
import pandas as pd

hho_files = [
    "curve_hho_run1.npy",
    "curve_hho_run2.npy",
    "curve_hho_run3.npy",
    "curve_hho_run4.npy",
    "curve_hho_run5.npy"
]

mhho_files = [
    "curve_mhho_run1.npy",
    "curve_mhho_run2.npy",
    "curve_mhho_run3.npy",
    "curve_mhho_run4.npy",
    "curve_mhho_run5.npy"
]

# Load data
hho_curves  = np.array([np.load(f"./results/{f}") for f in hho_files])
mhho_curves = np.array([np.load(f"./results/{f}") for f in mhho_files])
baseline_df = pd.read_csv("./results/detailed_results.csv")

baseline_f1 = baseline_df["Baseline_F1"].values  
baseline_time = baseline_df["Baseline_Time_sec"].values


n_runs, n_iter = hho_curves.shape

baseline_curves = np.repeat(
    baseline_f1[:, None],  
    n_iter,
    axis=1
)

assert hho_curves.shape == mhho_curves.shape, \
       "HHO and MHHO must have same shape"

n_runs, n_iter = hho_curves.shape
iterations = np.arange(1, n_iter + 1)
table_iii = pd.DataFrame({
    "Model": ["Base-LSTM", "LSTM-HHO", "LSTM-MHHO"],
    "Mean_F1": [
        baseline_curves[:, -1].mean(),
        hho_curves[:, -1].mean(),
        mhho_curves[:, -1].mean()
    ],
    "Std_F1": [
        baseline_curves[:, -1].std(),
        hho_curves[:, -1].std(),
        mhho_curves[:, -1].std()
    ],
    "Min_F1": [
        baseline_curves[:, -1].min(),
        hho_curves[:, -1].min(),
        mhho_curves[:, -1].min()
    ],
    "Max_F1": [
        baseline_curves[:, -1].max(),
        hho_curves[:, -1].max(),
        mhho_curves[:, -1].max()
    ]
})

table_iii.to_csv("Table_III_Performance_Comparison.csv", index=False)


table_iii.to_csv("Table_III_Performance_Comparison.csv", index=False)
print("✅ Table III saved")

def stability_stats(curves):
    final_scores = curves[:, -1]
    mean = final_scores.mean()
    std = final_scores.std()
    cov = (std / mean) * 100
    return mean, std, cov, final_scores.min(), final_scores.max()

base_stats = stability_stats(baseline_curves)
hho_stats  = stability_stats(hho_curves)
mhho_stats = stability_stats(mhho_curves)

table_iv = pd.DataFrame({
    "Model": ["Baseline", "HHO", "MHHO"],
    "Std_F1": [base_stats[1], hho_stats[1], mhho_stats[1]],
    "CoV_%":  [base_stats[2], hho_stats[2], mhho_stats[2]],
    "Worst_Run_F1": [base_stats[3], hho_stats[3], mhho_stats[3]],
    "Best_Run_F1":  [base_stats[4], hho_stats[4], mhho_stats[4]],
})

table_iv.to_csv("Table_IV_Stability_Analysis.csv", index=False)

print("✅ Table IV saved")


hho_improvement  = (hho_curves[:, -1]  - hho_curves[:, 0])  / hho_curves[:, 0] * 100
mhho_improvement = (mhho_curves[:, -1] - mhho_curves[:, 0]) / mhho_curves[:, 0] * 100

table_v = pd.DataFrame({
    "Model": ["HHO", "MHHO"],
    "Mean_Improvement_%": [
        hho_improvement.mean(),
        mhho_improvement.mean()
    ],
    "Std_Improvement_%": [
        hho_improvement.std(),
        mhho_improvement.std()
    ],
    "Gain_over_HHO_%": [
        0.0,
        mhho_improvement.mean() - hho_improvement.mean()
    ]
})

table_v.to_csv("Table_V_Improvement_Analysis.csv", index=False)
print("✅ Table V saved")

key_iters = [1, n_iter // 2, n_iter]

def mean_std_at(curves, idx):
    return curves[:, idx-1].mean(), curves[:, idx-1].std()

rows = []
for it in key_iters:
    b_mean, b_std = mean_std_at(baseline_curves, it)
    h_mean, h_std = mean_std_at(hho_curves, it)
    m_mean, m_std = mean_std_at(mhho_curves, it)

    rows.append({
        "Iteration": it,
        "Baseline_Mean_F1": b_mean,
        "Baseline_Std": b_std,
        "HHO_Mean_F1": h_mean,
        "HHO_Std": h_std,
        "MHHO_Mean_F1": m_mean,
        "MHHO_Std": m_std,
        "Δ_MHHO−HHO": m_mean - h_mean
    })

table_vi = pd.DataFrame(rows)
table_vi.to_csv("Table_VI_Convergence_Profile.csv", index=False)
