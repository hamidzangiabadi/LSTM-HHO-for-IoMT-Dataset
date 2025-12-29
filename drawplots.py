import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_curves(pattern):
    files = sorted(glob.glob(pattern))
    curves = [np.load(f) for f in files]
    return np.array(curves), files


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

# Load curves into arrays
hho_curves = np.array([np.load(f"./results/{f}") for f in hho_files])
mhho_curves = np.array([np.load(f"./results/{f}") for f in mhho_files])

print("HHO curves shape:", hho_curves.shape)
print("MHHO curves shape:", mhho_curves.shape)

print(f"HHO runs loaded: {len(hho_curves)}")
print(f"MHHO runs loaded: {len(mhho_curves)}")

iterations = np.arange(1, hho_curves.shape[1] + 1)

hho_mean = hho_curves.mean(axis=0)
hho_std  = hho_curves.std(axis=0)

mhho_mean = mhho_curves.mean(axis=0)
mhho_std  = mhho_curves.std(axis=0)

# Convergence Curves

df_convergence = pd.DataFrame({
    "Iteration": iterations,
    "HHO_Mean": hho_mean,
    "HHO_Std": hho_std,
    "MHHO_Mean": mhho_mean,
    "MHHO_Std": mhho_std,
})

df_convergence.to_csv("convergence_curves.csv", index=False)
print("convergence_curves.csv saved")


def extract_final_scores(curves, algo_name):
    data = []
    for i, curve in enumerate(curves, start=1):
        data.append({
            "Run": i,
            "Algorithm": algo_name,
            "Final_F1": curve[-1]
        })
    return data


rows = []
rows += extract_final_scores(hho_curves, "HHO")
rows += extract_final_scores(mhho_curves, "MHHO")

df_final = pd.DataFrame(rows)
df_final.to_csv("final_results_run_by_run.csv", index=False)

print("final_results_run_by_run.csv saved")



plt.figure(figsize=(8, 5))

plt.plot(iterations, hho_mean, label="HHO", linewidth=2)
plt.fill_between(iterations,
                 hho_mean - hho_std,
                 hho_mean + hho_std,
                 alpha=0.2)

plt.plot(iterations, mhho_mean, label="MHHO", linewidth=2)
plt.fill_between(iterations,
                 mhho_mean - mhho_std,
                 mhho_mean + mhho_std,
                 alpha=0.2)

plt.xlabel("Iteration")
plt.ylabel("F1-score")
plt.title("Convergence Behavior of HHO vs MHHO")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("convergence_plot.png", dpi=300)
plt.show()



df = pd.read_csv("final_results_run_by_run.csv")

plt.figure(figsize=(6,5))
df.boxplot(column="Final_F1", by="Algorithm", grid=False)

plt.title("Distribution of Final F1-Scores")
plt.suptitle("")
plt.xlabel("Algorithm")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig("boxplot_f1_scores.png", dpi=300)
plt.show()


df = pd.read_csv("convergence_curves.csv")

plt.figure(figsize=(8,5))

plt.plot(df["Iteration"], df["HHO_Mean"], label="HHO", linewidth=2)
plt.fill_between(df["Iteration"],
                 df["HHO_Mean"] - df["HHO_Std"],
                 df["HHO_Mean"] + df["HHO_Std"],
                 alpha=0.2)

plt.plot(df["Iteration"], df["MHHO_Mean"], label="MHHO", linewidth=2)
plt.fill_between(df["Iteration"],
                 df["MHHO_Mean"] - df["MHHO_Std"],
                 df["MHHO_Mean"] + df["MHHO_Std"],
                 alpha=0.2)

plt.xlabel("Iteration")
plt.ylabel("F1-score")
plt.title("Mean Convergence Curves of HHO and MHHO")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_convergence_curve.png", dpi=300)
plt.show()


df = pd.read_csv("./results/summary_results.csv")


summary_rows = []

for name, curves in [("HHO", hho_curves), ("MHHO", mhho_curves)]:
    final_scores = curves[:, -1]
    summary_rows.append({
        "Model": name,
        "Mean_F1": final_scores.mean(),
        "Std_F1": final_scores.std(),
        "Min_F1": final_scores.min(),
        "Max_F1": final_scores.max(),
        "N_Runs": len(final_scores)
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("summary_results.csv", index=False)

print("summary_results.csv saved")




plt.figure(figsize=(6,5))
plt.bar(df["Model"], df["Mean_F1"], yerr=df["Std_F1"],
        capsize=5, alpha=0.85)

plt.ylabel("Mean F1-score")
plt.title("Performance Comparison (Mean ± Std)")
plt.ylim(0.96, 0.99)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("bar_mean_f1.png", dpi=300)
plt.show()


hho_median = np.median(hho_curves, axis=0)
mhho_median = np.median(mhho_curves, axis=0)

plt.figure(figsize=(8,5))
plt.plot(iterations, hho_median, label="HHO (Median)", linewidth=2)
plt.plot(iterations, mhho_median, label="MHHO (Median)", linewidth=2)

plt.xlabel("Iteration")
plt.ylabel("F1-score")
plt.title("Median Convergence Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("median_convergence.png", dpi=300)
plt.show()

best_hho_idx = np.argmax(hho_curves[:, -1])
best_mhho_idx = np.argmax(mhho_curves[:, -1])

plt.figure(figsize=(8,5))
plt.plot(iterations, hho_curves[best_hho_idx], label="HHO (Best Run)", linewidth=2)
plt.plot(iterations, mhho_curves[best_mhho_idx], label="MHHO (Best Run)", linewidth=2)

plt.xlabel("Iteration")
plt.ylabel("F1-score")
plt.title("Best-Run Convergence Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("best_run_convergence.png", dpi=300)
plt.show()
