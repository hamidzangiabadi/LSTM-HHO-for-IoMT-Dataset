import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("convergence_curves_20251223_032808.csv")

plt.figure(figsize=(8, 5))


plt.plot(df["Iteration"], df["HHO_Mean"], label="HHO-LSTM", color="tab:blue")
plt.fill_between(
    df["Iteration"],
    df["HHO_Mean"] - df["HHO_Std"],
    df["HHO_Mean"] + df["HHO_Std"],
    color="tab:blue",
    alpha=0.2
)


plt.plot(df["Iteration"], df["MHHO_Mean"], label="MHHO-LSTM", color="tab:orange")
plt.fill_between(
    df["Iteration"],
    df["MHHO_Mean"] - df["MHHO_Std"],
    df["MHHO_Mean"] + df["MHHO_Std"],
    color="tab:orange",
    alpha=0.2
)

plt.xlabel("Iteration")
plt.ylabel("Best Fitness (F1-Score)")
plt.title("Convergence Curve of HHO vs MHHO")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("convergence_curve.png", dpi=300)
plt.show()

# ==============================================

df = pd.read_csv("summary_results_20251223_032808.csv")

models = df["Model"]
mean_f1 = df["Mean_F1"]
std_f1 = df["Std_F1"]

plt.figure(figsize=(7, 5))

plt.bar(models, mean_f1, color=["gray", "tab:blue", "tab:orange"], yerr=std_f1, capsize=6)

plt.ylabel("Mean F1-Score")
plt.title("Performance Comparison of Models")
plt.xticks(rotation=10)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=300)
plt.show()


# ==================================================

df = pd.read_csv("summary_results_20251223_032808.csv")

models = df["Model"]
mean_f1 = df["Mean_F1"]
std_f1 = df["Std_F1"]

x = np.arange(len(models))

plt.figure(figsize=(7, 5))

plt.errorbar(
    x,
    mean_f1,
    yerr=std_f1,
    fmt='o',
    ecolor='black',
    capsize=8,
    markersize=8
)

plt.xticks(x, models, rotation=10)
plt.ylabel("F1-Score")
plt.title("Stability Analysis of Models (Mean ± Std)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("stability_analysis.png", dpi=300)
plt.show()

# =======================================

df = pd.read_csv("detailed_results_20251223_032808.csv")

data = [
    df["Baseline_F1"],
    df["HHO_F1"],
    df["MHHO_F1"]
]

labels = ["Baseline LSTM", "HHO-LSTM", "MHHO-LSTM"]

plt.figure(figsize=(7, 5))

plt.boxplot(
    data,
    labels=labels,
    showmeans=True,
    meanline=True
)

plt.ylabel("F1-Score")
plt.title("Distribution of F1-Scores Across Multiple Runs")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("boxplot_f1_scores.png", dpi=300)
plt.show()
