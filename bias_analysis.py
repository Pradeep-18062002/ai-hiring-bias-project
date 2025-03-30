import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the decisions from LLaMA
df = pd.read_csv('data/llm_decisions.csv')

# Ensure the decision column is in integer format
df['LLM_Decision'] = df['LLM_Decision'].astype(int)

df['Truth'] = df['Employed']

# -----------------------
# Demographic Parity
# -----------------------
demographic_parity = df.groupby('Race')['LLM_Decision'].mean()

# -----------------------
# Predictive Parity (Precision)
# -----------------------
def precision(group):
    tp = ((group['LLM_Decision'] == 1) & (group['Truth'] == 1)).sum()
    pred_pos = (group['LLM_Decision'] == 1).sum()
    return tp / pred_pos if pred_pos > 0 else np.nan

predictive_parity = df.groupby('Race').apply(precision)

# -----------------------
# Equalized Odds: TPR and FPR
# -----------------------
def true_positive_rate(group):
    tp = ((group['LLM_Decision'] == 1) & (group['Truth'] == 1)).sum()
    fn = ((group['LLM_Decision'] == 0) & (group['Truth'] == 1)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan

def false_positive_rate(group):
    fp = ((group['LLM_Decision'] == 1) & (group['Truth'] == 0)).sum()
    tn = ((group['LLM_Decision'] == 0) & (group['Truth'] == 0)).sum()
    return fp / (fp + tn) if (fp + tn) > 0 else np.nan

tpr = df.groupby('Race').apply(true_positive_rate)
fpr = df.groupby('Race').apply(false_positive_rate)

# -----------------------
# Chi-Square Test
# -----------------------
contingency = pd.crosstab(df['Race'], df['LLM_Decision'])
chi2, p, dof, expected = chi2_contingency(contingency)

# -----------------------
# Print Bias Metrics
# -----------------------
print("\n📊 Demographic Parity (Positive Decision Rate by Race):\n", demographic_parity)
print("\n🎯 Predictive Parity (Precision by Race):\n", predictive_parity)
print("\n✅ True Positive Rate by Race:\n", tpr)
print("\n❌ False Positive Rate by Race:\n", fpr)
print("\n🧪 Chi-squared Test for Race vs LLM Decision:")
print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}, dof = {dof}")

# -----------------------
# Visualize Bias Metrics
# -----------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Bias Metrics by Race", fontsize=16)

# Demographic Parity
axs[0, 0].bar(demographic_parity.index, demographic_parity.values)
axs[0, 0].set_title("Demographic Parity\n(Positive Decision Rate)")
axs[0, 0].set_ylabel("Rate")

# Predictive Parity
axs[0, 1].bar(predictive_parity.index, predictive_parity.values)
axs[0, 1].set_title("Predictive Parity (Precision)")
axs[0, 1].set_ylabel("Precision")

# TPR
axs[1, 0].bar(tpr.index, tpr.values)
axs[1, 0].set_title("True Positive Rate (TPR)")
axs[1, 0].set_ylabel("TPR")

# FPR
axs[1, 1].bar(fpr.index, fpr.values)
axs[1, 1].set_title("False Positive Rate (FPR)")
axs[1, 1].set_ylabel("FPR")

for ax in axs.flat:
    ax.set_xlabel("Race")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(df['Race'].unique())))
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("bias_metrics_by_race.png")
plt.show()
