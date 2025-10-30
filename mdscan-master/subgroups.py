import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

dff_ag = pd.read_csv("dff_ag1.csv")
dff_nai = pd.read_csv("dff_nai1.csv")
dff_nan = pd.read_csv("dff_nan1.csv")

# Assuming your DataFrame is loaded as dff_ag
dff = dff_ag.copy()

# --------------------------------------------------------------------
# 1. Define AUTOSTRAT-DISCOVERED SUBGROUP MASKS
# --------------------------------------------------------------------
# subgroup_mask = (
#     (dff_ag["waist_hip_r_c_qc"] >= 0.9) & (dff_ag["waist_hip_r_c_qc"] <= 1.36) &
#     (dff_ag["diabetes_history_qc"] == 1) &
#     (dff_ag["bmi_c_qc"] >= 21.55) & (dff_ag["bmi_c_qc"] <= 68.02)
# )
model_filter1 = (
    (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)


model_filter2 = (
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)

model_filter3 = (
    (dff["diabetes_history_qc"] == 1) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)


model_subgroups = {
    "Model 1": model_filter1, # "mvpa, diabetes history, waist-hip, bmi"
    "Model 2": model_filter2, # "smoking, age, waist-hip, waist circumference, sex"
    "Model 3": model_filter3 # "waist-hip, age"
}

# --------------------------------------------------------------------
# 2. Define LITERATURE-BASED SUBGROUP MASKS
# --------------------------------------------------------------------

# Waist circumference cutoffs in mm
male_waist_cutoff = 940
female_waist_cutoff = 800

study_filter1 = (
    (dff["diabetes_history_qc"] == 1) &
    (dff["bmi_c_qc"] >= 30)
)

study_filter2 = (
    (
        ((dff["sex"] == 1) & (dff["waist_circumference_qc"] >= male_waist_cutoff)) |
        ((dff["sex"] == 0) & (dff["waist_circumference_qc"] >= female_waist_cutoff))
    ) &
    (dff["mvpa_c"] < 600)
)

study_filter3 = (
    (dff["age"] >= 45) & study_filter2
)

study_subgroups = {
    "Study 1": study_filter1, # "bmi >= 30, diabetes history"
    "Study 2": study_filter2, # "waist circumference, mvpa < 600"
    "Study 3": study_filter3 # "age >= 45, waist circumference, mvpa < 600"
}

# --------------------------------------------------------------------
# 3. Function to Compute OR, CI, and P-value using Logistic Regression
# --------------------------------------------------------------------
def compute_or_ci_p(data, mask, outcome_col="diabetes_status_c_qc"):
    temp = data.copy()
    temp["in_subgroup"] = mask.astype(int)
    
    model = smf.glm(formula=f"{outcome_col} ~ in_subgroup", 
                    data=temp, 
                    family=sm.families.Binomial()).fit()
    
    coef = model.params["in_subgroup"]
    se = model.bse["in_subgroup"]
    pval = model.pvalues["in_subgroup"]

    OR = np.exp(coef)
    CI_lower = np.exp(coef - 1.96 * se)
    CI_upper = np.exp(coef + 1.96 * se)

    return OR, CI_lower, CI_upper, pval

# --------------------------------------------------------------------
# 4. Loop through all subgroups and compile stats
# --------------------------------------------------------------------
results = []

for name, flt in model_subgroups.items():
    or_val, ci_lo, ci_hi, p_val = compute_or_ci_p(dff, flt)
    results.append({
        "Subgroup Name": name,
        "Method": "Autostrat",
        "OR": round(or_val, 2),
        "CI": f"({round(ci_lo, 2)}, {round(ci_hi, 2)})",
        "p-value": f"{p_val:.2e}"
    })

for name, flt in study_subgroups.items():
    or_val, ci_lo, ci_hi, p_val = compute_or_ci_p(dff, flt)
    results.append({
        "Subgroup Name": name,
        "Method": "Literature",
        "OR": round(or_val, 2),
        "CI": f"({round(ci_lo, 2)}, {round(ci_hi, 2)})",
        "p-value": f"{p_val:.2e}"
    })

# Convert to DataFrame
comparison_df = pd.DataFrame(results)

# Optional: sort by method and subgroup name
comparison_df.sort_values(by=["Method", "Subgroup Name"], inplace=True)

# Make a copy
combined_results = comparison_df.copy()

# --- Split CI string into CI_lower and CI_upper ---
combined_results[["CI_lower", "CI_upper"]] = combined_results["CI"].str.strip("()").str.split(",", expand=True).astype(float)

# --- Convert p-value string to float (if needed) ---
combined_results["p-value"] = combined_results["p-value"].astype(float)

# Optional: sort if needed
combined_results = combined_results.sort_values(by="Method", ascending=False).reset_index(drop=True)

combined_results = combined_results.iloc[::-1].reset_index(drop=True)

import matplotlib.pyplot as plt
import numpy as np

# Ensure required columns exist: OR, CI_lower, CI_upper, p-value
# combined_results = comparison_df.copy()

# Optional: Reverse if you want the top entries to be plotted lowest
# combined_results = combined_results.iloc[::-1].reset_index(drop=True)

# Indexing for plotting
n_groups = len(combined_results)
y_positions = np.arange(n_groups)

# Create subplots
fig, (ax_forest, ax_pvals) = plt.subplots(
    1, 2, figsize=(10, 6),
    gridspec_kw={"width_ratios": [2.8, 1.2]},
    dpi=300
)

# ----------------------------
# FOREST PLOT (LEFT)
# ----------------------------
for i, row in combined_results.iterrows():
    # Plot 95% CI line
    ax_forest.plot(
        [row["CI_lower"], row["CI_upper"]],
        [y_positions[i]] * 2,
        color="darkblue", lw=2
    )
    # Plot OR marker
    ax_forest.plot(
        row["OR"], y_positions[i],
        "s", color="darkred", markersize=6
    )
    # Annotate OR
    ax_forest.text(
        row["OR"], y_positions[i] + 0.1,
        f"{row['OR']:.2f}", ha="center",
        color="darkred", fontsize=9
    )
    # Annotate CI
    ci_label = f"({row['CI_lower']:.2f}, {row['CI_upper']:.2f})"
    ax_forest.text(
        row["CI_upper"] + 0.1, y_positions[i],
        ci_label, va="center", fontsize=9, color="black"
    )

# Add reference line at OR=1
ax_forest.axvline(x=1.0, color="gray", linestyle="--", lw=1)

labs = [
    "WHR$\geqslant$0.9 \nT2D Family History",
    "MVPA$\leqslant$2448 \nWHR$\geqslant$0.9 \nT2D Family History", 
    "MVPA$\leqslant$2448 \nWHR$\geqslant$0.9 \nBMI$\geqslant$21.37 \nT2D Family History",
    "WC: W$\geqslant$800, M$\geqslant$940 \nMVPA$\leqslant$150",
    "Age$\geqslant$45 \nWC: W$\geqslant$800, M$\geqslant$940 \nMVPA$\leqslant$150",
    "BMI ≥30 \nT2D Family History",
]

ax_forest.set_yticks(y_positions)
ax_forest.set_yticklabels(labs, fontsize=10)
# Custom y-labels
# ax_forest.set_yticks(y_positions)
# ax_forest.set_yticklabels(combined_results["Subgroup Name"], fontsize=10)
for label in ax_forest.yaxis.get_ticklabels():
    label.set_bbox(dict(facecolor='#DEDEE0', edgecolor='#DAD6D6',
                        alpha=0.5, pad=3, boxstyle='round,pad=0.5'))

ax_forest.set_xlabel("Odds Ratio (95% CI)", fontsize=11)
for spine in ["top", "right"]:
    ax_forest.spines[spine].set_visible(False)
ax_forest.set_xlim(
    min(combined_results["CI_lower"].min(), 0.5) - 0.1,
    combined_results["CI_upper"].max() + 1.0
)
ax_forest.set_ylim(-0.5, n_groups - 0.5)
ax_forest.set_xlim(0, 22)
# Hide the y-axis ticks
ax_forest.tick_params(axis='y', which='both', left=False, right=False)

# Shading and labeling regions
ax_forest.axhspan(2.5, 5.5, color="lightblue", alpha=0.1)
ax_forest.text(
    ax_forest.get_xlim()[0] + 15.5, 4.5,
    "Study-Defined", fontsize=13, color="#007991",
    va="center", ha="left", fontweight='bold'
)
ax_forest.axhspan(-0.5, 2.5, color="lightgreen", alpha=0.1)
ax_forest.text(
    ax_forest.get_xlim()[0] + 15.5, 1.5,
    "Model-Derived", fontsize=13, color="green",
    va="center", ha="left", fontweight='bold'
)

# Label the whole left section
ax_forest.text(
    ax_forest.get_xlim()[0] - 4, n_groups - 0.6,
    'Subgroups', fontsize=10, fontweight='bold',
    ha='left', va='bottom'
)

# ----------------------------
# P-VALUE BAR CHART (RIGHT)
# ----------------------------
# Clip very small p-values to avoid log(0)
pvals = combined_results["p-value"].astype(float).clip(lower=1e-300)
neg_log_pvals = -np.log10(pvals)

# Horizontal bars
ax_pvals.barh(
    y_positions, neg_log_pvals,
    height=0.3, color="#439A86"
)
# Annotate numeric p-values
for i, val in enumerate(neg_log_pvals):
    raw_pval = combined_results["p-value"].iloc[i]
    try:
        pval_text = f"{float(raw_pval):.1e}"
    except:
        pval_text = str(raw_pval)
    ax_pvals.text(
        val + 0.05, y_positions[i],
        pval_text, va="center", fontsize=9, color="black"
    )

# Styling
for spine in ["left", "right", "top"]:
    ax_pvals.spines[spine].set_visible(False)
ax_pvals.set_xlabel("-log10(p-value)", fontsize=11)
ax_pvals.set_ylim(ax_forest.get_ylim())
ax_pvals.set_yticks(y_positions)
ax_pvals.set_yticklabels([])

# ----------------------------
# Final Layout & Save
# ----------------------------
plt.tight_layout()
# plt.savefig("forest_plot_age.pdf", dpi=350)
plt.show()

#For Nairobi

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Assuming your DataFrame is loaded as dff_ag
dff = dff_nai.copy()

# --------------------------------------------------------------------
# 1. Define AUTOSTRAT-DISCOVERED SUBGROUP MASKS
# --------------------------------------------------------------------
# Waist circumference cutoffs in mm,
model_filter1 = (
    (dff["waist_hip_r_c_qc"] > 0.85) & (dff["waist_hip_r_c_qc"] <= 9.02) &
    (dff["hip_circumference_qc"] >= 887.0) & (dff["hip_circumference_qc"] <= 1494.0) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["age"] > 54.0) & (dff["age"] <= 60.0)
)


model_filter2 = (
    (dff["bmi_c_qc"] >= 20.49) & (dff["bmi_c_qc"] <= 62.8) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["age"] > 54.0) & (dff["age"] <= 60.0)
)

model_filter3 = (
    (dff["diabetes_history_qc"] == 1) &
    (dff["age"] > 54.0) & (dff["age"] <= 60.0)
)


model_subgroups = {
    "Model 1": model_filter1, # "mvpa, diabetes history, waist-hip, bmi"
    "Model 2": model_filter2, # "smoking, age, waist-hip, waist circumference, sex"
    "Model 3": model_filter3 # "waist-hip, age"
}

# --------------------------------------------------------------------
# 2. Define LITERATURE-BASED SUBGROUP MASKS
# --------------------------------------------------------------------

# Waist circumference cutoffs in mm
male_waist_cutoff = 940
female_waist_cutoff = 800

study_filter1 = (
    (dff["diabetes_history_qc"] == 1) &
    (dff["bmi_c_qc"] >= 30)
)

study_filter2 = (
    (
        ((dff["sex"] == 1) & (dff["waist_circumference_qc"] >= male_waist_cutoff)) |
        ((dff["sex"] == 0) & (dff["waist_circumference_qc"] >= female_waist_cutoff))
    ) &
    (dff["mvpa_c"] < 150)
)

study_filter3 = (
    (dff["age"] >= 45) & study_filter2
)

study_subgroups = {
    "Study 1": study_filter1, # "bmi >= 30, diabetes history"
    "Study 2": study_filter2, # "waist circumference, mvpa < 150"
    "Study 3": study_filter3 # "age >= 45, waist circumference, mvpa < 150"
}

# --------------------------------------------------------------------
# 3. Function to Compute OR, CI, and P-value using Logistic Regression
# --------------------------------------------------------------------
def compute_or_ci_p(data, mask, outcome_col="diabetes_status_c_qc"):
    temp = data.copy()
    temp["in_subgroup"] = mask.astype(int)
    
    model = smf.glm(formula=f"{outcome_col} ~ in_subgroup", 
                    data=temp, 
                    family=sm.families.Binomial()).fit()
    
    coef = model.params["in_subgroup"]
    se = model.bse["in_subgroup"]
    pval = model.pvalues["in_subgroup"]

    OR = np.exp(coef)
    CI_lower = np.exp(coef - 1.96 * se)
    CI_upper = np.exp(coef + 1.96 * se)

    return OR, CI_lower, CI_upper, pval

# --------------------------------------------------------------------
# 4. Loop through all subgroups and compile stats
# --------------------------------------------------------------------
results = []

for name, flt in model_subgroups.items():
    or_val, ci_lo, ci_hi, p_val = compute_or_ci_p(dff, flt)
    results.append({
        "Subgroup Name": name,
        "Method": "Autostrat",
        "OR": round(or_val, 2),
        "CI": f"({round(ci_lo, 2)}, {round(ci_hi, 2)})",
        "p-value": f"{p_val:.2e}"
    })

for name, flt in study_subgroups.items():
    or_val, ci_lo, ci_hi, p_val = compute_or_ci_p(dff, flt)
    results.append({
        "Subgroup Name": name,
        "Method": "Literature",
        "OR": round(or_val, 2),
        "CI": f"({round(ci_lo, 2)}, {round(ci_hi, 2)})",
        "p-value": f"{p_val:.2e}"
    })

# Convert to DataFrame
comparison_df = pd.DataFrame(results)

# Optional: sort by method and subgroup name
comparison_df.sort_values(by=["Method", "Subgroup Name"], inplace=True)

# Make a copy
combined_results = comparison_df.copy()

# --- Split CI string into CI_lower and CI_upper ---
combined_results[["CI_lower", "CI_upper"]] = combined_results["CI"].str.strip("()").str.split(",", expand=True).astype(float)

# --- Convert p-value string to float (if needed) ---
combined_results["p-value"] = combined_results["p-value"].astype(float)

# Optional: sort if needed
combined_results = combined_results.sort_values(by="Method", ascending=False).reset_index(drop=True)

combined_results = combined_results.iloc[::-1].reset_index(drop=True)

# For Nanoro

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Assuming your DataFrame is loaded as dff_ag
dff = dff_nan.copy()

# --------------------------------------------------------------------
# 1. Define AUTOSTRAT-DISCOVERED SUBGROUP MASKS
# --------------------------------------------------------------------
# --------------------------------------------------------
# Model Filter 1 (from Option 1)
model_filter1 = (
    (dff["days_veg_qc"] == 7) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
    (dff["sex"] == 1) &
    (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0)
)

# --------------------------------------------------------
# Model Filter 2 (from Option 2)
model_filter2 = (
    (dff["sex"] == 1) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
    (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0)
)

# --------------------------------------------------------
# Model Filter 3 (from Option 3)
model_filter3 = (
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
    (dff["waist_circumference_qc"] > 851.0) & (dff["waist_circumference_qc"] <= 1396.0)
)


model_subgroups = {
    "Model 1": model_filter1, # "mvpa, diabetes history, waist-hip, bmi"
    "Model 2": model_filter2, # "smoking, age, waist-hip, waist circumference, sex"
    "Model 3": model_filter3 # "waist-hip, age"
}

# --------------------------------------------------------------------
# 2. Define LITERATURE-BASED SUBGROUP MASKS
# --------------------------------------------------------------------

# Waist circumference cutoffs in mm
male_waist_cutoff = 940
female_waist_cutoff = 800

study_filter1 = (
    (dff["diabetes_history_qc"] == 1) &
    (dff["bmi_c_qc"] >= 23.0)
)

study_filter2 = (
    (
        ((dff["sex"] == 1) & (dff["waist_circumference_qc"] >= male_waist_cutoff)) |
        ((dff["sex"] == 0) & (dff["waist_circumference_qc"] >= female_waist_cutoff))
    ) &
    (dff["mvpa_c"] < 150)
)

study_filter3 = (
    (dff["age"] >= 45) & study_filter2
)

study_subgroups = {
    "Study 1": study_filter1, # "bmi >= 30, diabetes history"
    "Study 2": study_filter2, # "waist circumference, mvpa < 150"
    "Study 3": study_filter3 # "age >= 45, waist circumference, mvpa < 150"
}

# --------------------------------------------------------------------
# 3. Function to Compute OR, CI, and P-value using Logistic Regression
# --------------------------------------------------------------------
def compute_or_ci_p(data, mask, outcome_col="diabetes_status_c_qc"):
    temp = data.copy()
    temp["in_subgroup"] = mask.astype(int)
    
    model = smf.glm(formula=f"{outcome_col} ~ in_subgroup", 
                    data=temp, 
                    family=sm.families.Binomial()).fit()
    
    coef = model.params["in_subgroup"]
    se = model.bse["in_subgroup"]
    pval = model.pvalues["in_subgroup"]

    OR = np.exp(coef)
    CI_lower = np.exp(coef - 1.96 * se)
    CI_upper = np.exp(coef + 1.96 * se)

    return OR, CI_lower, CI_upper, pval

# --------------------------------------------------------------------
# 4. Loop through all subgroups and compile stats
# --------------------------------------------------------------------
results = []

for name, flt in model_subgroups.items():
    or_val, ci_lo, ci_hi, p_val = compute_or_ci_p(dff, flt)
    results.append({
        "Subgroup Name": name,
        "Method": "Autostrat",
        "OR": round(or_val, 2),
        "CI": f"({round(ci_lo, 2)}, {round(ci_hi, 2)})",
        "p-value": f"{p_val:.2e}"
    })

for name, flt in study_subgroups.items():
    or_val, ci_lo, ci_hi, p_val = compute_or_ci_p(dff, flt)
    results.append({
        "Subgroup Name": name,
        "Method": "Literature",
        "OR": round(or_val, 2),
        "CI": f"({round(ci_lo, 2)}, {round(ci_hi, 2)})",
        "p-value": f"{p_val:.2e}"
    })

# Convert to DataFrame
comparison_df = pd.DataFrame(results)

# Optional: sort by method and subgroup name
comparison_df.sort_values(by=["Method", "Subgroup Name"], inplace=True)

# Make a copy
combined_results = comparison_df.copy()

# --- Split CI string into CI_lower and CI_upper ---
combined_results[["CI_lower", "CI_upper"]] = combined_results["CI"].str.strip("()").str.split(",", expand=True).astype(float)

# --- Convert p-value string to float (if needed) ---
combined_results["p-value"] = combined_results["p-value"].astype(float)

# Optional: sort if needed
combined_results = combined_results.sort_values(by="Method", ascending=False).reset_index(drop=True)

combined_results = combined_results.iloc[::-1].reset_index(drop=True)