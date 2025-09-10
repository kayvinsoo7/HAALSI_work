#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AG_autostrat_cli.py

A cleaned, modular CLI version of your Autostrat workflow that:
- Loads ONE combined CSV (must include column 'site' with codes).
- Subsets by site for discovery/validation/transferability.
- Reproduces your MDSS scan + penalty sweeps + subgroup rules.
- Computes OR/CI/p (GLM and 2x2), PSM validation vs DIMAMO, heterogeneity.
- Plots (forest, venn, ROC, prevalence) similar to your notebook.

Usage examples
--------------
# Discovery on Agincourt (site_id=1), with penalty sweep + figures
python AG_autostrat_cli.py \
  --combined_csv merged_mock.csv \
  --site_id 1 \
  --phase discovery \
  --outdir results_agincourt \
  --make_plots

# Validation on DIMAMO (site_id=2) using Agincourt model rules
python AG_autostrat_cli.py \
  --combined_csv merged_mock.csv \
  --site_id 2 \
  --phase validation \
  --rules_from_site 1 \
  --outdir results_dimamo \
  --make_plots

# Transferability (Agincourt rules) across Nairobi (3) & Nanoro (4)
python AG_autostrat_cli.py \
  --combined_csv merged_mock.csv \
  --site_id 3 \
  --phase transfer \
  --rules_from_site 1 \
  --outdir results_nairobi \
  --make_plots

Dependencies
------------
pandas numpy matplotlib seaborn scipy statsmodels scikit-learn mdss matplotlib-venn
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import is_numeric_dtype
from scipy import stats
from scipy.stats import chi2
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf

# MDSS / Autostrat
from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.MDSS import MDSS

# Optional venn
try:
    from matplotlib_venn import venn3
except Exception:
    venn3 = None


# -----------------------
# Config / Schema
# -----------------------

SITE_NAME = {1: "Agincourt", 2: "Dimamo", 3: "Nairobi", 4: "Nanoro"}

# Columns you guaranteed in the mock set
CORE_FEATURES = [
    "age","sex","highest_level_of_education_qc","partnership_status_c_qc",
    "ses_site_quintile_c","occupation_qc","alcohol_use_status_c_qc","mvpa_c",
    "smoking_status_c_qc","diabetes_history_qc","bmi_c_qc","waist_hip_r_c_qc",
    "hip_circumference_qc","waist_circumference_qc","days_fruit_qc","days_veg_qc",
]
OUTCOME = "diabetes_status_c_qc"

# Extra columns used in your notebook if present (script degrades gracefully if missing)
EXTRA_FEATURES = [
    "ldl_qc","weight_qc","bp_sys_average_qc","bp_dia_average_qc","fruit_servings_qc",
    "glucose_qc","ur_creatinine_qc","triglycerides_qc","visceral_fat_qc","servings_veg_qc",
    "hdl_qc","cholesterol_qc","friedewald_ldl_c_c_qc","egfr_c_qc","acr_qc","use_drug_qc"
]
POSSIBLE_STUDY_ID = ["study_id"]

CONTINUOUS_CANDIDATES = [
    "age","weight_qc","bmi_c_qc","waist_hip_r_c_qc","visceral_fat_qc",
    "waist_circumference_qc","bp_sys_average_qc","bp_dia_average_qc","glucose_qc",
    "ur_creatinine_qc","mvpa_c","ldl_qc","hip_circumference_qc","hdl_qc","cholesterol_qc",
    "friedewald_ldl_c_c_qc","triglycerides_qc","egfr_c_qc","acr_qc"
]

SENTINELS = {-999: np.nan, -555: np.nan, -222: np.nan, -111: np.nan, 999: np.nan}


# -----------------------
# IO & preprocessing
# -----------------------

def load_combined_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Combined CSV not found: {p}")
    df = pd.read_csv(p)
    if "site" not in df.columns:
        raise ValueError("Combined CSV is missing column 'site'.")
    if OUTCOME not in df.columns:
        raise ValueError(f"Combined CSV is missing outcome column '{OUTCOME}'.")
    return df


def subset_site(df: pd.DataFrame, site_id: int) -> pd.DataFrame:
    d = df.loc[df["site"] == site_id].copy()
    if d.empty:
        raise ValueError(f"No rows found for site_id={site_id}")
    return d


def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(SENTINELS)


def iter_impute_continuous(df: pd.DataFrame, cont_cols: List[str]) -> pd.DataFrame:
    if not cont_cols:
        return df
    cols = [c for c in cont_cols if c in df.columns]
    if not cols:
        return df
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        initial_strategy="mean",
        max_iter=10,
        imputation_order="ascending",
        skip_complete=True,
        random_state=0,
    )
    arr = imputer.fit_transform(df[cols])
    df_imputed = df.copy()
    df_imputed[cols] = arr
    # ensure age stays int
    if "age" in df_imputed.columns:
        df_imputed["age"] = df_imputed["age"].round().astype(int)
    return df_imputed


def filter_age_sex(df: pd.DataFrame, age_min: int, age_max: int, sex_filter: int | None) -> pd.DataFrame:
    out = df.copy()
    if "age" in out.columns:
        out = out[(out["age"] >= age_min) & (out["age"] <= age_max)]
    if sex_filter is not None and "sex" in out.columns:
        out = out[out["sex"] == sex_filter]
    return out


# -----------------------
# Helpers for MDSS scan
# -----------------------

def numeric_scan_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 10:
            cols.append(c)
    return cols


def get_str_bin(x: pd.Interval) -> str:
    return f"{round(x.left, 2)} - {round(x.right, 2)}"


def custom_qcut_series(ser: pd.Series, q: int = 10, contiguous: bool = True) -> Tuple[pd.Series, List[str]]:
    sub = ser.copy()
    sub = sub.replace(SENTINELS)
    real = sub.dropna()
    if real.empty:
        return ser.astype(str), []
    binned = pd.qcut(real, q=q, duplicates="drop")
    labels = binned.apply(get_str_bin).astype(str)
    out = ser.astype(object)
    out.loc[labels.index] = labels.values
    bins = sorted(labels.unique(), key=lambda s: float(s.split(" - ")[0]))
    return out.astype(str), bins


def compress_contiguous(subset: Dict[str, List[str]], cont_bins: Dict[str, List[str]]) -> Dict[str, List[str]]:
    new = {}
    for col, values in subset.items():
        vals = list(values)
        if col in cont_bins and vals:
            # merge contiguous ranges where possible
            try:
                start = vals[0].split(" - ")[0]
                end = vals[-1].split(" - ")[-1] if isinstance(vals[-1], str) else vals[-2].split(" - ")[-1]
                merged = f"{start} - {end}"
                if isinstance(vals[-1], str):
                    new[col] = [merged]
                else:
                    new[col] = [merged, str(vals[-1])]
            except Exception:
                new[col] = [str(v) for v in vals]
        else:
            new[col] = [str(v) for v in vals]
    return new


def translate_subset_to_rule(subset: Dict[str, List[str]]) -> str:
    parts = []
    for k, vs in subset.items():
        parts.append(f"{k}[{' OR '.join(vs)}]")
    return " AND ".join(parts).replace("_", " ")


def count_literals(rule_str: str) -> int:
    return rule_str.count("AND") + 1 if rule_str else 0


# -----------------------
# Stats helpers
# -----------------------

def glm_or_ci_p(df: pd.DataFrame, mask: pd.Series, outcome_col: str = OUTCOME) -> Tuple[float, float, float, float]:
    tmp = df.copy()
    tmp["in_subgroup"] = mask.astype(int)
    model = smf.glm(formula=f"{outcome_col} ~ in_subgroup", data=tmp, family=smf.families.Binomial()).fit(disp=0)
    coef = model.params["in_subgroup"]
    se = model.bse["in_subgroup"]
    pval = model.pvalues["in_subgroup"]
    OR = float(np.exp(coef))
    CI_lo = float(np.exp(coef - 1.96 * se))
    CI_hi = float(np.exp(coef + 1.96 * se))
    return OR, CI_lo, CI_hi, float(pval)


def table2x2_or_ci_p(df: pd.DataFrame, mask: pd.Series, outcome_col: str = OUTCOME) -> Tuple[float, float, float, float, int, float]:
    sub = df[mask]
    out = df[~mask]
    a = int(sub[outcome_col].sum())
    b = int(len(sub) - a)
    c = int(out[outcome_col].sum())
    d = int(len(out) - c)
    n = len(sub)
    prev = (a / n) if n else 0.0
    if 0 in (a, b, c, d):
        return np.inf, np.inf, np.inf, 1.0, n, prev
    odds_ratio = (a / b) / (c / d)
    log_or = np.log(odds_ratio)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lo = np.exp(log_or - 1.96 * se)
    ci_hi = np.exp(log_or + 1.96 * se)
    z = log_or / se
    p = 2 * stats.norm.sf(abs(z))
    return float(odds_ratio), float(ci_lo), float(ci_hi), float(p), n, float(prev)


def psm_match(df: pd.DataFrame, mask: pd.Series, covars: List[str], outcome_col: str = OUTCOME) -> pd.DataFrame:
    tmp = df.copy()
    tmp["in_subgroup"] = mask.astype(int)
    tmp = tmp.dropna(subset=covars + ["in_subgroup"])
    if tmp.empty or tmp["in_subgroup"].nunique() < 2:
        return tmp
    X = tmp[covars]
    y = tmp["in_subgroup"]
    logit = LogisticRegression(solver="lbfgs", max_iter=1000)
    logit.fit(X, y)
    tmp["propensity"] = logit.predict_proba(X)[:, 1]
    treated = tmp[tmp["in_subgroup"] == 1].copy()
    control = tmp[tmp["in_subgroup"] == 0].copy()
    if treated.empty or control.empty:
        return tmp
    nbrs = NearestNeighbors(n_neighbors=1).fit(control[["propensity"]])
    _, idx = nbrs.kneighbors(treated[["propensity"]])
    matched_control = control.iloc[idx.flatten()]
    matched = pd.concat([treated, matched_control], ignore_index=True)
    return matched


# -----------------------
# Model & study subgroups (your thresholds)
# -----------------------

def model_rules() -> Dict[str, Dict]:
    # Matches your three Agincourt model-derived masks (WHR>0.9, +MVPA<=2448, +BMI>=21.37)
    return {
        "Model 3 (BMI+MVPA+FHx+WHR)": {
            "all": [
                (">=", "bmi_c_qc", 21.37),
                ("<=", "mvpa_c", 2448.0),
                ("==", "diabetes_history_qc", 1),
                (">", "waist_hip_r_c_qc", 0.9),
                ("<=", "waist_hip_r_c_qc", 1.16),
            ]
        },
        "Model 2 (MVPA+FHx+WHR)": {
            "all": [
                ("<=", "mvpa_c", 2448.0),
                ("==", "diabetes_history_qc", 1),
                (">", "waist_hip_r_c_qc", 0.9),
                ("<=", "waist_hip_r_c_qc", 1.16),
            ]
        },
        "Model 1 (FHx+WHR)": {
            "all": [
                ("==", "diabetes_history_qc", 1),
                (">", "waist_hip_r_c_qc", 0.9),
                ("<=", "waist_hip_r_c_qc", 1.16),
            ]
        },
    }


def study_rules() -> Dict[str, Dict]:
    # IDF-like WC cutoffs (mm): women>=800, men>=940; MVPA < 600 MET-min (~150 min)
    return {
        "Study 1 (BMI>=30 + FHx)": {
            "all": [
                (">=", "bmi_c_qc", 30.0),
                ("==", "diabetes_history_qc", 1),
            ]
        },
        "Study 2 (WC high + MVPA<600)": {
            "anyall": [  # sex-specific WC + MVPA
                {
                    "all": [
                        ("==", "sex", 1),
                        (">=", "waist_circumference_qc", 940.0),
                        ("<", "mvpa_c", 600.0),
                    ]
                },
                {
                    "all": [
                        ("==", "sex", 0),
                        (">=", "waist_circumference_qc", 800.0),
                        ("<", "mvpa_c", 600.0),
                    ]
                },
            ]
        },
        "Study 3 (Age>=45 + WC high + MVPA<600)": {
            "anyall": [
                {
                    "all": [
                        (">=", "age", 45),
                        ("==", "sex", 1),
                        (">=", "waist_circumference_qc", 940.0),
                        ("<", "mvpa_c", 600.0),
                    ]
                },
                {
                    "all": [
                        (">=", "age", 45),
                        ("==", "sex", 0),
                        (">=", "waist_circumference_qc", 800.0),
                        ("<", "mvpa_c", 600.0),
                    ]
                },
            ]
        },
    }


def apply_rule(df: pd.DataFrame, rule: Dict) -> pd.Series:
    # Supports {"all":[(op,col,thr), ...]} and {"anyall":[{"all":[...]}, {"all":[...]}]}
    def op_mask(df, op, col, thr):
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        if op == ">":  return df[col] > thr
        if op == ">=": return df[col] >= thr
        if op == "<":  return df[col] < thr
        if op == "<=": return df[col] <= thr
        if op == "==": return df[col] == thr
        if op == "!=": return df[col] != thr
        raise ValueError(f"Unsupported op: {op}")

    if "all" in rule:
        m = pd.Series(True, index=df.index)
        for op, col, thr in rule["all"]:
            m &= op_mask(df, op, col, thr)
        return m

    if "anyall" in rule:
        m = pd.Series(False, index=df.index)
        for block in rule["anyall"]:
            if "all" not in block:
                continue
            m_block = pd.Series(True, index=df.index)
            for op, col, thr in block["all"]:
                m_block &= op_mask(df, op, col, thr)
            m |= m_block
        return m

    raise ValueError("Rule must contain 'all' or 'anyall'.")


# -----------------------
# Visualization helpers
# -----------------------

def forest_plot(df_or: pd.DataFrame, title: str, out_png: Path):
    dfp = df_or.copy()
    dfp["rule"] = dfp.index
    dfp = dfp.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.55 * len(dfp))), dpi=300)
    y = np.arange(len(dfp))
    ax.errorbar(dfp["OR"], y, xerr=[dfp["OR"] - dfp["CI_low"], dfp["CI_high"] - dfp["OR"]],
                fmt="s", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1, color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels(dfp["rule"])
    ax.set_xlabel("Odds Ratio (log scale)")
    ax.set_xscale("log")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def venn_wrap(ax, set_list, labels, title=""):
    if venn3 is None:
        ax.text(0.5, 0.5, "matplotlib-venn not installed", ha="center", va="center")
        return
    v = venn3(subsets=set_list, set_labels=labels, ax=ax, alpha=0.4)
    ax.set_title(title)


def prevalence_barplot(df: pd.DataFrame, out_png: Path):
    # Uses the combined df to plot crude prevalence by site 1..4
    df = df[df["site"].isin([1,2,3,4])]
    df["Location"] = df["site"].map(SITE_NAME)
    sns.set(style="whitegrid", context="talk", font_scale=1.1)
    plt.figure(figsize=(10,6), dpi=300)
    ax = sns.barplot(x="Location", y=OUTCOME, data=df, edgecolor=".2", palette="Pastel1")
    ticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(100*t)}%" for t in ticks])
    ax.set_ylabel("Prevalence of T2D (%)")
    ax.set_xlabel("Cohorts")
    ax.grid(True, axis="y", linestyle="--", alpha=0.65, linewidth=0.6)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# -----------------------
# Main routines
# -----------------------

def run_discovery(
    df_site: pd.DataFrame,
    site_id: int,
    outdir: Path,
    penalties: List[float],
    num_iters: int = 10,
    make_plots: bool = False,
):
    site_name = SITE_NAME.get(site_id, f"site{site_id}")

    # 1) Replace sentinels, impute continuous
    df_site = replace_sentinels(df_site)
    cont_cols = [c for c in CONTINUOUS_CANDIDATES if c in df_site.columns]
    df_site = iter_impute_continuous(df_site, cont_cols)

    # 2) Expectation + targets
    if OUTCOME not in df_site.columns:
        raise ValueError(f"Outcome '{OUTCOME}' missing.")
    df_site["output"] = (df_site[OUTCOME] == 1).astype(int)
    df_site["expectation"] = df_site["output"].mean()

    # 3) Scan space: numeric columns, excluding bookkeeping
    exclude = set(POSSIBLE_STUDY_ID + ["site","output","expectation",OUTCOME])
    scan_cols = numeric_scan_columns(df_site, exclude=list(exclude))

    # Quantile bin continuous-like scan columns; track bin lists for contiguity compression
    dscan = df_site.copy()
    contiguous_bins: Dict[str, List[str]] = {}
    for c in scan_cols:
        binned, bins = custom_qcut_series(dscan[c], q=10, contiguous=True)
        dscan[c] = binned
        if bins:
            contiguous_bins[c] = bins

    # 4) Run MDSS across penalties
    scoring_function = Bernoulli(direction="positive")
    scanner = MDSS(scoring_function)

    rows = []
    df_by_penalty = {}
    for penalty in penalties:
        subset, score = scanner.scan(
            dscan[scan_cols],
            dscan[[OUTCOME]],
            dscan["expectation"],
            cpu=0.99,
            penalty=penalty,
            num_iters=num_iters,
            contiguous=contiguous_bins.copy(),
        )
        to_choose = dscan[subset.keys()].isin(subset).all(axis=1)
        temp_df = dscan.loc[to_choose]
        not_temp_df = dscan.loc[~to_choose]
        size = int(len(temp_df))
        tot = int(len(dscan))

        # 2x2 table over OUTCOME
        a = int(temp_df[OUTCOME].sum()); b = size - a
        c = int(not_temp_df[OUTCOME].sum()); d = (tot - size) - c

        if 0 in (a,b,c,d):
            or_val, ci, p = np.inf, (np.inf, np.inf), 0.0
        else:
            odds_temp = a / b
            odds_not  = c / d
            or_val = odds_temp / odds_not if odds_not != 0 else np.inf
            log_or = np.log(or_val); se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            ci = (float(np.exp(log_or - 1.96*se)), float(np.exp(log_or + 1.96*se)))
            z = log_or / se; p = float(2 * stats.norm.sf(abs(z)))

        rule_str = translate_subset_to_rule(
            compress_contiguous(subset, contiguous_bins)
        )

        rows.append({
            "Penalty": penalty,
            "Literals": count_literals(rule_str),
            "Subset": rule_str,
            "Size": size,
            "Size_%": round(100*size/tot, 2),
            "OR": np.round(or_val, 2) if np.isfinite(or_val) else np.inf,
            "CI": (round(ci[0],2) if np.isfinite(ci[0]) else np.inf,
                   round(ci[1],2) if np.isfinite(ci[1]) else np.inf),
            "Score": round(float(score), 3),
            "P_value": p,
            "P(T2D|S)%": round(100*a/size, 2) if size>0 else np.nan,
            "P(S|T2D)%": round(100*a/max(1, int(dscan[OUTCOME].sum())), 2),
        })
        df_by_penalty[f"df_{penalty}"] = temp_df

    res = pd.DataFrame(rows)
    res_path = outdir / f"{site_name}_autostrat_penalty_sweep.csv"
    res.to_csv(res_path, index=False)
    print(f"[{site_name}] Saved penalty sweep -> {res_path}")

    # 5) Save discovered rules JSON (top-3 distinct by literals as example)
    discovered_rules = []
    seen = set()
    for _, r in res.sort_values(["Score"], ascending=False).iterrows():
        if r["Subset"] in seen: continue
        seen.add(r["Subset"])
        discovered_rules.append({"penalty": r["Penalty"], "rule": r["Subset"]})
        if len(discovered_rules) >= 5:
            break
    json_path = outdir / f"{site_name}_discovered_rules.json"
    with open(json_path, "w") as f:
        json.dump(discovered_rules, f, indent=2)
    print(f"[{site_name}] Saved discovered rules -> {json_path}")

    # 6) Optional quick forest (model+study subgroups)
    if make_plots:
        mrules = model_rules()
        srules = study_rules()
        all_rules = {**mrules, **srules}
        rows = []
        for name, rule in all_rules.items():
            mask = apply_rule(df_site, rule)
            OR, lo, hi, p = glm_or_ci_p(df_site, mask)
            rows.append({"rule": name, "OR": OR, "CI_low": lo, "CI_high": hi, "p": p})
        df_or = pd.DataFrame(rows).set_index("rule")
        forest_plot(df_or, f"{SITE_NAME.get(site_id)}: Subgroup ORs", outdir / f"{site_name}_forest.png")
        print(f"[{site_name}] Saved forest plot -> {outdir / f'{site_name}_forest.png'}")


def run_validation_dimamo(
    df_ag: pd.DataFrame,
    df_dim: pd.DataFrame,
    outdir: Path,
    psm_covars: List[str],
    make_plots: bool = False,
):
    # Model vs Study masks (discovery on Agincourt, evaluate on Dimamo pre+post PSM)
    mrules = model_rules()
    srules = study_rules()
    all_rules = list(mrules.items()) + list(srules.items())

    rows = []
    for name, rule in all_rules:
        m_ag = apply_rule(df_ag, rule)
        m_di = apply_rule(df_dim, rule)

        # Discovery OR (Agincourt)
        OR_disc, _, _, _ = glm_or_ci_p(df_ag, m_ag)

        # Dimamo pre-PSM
        OR_pre, lo_pre, hi_pre, p_pre = glm_or_ci_p(df_dim, m_di)

        # Dimamo post-PSM
        matched_di = psm_match(df_dim, m_di, [c for c in psm_covars if c in df_dim.columns])
        OR_post, lo_post, hi_post, p_post = glm_or_ci_p(matched_di, matched_di["in_subgroup"] == 1)

        # Agincourt post-PSM (for heterogeneity symmetry)
        matched_ag = psm_match(df_ag, m_ag, [c for c in psm_covars if c in df_ag.columns])
        OR_post_ag, lo_post_ag, hi_post_ag, p_post_ag = glm_or_ci_p(matched_ag, matched_ag["in_subgroup"] == 1)

        rows.append({
            "Subgroup": name,
            "Method": "Model" if name.startswith("Model") else "Study",
            "OR_discovery": OR_disc,
            "OR_pre": OR_pre, "CI_low_pre": lo_pre, "CI_high_pre": hi_pre, "p_value_pre": p_pre,
            "OR_post": OR_post, "CI_low_post": lo_post, "CI_high_post": hi_post, "p_value_post": p_post,
            "OR_post_ag": OR_post_ag, "CI_low_post_ag": lo_post_ag, "CI_high_post_ag": hi_post_ag, "p_value_post_ag": p_post_ag
        })

    val = pd.DataFrame(rows)
    path = outdir / "validation_dimamo.csv"
    val.to_csv(path, index=False)
    print(f"[Validation] Saved -> {path}")

    # Minimal forest plot (pre vs post)
    if make_plots:
        # order: model top, then study
        val_sorted = pd.concat([
            val[val["Method"] == "Model"],
            val[val["Method"] == "Study"]
        ], axis=0).reset_index(drop=True)

        fig, (ax_left, ax_center, ax_right) = plt.subplots(1, 3, figsize=(15, 6), dpi=300,
                                                           gridspec_kw={"width_ratios": [0.5, 2.5, 1.5]})
        n = len(val_sorted); y = np.arange(n)

        # left: discovery OR box
        ax_left.set_ylim(-0.5, n-0.5); ax_left.set_xlim(0, 1); ax_left.set_xticks([]); ax_left.set_xticklabels([])
        for i, r in val_sorted.iterrows():
            color = "lightgreen" if r["Method"] == "Model" else "lightblue"
            ax_left.plot(0.5, y[i], "s", color=color, markersize=20, alpha=0.85)
            ax_left.text(0.5, y[i], f"{r['OR_discovery']:.2f}", ha="center", va="center", fontsize=9)
        ax_left.invert_yaxis()
        ax_left.set_yticks(y)
        ax_left.set_yticklabels(val_sorted["Subgroup"], fontsize=10)
        ax_left.set_title("Discovery (Agincourt)")
        for spine in ["top","right","bottom"]: ax_left.spines[spine].set_visible(False)

        # center: pre vs post CIs
        ax_center.set_ylim(-0.5, n-0.5)
        x_min = float(min(val_sorted["CI_low_pre"].min(), val_sorted["CI_low_post"].min()) * 0.8)
        x_max = float(max(val_sorted["CI_high_pre"].max(), val_sorted["CI_high_post"].max()) * 1.2)
        ax_center.set_xlim(x_min, x_max)
        off = 0.15
        for i, r in val_sorted.iterrows():
            ypre = y[i] - off; ypost = y[i] + off
            ax_center.plot([r["CI_low_pre"], r["CI_high_pre"]], [ypre, ypre], color="darkblue", lw=2)
            ax_center.plot(r["OR_pre"], ypre, "o", color="darkblue", markersize=5)
            ax_center.plot([r["CI_low_post"], r["CI_high_post"]], [ypost, ypost], color="darkred", lw=2)
            ax_center.plot(r["OR_post"], ypost, "o", color="darkred", markersize=5)
        ax_center.axvline(1.0, color="gray", linestyle="--", lw=1)
        ax_center.invert_yaxis()
        ax_center.set_yticks(y); ax_center.set_yticklabels([])
        ax_center.set_xlabel("Odds Ratio (95% CI)"); ax_center.set_title("Dimamo: Pre vs Post PSM")
        for spine in ["top","right"]: ax_center.spines[spine].set_visible(False)

        # right: -log10 p-values
        ppre = -np.log10(np.clip(val_sorted["p_value_pre"].values.astype(float), 1e-300, 1))
        ppost = -np.log10(np.clip(val_sorted["p_value_post"].values.astype(float), 1e-300, 1))
        bh = 0.35
        ax_right.set_ylim(-0.5, n-0.5); ax_right.invert_yaxis()
        ax_right.barh(y - bh/2, ppre, height=bh, color="steelblue", label="Pre-PSM")
        ax_right.barh(y + bh/2, ppost, height=bh, color="tomato", label="Post-PSM")
        ax_right.axvline(-np.log10(0.05), color="brown", linestyle="--", lw=1)
        ax_right.set_xlabel("-log10(p-value)"); ax_right.set_title("Significance (Dimamo)")
        ax_right.legend(loc="lower right", fontsize=9)
        for spine in ["top","right"]: ax_right.spines[spine].set_visible(False)

        plt.tight_layout()
        fig_path = outdir / "forest_validation_dimamo.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[Validation] Saved plot -> {fig_path}")


def run_transferability(
    df_ag: pd.DataFrame,
    df_nai: pd.DataFrame,
    df_nan: pd.DataFrame,
    outdir: Path,
    make_plots: bool = False,
):
    # Use the three model rules and compute ORs/prevalence for Agincourt, Nairobi, Nanoro
    mrules = model_rules()
    order = ["Model 1 (FHx+WHR)", "Model 2 (MVPA+FHx+WHR)", "Model 3 (BMI+MVPA+FHx+WHR)"]
    rows = []
    for name in order:
        rule = mrules[name]
        for pop_name, d in [("Agincourt", df_ag), ("Nairobi", df_nai), ("Nanoro", df_nan)]:
            m = apply_rule(d, rule)
            orv, lo, hi, p, size, prev = table2x2_or_ci_p(d, m)
            rows.append({
                "Group": name, "Population": pop_name,
                "OR": orv, "CI_low": lo, "CI_high": hi, "p_value": p,
                "subgroup_size": size, "t2d_prev": prev
            })
    df_plot = pd.DataFrame(rows)
    path = outdir / "transferability_table.csv"
    df_plot.to_csv(path, index=False)
    print(f"[Transferability] Saved -> {path}")

    if make_plots:
        # Simple forest per group with bars of prevalence
        colors = {"Agincourt": "#F06543", "Nairobi": "#A40E4C", "Nanoro": "#70A9A1"}
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        y_levels = list(reversed(order))
        y_ticks = []
        for i, g in enumerate(y_levels):
            base_y = i
            for k, pop in enumerate(["Agincourt","Nairobi","Nanoro"]):
                r = df_plot[(df_plot["Group"]==g)&(df_plot["Population"]==pop)].iloc[0]
                y = base_y + (k-1)*0.15
                ax.plot([r["CI_low"], r["CI_high"]], [y,y], color=colors[pop], lw=2)
                ax.plot(r["OR"], y, "s", color=colors[pop], markersize=6)
                ax.text(r["CI_high"]*1.03, y, f"({r['CI_low']:.2f}, {r['CI_high']:.2f})", va="center", fontsize=9)
            y_ticks.append(base_y)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_levels)
        ax.axvline(1.0, color="gray", linestyle="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Odds Ratio (95% CI)")
        ax.set_title("Transferability: ORs across populations")
        plt.tight_layout()
        fig_path = outdir / "transferability_forest.png"
        plt.savefig(fig_path, dpi=300); plt.close()
        print(f"[Transferability] Saved plot -> {fig_path}")


# -----------------------
# CLI
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_csv", required=True)
    ap.add_argument("--site_id", type=int, required=True, help="1=Agincourt, 2=Dimamo, 3=Nairobi, 4=Nanoro")
    ap.add_argument("--phase", choices=["discovery","validation","transfer","prevalence"], default="discovery")
    ap.add_argument("--rules_from_site", type=int, default=1, help="For validation/transfer: which site’s model rules to assume")
    ap.add_argument("--age_min", type=int, default=40)
    ap.add_argument("--age_max", type=int, default=60)
    ap.add_argument("--sex_filter", type=int, choices=[0,1], default=None, help="Optional: 0=female, 1=male")
    ap.add_argument("--outdir", default="results_out")
    ap.add_argument("--penalties", default="0.4,0.5,1.0,1.2,1.5,3.0,12.5")
    ap.add_argument("--make_plots", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load all
    df_all = load_combined_csv(args.combined_csv)

    # Optional high-level prevalence plot
    if args.phase == "prevalence":
        prevalence_barplot(df_all, outdir / "prevalence_by_site.png")
        print(f"[All sites] Saved -> {outdir / 'prevalence_by_site.png'}")
        return

    # Site subset + basic cleaning for the requested phase
    # Also prepare specific site frames for validation/transfer
    df_site = subset_site(df_all, args.site_id)
    df_site = replace_sentinels(df_site)
    df_site = iter_impute_continuous(df_site, [c for c in CONTINUOUS_CANDIDATES if c in df_site.columns])
    df_site = filter_age_sex(df_site, args.age_min, args.age_max, args.sex_filter)

    if args.phase == "discovery":
        penalties = [float(x.strip()) for x in args.penalties.split(",") if x.strip()]
        run_discovery(df_site, args.site_id, outdir, penalties, num_iters=10, make_plots=args.make_plots)
        return

    if args.phase == "validation":
        # Agincourt = discovery; Dimamo = validation
        df_ag = subset_site(df_all, 1)
        df_dim = subset_site(df_all, 2)
        df_ag = iter_impute_continuous(replace_sentinels(df_ag), [c for c in CONTINUOUS_CANDIDATES if c in df_ag.columns])
        df_dim = iter_impute_continuous(replace_sentinels(df_dim), [c for c in CONTINUOUS_CANDIDATES if c in df_dim.columns])
        df_ag = filter_age_sex(df_ag, args.age_min, args.age_max, args.sex_filter)
        df_dim = filter_age_sex(df_dim, args.age_min, args.age_max, args.sex_filter)

        # PSM covariates used in your code; fallback if missing
        psm_covars = [c for c in ["age","sex","triglycerides_qc","bmi_c_qc"] if c in df_dim.columns]
        if not psm_covars:
            psm_covars = [c for c in ["age","sex","bmi_c_qc"] if c in df_dim.columns]

        run_validation_dimamo(df_ag, df_dim, outdir, psm_covars, make_plots=args.make_plots)
        return

    if args.phase == "transfer":
        df_ag = subset_site(df_all, 1)
        df_nai = subset_site(df_all, 3)
        df_nan = subset_site(df_all, 4)
        df_ag = filter_age_sex(iter_impute_continuous(replace_sentinels(df_ag), [c for c in CONTINUOUS_CANDIDATES if c in df_ag.columns]),
                               args.age_min, args.age_max, args.sex_filter)
        df_nai = filter_age_sex(iter_impute_continuous(replace_sentinels(df_nai), [c for c in CONTINUOUS_CANDIDATES if c in df_nai.columns]),
                                args.age_min, args.age_max, args.sex_filter)
        df_nan = filter_age_sex(iter_impute_continuous(replace_sentinels(df_nan), [c for c in CONTINUOUS_CANDIDATES if c in df_nan.columns]),
                                args.age_min, args.age_max, args.sex_filter)

        run_transferability(df_ag, df_nai, df_nan, outdir, make_plots=args.make_plots)
        return


if __name__ == "__main__":
    main()
