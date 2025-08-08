#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AG_autostrat_clean.py

Reproducible, CLI-driven version of your notebook pipeline, adapted to a single
combined CSV with a 'site' column (Option B). Closely follows your code.

Key steps per site:
- load combined CSV, subset by --site_id
- sentinel cleanup + MICE (BayesianRidge) on continuous vars
- optional filters (--age_min/--age_max, --sex_filter)
- MDSS scan across penalties -> CSV
- build model-derived & study rules -> OR/CI/p comparisons -> CSV (+ forest plot)
- optional DIMAMO validation with PSM & heterogeneity (+ plots)

Examples:
  # Discovery on Agincourt (site_id=1)
  python AG_autostrat_clean.py --combined_csv merged_mock.csv --site_id 1 \
      --phase discovery --outdir results_ag --make_plots

  # Validation on Dimamo (site_id=2) applying built-in Agincourt rules
  python AG_autostrat_clean.py --combined_csv merged_mock.csv --site_id 2 \
      --phase validation --apply_ag_rules --outdir results_dim --make_plots

Requires:
  pandas numpy scikit-learn statsmodels matplotlib seaborn matplotlib-venn mdss
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

# imputation
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.neighbors import NearestNeighbors

# stats
from scipy import stats
import statsmodels.formula.api as smf

# mdss
from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.MDSS import MDSS


# -----------------------
# Config / Schema
# -----------------------

REQUIRED_COLS = [
    "age","sex","highest_level_of_education_qc","partnership_status_c_qc",
    "ses_site_quintile_c","occupation_qc","alcohol_use_status_c_qc","mvpa_c",
    "smoking_status_c_qc","diabetes_history_qc","bmi_c_qc","waist_hip_r_c_qc",
    "hip_circumference_qc","waist_circumference_qc","days_fruit_qc","days_veg_qc",
    "diabetes_status_c_qc","site"
]

CONTINUOUS = [
    "age", "weight_qc", "bmi_c_qc", "waist_hip_r_c_qc","visceral_fat_qc",
    "waist_circumference_qc", "bp_sys_average_qc","bp_dia_average_qc",
    "glucose_qc", "ur_creatinine_qc","mvpa_c", "ldl_qc", "hip_circumference_qc",
    "hdl_qc", "cholesterol_qc", "friedewald_ldl_c_c_qc","triglycerides_qc",
    "egfr_c_qc","acr_qc"
]

# this is the superset of features you used
FEATURES = [
    'age', 'sex', 'highest_level_of_education_qc','partnership_status_c_qc',
    'ses_site_quintile_c','occupation_qc','alcohol_use_status_c_qc','mvpa_c','ldl_qc',
    'smoking_status_c_qc','diabetes_history_qc','bmi_c_qc','waist_hip_r_c_qc',
    'weight_qc','hip_circumference_qc','waist_circumference_qc','bp_sys_average_qc',
    'bp_dia_average_qc','fruit_servings_qc','glucose_qc','ur_creatinine_qc',
    'triglycerides_qc','visceral_fat_qc','servings_veg_qc','hdl_qc','cholesterol_qc',
    'friedewald_ldl_c_c_qc','days_fruit_qc','days_veg_qc','egfr_c_qc','acr_qc','use_drug_qc'
]

TARGET = ["diabetes_status_c_qc"]

SITE_NAME = {1: "Agincourt", 2: "Dimamo", 3: "Nairobi", 4: "Nanoro"}

SENTINELS = {-999: np.nan, -555: np.nan, -222: np.nan, -111: np.nan, 999: np.nan}


# -----------------------
# IO & preprocessing
# -----------------------

def load_combined_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Combined CSV is missing columns: {missing}")
    return df

def subset_site(df: pd.DataFrame, site_id: int) -> pd.DataFrame:
    out = df.loc[df["site"] == site_id].copy()
    if out.empty:
        raise ValueError(f"No rows for site_id={site_id}")
    return out

def sentinel_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(SENTINELS)

def mice_impute(df: pd.DataFrame, cont_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cont_cols if c in df.columns]
    if not cols:
        return df
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        initial_strategy="mean",
        max_iter=10,
        imputation_order="ascending",
        skip_complete=True,
        random_state=0
    )
    imputer.fit(df[cols])
    imputed = pd.DataFrame(imputer.transform(df[cols]), columns=cols, index=df.index)
    df = df.copy()
    df[cols] = imputed[cols]
    if "age" in df.columns:
        df["age"] = df["age"].round().astype(int)
    return df

def apply_filters(df: pd.DataFrame, age_min: int|None, age_max: int|None, sex_filter: int|None) -> pd.DataFrame:
    out = df.copy()
    if age_min is not None:
        out = out[out["age"] >= age_min]
    if age_max is not None:
        out = out[out["age"] <= age_max]
    if sex_filter is not None:
        out = out[out["sex"] == sex_filter]
    return out


# -----------------------
# Binning / Search space
# -----------------------

def is_numeric_dtype_series(s: pd.Series) -> bool:
    return s.dtype.kind in "biufc"

def get_numeric_columns_for_scan(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    return [c for c in df.columns if is_numeric_dtype_series(df[c]) and c not in exclude and df[c].nunique() > 10]

def get_str_bin(x: pd.Interval) -> str:
    return f"{round(x.left,2)} - {round(x.right,2)}"

def custom_qcut(series: pd.Series, contiguous: bool=True, q: int=10) -> Tuple[pd.Series, List[str]]:
    ser = series.copy()
    sub = ser[~ser.isna()]
    if sub.empty:
        return ser.astype(object), []
    bins = pd.qcut(sub, q, duplicates="drop")
    lab = bins.apply(get_str_bin).astype(str)
    ser.loc[lab.index] = lab
    uniq = list(pd.unique(lab))
    uniq_sorted = sorted(uniq, key=lambda s: float(s.split(" - ")[0]))
    return ser.astype(object), uniq_sorted

def compress_contiguous(subset: Dict[str, List[str]], cont_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    new = {}
    for col, vals in subset.items():
        if col in cont_map and vals:
            # merge if sequential intervals
            try:
                start = vals[0].split(" - ")[0]
                end = vals[-1].split(" - ")[-1]
                merged = [f"{start} - {end}"] if len(vals) > 1 else [vals[0]]
                new[col] = merged
            except Exception:
                new[col] = [str(v) for v in vals]
        else:
            new[col] = [str(v) for v in vals]
    return new

def translate_subset_to_rule(subset: Dict[str, List[str]]) -> str:
    parts = []
    for k, v in subset.items():
        parts.append(f"{k}[{' OR '.join(v)}]")
    return " AND ".join(parts)

def count_conditions(rule_str: str) -> int:
    return len(rule_str.replace("AND", "OR").split("OR"))


# -----------------------
# MDSS scanning
# -----------------------

def run_mdss_scan(dff2: pd.DataFrame, search_cols: List[str], expectation: pd.Series,
                  contiguous_map: Dict[str, List[str]], penalties: List[float], iters: int=10):
    scoring_function = Bernoulli(direction="positive")
    scanner = MDSS(scoring_function)

    rows = []
    for pen in penalties:
        subset, score = scanner.scan(
            dff2[search_cols], dff2[TARGET], expectation,
            cpu=0.99, penalty=pen, num_iters=iters, contiguous=contiguous_map.copy()
        )
        to_choose = dff2[subset.keys()].isin(subset).all(axis=1)
        temp_df = dff2.loc[to_choose]
        not_temp = dff2.loc[~to_choose]

        # 2x2 table
        a = float(temp_df[TARGET].sum(numeric_only=True).sum())
        b = float(len(temp_df) - a)
        c = float(not_temp[TARGET].sum(numeric_only=True).sum())
        d = float(len(not_temp) - c)

        if min(a,b,c,d) == 0:
            or_val = np.inf; lcl=ucl=np.inf; pval=0.0
        else:
            odds_temp = a/b; odds_out = c/d
            or_val = odds_temp/odds_out if odds_out != 0 else np.inf
            log_or = np.log(or_val)
            se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            lcl = float(np.exp(log_or - 1.96*se))
            ucl = float(np.exp(log_or + 1.96*se))
            z = log_or / se
            pval = 2*stats.norm.sf(abs(z))

        rule_str = translate_subset_to_rule(compress_contiguous(subset, contiguous_map))
        rows.append({
            "Penalty": pen,
            "Literals": count_conditions(rule_str),
            "Subset": rule_str,
            "Size": len(temp_df),
            "Size_%": round(100*len(temp_df)/len(dff2), 2),
            "OR": np.round(or_val, 2) if np.isfinite(or_val) else np.inf,
            "CI": (None if not np.isfinite(or_val) else (round(lcl,2), round(ucl,2))),
            "Score": round(score,3),
            "P_value": pval
        })
    return pd.DataFrame(rows)


# -----------------------
# Rule definitions & evaluation
# -----------------------

def glm_or_ci(df: pd.DataFrame, mask: pd.Series, outcome="diabetes_status_c_qc"):
    tmp = df.copy()
    tmp["in_subgroup"] = mask.astype(int)
    model = smf.glm(formula=f"{outcome} ~ in_subgroup", data=tmp,
                    family=__import__("statsmodels.api").api.families.Binomial()).fit(disp=0)
    coef = model.params["in_subgroup"]; se = model.bse["in_subgroup"]; pval = model.pvalues["in_subgroup"]
    OR = float(np.exp(coef)); lcl = float(np.exp(coef - 1.96*se)); ucl = float(np.exp(coef + 1.96*se))
    return OR, lcl, ucl, pval

def model_derived_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    # These mirror your “Model 1–3” masks
    m1 = ((df["bmi_c_qc"] >= 21.37) & (df["bmi_c_qc"] <= 68.02) &
          (df["mvpa_c"] >= 0.0) & (df["mvpa_c"] <= 2448.0) &
          (df["diabetes_history_qc"] == 1) &
          (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16))
    m2 = ((df["mvpa_c"] >= 0.0) & (df["mvpa_c"] <= 2448.0) &
          (df["diabetes_history_qc"] == 1) &
          (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16))
    m3 = ((df["diabetes_history_qc"] == 1) &
          (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16))
    return {"Model 3": m3, "Model 2": m2, "Model 1": m1}  # order to match your plotting

def study_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    male_wc, female_wc = 940, 800
    s1 = ((df["diabetes_history_qc"] == 1) & (df["bmi_c_qc"] >= 30))
    s2 = ((((df["sex"] == 1) & (df["waist_circumference_qc"] >= male_wc)) |
           ((df["sex"] == 0) & (df["waist_circumference_qc"] >= female_wc))) & (df["mvpa_c"] < 600))
    s3 = ((df["age"] >= 45) & s2)
    return {"Study 3": s3, "Study 2": s2, "Study 1": s1}

def evaluate_masks(df: pd.DataFrame, masks: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, m in masks.items():
        orv, lcl, ucl, p = glm_or_ci(df, m)
        rows.append({"Subgroup": name, "OR": orv, "CI_low": lcl, "CI_high": ucl, "p_value": p})
    return pd.DataFrame(rows)


# -----------------------
# Validation (Dimamo PSM)
# -----------------------

def propensity_match(df: pd.DataFrame, mask: pd.Series, covars: List[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["in_subgroup"] = mask.astype(int)
    tmp = tmp.dropna(subset=covars + ["in_subgroup"])
    if tmp["in_subgroup"].nunique() < 2:
        return tmp
    X = tmp[covars]; y = tmp["in_subgroup"]
    logit = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, y)
    tmp["propensity"] = logit.predict_proba(X)[:,1]
    treated = tmp[tmp["in_subgroup"]==1]; control = tmp[tmp["in_subgroup"]==0]
    if treated.empty or control.empty:
        return tmp
    nn = NearestNeighbors(n_neighbors=1).fit(control[["propensity"]])
    _, idx = nn.kneighbors(treated[["propensity"]])
    match_ctrl = control.iloc[idx.flatten()]
    return pd.concat([treated, match_ctrl], axis=0)

def run_validation(ag_df: pd.DataFrame, dim_df: pd.DataFrame,
                   covars: List[str], outdir: Path, make_plots: bool):
    ag_masks = model_derived_masks(ag_df)
    rows = []
    for name, ag_m in ag_masks.items():
        # discovery (Ag)
        or_disc, lcl_d, ucl_d, p_d = glm_or_ci(ag_df, ag_m)

        # apply to Dimamo
        dim_m = model_derived_masks(dim_df)[name]
        or_pre, lcl_pre, ucl_pre, p_pre = glm_or_ci(dim_df, dim_m)

        # PSM in Dimamo
        matched_dim = propensity_match(dim_df, dim_m, covars)
        or_post, lcl_post, ucl_post, p_post = glm_or_ci(matched_dim, matched_dim["in_subgroup"].astype(bool))

        # PSM in Ag (for heterogeneity symmetry)
        matched_ag = propensity_match(ag_df, ag_m, covars)
        or_post_ag, lcl_post_ag, ucl_post_ag, p_post_ag = glm_or_ci(matched_ag, matched_ag["in_subgroup"].astype(bool))

        rows.append({
            "Subgroup": name, "Method": "Model",
            "OR_discovery": or_disc,
            "OR_pre": or_pre, "CI_low_pre": lcl_pre, "CI_high_pre": ucl_pre, "p_value_pre": p_pre,
            "OR_post_ag": or_post_ag, "CI_low_post_ag": lcl_post_ag, "CI_high_post_ag": ucl_post_ag, "p_value_post_ag": p_post_ag,
            "OR_post": or_post, "CI_low_post": lcl_post, "CI_high_post": ucl_post, "p_value_post": p_post
        })
    val_df = pd.DataFrame(rows)
    val_df.to_csv(outdir/"validation_dimamo.csv", index=False)
    print(f"[Validation] Saved -> {outdir/'validation_dimamo.csv'}")
    if make_plots:
        try:
            plot_validation_forest(val_df, outdir/outdir.name+"_validation_forest.pdf")
        except Exception as e:
            print(f"[warn] validation plot failed: {e}")
    return val_df


# -----------------------
# Plotting
# -----------------------

def plot_prevalence_by_site(df_all: pd.DataFrame, out_png: Path):
    sns.set(style="whitegrid", context="talk", font_scale=1.1)
    mp = {1:"Agincourt",2:"Dimamo",3:"Nairobi",4:"Nanoro"}
    df = df_all[df_all["site"].isin([1,2,3,4])].copy()
    df["Location"] = df["site"].map(mp)
    plt.figure(figsize=(8,5), dpi=300)
    ax = sns.barplot(x="Location", y="diabetes_status_c_qc", data=df, edgecolor=".2")
    ax.set_ylabel("Prevalence of T2D (%)")
    ax.set_xlabel("Cohorts")
    ax.set_yticklabels([f"{int(t*100)}%" for t in ax.get_yticks()])
    ax.grid(True, axis="y", linestyle="--", alpha=0.65, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_forest_or(df_cmp: pd.DataFrame, out_png: Path):
    # expects columns: Subgroup, Method, OR, CI_low, CI_high, p_value
    df = df_cmp.copy()
    df["Label"] = df["Subgroup"]
    df = df.sort_values(["Method","Subgroup"], ascending=[False, True])
    y = np.arange(len(df))
    fig, (ax_forest, ax_pvals) = plt.subplots(1,2, figsize=(10,6), dpi=300, gridspec_kw={"width_ratios":[3,1]})
    for i, r in df.iterrows():
        ax_forest.plot([r["CI_low"], r["CI_high"]],[i,i], color="darkblue")
        ax_forest.plot(r["OR"], i, "s", color="darkred", ms=5)
        ax_forest.text(r["CI_high"]*1.02, i, f"({r['CI_low']:.2f}, {r['CI_high']:.2f})", va="center", fontsize=9)
    ax_forest.axvline(1.0, ls="--", c="gray")
    ax_forest.set_yticks(y)
    ax_forest.set_yticklabels(df["Label"])
    ax_forest.set_xscale("log")
    ax_forest.set_xlabel("Odds Ratio (95% CI)")
    ax_forest.set_title("Subgroup ORs")
    # pvals
    neglogp = -np.log10(np.clip(df["p_value"].astype(float), 1e-300, None))
    ax_pvals.barh(y, neglogp, height=0.35)
    ax_pvals.set_xlabel("-log10(p)")
    ax_pvals.set_yticks(y); ax_pvals.set_yticklabels([])
    ax_pvals.axvline(-np.log10(0.05), ls="--", c="brown")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_validation_forest(val_df: pd.DataFrame, out_pdf: Path):
    df = val_df.iloc[::-1].reset_index(drop=True).copy()
    n = len(df)
    fig, (axL, axC, axR) = plt.subplots(1,3, figsize=(15,6), dpi=300, gridspec_kw={"width_ratios":[0.5,2.5,1.5]})
    y = np.arange(n)

    # Left: discovery OR
    axL.set_ylim(-0.5, n-0.5); axL.set_xlim(0,1); axL.set_xticks([]); axL.set_xticklabels([])
    for i, r in df.iterrows():
        color = "lightgreen" if r["Method"]=="Model" else "lightblue"
        axL.plot(0.5, y[i], "s", color=color, ms=18, alpha=0.9)
        axL.text(0.5, y[i], f"{r['OR_discovery']:.2f}", ha="center", va="center", fontsize=9, color="black")
    axL.invert_yaxis(); axL.set_yticks(y); axL.set_yticklabels(df["Subgroup"], fontsize=10)
    axL.set_title("Discovery (Agincourt)"); [axL.spines[s].set_visible(False) for s in ["top","right","bottom"]]

    # Center: pre vs post PSM in Dimamo
    low = float(pd.concat([df["CI_low_pre"], df["CI_low_post"]]).min())*0.8
    high = float(pd.concat([df["CI_high_pre"], df["CI_high_post"]]).max())*1.2
    axC.set_ylim(-0.5, n-0.5); axC.set_xlim(low, high)
    off = 0.15
    for i, r in df.iterrows():
        ypre, ypost = y[i]-off, y[i]+off
        axC.plot([r["CI_low_pre"], r["CI_high_pre"]],[ypre, ypre], c="navy", lw=2)
        axC.plot(r["OR_pre"], ypre, "o", c="navy", ms=5)
        axC.plot([r["CI_low_post"], r["CI_high_post"]],[ypost, ypost], c="darkred", lw=2)
        axC.plot(r["OR_post"], ypost, "o", c="darkred", ms=5)
    axC.axvline(1.0, ls="--", c="gray"); axC.invert_yaxis()
    axC.set_yticks(y); axC.set_yticklabels([]); axC.set_xlabel("Odds Ratio (95% CI)")
    axC.set_title("Validation (Dimamo): Pre vs Post PSM")
    [axC.spines[s].set_visible(False) for s in ["top","right"]]

    # Right: p-values
    neglog_pre = -np.log10(np.clip(df["p_value_pre"].astype(float), 1e-300, None))
    neglog_post = -np.log10(np.clip(df["p_value_post"].astype(float), 1e-300, None))
    bh = 0.35
    axR.set_ylim(-0.5, n-0.5); axR.invert_yaxis()
    axR.barh(y-bh/2, neglog_pre, height=bh, color="steelblue", label="Pre-PSM")
    axR.barh(y+bh/2, neglog_post, height=bh, color="tomato", label="Post-PSM")
    axR.axvline(-np.log10(0.05), ls="--", c="brown")
    axR.set_xlabel("-log10(p)"); axR.set_yticks(y); axR.set_yticklabels([]); axR.legend()
    [axR.spines[s].set_visible(False) for s in ["top","right"]]

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_csv", required=True)
    ap.add_argument("--site_id", type=int, required=True, help="1=Agincourt, 2=Dimamo, 3=Nairobi, 4=Nanoro")
    ap.add_argument("--phase", choices=["discovery","validation","transfer"], default="discovery")
    ap.add_argument("--age_min", type=int, default=None)
    ap.add_argument("--age_max", type=int, default=None)
    ap.add_argument("--sex_filter", type=int, default=None, help="0=female, 1=male, omit for both")
    ap.add_argument("--outdir", default="results_out")
    ap.add_argument("--make_plots", action="store_true")
    ap.add_argument("--apply_ag_rules", action="store_true",
                    help="For validation/transfer: evaluate the fixed Agincourt (model-derived) masks")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 0) prevalence plot (all sites) if requested
    df_all = load_combined_csv(args.combined_csv)
    if args.make_plots:
        try:
            plot_prevalence_by_site(df_all, outdir/"prevalence_by_site.png")
        except Exception as e:
            print(f"[warn] prevalence plot failed: {e}")

    # 1) subset this site + keep only needed cols
    keep_cols = list({*FEATURES, *TARGET, "site"})
    keep_cols = [c for c in keep_cols if c in df_all.columns]
    df_site = subset_site(df_all[keep_cols], args.site_id)

    # 2) clean sentinels & impute continuous
    df_site = sentinel_cleanup(df_site)
    df_site = mice_impute(df_site, CONTINUOUS)

    # 3) optional filters (align with your notebook defaults if desired)
    df_site = apply_filters(df_site, args.age_min, args.age_max, args.sex_filter)

    # 4) build numeric columns & search space (like your notebook)
    numeric_cols = get_numeric_columns_for_scan(df_site, exclude=TARGET)
    dff2 = df_site.copy()
    contiguous_map = {}
    search_space = []
    for col in numeric_cols:
        if col in df_site.columns and col not in ["site"]:
            ser, bins = custom_qcut(dff2[col])
            dff2[col] = ser
            if bins:
                contiguous_map[col] = bins
                search_space.append(col)

    # 5) define output & expectation
    dff2["output"] = (dff2[TARGET] == 1)
    dff2["expectation"] = dff2["output"].mean()

    # 6) MDSS scan (discovery)
    if args.phase == "discovery":
        penalties = [0.4, 0.5, 1.0, 1.2, 1.5, 3.0, 12.5]
        res_df = run_mdss_scan(dff2, search_space, dff2["expectation"], contiguous_map, penalties, iters=10)
        res_df.to_csv(outdir/f"{SITE_NAME.get(args.site_id,'site')}_mdss_results.csv", index=False)
        print(f"[MDSS] Saved -> {outdir/(SITE_NAME.get(args.site_id,'site')+'_mdss_results.csv')}")

    # 7) Compare model-derived vs. study rules for this site
    mdl = evaluate_masks(df_site, model_derived_masks(df_site))
    mdl["Method"] = "Autostrat"
    std = evaluate_masks(df_site, study_masks(df_site))
    std["Method"] = "Literature"
    cmp_df = pd.concat([mdl.assign(Subgroup=k) for k in mdl["Subgroup"].unique()] + [std], ignore_index=True)
    # ensure columns
    cmp_df = pd.concat([mdl, std], ignore_index=True)
    cmp_df.to_csv(outdir/f"{SITE_NAME.get(args.site_id,'site')}_subgroup_ORs.csv", index=False)
    print(f"[Compare] Saved -> {outdir/(SITE_NAME.get(args.site_id,'site')+'_subgroup_ORs.csv')}")
    if args.make_plots:
        try:
            plot_forest_or(cmp_df.assign(Method=cmp_df["Method"]), outdir/f"{SITE_NAME.get(args.site_id,'site')}_forest.png")
        except Exception as e:
            print(f"[warn] forest plot failed: {e}")

    # 8) Optional validation only makes sense if we have Agincourt & Dimamo
    if args.phase == "validation" and args.apply_ag_rules:
        if args.site_id != 2:
            print("[Validation] Tip: run this on Dimamo (site_id=2) to mirror your paper.")
        try:
            ag_df = subset_site(df_all[keep_cols].pipe(sentinel_cleanup).pipe(lambda d: mice_impute(d, CONTINUOUS)), 1)
            dim_df = subset_site(df_all[keep_cols].pipe(sentinel_cleanup).pipe(lambda d: mice_impute(d, CONTINUOUS)), 2)
            # optional same age/sex filters for both
            ag_df = apply_filters(ag_df, args.age_min, args.age_max, args.sex_filter)
            dim_df = apply_filters(dim_df, args.age_min, args.age_max, args.sex_filter)
            run_validation(ag_df, dim_df, covars=["age","sex","triglycerides_qc"], outdir=outdir, make_plots=args.make_plots)
        except Exception as e:
            print(f"[warn] validation step failed: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
