#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AG_autostrat_runner.py

Runs your Autostrat pipeline on a COMBINED CSV (with a 'site' column) while following
the same structure you used in the notebooks / AG_autostrat.py.

Usage:
  python AG_autostrat_runner.py \
      --combined_csv mock_t2d_datasets/combined_mock.csv \
      --site_id 1 \
      --outdir results_agincourt
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ---- sklearn bits you used in notebooks ----
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# If you import MDSS/Autostrat in your code, keep it here too:
# from mdss.ScoringFunctions.Bernoulli import Bernoulli
# from mdss.SubsetScanning import MDSS

# -----------------------
# Config matching your schema
# -----------------------
FEATURES = [
    'age', 'sex', 'highest_level_of_education_qc', 'partnership_status_c_qc',
    'ses_site_quintile_c', 'occupation_qc', 'alcohol_use_status_c_qc',
    'mvpa_c', 'smoking_status_c_qc', 'diabetes_history_qc', 'bmi_c_qc',
    'waist_hip_r_c_qc', 'hip_circumference_qc', 'waist_circumference_qc',
    'days_fruit_qc', 'days_veg_qc'
]
TARGET = ['diabetes_status_c_qc']
REQUIRED = FEATURES + TARGET + ['site']

SENTINELS = {-999: np.nan, -222: np.nan, -111: np.nan, 999: np.nan}

SITE_NAME = {1: "Agincourt", 2: "Dimamo", 3: "Nairobi", 4: "Nanoro"}


def load_combined(combined_csv: str) -> pd.DataFrame:
    p = Path(combined_csv)
    if not p.exists():
        raise FileNotFoundError(f"Combined CSV not found: {p}")
    df = pd.read_csv(p)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Combined CSV missing columns: {missing}")
    # stabilize order
    return df[REQUIRED].copy()


def subset_and_preprocess(df_all: pd.DataFrame, site_id: int) -> pd.DataFrame:
    df = df_all.loc[df_all['site'] == site_id].copy()
    if df.empty:
        raise ValueError(f"No rows for site_id={site_id}")
    # replace sentinels
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].replace(SENTINELS)
    # iterative imputation on FEATURES only (never impute target)
    X = df[FEATURES].copy()
    imp = IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=25, sample_posterior=False)
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    df_imp = pd.concat([X_imp, df[TARGET]], axis=1)
    return df_imp


def compute_2x2_or_ci_p(y: np.ndarray, mask: np.ndarray):
    """
    Returns dict with a,b,c,d, OR, LCL, UCL, p, prevalence in S, size %.
    Fisher exact for p; Wald CI with Haldane-Anscombe if zeros.
    """
    from scipy.stats import fisher_exact, norm
    a = int(((y == 1) & mask).sum())
    b = int(((y == 0) & mask).sum())
    c = int(((y == 1) & (~mask)).sum())
    d = int(((y == 0) & (~mask)).sum())
    # continuity correction if zeros
    a_, b_, c_, d_ = (a, b, c, d)
    if 0 in [a, b, c, d]:
        a_, b_, c_, d_ = a+0.5, b+0.5, c+0.5, d+0.5
    OR = (a_ * d_) / (b_ * c_)
    se = np.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
    z = norm.ppf(0.975)
    LCL = np.exp(np.log(OR) - z*se)
    UCL = np.exp(np.log(OR) + z*se)
    _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    prev_S = a / (a + b) if (a + b) else np.nan
    size_pct = 100.0 * ((a + b) / (a + b + c + d))
    return dict(a=a, b=b, c=c, d=d, OR=float(OR), LCL=float(LCL), UCL=float(UCL),
                p=float(p), P_T2D_given_S=float(prev_S), size_pct=float(size_pct))


def eval_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    """
    Rule format:
      {"all":[{">":"waist_hip_r_c_qc", "thr":0.9}, {"<=":"mvpa_c","thr":2448}, {"==":"diabetes_history_qc","thr":1}]}
      {"any":[{"==":"sex","thr":1}, {">=":"bmi_c_qc","thr":30}]}
    """
    def _apply_atom(atom):
        if ">" in atom:  return df[atom[">"]]  >  atom["thr"]
        if ">=" in atom: return df[atom[">="]] >= atom["thr"]
        if "<" in atom:  return df[atom["<"]]  <  atom["thr"]
        if "<=" in atom: return df[atom["<="]] <= atom["thr"]
        if "==" in atom: return df[atom["=="]] == atom["thr"]
        if "!=" in atom: return df[atom["!="]] != atom["thr"]
        raise ValueError(f"Bad atom {atom}")

    if "all" in rule:
        m = pd.Series(True, index=df.index)
        for a in rule["all"]:
            m &= _apply_atom(a)
        return m
    if "any" in rule:
        m = pd.Series(False, index=df.index)
        for a in rule["any"]:
            m |= _apply_atom(a)
        return m
    raise ValueError("Rule must contain 'all' or 'any'.")


def evaluate_ruleset(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    y = df['diabetes_status_c_qc'].to_numpy()
    rows = []
    for name, rule in rules.items():
        mask = eval_rule(df, rule)
        stats = compute_2x2_or_ci_p(y, mask.to_numpy())
        stats.update(rule=name, n_in_s=int(mask.sum()))
        rows.append(stats)
    out = pd.DataFrame(rows).set_index("rule").sort_values("OR", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_csv", required=True)
    ap.add_argument("--site_id", type=int, required=True, help="1=Agincourt, 2=Dimamo, 3=Nairobi, 4=Nanoro")
    ap.add_argument("--outdir", default="results_autostrat")
    ap.add_argument("--export_rules_json", default=None,
                    help="If set during discovery, save discovered rules to this JSON")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    site_name = SITE_NAME.get(args.site_id, f"site{args.site_id}")

    # 1) Load combined; subset; preprocess
    df_all = load_combined(args.combined_csv)
    df = subset_and_preprocess(df_all, args.site_id)

    # 2) === PASTE YOUR AUTOSTRAT SCAN HERE =================================
    # Your code likely defines discovered subgroups based on MDSS scan (Autostrat).
    # Exactly paste the scan and the step that turns best subgroups into a list of rules.
    #
    # IMPORTANT: produce 'discovered_rules' as a dict:
    # discovered_rules = {
    #   "AS_g1_MVPA<=2448 + FHx + WHR>0.9": {"all":[{"<=":"mvpa_c","thr":2448},{"==":"diabetes_history_qc","thr":1},{">":"waist_hip_r_c_qc","thr":0.9}]},
    #   "AS_g2_BMI>21.4 + MVPA<=2448 + FHx + WHR>0.9": {...},
    #   "AS_g3_Age>42 + BMI>21.4 + MVPA<=2448 + FHx + WHR>0.9": {...},
    # }
    #
    # For now I’ll include the paper’s representative rules as placeholders. Replace with your scan output.
    discovered_rules = {
        "AS_g1_MVPA<=2448 + FHx + WHR>0.9": {
            "all": [
                {"<=": "mvpa_c", "thr": 2448},
                {"==": "diabetes_history_qc", "thr": 1},
                {">": "waist_hip_r_c_qc", "thr": 0.9},
            ]
        },
        "AS_g2_BMI>21.4 + MVPA<=2448 + FHx + WHR>0.9": {
            "all": [
                {">": "bmi_c_qc", "thr": 21.4},
                {"<=": "mvpa_c", "thr": 2448},
                {"==": "diabetes_history_qc", "thr": 1},
                {">": "waist_hip_r_c_qc", "thr": 0.9},
            ]
        },
        "AS_g3_Age>42 + BMI>21.4 + MVPA<=2448 + FHx + WHR>0.9": {
            "all": [
                {">": "age", "thr": 42},
                {">": "bmi_c_qc", "thr": 21.4},
                {"<=": "mvpa_c", "thr": 2448},
                {"==": "diabetes_history_qc", "thr": 1},
                {">": "waist_hip_r_c_qc", "thr": 0.9},
            ]
        },
    }
    # =========================================================================

    # 3) Export discovered rules if requested (for validation/transfer phases later)
    if args.export_rules_json:
        with open(args.export_rules_json, "w") as f:
            json.dump(discovered_rules, f, indent=2)

    # 4) Evaluate discovered rules (plus optionally your study-defined comparators)
    df_or = evaluate_ruleset(df, discovered_rules)
    df_or_path = outdir / f"{site_name}_or_table.csv"
    df_or.to_csv(df_or_path)
    print(f"[{site_name}] Saved OR table -> {df_or_path}")

    # If you want to evaluate literature comparators as well, add them here (same format):
    # study_rules = {...}; df_or2 = evaluate_ruleset(df, study_rules); df_or2.to_csv(...)

if __name__ == "__main__":
    main()
