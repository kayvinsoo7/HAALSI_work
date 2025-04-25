## **Study: Uncovering Hidden Risk Factors: A Bernoulli-based Stratification Method for Identifying High-Risk Type 2 Diabetes Subgroups**

### Research question

*Can we identify latent high-risk subgroups for type 2 diabetes and uncover significant interactions using easily accessible features across sub-Saharan Africa to improve screening and clinical decision-making?*
<!-- **Can we uncover latent high-risk subgroups for type 2 diabetes and identify meaningful interactions between anthropometric and lifestyle-based risk factor cut-offs across sub-Saharan African populations—and to what extent are these subgroups transferable across regions and sex, while improving risk prediction beyond conventional models?** -->

> ### **Objective 1 — Identify high-risk T2D subgroups in the Agincourt cohort using both comprehensive and clinically accessible feature sets**

- **Table A1** All-feature solution set (penalty, literals, n, prevalence, divergence score).  
- **Table 1** Easily accessible solution set (same columns)-to be used hereafter.  
- **Figure 1** Cumulative-distribution overlays (BMI, age, waist) showing the distribution difference within and outside of each discovered subgroup.  
<!-- - **Supplement S1** Autostrat search-path flowchart and stability filter that yields the final β subgroups. -->

---

> ### **Objective 2 — Validate the discovered subgroups, evaluating the effects of their risk factor cutoffs and benchmark them against established screening thresholds**

<!-- - **Table B1** Adjusted OR, 95 % CI, p-value for each subgroup. -->
<!-- , calibration slope, Hosmer–Lemeshow p for each subgroup.   -->
- **Figure 2A** Forest plot of OR, 95% CI, p-value, comparing our subgroups' ORs with established cut-offs in literature.  
- **Figure 2B** Venn diagrams and Jaccard similarity matrix score for overlapping subgroups.
- **Figure 3** Forest plots showing the effect sizes of all subgroups when assessed in a similar population (DIMAMO) before and after accounting for confounders using propensity score matching, and their corresponding significance level.
- **Table 2** Heterogeneity test using Cochran’s Q to estimate the similarity of subgroup individuals with the same characteristics in the Agincourt and DIMAMO cohort.
<!-- - **Supplement S4** Jaccard similarity matrix and descriptive overlap statistics. -->
<!-- - **Supplement S2** Complete literature rule bank with re-estimated ORs in Agincourt. -->

---

> ### **Objective 3 — Assess the cross-regional generalizability of discovered subgroups through reciprocal discovery-transfer analysis**

1. **Path 3a:** Assess the consistency of the effects of the discovered Agincourt subgroups in Nairobi and Nanoro.  
   - **Figure 4A** Forest plot stacking ORs across the three cohorts.  
   - **Figure 4B** Prevalence shifts (Nairobi/Nanoro relative to Agincourt).

2. **Path 3b:** Discover in Nairobi, and assess the consistency of the effects in Agincourt and Nanoro.  
   - **Figure 5A** Forest plot stacking ORs across the three cohorts.  
   - **Figure 5B** Prevalence shifts (Agincourt/Nanoro relative to Nairobi).

3. **Path 3c:** Discover in Nanoro, and assess the consistency of the effects in Agincourt and Nairobi.  
   - **Figure 6A** Parallel forest plot.  
   - **Figure 6B** Prevalence shifts (Agincourt/Nairobi relative to Nanoro).

4. **Predictive impact in targets**  
   - **Figures 7, 8, and 9** ROC curves and ΔAUC for each transfer path.  
   <!-- - **Supplement S3** Calibration diagnostics for all transferred models. -->

---

> ### **Objective 4 — Determine which risk-factor cut-offs are common across regions and which remain context-specific**

- **Figure 10** Heat-map (or Upset plot) marking the presence of every literal cut-off across Agincourt-accessible, Nairobi, and Nanoro discovery runs.  
<!-- - **Supplement S4** Jaccard similarity matrix and descriptive overlap statistics. -->

---

> ### **Objective 5 — Discover and validate sex-specific high-risk subgroups across the study regions**

- **Table A2** Male-specific and female-specific literals with ORs and prevalence (discovered in pooled Nairobi + Agincourt).  
- **Figure 6** Side-by-side forest plots showing sex-stratified effect sizes across regions.  
- **Supplement S5** Interaction model outputs (sex × subgroup term, CIs, p-values).

<!-- ### **Objective 6 — Quantify the incremental predictive value of subgroup membership beyond baseline risk factors**

- **Figure 7** For Agincourt: ROC panel comparing the base model (age, sex, BMI) against successive additions of each subgroup mask.  
- **Figure 7-supplement** Analogous ROC sets for Nairobi and Nanoro.  
- **Supplement S6** Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI) tables for every cohort. -->

<!-- **Integration Note.**  
- Tables A2, B1, C1–C2 and Figures 1–5, 7 reside in the main manuscript.  
- All “A1”, “D1” and “S-” materials go to the online appendix.   -->
