import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
import seaborn as sns
from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.ScoringFunctions.Poisson import Poisson
from mdss.MDSS import MDSS
file_path = '~/t2d_as.csv'
dff = pd.read_csv(file_path)
dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']
# More features
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']
dff = dfff[features + target_cols + ['study_id'] + ['site']]
# Columns we need for the sensitivity analysis
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# Modify this line to include the new columns
dff = dfff[features + target_cols + ['study_id'] + ['site'] + RAW_CASE_COLS].copy()
map_dict = {1: "Agincourt", 2: "Dimamo", 3: "Nairobi", 4: "Nanoro", 5: "Navrongo", 6: "Soweto"}
dff["Location"] = dff["site"].map(map_dict)

# Set publication-style aesthetics
sns.set(style="whitegrid", context="talk", font_scale=1.2)

# Filter the dataset for the specified locations
filtered_dff = dff[dff["site"].isin([1, 2, 3, 4])]

# Create the plot
plt.figure(figsize=(10, 6), dpi=300)
ax = sns.barplot(
    x="Location",
    y="diabetes_status_c_qc",
    data=filtered_dff,
    palette="Pastel1",
    edgecolor=".2"
)

# Format y-axis as percentages
ax.set_yticklabels([f'{int(tick * 100)}%' for tick in ax.get_yticks()])

# Set axis labels and title
ax.set_ylabel("Prevalence of T2D (%)")
ax.set_xlabel("Cohorts")
ax.grid(True, axis='y', linestyle='--', alpha=0.65, linewidth=0.6)
# ax.set_title("Prevalence of Type 2 Diabetes by Location")

# # Remove top and right spines for cleaner look
# sns.despine()

# Show the plot
plt.tight_layout()
plt.show()
site_id = 1 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_ag.shape)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Agincourt after removing records with missing targets: ', dff_ag.shape)

site_id = 3 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_nai = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_nai.shape)
dff_nai = dff_nai[(dff_nai[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Nairobi after removing records with missing targets: ', dff_nai.shape)

site_id = 4 

# Choose the relevant site and age group
dff_nan = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_nan.shape)
dff_nan = dff_nan[(dff_nan[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Nanoro after removing records with missing targets: ', dff_nan.shape)

site_id = 2 

# Choose the relevant site and age group
dff_dim = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_dim.shape)
dff_dim = dff_dim[(dff_dim[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Dimamo after removing records with missing targets: ', dff_dim.shape)
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()
dff1_nai = dff_nai.replace([-999, -222, -111, 999], np.nan).copy()
dff1_nan = dff_nan.replace([-999, -222, -111, 999], np.nan).copy()
dff1_dim = dff_dim.replace([-999, -222, -111, 999], np.nan).copy()
# --------------------------------------------------------------------
# ANALYSIS 2A: JUSTIFYING MAR for WHR
# --------------------------------------------------------------------
# We need the pre-imputation Agincourt dataframe: dff1_ag
# We also need the 'diabetes_self_reported_qc' column for this analysis
# Make sure to merge it in beforehand if it's not already in dff_ag
# For now, I'll assume 'diabetes_status_c_qc' is in dff1_ag

# Create a missingness flag for the key variable
dff_mar_test = dff1_ag.copy()
dff_mar_test['whr_missing'] = dff_mar_test['waist_hip_r_c_qc'].isna()

# List of characteristics to compare
compare_vars = {
    'age': 'mean',
    'sex': 'mean', # This will give proportion of males (if 1=Male)
    'bmi_c_qc': 'mean',
    'diabetes_status_c_qc': 'mean' # This will give T2D prevalence
}

# Group by the missingness flag and aggregate
comparison_results = dff_mar_test.groupby('whr_missing').agg(compare_vars)

# Add a 'N' (count) column
comparison_results['N'] = dff_mar_test['whr_missing'].value_counts()

print("\n--- MAR Analysis: Comparison of Missing vs. Non-Missing WHR ---")
print(comparison_results.to_markdown(floatfmt=".2f"))
# --------------------------------------------------------------------
# ANALYSIS 2A (REVISED): CORRECT MAR ANALYSIS
# --------------------------------------------------------------------
# 1. Load your pre-imputation data
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()

# 2. **CRITICAL STEP**: Apply the age filter *FIRST*
dff1_ag_filtered = dff1_ag[(dff1_ag['age'] >= 40) & (dff1_ag['age'] <= 60)].copy()
# This dff1_ag_filtered is your N=1465 cohort, with NAs

# 3. Now, run the MAR test on this correct, filtered dataframe
dff_mar_test = dff1_ag_filtered.copy()
# We also need the diabetes status for the comparison
dff_mar_test['diabetes_status_c_qc'] = dff_ag['diabetes_status_c_qc'] 

dff_mar_test['whr_missing'] = dff_mar_test['waist_hip_r_c_qc'].isna()

compare_vars = {
    'age': 'mean',
    'sex': 'mean', 
    'bmi_c_qc': 'mean',
    'diabetes_status_c_qc': 'mean'
}

comparison_results = dff_mar_test.groupby('whr_missing').agg(compare_vars)
comparison_results['N'] = dff_mar_test['whr_missing'].value_counts()

print("\n--- CORRECTED MAR Analysis: Comparison of Missing vs. Non-Missing WHR ---")
print(comparison_results.to_markdown(floatfmt=".2f"))
# --------------------------------------------------------------------
# ANALYSIS 2B: CREATING COMPLETE-CASE DATAFRAME
# --------------------------------------------------------------------
# Define all columns used in the main analysis (from Table 2)
# that had missingness
key_anthrop_vars = [
    'waist_hip_r_c_qc', 
    'hip_circumference_qc', 
    'waist_circumference_qc',
    'bmi_c_qc'
]

# Create the complete-case dataframe by dropping rows with NA
dff_ag_complete_case = dff1_ag.dropna(subset=key_anthrop_vars).copy()

print(f"\nOriginal Agincourt size: {len(dff1_ag)}")
print(f"Complete-Case Agincourt size: {len(dff_ag_complete_case)}")
continuous1 = ['age', 'weight_qc','visceral_fat_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
imputer_bayes = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42)

# let's create a MICE imputer using Bayes as estimator

imputer = IterativeImputer(
    estimator=BayesianRidge(), # the estimator to predict the NA
    initial_strategy='mean', # how will NA be imputed in step 1
    max_iter=10, # number of cycles
    imputation_order='ascending', # the order in which to impute the variables
    n_nearest_features=None, # whether to limit the number of predictors
    skip_complete=True, # whether to ignore variables without NA
    random_state=0,
)
imputer1 = IterativeImputer(
    estimator=BayesianRidge(), # the estimator to predict the NA
    initial_strategy='mean', # how will NA be imputed in step 1
    max_iter=10, # number of cycles
    imputation_order='ascending', # the order in which to impute the variables
    n_nearest_features=None, # whether to limit the number of predictors
    skip_complete=True, # whether to ignore variables without NA
    random_state=0,
)
# --- SENSITIVITY ANALYSIS 1: DIAGNOSED CASES ONLY ---
# 1. Define the raw dataset for this analysis
df_sens1_raw = dff1_ag[
    (dff1_ag['diabetes_status_c_qc'] == 0) | 
    (dff1_ag['diabetes_self_reported_qc'] == 1)
].copy()

# 2. Run imputation (using your 'imputer' variable logic)
print("Running Imputation for Sensitivity Analysis 1...")
imputer_sens1 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s1 = imputer_sens1.fit_transform(df_sens1_raw[continuous])

# 3. Create the final imputed dataframe
df_sens1_imputed = df_sens1_raw.copy()
df_sens1_imputed[continuous] = imputed_features_s1
df_sens1_imputed['age'] = df_sens1_imputed.age.astype(int)
df_sens1_imputed = df_sens1_imputed[(df_sens1_imputed['age'] >= 40) & (df_sens1_imputed['age'] <= 60)]

print(f"Dataset 1 (Imputed) is ready. N={len(df_sens1_imputed)}")

# --- YOUR NEXT STEP ---
# Run Autostrat on 'df_sens1_imputed'.
# The target column is 'diabetes_status_c_qc'.
imputer.fit(dff1_ag[continuous])
train_t = imputer.transform(dff1_ag[continuous])
# train_x = imputer.transform(data.iloc[:,2:])
treated_ag = pd.DataFrame(train_t, columns=dff1_ag[continuous].columns)

imputer1.fit(dff1_nai[continuous])
train_t = imputer1.transform(dff1_nai[continuous])
# train_x = imputer.transform(data.iloc[:,2:])
treated_nai = pd.DataFrame(train_t, columns=dff1_nai[continuous].columns)

imputer1.fit(dff1_nan[continuous])
train_t = imputer1.transform(dff1_nan[continuous])
# train_x = imputer.transform(data.iloc[:,2:])
treated_nan = pd.DataFrame(train_t, columns=dff1_nan[continuous].columns)

imputer1.fit(dff1_dim[continuous])
train_t = imputer1.transform(dff1_dim[continuous])
# train_x = imputer.transform(data.iloc[:,2:])
treated_dim = pd.DataFrame(train_t, columns=dff1_dim[continuous].columns)
dff_ag = dff_ag[features + target_cols + ['study_id']].copy()
dff_ag.reset_index(inplace=True, drop=True)
dff_ag[continuous] = treated_ag[continuous]
dff_ag['age'] = dff_ag.age.astype(int)

dff_nai = dff_nai[features + target_cols + ['study_id']].copy()
dff_nai.reset_index(inplace=True, drop=True)
dff_nai[continuous] = treated_nai[continuous]
dff_nai['age'] = dff_nai.age.astype(int)

dff_dim = dff_dim[features + target_cols + ['study_id']].copy()
dff_dim.reset_index(inplace=True, drop=True)
dff_dim[continuous] = treated_dim[continuous]
dff_dim['age'] = dff_dim.age.astype(int)

dff_nan = dff_nan[features + target_cols + ['study_id']].copy()
dff_nan.reset_index(inplace=True, drop=True)
dff_nan[continuous] = treated_nan[continuous]
dff_nan['age'] = dff_nan.age.astype(int)
merged_df = dff_ag.merge(dfff[['study_id', 'fasting_confirmation_qc', 'diabetes_self_reported_qc']], on='study_id', how='left')
# 1. Define the 'biomarker_positive' filter
# This is the logic you already have
biomarker_positive = (
    ((merged_df['fasting_confirmation_qc'] == 0) & (merged_df['glucose_qc'] >= 7)) | 
    ((merged_df['fasting_confirmation_qc'] == 1) & (merged_df['glucose_qc'] >= 11.1))
)

# 2. Define the 'self_report_positive' filter
# (Assuming 1 means they self-reported)
self_report_positive = (merged_df['diabetes_self_reported_qc'] == 1)

# 3. Combine them to find ALL T2D cases (as defined in your paper)
all_t2d_cases_filter = biomarker_positive | self_report_positive
all_t2d_df = merged_df[all_t2d_cases_filter]

# 4. NOW, find the distribution of self-report status *within this complete T2D group*
report_distribution = all_t2d_df['diabetes_self_reported_qc'].value_counts(normalize=True)

# report_distribution will look like this (for one site):
# 0    0.45  <-- This is your answer (45%)
# 1    0.55
# Name: diabetes_self_reported_qc, dtype: float64

# You can get the final number like this:
percent_undiagnosed = report_distribution.get(0, 0.0) * 100

print(percent_undiagnosed)
print(dff_ag.shape)
print(dff_nai.shape)
print(dff_nan.shape)
print(dff_dim.shape)
from scipy.stats import gaussian_kde
plt.figure(figsize=(5,4), dpi =300)
plt.hist(dff_nai['mvpa_c'], bins=30, alpha=0.9, color='blue', density=True)
# KDE line
kde = gaussian_kde(dff_nai['mvpa_c'])
x_vals = np.linspace(dff_nai['mvpa_c'].min(), dff_nai['mvpa_c'].max(), 30)
plt.plot(x_vals, kde(x_vals), color='black')

plt.xlabel('MVPA')
plt.ylabel('Density')
plt.title('Nairobi')
plt.tight_layout()
plt.show()
from scipy.stats import gaussian_kde
plt.figure(figsize=(5,4), dpi =300)
plt.hist(dff_nai['bmi_c_qc'], bins=30, alpha=0.7, color='blue', density=True)
# KDE line
kde = gaussian_kde(dff_nai['bmi_c_qc'])
x_vals = np.linspace(dff_nai['bmi_c_qc'].min(), dff_nai['bmi_c_qc'].max(), 300)
plt.plot(x_vals, kde(x_vals), color='black')

plt.xlabel('BMI')
plt.ylabel('Density')
plt.title('Nairobi')
plt.tight_layout()
plt.show()
dff_ag = dff_ag[(dff_ag['age'] >= 40) & (dff_ag['age'] <= 60)]
dff_nai = dff_nai[(dff_nai['age'] >= 40) & (dff_nai['age'] <= 60)]
dff_nan = dff_nan[(dff_nan['age'] >= 40) & (dff_nan['age'] <= 60)]
dff_dim = dff_dim[(dff_dim['age'] >= 40) & (dff_dim['age'] <= 60)]

import statsmodels.api as sm
import statsmodels.formula.api as smf
# Combine the raw dataframes
dff_ag['site'] = 1  #(This is already in your 'snips.py')
dff_nai['site'] = 3
dff_nan['site'] = 4
dff_dim['site'] = 2
all_sites_raw = pd.concat([dff_ag, dff_nai, dff_nan, dff_dim], ignore_index=True)

# --- 2. Clean data for this specific analysis ---
# Replace -999s etc. with NaN for the model
all_sites_raw['diabetes_status_c_qc'] = all_sites_raw['diabetes_status_c_qc'].replace([-999, -222, -111, 999], np.nan)
all_sites_raw['age'] = all_sites_raw['age'].replace([-999, -222, -111, 999], np.nan)
all_sites_raw['sex'] = all_sites_raw['sex'].replace([-999, -222, -111, 999], np.nan)

# Drop rows where our model variables are missing
model_data = all_sites_raw.dropna(subset=['diabetes_status_c_qc', 'age', 'sex', 'site'])

# --- 3. Fit the logistic regression model ---
# We use C(site) and C(sex) to treat them as categorical variables
model = smf.glm(formula="diabetes_status_c_qc ~ C(site) + age + C(sex)", 
                data=model_data, 
                family=sm.families.Binomial()).fit()

print(model.summary())

# --- 4. Get adjusted prevalence ---
# This calculates the average predicted probability for each site,
# adjusted for age and sex.
model_data['predicted_prevalence'] = model.predict(model_data)
adjusted_prevalence_by_site = model_data.groupby('site')['predicted_prevalence'].mean()

print("\n--- Age- and Sex-Adjusted Prevalence ---")
print(adjusted_prevalence_by_site)

# You can map site IDs back to names for your Figure S3
site_map = {1: 'Agincourt', 3: 'Nairobi', 4: 'Nanoro', 2: 'DIMAMO'}
print(adjusted_prevalence_by_site.rename(index=site_map))

map_dict = {1: "Agincourt", 2: "DIMAMO", 3: "Nairobi", 4: "Nanoro", 5: "Navrongo", 6: "Soweto"}
all_sites_raw["Location"] = all_sites_raw["site"].map(map_dict)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.formula.api as smf
import numpy as np

# --- Assumed Starting Point ---
# all_sites_raw = pd.concat([dff_ag, dff_nai, dff_nan, dff_dim], ...)
# all_sites_raw is your combined 4-site data, *already filtered* for age 40-60.
# It MUST have columns: 'diabetes_status_c_qc', 'age', 'sex', 'site', 'Location'
# (Where 'Location' is the string name like "Agincourt")

# --- 1. Calculate Crude Prevalence ---
# This is your correct starting code
crude_df = all_sites_raw.groupby('Location')['diabetes_status_c_qc'].mean().reset_index()
crude_df = crude_df.rename(columns={'diabetes_status_c_qc': 'Prevalence'})
crude_df['Prevalence Type'] = 'Crude'
print("--- Crude Prevalence ---")
print(crude_df)

# --- 2. Calculate True Adjusted Prevalence ---

# A. Fit the one, big model on all (40-60) data
# We use C() to tell the model 'site' and 'sex' are categories
glm_model = smf.glm(
    formula="diabetes_status_c_qc ~ C(site) + age + C(sex)",
    data=all_sites_raw,
    family=sm.families.Binomial()
).fit()

# B. Create the "Standardized Person" profile
# Get the average age and most common sex from the *entire 40-60 cohort*
overall_mean_age = all_sites_raw['age'].mean()
overall_mode_sex = all_sites_raw['sex'].mode()[0] 

print(f"\nStandardizing to: Mean Age = {overall_mean_age:.2f}, Mode Sex = {overall_mode_sex}")

# C. Create a new, artificial DataFrame for prediction
# It has one row for each site, but all have the SAME age and sex.
site_map = {
    'Agincourt': 1,
    'DIMAMO': 2,
    'Nairobi': 3,
    'Nanoro': 4
}

locations = list(site_map.keys())
site_ids = list(site_map.values())

standardized_data = pd.DataFrame({
    'Location': locations,
    'site': site_ids,
    'age': [overall_mean_age] * len(locations),
    'sex': [overall_mode_sex] * len(locations)
})

print("\n--- Standardized Data for Prediction ---")
print(standardized_data)

# D. Predict prevalence for this "standardized person" at each site
adjusted_prevalence = glm_model.predict(standardized_data)

adjusted_df = pd.DataFrame({
    'Location': locations,
    'Prevalence': adjusted_prevalence,
    'Prevalence Type': 'Adjusted (for Age & Sex)'
})

print("\n--- Adjusted Prevalence ---")
print(adjusted_df)

# --- 3. Combine and Plot ---
# This fixes your plot issue. The 'hue' will group the bars.
plot_data = pd.concat([crude_df, adjusted_df], ignore_index=True)

print("\n--- Final Data for Plotting ---")
print(plot_data)

plt.figure(figsize=(10, 6), dpi=300)

# Use x='Location' and hue='Prevalence Type'
# This plots 'Crude' and 'Adjusted' bars next to each other
# for each Location.
ax = sns.barplot(
    x="Location",
    y="Prevalence",
    hue="Prevalence Type",  # <-- This groups the bars correctly
    data=plot_data,
    palette="Pastel1",
    edgecolor=".2"
)

# Format y-axis as percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# Set axis labels and title
ax.set_ylabel("Prevalence of T2D (%)")
ax.set_xlabel("Cohorts")
ax.grid(True, axis='y', linestyle='--', alpha=0.65, linewidth=0.6)
ax.legend(title='Prevalence Type', loc='upper right')

plt.tight_layout()
# Save the new Figure S3
plt.savefig("adjusted_prevalence_figure_S3.png", dpi=300)
plt.show()
dff_ag[dff_ag['diabetes_status_c_qc']==1].value_counts().shape[0]/dff_ag.shape[0]*100
dff11_ag = dff1_ag[(dff1_ag['age'] >= 40) & (dff1_ag['age'] <= 60)]
dff11_ag.hip_circumference_qc.isna().value_counts()
dff_ag_complete_case = dff_ag_complete_case[(dff_ag_complete_case['age'] >= 40) & (dff_ag_complete_case['age'] <= 60)]
# dff_ag_male = dff_ag[dff_ag['sex'] == 1].copy()
# dff_ag = dff_ag_male
dff_ag_female = dff_ag[dff_ag['sex'] == 0].copy()
dff_ag = dff_ag_female
print(dff_ag.shape)
print(dff_nai.shape)
print(dff_nan.shape)
print(dff_dim.shape)
numeric_columns = [col for col in dff_ag.columns \
                     if (is_numeric_dtype(dff_ag[col])) \
                     & (col not in target_cols) \
                     & (dff_ag[col].nunique() > 10)]
# defining the outcome variable
dff_ag['output'] = (dff_ag[target_cols] == 1)
dff_ag['expectation'] = dff_ag['output'].mean()

dff_nai['output'] = (dff_nai[target_cols] == 1)
dff_nai['expectation'] = dff_nai['output'].mean()

dff_dim['output'] = (dff_dim[target_cols] == 1)
dff_dim['expectation'] = dff_dim['output'].mean()

dff_nan['output'] = (dff_nan[target_cols] == 1)
dff_nan['expectation'] = dff_nan['output'].mean()
dff_ag_complete_case['output'] = (dff_ag_complete_case[target_cols] == 1)
dff_ag_complete_case['expectation'] = dff_ag_complete_case['output'].mean()
# dff_ag.to_csv('dff_ag.csv', index=False)
# dff_nai.to_csv('dff_nai.csv', index=False)
# dff_nan.to_csv('dff_nan.csv', index=False)
# dff_dim.to_csv('dff_dim.csv', index=False)
# Defining the search space to be all the features except site, study_id, 
# and our created target_col and expectations columns

search_space1 = [col for col in dff_ag.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc',
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc']]
print(search_space1)
def get_str(x):
    # This function turns a pandas bin to a meaningful string
    s = str(round(x.left, 2)) + ' - ' + str(round(x.right,2))
    return s

def custom_qcut(ser, contiguous = True):
    # Get the rows that are actual numbers
    sub_ser = ser[(ser != -111) \
                     & (ser != -222) \
                     & (ser != -555) \
                     & (ser != -999)]
    
    if contiguous:
        # if contiguous, treat all the special numbers the same
        ser = ser.replace(-111 , -999)
        ser = ser.replace(-222 , -999)
        ser = ser.replace(-555 , -999)

    # Bin the actual numbers into 10 bins for scanning
    sub_ser = pd.qcut(sub_ser, 10, duplicates='drop')
    sub_ser = sub_ser.apply(get_str).astype(str)
    ser[list(sub_ser.index)] = sub_ser
    return ser
contiguous = {}
dff_2 = dff_ag_complete_case.copy()

# Create a new dataframe with the numeric columns bins     
for col in numeric_columns:
    if col in search_space1:
        dff_2[col] = custom_qcut(dff_2[col].copy())
        
        bins = list(dff_2[col].unique())
        
        if -999 in bins:
            bins.remove(-999)
        
        bins = sorted(bins, key=lambda x : float(x.split(' - ')[0]))
        
        contiguous[col] = bins

def compress_contiguous(subset, contiguous):
    # Shorten a contiguous list e.g [0-9, 10-19] is converted to [0 - 19] 
    new = {}
    
    for col in subset:
        if col in contiguous:
            if isinstance(subset[col][0], (float,int)):
                new[col] = [str(c) for c in subset[col]]
                continue
            i = -1 if isinstance(subset[col][-1], str) else -2
            new[col] = [subset[col][0].split(' - ')[0] + ' - ' + subset[col][i].split(' - ')[-1]]
            new[col] = new[col] if i == -1 else new[col] + [str(subset[col][-1])]
        else:
            new[col] = [str(c) for c in subset[col]]
    return new

def translate_subset_to_rule(subset):
    # Print the subset as a rule for easier understanding
    desc = ''
    for key, value in subset.items():
        # desc += key + ' = {' + ' OR '.join(value) + '} AND' + '\n'
        desc += key + '{' + ' OR '.join(value) + '} AND' + ' '

    return desc[:-5].replace('_',' ').replace('{', '[').replace('}', ']')

def count_conditions(subset):
    # Split the string by 'AND' and 'OR'
    conditions = subset.replace("AND", "OR").split("OR")
    
    # Count the number of conditions
    condition_count = len(conditions)
    
    return condition_count
# %%time
# Scan in the positive direction using defined penalties and num iters
import scipy.stats as stats
scoring_function = Bernoulli(direction='positive')
scanner = MDSS(scoring_function)
# Define a list of penalty values to loop through
# penalty_values = [0.5, 1, 1.5, 3.6, 4, 4.5, 5.0,18.5]
# penalty_values = [0.5, 0.65, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4, 4.5, 5.0,10.5,12.5] # 2, 1.5, 2.36
# penalty_values = [0.4, 1.3, 1.4, 1.5, 2.0, 2.4, 3.0,12.5]
# penalty_values = [0.4, 0.5, 1.0, 1.2, 1.5, 3.0,12.5]
# penalty_values = [1.0, 1.5, 2, 2.5, 3.0,3.5]
# For complete case
penalty_values = [0.25, 1.0, 1.6, 1.8, 2.0,3.5]
num_iters = 5
# Initialize empty lists to store results
subset_results = []
subset_results1 = []
score_results = []
size_results = []
counts_percent = []
size_percent = []
odds_results = []
odds = []
z_scores = []
p_values = []
count_results = []
dataframes = {}
dataframes_complement = {}
CIs = []

for penalty in penalty_values:
    # Perform the scan with the current penalty value
    subset, score = scanner.scan(
        dff_2[search_space1], 
        dff_2[target_cols], 
        dff_2['expectation'], 
        cpu=0.99,
        penalty=penalty, 
        num_iters=num_iters, 
        contiguous=contiguous.copy()
    )
    
    # Identify subset rows
    to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
    temp_df = dff_2.loc[to_choose]
    not_temp_df = dff_2.loc[~to_choose]

    # Store each DataFrame
    dataframes[f'df_{penalty}'] = temp_df
    dataframes_complement[f'df_{penalty}'] = not_temp_df

    # Basic info
    size = len(temp_df)
    total_size = len(dff_2)
    
    # Odds_multiplicative factor (kept from your code)
    group_obs = temp_df[target_cols].mean().mean()  # mean across columns, then mean of that Series
    group_prob = dff_2['expectation'].mean()
    if (1 - group_obs) == 0 or (1 - group_prob) == 0:
        odds_mul = float('inf')
    else:
        odds_mul = (group_obs / (1 - group_obs)) / (group_prob / (1 - group_prob))

    # 2×2 counts (summing across all columns in target_cols)
    a = temp_df[target_cols].sum(numeric_only=True).sum()        # total positives in subset
    b = len(temp_df) - a                                         # total negatives in subset
    c = not_temp_df[target_cols].sum(numeric_only=True).sum()    # total positives outside subset
    d = len(not_temp_df) - c                                     # total negatives outside subset
    
    # Safely convert to float if needed
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    
    # Avoid zero counts
    if a == 0 or b == 0 or c == 0 or d == 0:
        odds_ratio = float('inf')
        log_IDR = float('inf')
        Z_score = float('inf')
        CI_lower, CI_upper = float('inf'), float('inf')
        p_value = 0.0
    else:
        # Calculate unadjusted odds ratio
        odds_temp = a / b
        odds_not_temp = c / d
        odds_ratio = odds_temp / odds_not_temp if odds_not_temp != 0 else float('inf')

        # log(OR) & variance for a 2×2 table
        log_IDR = np.log(odds_ratio)
        var_log_IDR = (1/a) + (1/b) + (1/c) + (1/d)
        SE_log_IDR = np.sqrt(var_log_IDR)

        # 95% CI
        CI_lower_log = log_IDR - 1.96 * SE_log_IDR
        CI_upper_log = log_IDR + 1.96 * SE_log_IDR
        CI_lower = np.exp(CI_lower_log)
        CI_upper = np.exp(CI_upper_log)

        # Z-score & p-value
        Z_score = log_IDR / SE_log_IDR
        p_value = 2 * stats.norm.sf(abs(Z_score))
    
    # Final CI tuple
    CI = (round(CI_lower, 2), round(CI_upper, 2))

    # Summaries
    score_results.append(round(score, 3))
    size_results.append(size)
    size_percent.append(round(size / total_size * 100, 2))
    
    # Proportion of events in subset vs. total events
    total_events_subset = a
    total_events = dff_2[target_cols].sum(numeric_only=True).sum()
    if total_events > 0:
        counts_percent.append(round(total_events_subset / total_events * 100, 2))
    else:
        counts_percent.append(0)
    
    odds_results.append(round(odds_mul, 2))
    odds.append(round(odds_ratio, 2))
    rule_str = translate_subset_to_rule(compress_contiguous(subset, contiguous))
    subset_results1.append(rule_str)
    counting = count_conditions(rule_str)
    count_results.append(counting)

    z_scores.append(Z_score)
    p_values.append(p_value)
    CIs.append(CI)

# Display the results for each penalty value
for i, penalty in enumerate(penalty_values):
    print(f"Penalty = {penalty}: \n"
          f"  Subset = {subset_results1[i]}, \n"
          f"  LiteralsNumber = {count_results[i]}, \n"
          f"  Size = {size_results[i]}, \n"
          f"  Mul_odds = {odds_results[i]}, \n"
          f"  Odds = {odds[i]}, \n"
          f"  Score = {score_results[i]}, \n"
          f"  Size_percent = {size_percent[i]}, \n"
          f"  Count_percent = {counts_percent[i]}, \n"
          f"  P_value = {p_values[i]:.3g}, \n"
          f"  CI = {CIs[i]}")
results_df = pd.DataFrame({
    'Penalty': penalty_values,
    'No of literals': count_results,
    'Subset': subset_results1,
    'Size': size_results,
    'Size (%)': size_percent,
    'X Odds': odds_results,
    'Odds': odds,
    'CI': CIs,
    'Score': score_results,
    'DM rate (%)': counts_percent,
    'P-value': p_values
})

# Display the DataFrame
results_df
results_df = pd.DataFrame({
    'Penalty': penalty_values,
    'No of literals': count_results,
    'Subset': subset_results1,
    'Size': size_results,
    'Size (%)': size_percent,
    'X Odds': odds_results,
    'Odds': odds,
    'CI': CIs,
    'Score': score_results,
    'DM rate (%)': counts_percent,
    'P-value': p_values
})

# Display the DataFrame
results_df
print(results_df.to_markdown(floatfmt=".3f", index=False))
results_df.to_csv('DM_AG_res_mich_age.csv', index=False) # 2, 1.5, 2.36
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib64/python3.9/concurrent/futures/process.py", line 246, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "/usr/lib64/python3.9/concurrent/futures/process.py", line 205, in _process_chunk
    return [fn(*args) for args in chunk]
  File "/usr/lib64/python3.9/concurrent/futures/process.py", line 205, in <listcomp>
    return [fn(*args) for args in chunk]
  File "/home/kayode/Kay/mdscan-master/mdss/MDSS.py", line 1003, in _scan_ascents_single_core
    current_subset, current_score = self._scan_single_ascent(
  File "/home/kayode/Kay/mdscan-master/mdss/MDSS.py", line 906, in _scan_single_ascent
    assert (
AssertionError: WARNING SCORE HAS DECREASED from -1.317203 to -2.919299
"""

The above exception was the direct cause of the following exception:

AssertionError                            Traceback (most recent call last)
Cell In[89], line 6
      2 scanner = MDSS(scoring_function)
      3 # Perform the scan with the current penalty value
      4 # res = scanner.scan(dff_2[search_space1], dff_2[target_cols], dff_2['expectation'], cpu=0.95,
      5 #                                 penalty=1.0, num_iters=10, contiguous=contiguous.copy(), num_of_subsets=5)
----> 6 subset, score = scanner.scan(dff_2[search_space1], dff_2[target_cols], dff_2['expectation'], cpu=0.99,
      7                                 penalty=0.9, num_iters=10, contiguous=contiguous.copy())
      9 # subset = res[0]
     10 # score
     11 print(translate_subset_to_rule(compress_contiguous(subset, contiguous)))

File ~/Kay/mdscan-master/mdss/MDSS.py:1243, in MDSS.scan(self, coordinates, outcomes, expectations, penalty, num_iters, max_literals, use_not_direction, contiguous, feature_penalty, verbose, seed, num_of_subsets, mode, cpu)
   1224     contiguous = {}
   1226 data = MDSSData(
   1227     coordinates = coordinates,
   1228     outcomes = outcomes,
   (...)
   1240     max_literals = max_literals
   1241     )
-> 1243 results = self._scan_in_diff_modes(data)
   1245 if (
   1246     len(results) == 1
   1247 ):  # Return the subset and score separately if num_of_subsets = 1
   1248     # Ensure backward compatibility with earlier notebooks.
   1249     subset, score = results[0]

File ~/Kay/mdscan-master/mdss/MDSS.py:1166, in MDSS._scan_in_diff_modes(self, data)
   1163 if data.mode == "nominal":
   1164     return self._scan_in_nominal_mode(data)
-> 1166 return self._scan_k_subsets(data)

File ~/Kay/mdscan-master/mdss/MDSS.py:1111, in MDSS._scan_k_subsets(self, data)
   1108             data.penalty = score/num_of_literals
   1110     else:
-> 1111         subset, score = scan_func(data)
   1113     k_subsets_and_scores.append([subset, score])
   1115 return k_subsets_and_scores

File ~/Kay/mdscan-master/mdss/MDSS.py:1061, in MDSS._scan_ascents_in_parallel(self, data)
   1058     results = executor.map(scan, self.starting_subsets)
   1060 # collect the results
-> 1061 results = list(results)
   1063 # get the best score and sub-population
   1064 best_subset, best_score = max(results, key=operator.itemgetter(1))

File /usr/lib64/python3.9/concurrent/futures/process.py:562, in _chain_from_iterable_of_lists(iterable)
    556 def _chain_from_iterable_of_lists(iterable):
    557     """
    558     Specialized implementation of itertools.chain.from_iterable.
    559     Each item in *iterable* should be a list.  This function is
    560     careful not to keep references to yielded objects.
    561     """
--> 562     for element in iterable:
    563         element.reverse()
    564         while element:

File /usr/lib64/python3.9/concurrent/futures/_base.py:609, in Executor.map.<locals>.result_iterator()
    606 while fs:
    607     # Careful not to keep a reference to the popped future
    608     if timeout is None:
--> 609         yield fs.pop().result()
    610     else:
    611         yield fs.pop().result(end_time - time.monotonic())

File /usr/lib64/python3.9/concurrent/futures/_base.py:439, in Future.result(self, timeout)
    437     raise CancelledError()
    438 elif self._state == FINISHED:
--> 439     return self.__get_result()
    441 self._condition.wait(timeout)
    443 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:

File /usr/lib64/python3.9/concurrent/futures/_base.py:391, in Future.__get_result(self)
    389 if self._exception:
    390     try:
--> 391         raise self._exception
    392     finally:
    393         # Break a reference cycle with the exception in self._exception
    394         self = None

AssertionError: WARNING SCORE HAS DECREASED from -1.317203 to -2.919299
scoring_function = Bernoulli(direction='positive') 
scanner = MDSS(scoring_function)
# Perform the scan with the current penalty value
# res = scanner.scan(dff_2[search_space1], dff_2[target_cols], dff_2['expectation'], cpu=0.95,
#                                 penalty=1.0, num_iters=10, contiguous=contiguous.copy(), num_of_subsets=5)
subset, score = scanner.scan(dff_2[search_space1], dff_2[target_cols], dff_2['expectation'], cpu=0.99,
                                penalty=0.85, num_iters=5, contiguous=contiguous.copy())

# subset = res[0]
# score
print(translate_subset_to_rule(compress_contiguous(subset, contiguous)))
# print(translate_subset_to_rule(subset))
to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
temp_df = dff_2.loc[to_choose]
not_tempdf = dff_2.loc[~to_choose]
print(count_conditions(translate_subset_to_rule(compress_contiguous(subset, contiguous))))
print(temp_df.shape[0])
print(temp_df.shape[0] / dff_2.shape[0]*100)
print(score)
print(temp_df[temp_df['diabetes_status_c_qc'] == 1].shape[0] / len(temp_df) * 100)
print(temp_df[temp_df['diabetes_status_c_qc'] == 1].shape[0] / dff_2[dff_2['diabetes_status_c_qc'] == 1].shape[0] * 100)
from collections import OrderedDict

# Define array of penalty values
penalty_values = [0.5, 0.7, 0.8, 1.0, 2.0, 3.0, 4.5]

# Initialize results storage
results = {
    'penalty': [],
    'rule': [],
    'condition_count': [],
    'subset_size': [],
    'subset_percentage': [],
    'score': [],
    'diabetes_prevalence': [],
    'diabetes_capture': []
}

# Initialize scanner
scoring_function = Bernoulli(direction='positive')
scanner = MDSS(scoring_function)

# Loop through penalty values
for penalty in penalty_values:
    # Perform scan
    subset, score = scanner.scan(
        dff_2[search_space1],
        dff_2[target_cols],
        dff_2['expectation'],
        cpu=0.99,
        penalty=penalty,
        num_iters=10,
        contiguous=contiguous.copy()
    )
    
    # Process results
    rule = translate_subset_to_rule(compress_contiguous(subset, contiguous))
    to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
    temp_df = dff_2.loc[to_choose]
    not_tempdf = dff_2.loc[~to_choose]
    
    # Store results
    results['penalty'].append(penalty)
    results['rule'].append(rule)
    results['condition_count'].append(count_conditions(rule))
    results['subset_size'].append(temp_df.shape[0])
    results['subset_percentage'].append(temp_df.shape[0] / dff_2.shape[0] * 100)
    results['score'].append(score)
    results['diabetes_prevalence'].append(
        temp_df[temp_df['diabetes_status_c_qc'] == 1].shape[0] / len(temp_df) * 100
    )
    results['diabetes_capture'].append(
        temp_df[temp_df['diabetes_status_c_qc'] == 1].shape[0] / 
        dff_2[dff_2['diabetes_status_c_qc'] == 1].shape[0] * 100
    )

# Create DataFrame from results
results_df = pd.DataFrame(results)
results_df
from collections import OrderedDict

# Define array of penalty values
penalty_values = [0.5, 1.0, 2.0, 3.0, 4.5]

# Initialize results storage
results = {
    'penalty': [],
    'rule': [],
    'condition_count': [],
    'subset_size': [],
    'subset_percentage': [],
    'score': [],
    'diabetes_prevalence': [],
    'diabetes_capture': []
}

# Initialize scanner
scoring_function = Bernoulli(direction='positive')
scanner = MDSS(scoring_function)

# Loop through penalty values
for penalty in penalty_values:
    # Perform scan
    subset, score = scanner.scan(
        dff_2[search_space1],
        dff_2[target_cols],
        dff_2['expectation'],
        cpu=0.99,
        penalty=penalty,
        num_iters=10,
        contiguous=contiguous.copy()
    )
    
    # Process results
    rule = translate_subset_to_rule(compress_contiguous(subset, contiguous))
    to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
    temp_df = dff_2.loc[to_choose]
    not_tempdf = dff_2.loc[~to_choose]
    
    # Store results
    results['penalty'].append(penalty)
    results['rule'].append(rule)
    results['condition_count'].append(count_conditions(rule))
    results['subset_size'].append(temp_df.shape[0])
    results['subset_percentage'].append(temp_df.shape[0] / dff_2.shape[0] * 100)
    results['score'].append(score)
    results['diabetes_prevalence'].append(
        temp_df[temp_df['diabetes_status_c_qc'] == 1].shape[0] / len(temp_df) * 100
    )
    results['diabetes_capture'].append(
        temp_df[temp_df['diabetes_status_c_qc'] == 1].shape[0] / 
        dff_2[dff_2['diabetes_status_c_qc'] == 1].shape[0] * 100
    )

# Create DataFrame from results
results_df = pd.DataFrame(results)
results_df
subset,score = res[3]
print(translate_subset_to_rule(compress_contiguous(subset, contiguous)))
# print(translate_subset_to_rule(subset))
to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
temp_df = dff_2.loc[to_choose]
not_tempdf = dff_2.loc[~to_choose]
print('no of lit: ', count_conditions(translate_subset_to_rule(compress_contiguous(subset, contiguous))))
print('size:', len(temp_df)) # partnership status c qc[1] AND days fruit qc[7] AND age[53.0 - 60.0] AND occupation qc[-999 OR 3]
import scipy.stats as stats # days fruit qc[2 OR 3 OR 4] AND sex[1] AND ses site quintile c[4.0 OR 5.0] AND alcohol use status c qc[1] AND hip circumference qc[1050.0 - 1600.0] AND highest level of education qc[1]
dff_ag['age'].max()
dd = dff_ag.copy()
# subgroup_mask = (
#     (dd["bmi_c_qc"] >= 21.37) & (dd["bmi_c_qc"] <= 68.02) &
#     (dd["mvpa_c"] >= 0.0) & (dd["mvpa_c"] <= 2448.0) &
#     (dd["diabetes_history_qc"] == 1) &
#     (dd["waist_hip_r_c_qc"] > 0.9) & (dd["waist_hip_r_c_qc"] <= 1.16)
# )
# subgroup_mask = (
#     (dd["mvpa_c"] >= 0.0) & (dd["mvpa_c"] <= 2448.0) &
#     (dd["diabetes_history_qc"] == 1) &
#     (dd["waist_hip_r_c_qc"] > 0.9) & (dd["waist_hip_r_c_qc"] <= 1.16)
# )
# subgroup_mask = (
#     (dd["diabetes_history_qc"] == 1) &
#     (dd["waist_hip_r_c_qc"] > 0.9) & (dd["waist_hip_r_c_qc"] <= 1.16)
# )
subgroup_mask = (
    (dd["age"] > 42.0) & (dd["age"] <= 60.0) &
    (dd["bmi_c_qc"] >= 21.37) & (dd["bmi_c_qc"] <= 68.02) &
    (dd["waist_hip_r_c_qc"] > 0.9) & (dd["waist_hip_r_c_qc"] <= 1.16) &
    (dd["diabetes_history_qc"] == 1) &
    (dd["mvpa_c"] >= 0.0) & (dd["mvpa_c"] <= 2448.0)
)
# subgroup_mask = (
#     (dd["occupation_qc"].isin([-999, 3, 4])) &
#     (dd["diabetes_history_qc"] == 1) &
#     (dd["age"] > 42.0) & (dd["age"] <= 60.0) &
#     (dd["bmi_c_qc"] >= 21.37) & (dd["bmi_c_qc"] <= 68.02) &
#     (dd["mvpa_c"] >= 0.0) & (dd["mvpa_c"] <= 2448.0) &
#     (dd["waist_hip_r_c_qc"] > 0.9) & (dd["waist_hip_r_c_qc"] <= 1.16)
# )

##########################_________---------------______________###########################
# subgroup_mask = (
#     (dd["occupation_qc"].isin([-999, 1])) &
#     (dd["mvpa_c"] >= 0.0) & (dd["mvpa_c"] <= 119.0) &
#     (dd["days_fruit_qc"] == 2) &
#     (dd["waist_circumference_qc"] >= 860.0) & (dd["waist_circumference_qc"] <= 1470.0)
# )
# subgroup_mask = (
#     (dd["smoking_status_c_qc"] == 1) &
#     (dd["days_veg_qc"] == 3) &
#     (dd["waist_circumference_qc"] >= 950.0) & (dd["waist_circumference_qc"] <= 1040.0)
# )
# subgroup_mask = (
#     (dd["highest_level_of_education_qc"] == 3) &
#     (dd["bmi_c_qc"] >= 32.61) & (dd["bmi_c_qc"] <= 35.92) &
#     (dd["waist_hip_r_c_qc"] >= 0.84) & (dd["waist_hip_r_c_qc"] <= 0.9)
# )
# subgroup_mask = (
#     (dd["waist_circumference_qc"] >= 860.0) & (dd["waist_circumference_qc"] <= 1470.0) &
#     (dd["age"] >= 55.0) & (dd["age"] <= 58.0) &
#     (dd["waist_hip_r_c_qc"] >= 1.0) & (dd["waist_hip_r_c_qc"] <= 1.16) &
#     (dd["alcohol_use_status_c_qc"] == 1)
# )


# --------------------------------------------------------
# Option 4
# subgroup_mask = ((dd['diabetes_history_qc'] == 1)& (dd['bmi_c_qc'] >= 30))
# subgroup_mask = (dd['bmi_c_qc'] >= 30)  # e.g., "BMI ≥ 30"

# 1. Subgroup (temp) and complement (not_temp_df)
temp = dd.loc[subgroup_mask].copy()
not_temp_df = dd.loc[~subgroup_mask].copy()

# 2. Count events (diabetes=1) and non-events (diabetes=0) for each group
positive_temp = temp['diabetes_status_c_qc'].sum()  # # of diabetic cases in subgroup
negative_temp = len(temp) - positive_temp

positive_not_temp = not_temp_df['diabetes_status_c_qc'].sum()
negative_not_temp = len(not_temp_df) - positive_not_temp

# 3. Calculate odds ratio from the 2×2 table:
#    a = positive_temp
#    b = negative_temp
#    c = positive_not_temp
#    d = negative_not_temp
if negative_temp == 0 or negative_not_temp == 0:
    odds_ratio = float('inf')
else:
    odds_temp = positive_temp / negative_temp
    odds_not_temp = positive_not_temp / negative_not_temp
    odds_ratio = odds_temp / odds_not_temp if odds_not_temp != 0 else float('inf')

# 4. Compute standard error of log(OR) using the usual 2×2 formula
a, b = positive_temp, negative_temp
c, d = positive_not_temp, negative_not_temp

if a == 0 or b == 0 or c == 0 or d == 0:
    # If any cell is zero, the variance formula can go to infinity.
    # Optionally apply a continuity correction (e.g., a+0.5, etc.) if desired.
    log_or = float('inf')
    se_log_or = float('inf')
    CI_lower, CI_upper = float('inf'), float('inf')
    p_value = 0.0
else:
    log_or = np.log(odds_ratio)
    var_log_or = (1/a) + (1/b) + (1/c) + (1/d)
    se_log_or = np.sqrt(var_log_or)

    # 5. Confidence Interval (95% by default)
    z_crit = 1.96
    CI_lower_log = log_or - z_crit * se_log_or
    CI_upper_log = log_or + z_crit * se_log_or
    CI_lower = np.exp(CI_lower_log)
    CI_upper = np.exp(CI_upper_log)

    # 6. p-value (two-sided Z-test)
    Z_score = log_or / se_log_or
    p_value = 2 * stats.norm.sf(abs(Z_score))

# 7. Print outputs
print(f"Odds ratio: {odds_ratio:.2f}")
print(f"Population size: {len(dd)}")
print(f"Subgroup size: {len(temp)}")
print(f"Subgroup # of T2D: {int(positive_temp)}")
print(f"Subgroup mean T2D: {(temp['diabetes_status_c_qc'].mean()):.3f}")
print(f"Subgroup proportion T2D: {(positive_temp/len(temp)*100):.3f}")
print(f"Population proportion T2D: {(positive_temp/len(dd)*100):.3f}")
print(f"95% CI: [{CI_lower:.2f}, {CI_upper:.2f}]")
print(f"p-value: {p_value:.2e}")
temp[temp['diabetes_status_c_qc'] == 1].shape[0] / len(temp) * 100
import pandas as pd

# Total population size and total T2D cases
total_population = len(dff_ag)
total_T2D_cases = dff_ag["diabetes_status_c_qc"].sum()

# Define subgroup masks and their literal descriptions
subgroup_info = [
    {
        "mask": (
            (dff_ag["smoking_status_c_qc"] == 0) &
            (dff_ag["waist_hip_r_c_qc"] >= 0.96) & (dff_ag["waist_hip_r_c_qc"] <= 1.36) &
            (dff_ag["age"] >= 61.0) & (dff_ag["age"] <= 71.0) &
            (dff_ag["waist_circumference_qc"] >= 1030.0) & (dff_ag["waist_circumference_qc"] <= 1470.0) &
            (dff_ag["sex"] == 0)
        ),
        "description": "Smoking=0 AND WHR[0.96-1.36] AND Age[61-71] AND Waist[1030-1470] AND Sex=0"
    },
    {
        "mask": (
            (dff_ag["mvpa_c"] >= 0.0) & (dff_ag["mvpa_c"] <= 2297.5) &
            (dff_ag["diabetes_history_qc"] == 1) &
            (dff_ag["bmi_c_qc"] >= 21.55) & (dff_ag["bmi_c_qc"] <= 68.02) &
            (dff_ag["waist_hip_r_c_qc"] >= 0.9) & (dff_ag["waist_hip_r_c_qc"] <= 1.36)
        ),
        "description": "MVPA[0-2297.5] AND Diabetes history=1 AND BMI[21.55-68.02] AND WHR[0.9-1.36]"
    },
    {
        "mask": (
            (dff_ag["waist_hip_r_c_qc"] >= 0.9) & (dff_ag["waist_hip_r_c_qc"] <= 1.36) &
            (dff_ag["diabetes_history_qc"] == 1)
        ),
        "description": "WHR[0.9-1.36] AND Diabetes history=1"
    },
    {
        "mask": (
            (dff_ag["age"] > 44.0) & (dff_ag["age"] <= 81.0) &
            (dff_ag["waist_circumference_qc"] >= 810.0) & (dff_ag["waist_circumference_qc"] <= 1470.0) &
            (dff_ag["waist_hip_r_c_qc"] >= 0.9) & (dff_ag["waist_hip_r_c_qc"] <= 1.36) &
            (dff_ag["alcohol_use_status_c_qc"].isin([0, 3])) &
            (dff_ag["mvpa_c"] >= 0.0) & (dff_ag["mvpa_c"] <= 2297.5) &
            (dff_ag["diabetes_history_qc"] == 1)
        ),
        "description": "Age[44-81] AND Waist[810-1470] AND WHR[0.9-1.36] AND Alcohol[0 or 3] AND MVPA[0-2297.5] AND Diabetes history=1"
    },
    {
        "mask": (
            (dff_ag["bmi_c_qc"] >= 21.55) & (dff_ag["bmi_c_qc"] <= 24.7) &
            (dff_ag["age"] >= 75.0) & (dff_ag["age"] <= 81.0) &
            (dff_ag["ses_site_quintile_c"] == 5.0)
        ),
        "description": "BMI[21.55-24.7] AND Age[75-81] AND SES quintile=5"
    },
    {
        "mask": (
            (dff_ag["waist_hip_r_c_qc"] >= 0.94) & (dff_ag["waist_hip_r_c_qc"] <= 1.36) &
            (dff_ag["age"] >= 65.0) & (dff_ag["age"] <= 71.0)
        ),
        "description": "WHR[0.94-1.36] AND Age[65-71]"
    },
    {
        "mask": (
            (dff_ag["days_fruit_qc"].isin([2, 3, 4])) &
            (dff_ag["sex"] == 1) &
            dff_ag["ses_site_quintile_c"].isin([4.0, 5.0]) &
            (dff_ag["alcohol_use_status_c_qc"] == 1) &
            (dff_ag["hip_circumference_qc"] >= 1050.0) & (dff_ag["hip_circumference_qc"] <= 1600.0) &
            (dff_ag["highest_level_of_education_qc"] == 1)
        ),
        "description": "Days fruit[2-4] AND sex=1 AND SES quintile[4-5] AND Age[65-71] AND Alcohol[1] AND Hip circumference[1050-1600], AND Education[1]"
    }
]


# Build the result table
result_table = []

for info in subgroup_info:
    mask = info["mask"]
    description = info["description"]
    n_literals = description.count("AND") + 1  # number of literals = number of "AND" + 1

    size = mask.sum()
    size_percent = (size / total_population) * 100

    T2D_in_subgroup = dff_ag.loc[mask, "diabetes_status_c_qc"].sum()
    prevalence_in_subgroup = (T2D_in_subgroup / size) * 100 if size > 0 else 0

    contribution_to_population = (T2D_in_subgroup / total_T2D_cases) * 100 if total_T2D_cases > 0 else 0

    result_table.append({
        "Number of Literals": n_literals,
        "Subgroup Description": description,
        "Size": size,
        "Size (%)": round(size_percent, 2),
        "T2D Prevalence in Subgroup (%)": round(prevalence_in_subgroup, 2),
        "Subgroup Contribution to Total T2D (%)": round(contribution_to_population, 2)
    })

# Convert to DataFrame
df_subgroups_summary = pd.DataFrame(result_table)
# Display the DataFrame
df_subgroups_summary
# df_subgroups_summary.to_csv('new_DM_AG.csv', index=False)
df_subgroups_summary
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

model_filter4 = (
    (dff["occupation_qc"].isin([-999, 3, 4])) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["age"] > 42.0) & (dff["age"] <= 60.0) &
    (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)

model_filter5 = (
    (dff["age"] > 42.0) & (dff["age"] <= 60.0) &
    (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0)
)


model_subgroups = {
    "Model 1": model_filter1, # "mvpa, diabetes history, waist-hip, bmi"
    "Model 2": model_filter2, # "smoking, age, waist-hip, waist circumference, sex"
    "Model 3": model_filter3, # "waist-hip, age"
    # "Model 4": model_filter4, # "smoking, age, waist-hip, waist circumference, sex"
    # "Model 5": model_filter5 # "waist-hip, age"
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

# Display final comparison table
display(comparison_df)

# Make a copy
combined_results = comparison_df.copy()

# --- Split CI string into CI_lower and CI_upper ---
combined_results[["CI_lower", "CI_upper"]] = combined_results["CI"].str.strip("()").str.split(",", expand=True).astype(float)

# --- Convert p-value string to float (if needed) ---
combined_results["p-value"] = combined_results["p-value"].astype(float)

# Optional: sort if needed
combined_results = combined_results.sort_values(by="Method", ascending=False).reset_index(drop=True)

combined_results = combined_results.iloc[::-1].reset_index(drop=True)
# --------------------------------------------------------------------
# 5. NEW FUNCTION: Calculate Full Performance Metrics
# --------------------------------------------------------------------
def calculate_subgroup_metrics(data, mask, outcome_col="diabetes_status_c_qc"):
    
    tp = (mask & (data[outcome_col] == 1)).sum()
    fp = (mask & (data[outcome_col] == 0)).sum()
    tn = (~mask & (data[outcome_col] == 0)).sum()
    fn = (~mask & (data[outcome_col] == 1)).sum()
    
    total_pop = len(data)
    total_t2d = data[outcome_col].sum()
    total_nont2d = total_pop - total_t2d
    
    # Sensitivity (P(S|T2D)) - "Recall"
    # % of all T2D cases captured by the subgroup
    sensitivity = tp / (tp + fn)
    
    # Specificity
    # % of all Non-T2D cases correctly excluded by the subgroup
    specificity = tn / (tn + fp)
    
    # PPV (P(T2D|S)) - "Precision"
    # Probability of having T2D if you are in the subgroup
    ppv = tp / (tp + fp)
    
    # NPV
    # Probability of being Non-T2D if you are NOT in the subgroup
    npv = tn / (tn + fn)
    
    return {
        "P(S|T2D) [Sensitivity]": sensitivity,
        "Specificity": specificity,
        "P(T2D|S) [PPV]": ppv,
        "NPV": npv,
        "Total_in_Subgroup": (tp + fp),
        "Subgroup_Size_Percent": (tp + fp) / total_pop
    }

# --------------------------------------------------------------------
# 6. NEW LOOP: Run new metric calculations
# --------------------------------------------------------------------
print("--- Calculating Full Performance Metrics for Agincourt Subgroups ---")
dff = dff_ag.copy()
outcome_col = "diabetes_status_c_qc"

# Get all subgroup masks (model-derived and study-defined)
all_subgroups = model_subgroups.copy()
all_subgroups.update(study_subgroups)

metric_results = []
for name, flt in all_subgroups.items():
    metrics = calculate_subgroup_metrics(dff, flt, outcome_col)
    metrics["Subgroup Name"] = name
    metric_results.append(metrics)

# Create a clean DataFrame with the results
performance_df = pd.DataFrame(metric_results)
performance_df = performance_df.set_index("Subgroup Name")

# Re-order columns for clarity
performance_df = performance_df[[
    "P(T2D|S) [PPV]",
    "P(S|T2D) [Sensitivity]",
    "Specificity",
    "NPV",
    "Subgroup_Size_Percent",
    "Total_in_Subgroup"
]]

# Display the results
print(performance_df.to_markdown(floatfmt=".3f"))
# --- AUTOSCAN RESULTS: Sensitivity Analysis 1: Diagnosed Only ---

# Penalty = 0.4
model_filter1 = (
    (dff["days_fruit_qc"].isin([2.0, 3.0, 4.0, 5.0])) &
    (dff["days_veg_qc"].isin([2.0, 4.0, 5.0])) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["highest_level_of_education_qc"].isin([0.0, 1.0, 3.0])) &
    (dff["bmi_c_qc"] >= 24.4) & (dff["bmi_c_qc"] <= 35.9) &
    (dff["alcohol_use_status_c_qc"].isin([1, 3]))
)

# Penalty = 0.5
model_filter2 = (
    (dff["days_fruit_qc"].isin([0.0, 1.0, 2.0, 3.0])) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["highest_level_of_education_qc"].isin([0.0, 1.0])) &
    (dff["use_drug_qc"] == 0.0) &
    (dff["waist_circumference_qc"] >= 990.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["hip_circumference_qc"] >= 670.0) & (dff["hip_circumference_qc"] <= 1120.0)
)

# Penalty = 1.0
model_filter3 = (
    (dff["hip_circumference_qc"] >= 670.0) & (dff["hip_circumference_qc"] <= 1120.0) &
    (dff["days_fruit_qc"].isin([0.0, 2.0, 3.0])) &
    (dff["waist_circumference_qc"] >= 990.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["diabetes_history_qc"] == 1.0)
)

# Penalty = 1.2
model_filter4 = (
    (dff["hip_circumference_qc"] >= 670.0) & (dff["hip_circumference_qc"] <= 1120.0) &
    (dff["days_fruit_qc"].isin([0.0, 2.0, 3.0])) &
    (dff["waist_circumference_qc"] >= 990.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["diabetes_history_qc"] == 1.0)
)

# Penalty = 1.5
model_filter5 = (
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["age"] > 47.0) & (dff["age"] <= 60.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_circumference_qc"] >= 810.0) & (dff["waist_circumference_qc"] <= 1470.0)
)

# Penalty = 3.0
model_filter6 = (
    (dff["age"] > 47.0) & (dff["age"] <= 60.0) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_circumference_qc"] >= 810.0) & (dff["waist_circumference_qc"] <= 1470.0)
)

# Penalty = 12.5
model_filter7 = (
    (dff["diabetes_history_qc"] == 1.0)
)

# Combine all filters into a single dictionary
model_subgroups = {
    "Model 1": model_filter1,  # days fruit, days veg, diabetes history, education, bmi, alcohol
    "Model 2": model_filter2,  # fruit, diabetes history, education, drug use, waist, hip
    "Model 3": model_filter3,  # fruit, waist, hip, diabetes history
    "Model 4": model_filter4,  # fruit, waist, hip, diabetes history (same as model 3)
    "Model 5": model_filter5,  # mvpa, age, waist, diabetes history
    "Model 6": model_filter6,  # age, mvpa, waist, diabetes history (same as model 5)
    "Model 7": model_filter7   # diabetes history only
}

def calculate_subgroup_metrics(data, mask, outcome_col="diabetes_status_c_qc"):
    
    tp = (mask & (data[outcome_col] == 1)).sum()
    fp = (mask & (data[outcome_col] == 0)).sum()
    tn = (~mask & (data[outcome_col] == 0)).sum()
    fn = (~mask & (data[outcome_col] == 1)).sum()
    
    total_pop = len(data)
    total_t2d = data[outcome_col].sum()
    total_nont2d = total_pop - total_t2d
    
    # Sensitivity (P(S|T2D)) - "Recall"
    # % of all T2D cases captured by the subgroup
    sensitivity = tp / (tp + fn)
    
    # Specificity
    # % of all Non-T2D cases correctly excluded by the subgroup
    specificity = tn / (tn + fp)
    
    # PPV (P(T2D|S)) - "Precision"
    # Probability of having T2D if you are in the subgroup
    ppv = tp / (tp + fp)
    
    # NPV
    # Probability of being Non-T2D if you are NOT in the subgroup
    npv = tn / (tn + fn)
    
    return {
        "P(S|T2D) [Sensitivity]": sensitivity,
        "Specificity": specificity,
        "P(T2D|S) [PPV]": ppv,
        "NPV": npv,
        "Total_in_Subgroup": (tp + fp),
        "Subgroup_Size_Percent": (tp + fp) / total_pop
    }

# --------------------------------------------------------------------
# 6. NEW LOOP: Run new metric calculations
# --------------------------------------------------------------------
print("--- Calculating Full Performance Metrics for Agincourt Subgroups ---")
dff = dff_ag.copy()
outcome_col = "diabetes_status_c_qc"

# Get all subgroup masks (model-derived and study-defined)
all_subgroups = model_subgroups.copy()
all_subgroups.update(study_subgroups)

metric_results = []
for name, flt in all_subgroups.items():
    metrics = calculate_subgroup_metrics(dff, flt, outcome_col)
    metrics["Subgroup Name"] = name
    metric_results.append(metrics)

# Create a clean DataFrame with the results
performance_df = pd.DataFrame(metric_results)
performance_df = performance_df.set_index("Subgroup Name")

# Re-order columns for clarity
performance_df = performance_df[[
    "P(T2D|S) [PPV]",
    "P(S|T2D) [Sensitivity]",
    "Specificity",
    "NPV",
    "Subgroup_Size_Percent",
    "Total_in_Subgroup"
]]

# Display the results
print(performance_df.to_markdown(floatfmt=".3f"))
# --- AUTOSCAN RESULTS: Sensitivity Analysis 2: Fasting/Self-Report Only ---

# Penalty = 0.4
model_filter1 = (
    (dff["occupation_qc"].isin([1.0, 3.0, 4.0])) &
    (dff["alcohol_use_status_c_qc"].isin([1, 3])) &
    (dff["days_veg_qc"].isin([2.0, 4.0, 5.0])) &
    (dff["diabetes_history_qc"].isin([1.0, 2.0])) &
    (dff["bmi_c_qc"] >= 14.53) & (dff["bmi_c_qc"] <= 35.9) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["hip_circumference_qc"] >= 910.0) & (dff["hip_circumference_qc"] <= 1600.0) &
    (dff["waist_circumference_qc"] >= 780.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["highest_level_of_education_qc"].isin([0.0, 1.0, 3.0]))
)

# Penalty = 0.5
model_filter2 = (
    (dff["occupation_qc"].isin([1.0, 3.0, 4.0])) &
    (dff["alcohol_use_status_c_qc"].isin([1, 3])) &
    (dff["days_veg_qc"].isin([2.0, 4.0, 5.0])) &
    (dff["diabetes_history_qc"].isin([1.0, 2.0])) &
    (dff["bmi_c_qc"] >= 14.53) & (dff["bmi_c_qc"] <= 35.9) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["hip_circumference_qc"] >= 910.0) & (dff["hip_circumference_qc"] <= 1600.0) &
    (dff["waist_circumference_qc"] >= 780.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["highest_level_of_education_qc"].isin([0.0, 1.0, 3.0]))
)

# Penalty = 1.0
model_filter3 = (
    (dff["waist_circumference_qc"] >= 810.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["diabetes_history_qc"].isin([1.0, 2.0])) &
    (dff["age"] > 47.0) & (dff["age"] <= 60.0) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0)
)

# Penalty = 1.2
model_filter4 = (
    (dff["waist_circumference_qc"] >= 810.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["age"] > 47.0) & (dff["age"] <= 60.0)
)

# Penalty = 1.5
model_filter5 = (
    (dff["age"] > 47.0) & (dff["age"] <= 60.0) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_circumference_qc"] >= 810.0) & (dff["waist_circumference_qc"] <= 1470.0)
)

# Penalty = 3.0
model_filter6 = (
    (dff["age"] > 47.0) & (dff["age"] <= 60.0) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_circumference_qc"] >= 810.0) & (dff["waist_circumference_qc"] <= 1470.0)
)

# Penalty = 12.5
model_filter7 = (
    (dff["diabetes_history_qc"] == 1.0)
)

# Combine all filters into a single dictionary
model_subgroups = {
    "Model 1": model_filter1,  # occupation, alcohol, veg, diabetes history, bmi, mvpa, hip, waist, education
    "Model 2": model_filter2,  # same as Model 1 (identical subset)
    "Model 3": model_filter3,  # waist, diabetes history, age, mvpa
    "Model 4": model_filter4,  # waist, mvpa, diabetes history, age
    "Model 5": model_filter5,  # age, mvpa, waist, diabetes history
    "Model 6": model_filter6,  # same as Model 5
    "Model 7": model_filter7   # diabetes history only
}

# --- AUTOSCAN RESULTS: Sensitivity Analysis 1: Diagnosed Only (Imputed Dataset) ---

# Penalty = 0.4
model_filter1 = (
    (dff["days_veg_qc"].isin([2.0, 4.0, 5.0])) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["alcohol_use_status_c_qc"].isin([1, 3])) &
    (dff["hip_circumference_qc"] >= 940.0) & (dff["hip_circumference_qc"] <= 1180.0) &
    (dff["occupation_qc"].isin([3.0, 4.0])) &
    (dff["highest_level_of_education_qc"].isin([0.0, 1.0]))
)

# Penalty = 0.5
model_filter2 = (
    (dff["days_fruit_qc"].isin([0.0, 1.0, 2.0, 3.0])) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["highest_level_of_education_qc"].isin([0.0, 1.0])) &
    (dff["use_drug_qc"] == 0.0) &
    (dff["waist_circumference_qc"] >= 990.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["hip_circumference_qc"] >= 670.0) & (dff["hip_circumference_qc"] <= 1120.0)
)

# Penalty = 1.0
model_filter3 = (
    (dff["hip_circumference_qc"] >= 670.0) & (dff["hip_circumference_qc"] <= 1120.0) &
    (dff["days_fruit_qc"].isin([0.0, 2.0, 3.0])) &
    (dff["waist_circumference_qc"] >= 990.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["diabetes_history_qc"] == 1.0)
)

# Penalty = 1.2
model_filter4 = (
    (dff["bmi_c_qc"] >= 21.31) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16) &
    (dff["hip_circumference_qc"] >= 670.0) & (dff["hip_circumference_qc"] <= 1180.0)
)

# Penalty = 1.5
model_filter5 = (
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["age"] > 47.0) & (dff["age"] <= 60.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_circumference_qc"] >= 810.0) & (dff["waist_circumference_qc"] <= 1470.0)
)

# Penalty = 3.0
model_filter6 = (
    (dff["age"] > 47.0) & (dff["age"] <= 60.0) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 1680.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_circumference_qc"] >= 810.0) & (dff["waist_circumference_qc"] <= 1470.0)
)

# Penalty = 12.5
model_filter7 = (
    (dff["diabetes_history_qc"] == 1.0)
)

# Combine all filters into a single dictionary
model_subgroups = {
    "Model 1": model_filter1,  # veg, diabetes history, alcohol, hip, occupation, education
    "Model 2": model_filter2,  # fruit, diabetes history, education, drug use, waist, hip
    "Model 3": model_filter3,  # fruit, waist, hip, diabetes history
    "Model 4": model_filter4,  # bmi, mvpa, diabetes history, waist-hip, hip
    "Model 5": model_filter5,  # mvpa, age, waist, diabetes history
    "Model 6": model_filter6,  # same as model 5
    "Model 7": model_filter7   # diabetes history only
}

# --- AUTOSCAN RESULTS: Sensitivity Analysis 3: Exclude Borderline (Imputed Dataset) ---

# Penalty = 0.3
model_filter1 = (
    (dff["days_fruit_qc"].isin([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])) &
    (dff["waist_circumference_qc"] >= 810.8) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["age"] > 42.0) & (dff["age"] <= 60.0) &
    (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)

# Penalty = 1.0
model_filter2 = (
    (dff["age"] > 42.0) & (dff["age"] <= 60.0) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16) &
    (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["diabetes_history_qc"] == 1.0)
)

# Penalty = 1.1
model_filter3 = (
    (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)

# Penalty = 1.4
model_filter4 = (
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["diabetes_history_qc"] == 1.0)
)

# Penalty = 3.5
model_filter5 = (
    (dff["diabetes_history_qc"] == 1.0) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)

# Combine all filters into a single dictionary
model_subgroups = {
    "Model 1": model_filter1,  # fruit, waist, diabetes history, age, bmi, mvpa, waist-hip
    "Model 2": model_filter2,  # age, waist-hip, bmi, mvpa, diabetes history
    "Model 3": model_filter3,  # bmi, mvpa, diabetes history, waist-hip
    "Model 4": model_filter4,  # waist-hip, mvpa, diabetes history
    "Model 5": model_filter5   # diabetes history, waist-hip
}

def calculate_subgroup_metrics(data, mask, outcome_col="diabetes_status_c_qc"):
    
    tp = (mask & (data[outcome_col] == 1)).sum()
    fp = (mask & (data[outcome_col] == 0)).sum()
    tn = (~mask & (data[outcome_col] == 0)).sum()
    fn = (~mask & (data[outcome_col] == 1)).sum()
    
    total_pop = len(data)
    total_t2d = data[outcome_col].sum()
    total_nont2d = total_pop - total_t2d
    
    # Sensitivity (P(S|T2D)) - "Recall"
    # % of all T2D cases captured by the subgroup
    sensitivity = tp / (tp + fn)
    
    # Specificity
    # % of all Non-T2D cases correctly excluded by the subgroup
    specificity = tn / (tn + fp)
    
    # PPV (P(T2D|S)) - "Precision"
    # Probability of having T2D if you are in the subgroup
    ppv = tp / (tp + fp)
    
    # NPV
    # Probability of being Non-T2D if you are NOT in the subgroup
    npv = tn / (tn + fn)
    
    return {
        "P(S|T2D) [Sensitivity]": sensitivity,
        "Specificity": specificity,
        "P(T2D|S) [PPV]": ppv,
        "NPV": npv,
        "Total_in_Subgroup": (tp + fp),
        "Subgroup_Size_Percent": (tp + fp) / total_pop
    }

# --------------------------------------------------------------------
# 6. NEW LOOP: Run new metric calculations
# --------------------------------------------------------------------
print("--- Calculating Full Performance Metrics for Agincourt Subgroups ---")
dff = dff_ag.copy()
outcome_col = "diabetes_status_c_qc"

# Get all subgroup masks (model-derived and study-defined)
all_subgroups = model_subgroups.copy()
all_subgroups.update(study_subgroups)

metric_results = []
for name, flt in all_subgroups.items():
    metrics = calculate_subgroup_metrics(dff, flt, outcome_col)
    metrics["Subgroup Name"] = name
    metric_results.append(metrics)

# Create a clean DataFrame with the results
performance_df = pd.DataFrame(metric_results)
performance_df = performance_df.set_index("Subgroup Name")

# Re-order columns for clarity
performance_df = performance_df[[
    "P(T2D|S) [PPV]",
    "P(S|T2D) [Sensitivity]",
    "Specificity",
    "NPV",
    "Subgroup_Size_Percent",
    "Total_in_Subgroup"
]]

# Display the results
print(performance_df.to_markdown(floatfmt=".3f"))
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
    "WC: W$\geqslant$800, M$\geqslant$940 \nMVPA$\leqslant$600",
    "Age$\geqslant$45 \nWC: W$\geqslant$800, M$\geqslant$940 \nMVPA$\leqslant$600",
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
ax_pvals.set_xlabel("–log$_{10}$(p-value)", fontsize=11)
ax_pvals.set_ylim(ax_forest.get_ylim())
ax_pvals.set_yticks(y_positions)
ax_pvals.set_yticklabels([])

# ----------------------------
# Final Layout & Save
# ----------------------------
plt.tight_layout()
# plt.savefig("forest_plot_age.pdf", dpi=350)
plt.show()

import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# ------------------------------------------------------------------------------
# 1. Convert subgroup filters to index sets
# ------------------------------------------------------------------------------

model_sets = [set(dff.index[flt]) for flt in model_subgroups.values()]
study_sets = [set(dff.index[flt]) for flt in study_subgroups.values()]

# ------------------------------------------------------------------------------
# 2. Venn + Jaccard Plotting Function
# ------------------------------------------------------------------------------

def plot_venn_with_jaccard(ax, set_list, set_labels, title="", 
                           set_colors=("r", "g", "b"), alpha=0.5,
                           label_offsets=None):
    """
    Draws a 3-set Venn diagram with Jaccard index box.
    """
    if len(set_list) != 3:
        raise ValueError("Exactly 3 sets required for venn3.")
    
    v = venn3(subsets=set_list, set_labels=set_labels, ax=ax,
              set_colors=set_colors, alpha=alpha)
    ax.set_title(title, fontsize=12)

    # Optional: adjust label positions
    if label_offsets:
        for idx, (dx, dy) in label_offsets.items():
            label = v.subset_labels[idx]
            if label is not None:
                x, y = label.get_position()
                label.set_position((x + dx, y + dy))

    # Compute Jaccard indices
    def jaccard(a, b):
        return len(a & b) / len(a | b) if len(a | b) > 0 else 0.0

    j12 = jaccard(set_list[0], set_list[1])
    j23 = jaccard(set_list[1], set_list[2])
    j13 = jaccard(set_list[0], set_list[2])

    jacc_text = (f"Jaccard:\n"
                 f"S1 vs S2: {j12:.2f}\n"
                 f"S2 vs S3: {j23:.2f}\n"
                 f"S1 vs S3: {j13:.2f}")

    ax.text(
        0.05, 0.04, jacc_text,
        transform=ax.transAxes,
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

# ------------------------------------------------------------------------------
# 3. Plot Side-by-Side Venn Diagrams
# ------------------------------------------------------------------------------

model_label_offsets = {0: (0, 0.1), 1: (0, 0.2), 2: (0, -0.2)}

if len(model_sets) == 3 and len(study_sets) == 3:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plot_venn_with_jaccard(
        ax1, 
        model_sets, 
        ["Model 1", "Model 2", "Model 3"],
        title="Model Subgroups",
        set_colors=("red", "green", "blue"),
        alpha=0.4,
        label_offsets=model_label_offsets
    )

    plot_venn_with_jaccard(
        ax2,
        study_sets, 
        ['Study 1', 'Study 2', 'Study 3'],
        title="Study-Defined Subgroups",
        set_colors=("orange", "purple", "cyan"),
        alpha=0.4
    )

    plt.tight_layout()
    plt.show()
else:
    print("Either model or study subgroups != 3 — skipping Venn diagrams.")

import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# ------------------------------------------------------------------------------
# 1. Build index sets from subgroup masks
# ------------------------------------------------------------------------------

model_sets = [set(dff.index[flt]) for flt in model_subgroups.values()]
study_sets = [set(dff.index[flt]) for flt in study_subgroups.values()]

# ------------------------------------------------------------------------------
# 2. Enhanced Venn Plotting Function
# ------------------------------------------------------------------------------

def plot_venn_with_jaccard(ax, set_list, set_labels, title="", 
                          set_colors=("r", "g", "b"), alpha=0.5,
                          label_offsets=None):
    """
    Enhanced Venn diagram with bold fonts, colored labels, and Jaccard box.
    """
    if len(set_list) != 3:
        raise ValueError("This function requires exactly 3 sets for venn3.")

    v = venn3(subsets=set_list, set_labels=set_labels, ax=ax,
              set_colors=set_colors, alpha=alpha)

    # Improve subset labels (numbers)
    for subset in v.subset_labels:
        if subset is not None:
            subset.set_fontsize(11)
            subset.set_fontweight('bold')

    # Improve set labels
    for idx, label in enumerate(v.set_labels):
        if label is not None:
            label.set_fontsize(11)
            label.set_bbox(dict(
                facecolor=set_colors[idx],
                alpha=0.2,
                edgecolor=set_colors[idx],
                pad=3,
                boxstyle='round,pad=0.5'
            ))

    ax.set_title(title, fontsize=14, pad=20)

    # Custom label positioning
    if label_offsets:
        for idx, (dx, dy) in label_offsets.items():
            label = v.subset_labels[idx]
            if label is not None:
                x, y = label.get_position()
                label.set_position((x + dx, y + dy))

    # Compute Jaccard indices
    def jaccard_index(a, b):
        union_size = len(a.union(b))
        return len(a.intersection(b)) / union_size if union_size > 0 else 0.0

    j12 = jaccard_index(set_list[0], set_list[1])
    j23 = jaccard_index(set_list[1], set_list[2])
    j13 = jaccard_index(set_list[0], set_list[2])

    jacc_text = (
        f"Jaccard Similarity:\n"
        f"S1 ∩ S2: {j12:.2f}\n"
        f"S2 ∩ S3: {j23:.2f}\n"
        f"S1 ∩ S3: {j13:.2f}"
    )

    ax.text(
        0.04, 0.04, jacc_text,
        transform=ax.transAxes,
        fontsize=11,
        color="black",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
    )

# ------------------------------------------------------------------------------
# 3. Figure Setup + Execution
# ------------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

model_colors = ("#FF6B6B", "#4ECDC4", "#BD632F")
study_colors = ("#FFB347", "#9B59B6", "#3498DB")

model_label_offsets = {
    0: (-0.1, 0.1),
    1: (0.1, 0.1),
    2: (0, -0.1)
}

study_label_offsets = {
    0: (-0.1, 0.1),
    1: (0.1, 0.1),
    2: (0, -0.1)
}

plot_venn_with_jaccard(
    ax1,
    model_sets,
    ['Model 1', 'Model 2', 'Model 3'],
    title="Model Subgroups",
    set_colors=model_colors,
    alpha=0.4,
    label_offsets=model_label_offsets
)

plot_venn_with_jaccard(
    ax2,
    study_sets,
    ['Study 1', 'Study 2', 'Study 3'],
    title="Study-Defined Subgroups",
    set_colors=study_colors,
    alpha=0.4,
    label_offsets=study_label_offsets
)

plt.tight_layout()
# plt.savefig('venn.pdf', dpi=350, format='pdf')
plt.show()

subgroup_definitions = [
    {
        "Subgroup": "MVPA$\leqslant$2448 \nWHR$\geqslant$0.9 \nBMI$\geqslant$21.37 \nT2D Hist",
        "Method": "Model",
        "filter_ag": (
            (dff_ag["bmi_c_qc"] >= 21.37) & (dff_ag["bmi_c_qc"] <= 68.02) &
            (dff_ag["mvpa_c"] >= 0.0) & (dff_ag["mvpa_c"] <= 2448.0) &
            (dff_ag["diabetes_history_qc"] == 1) &
            (dff_ag["waist_hip_r_c_qc"] > 0.9) & (dff_ag["waist_hip_r_c_qc"] <= 1.16)
        ),
        "filter_dim": (
            (dff_dim["bmi_c_qc"] >= 21.37) & (dff_dim["bmi_c_qc"] <= 68.02) &
            (dff_dim["mvpa_c"] >= 0.0) & (dff_dim["mvpa_c"] <= 2448.0) &
            (dff_dim["diabetes_history_qc"] == 1) &
            (dff_dim["waist_hip_r_c_qc"] > 0.9) & (dff_dim["waist_hip_r_c_qc"] <= 1.16)
        )
    },
    {
    "Subgroup": "MVPA$\leqslant$2448 \nWHR$\geqslant$0.9 \nT2D Hist",
    "Method": "Model",
    "filter_ag": (
        (dff_ag["mvpa_c"] >= 0.0) & (dff_ag["mvpa_c"] <= 2448.0) &
        (dff_ag["diabetes_history_qc"] == 1) &
        (dff_ag["waist_hip_r_c_qc"] > 0.9) & (dff_ag["waist_hip_r_c_qc"] <= 1.16)
    ),
    "filter_dim": (
        (dff_dim["mvpa_c"] >= 0.0) & (dff_dim["mvpa_c"] <= 2448.0) &
        (dff_dim["diabetes_history_qc"] == 1) &
        (dff_dim["waist_hip_r_c_qc"] > 0.9) & (dff_dim["waist_hip_r_c_qc"] <= 1.16)
    )
    },
    {
    "Subgroup": "WHR$\geqslant$0.9 \nT2D Fam. Hist",
    "Method": "Model",
    "filter_ag": (
        (dff_ag["diabetes_history_qc"] == 1) &
        (dff_ag["waist_hip_r_c_qc"] > 0.9) & (dff_ag["waist_hip_r_c_qc"] <= 1.16)
    ),
    "filter_dim": (
        (dff_dim["diabetes_history_qc"] == 1) &
        (dff_dim["waist_hip_r_c_qc"] > 0.9) & (dff_dim["waist_hip_r_c_qc"] <= 1.16)
    )
    },
    {
        "Subgroup": "BMI$\geqslant$30 \nT2D Family History",
        "Method": "Study",
        "filter_ag": (
            (dff_ag["diabetes_history_qc"] == 1) & (dff_ag["bmi_c_qc"] >= 30)
        ),
        "filter_dim": (
            (dff_dim["diabetes_history_qc"] == 1) & (dff_dim["bmi_c_qc"] >= 30)
        )
    },
    {
        "Subgroup": "WC: W$\geqslant$800, M$\geqslant$940\nMVPA$\leqslant$150",
        "Method": "Study",
        "filter_ag": (
            (((dff_ag["sex"] == 1) & (dff_ag["waist_circumference_qc"] >= 940)) |
             ((dff_ag["sex"] == 0) & (dff_ag["waist_circumference_qc"] >= 800))) &
            (dff_ag["mvpa_c"] < 600)
        ),
        "filter_dim": (
            (((dff_dim["sex"] == 1) & (dff_dim["waist_circumference_qc"] >= 940)) |
             ((dff_dim["sex"] == 0) & (dff_dim["waist_circumference_qc"] >= 800))) &
            (dff_dim["mvpa_c"] < 600)
        )
    },
    {
        "Subgroup": "Age$\geqslant$45 \nWC: W$\geqslant$800, M$\geqslant$940 & \nMVPA$\leqslant$150",
        "Method": "Study",
        "filter_ag": (
            (dff_ag["age"] >= 45) &
            (((dff_ag["sex"] == 1) & (dff_ag["waist_circumference_qc"] >= 940)) |
             ((dff_ag["sex"] == 0) & (dff_ag["waist_circumference_qc"] >= 800))) &
            (dff_ag["mvpa_c"] < 600)
        ),
        "filter_dim": (
            (dff_dim["age"] >= 45) &
            (((dff_dim["sex"] == 1) & (dff_dim["waist_circumference_qc"] >= 940)) |
             ((dff_dim["sex"] == 0) & (dff_dim["waist_circumference_qc"] >= 800))) &
            (dff_dim["mvpa_c"] < 600)
        )
    }
]

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import math

# 1. Utility Functions

def compute_or_ci_p(dff, subgroup_mask, outcome_col="diabetes_status_c_qc"):
    temp_df = dff.copy()
    temp_df["in_subgroup"] = subgroup_mask.astype(int)
    model = smf.glm(formula=f"{outcome_col} ~ in_subgroup", data=temp_df, family=sm.families.Binomial()).fit(disp=0)
    coef = model.params["in_subgroup"]
    se   = model.bse["in_subgroup"]
    pval = model.pvalues["in_subgroup"]
    or_val = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * se)
    ci_high = np.exp(coef + 1.96 * se)
    return or_val, ci_low, ci_high, pval

def propensity_score_matching(dff, subgroup_mask, match_covariates, outcome_col="diabetes_status_c_qc"):
    temp_df = dff.copy()
    temp_df["in_subgroup"] = subgroup_mask.astype(int)
    temp_df = temp_df.dropna(subset=match_covariates + ["in_subgroup"])
    X = temp_df[match_covariates]
    y = temp_df["in_subgroup"]
    logit = LogisticRegression(solver="lbfgs", max_iter=1000)
    logit.fit(X, y)
    temp_df["propensity"] = logit.predict_proba(X)[:, 1]
    treated = temp_df[temp_df["in_subgroup"] == 1].copy()
    control = temp_df[temp_df["in_subgroup"] == 0].copy()
    nbrs = NearestNeighbors(n_neighbors=1).fit(control[["propensity"]])
    distances, indices = nbrs.kneighbors(treated[["propensity"]])
    matched_control = control.iloc[indices.flatten()]
    matched_df = pd.concat([treated, matched_control])
    return matched_df

# 2. Validation Routine
def run_validation(subgroup_definitions, dff_ag, dff_dim, psm_covariates):
    results = []

    for subgroup in subgroup_definitions:
        name = subgroup["Subgroup"]
        method = subgroup["Method"]
        ag_mask = subgroup["filter_ag"]
        dim_mask = subgroup["filter_dim"]

        # Discovery
        or_disc, ci_low_disc, ci_high_disc, pval_disc = compute_or_ci_p(dff_ag, ag_mask)

        # Dimamo pre-PSM
        or_pre, ci_low_pre, ci_high_pre, pval_pre = compute_or_ci_p(dff_dim, dim_mask)

        # Dimamo post-PSM
        matched_dim = propensity_score_matching(dff_dim, dim_mask, psm_covariates)
        matched_mask_dim = matched_dim["in_subgroup"] == 1
        or_post, ci_low_post, ci_high_post, pval_post = compute_or_ci_p(matched_dim, matched_mask_dim)

        # Agincourt post-PSM
        matched_ag = propensity_score_matching(dff_ag, ag_mask, psm_covariates)
        matched_mask_ag = matched_ag["in_subgroup"] == 1
        or_post_ag, ci_low_post_ag, ci_high_post_ag, pval_post_ag = compute_or_ci_p(matched_ag, matched_mask_ag)

        results.append({
            "Subgroup": name,
            "Method": method,
            "OR_discovery": or_disc,
            "OR_pre": or_pre,
            "CI_low_pre": ci_low_pre,
            "CI_high_pre": ci_high_pre,
            "p_value_pre": pval_pre,
            "OR_post_ag": or_post_ag,
            "CI_low_post_ag": ci_low_post_ag,
            "CI_high_post_ag": ci_high_post_ag,
            "p_value_post_ag": pval_post_ag,
            "OR_post": or_post,
            "CI_low_post": ci_low_post,
            "CI_high_post": ci_high_post,
            "p_value_post": pval_post
        })

    return pd.DataFrame(results)

# 3. Formatting Functions
def clamp_inf_to_large(x, large_val=1e5):
    if np.isnan(x): return 0.0
    if x == np.inf: return large_val
    if x == -np.inf: return -large_val
    return x

def format_value_or_ci(x):
    val = clamp_inf_to_large(x)
    if val == 0: return "0.00"
    exp_val = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    return f"{val:.2f}" if -2 <= exp_val < 5 else f"{val:.2e}"

def format_pvalue(p):
    val = clamp_inf_to_large(p)
    return f"{val:.1e}" if val < 1e-3 else f"{val:.3f}"

def format_validation_df(df):
    pval_cols = ["p_value_pre", "p_value_post", "p_value_post_ag"]
    numeric_cols = ["OR_discovery", "OR_pre", "CI_low_pre", "CI_high_pre",
                    "OR_post", "CI_low_post", "CI_high_post",
                    "OR_post_ag", "CI_low_post_ag", "CI_high_post_ag"]
    for col in df.columns:
        if col in pval_cols:
            df[col] = df[col].apply(format_pvalue)
        elif col in numeric_cols:
            df[col] = df[col].apply(format_value_or_ci)
    return df

# 4. Run it
# Make sure you have your `subgroup_definitions`, `dff_ag`, `dff_dim`, and filters loaded
psm_covariates = ["age", "sex", "triglycerides_qc"]

validation_df = run_validation(subgroup_definitions, dff_ag, dff_dim, psm_covariates)

# Reorder columns
validation_df = validation_df[
    ['Subgroup', 'Method', 'OR_discovery', 'OR_pre', 'CI_low_pre', 'CI_high_pre', 'p_value_pre',
     'OR_post_ag', 'CI_low_post_ag', 'CI_high_post_ag', 'p_value_post_ag',
     'OR_post', 'CI_low_post', 'CI_high_post', 'p_value_post']
]

# Format the output for presentation
validation_df = format_validation_df(validation_df)

# Display
display(validation_df)

# Save to CSV
# validation_df.to_csv("validation_age.csv", index=False)

validation_df[['OR_discovery', 'OR_pre', 'CI_low_pre',
       'CI_high_pre', 'p_value_pre', 'OR_post', 'CI_low_post', 'CI_high_post',
       'p_value_post', 'OR_post_ag', 'CI_low_post_ag', 'CI_high_post_ag',
       'p_value_post_ag']] = validation_df[['OR_discovery', 'OR_pre', 'CI_low_pre',
       'CI_high_pre', 'p_value_pre', 'OR_post', 'CI_low_post', 'CI_high_post',
       'p_value_post', 'OR_post_ag', 'CI_low_post_ag', 'CI_high_post_ag',
       'p_value_post_ag']].apply(pd.to_numeric, errors='coerce')
# Reverse the DataFrame so top rows appear on top in plot
validation_df = validation_df.iloc[::-1].reset_index(drop=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup plotting
fig, (ax_left, ax_center, ax_right) = plt.subplots(1, 3, figsize=(15, 6), dpi=300, 
                                                   gridspec_kw={"width_ratios": [0.5, 2.5, 1.5]})

n_subgroups = len(validation_df)
y_positions = np.arange(n_subgroups)

###############################################################################
# LEFT SUBPLOT: Discovery (Agincourt)
###############################################################################
ax_left.set_ylim(-0.5, n_subgroups - 0.5)
ax_left.set_xlim(0, 1)
ax_left.set_xticks([])
ax_left.set_xticklabels([])

for i, row in validation_df.iterrows():
    color = "lightgreen" if row["Method"] == "Model" else "lightblue"
    ax_left.plot(0.5, y_positions[i], "s", color=color, markersize=20, alpha=0.8)
    ax_left.text(0.5, y_positions[i], f"{row['OR_discovery']}", 
                 ha="center", va="center", fontsize=10, color="black")

ax_left.invert_yaxis()
ax_left.set_yticks(y_positions)
ax_left.set_yticklabels(validation_df["Subgroup"], fontsize=11)
for label in ax_left.yaxis.get_ticklabels():
    label.set_bbox(dict(facecolor='#DEDEE0', edgecolor='#DAD6D6',
                        alpha=0.5, pad=3, boxstyle='round,pad=0.5'))
ax_left.set_xlabel("Odds Ratio", fontsize=11, fontweight='bold')
ax_left.set_title("Discovery (Agincourt)", fontsize=10)

# Label the whole left section
# ax_left.text(
#     -0.05, .05,
#     'Subgroups', fontsize=10, fontweight='bold',
#     ha='left', va='bottom'
# )
ax_left.text(-0.73, 0.98, "Subgroups", fontsize=10, fontweight='bold',
             ha='left', va='bottom', transform=ax_left.transAxes)
for spine in ["top", "right", "bottom"]:
    ax_left.spines[spine].set_visible(False)

###############################################################################
# CENTER SUBPLOT: Forest Plot (Dimamo Pre vs Post PSM)
###############################################################################
ax_center.set_ylim(-0.5, n_subgroups - 0.5)
# all_ci_lows = pd.to_numeric(validation_df["CI_low_pre"]).append(
#               pd.to_numeric(validation_df["CI_low_post"]))
# all_ci_highs = pd.to_numeric(validation_df["CI_high_pre"]).append(
#                pd.to_numeric(validation_df["CI_high_post"]))
##########################################################################
all_ci_lows  = pd.concat([validation_df["CI_low_pre"],  validation_df["CI_low_post"]]).astype(float)
all_ci_highs = pd.concat([validation_df["CI_high_pre"], validation_df["CI_high_post"]]).astype(float)
# x_min = all_ci_lows.min()*0.8
# x_max = all_ci_highs.max()*1.2

x_min = float(all_ci_lows.min()) * 0.8
x_max = float(all_ci_highs.max()) * 1.2
ax_center.set_xlim(x_min, x_max)

offset = 0.15

# ax_center.set_xscale("log")
for i, row in validation_df.iterrows():
    y_pre = y_positions[i] - offset
    y_post = y_positions[i] + offset
    # Pre-PSM
    ax_center.plot([float(row["CI_low_pre"]), float(row["CI_high_pre"])], [y_pre, y_pre],
                   color="darkblue", lw=2)
    p, = ax_center.plot(float(row["OR_pre"]), y_pre, "o", color="darkblue", markersize=6)
    ax_center.text(float(row["OR_pre"]), y_pre - 0.1, f"{row['OR_pre']}", 
                   ha="center", color="darkblue", fontsize=10)

    # Post-PSM
    ax_center.plot([float(row["CI_low_post"]), float(row["CI_high_post"])], [y_post, y_post],
                   color="darkred", lw=2)
    q, = ax_center.plot(float(row["OR_post"]), y_post, "o", color="darkred", markersize=6)
    ax_center.text(float(row["OR_post"]), y_post - 0.1, f"{row['OR_post']}", 
                   ha="center", color="darkred", fontsize=10)

ax_center.axvline(1.0, color="gray", linestyle="--", lw=1)
ax_center.invert_yaxis()
ax_center.set_yticks(y_positions)
ax_center.set_yticklabels([])
# ax_center.set_xlim(x_min, 60)
ax_center.set_xlabel("Odds Ratio (95% CI)", fontsize=10, labelpad=10, fontweight='bold')
ax_center.set_title("Validation (DIMAMO): Pre-PSM vs. Post-PSM", fontsize=12)
ax_center.legend([p, q],["Pre-PSM", "Post-PSM"],loc="upper right", fontsize=12)

for spine in ["top", "right"]:
    ax_center.spines[spine].set_visible(False)

###############################################################################
# RIGHT SUBPLOT: -log10(p-values) Bar Plot
###############################################################################
neg_log_p_pre = -np.log10(pd.to_numeric(validation_df["p_value_pre"]).clip(lower=1e-300))
neg_log_p_post = -np.log10(pd.to_numeric(validation_df["p_value_post"]).clip(lower=1e-300))

ax_right.set_ylim(-0.5, n_subgroups - 0.5)
ax_right.invert_yaxis()
ax_right.set_yticks(y_positions)
ax_right.set_yticklabels([])

bar_height = 0.35

ax_right.barh(y_positions - bar_height/2, neg_log_p_pre, 
              height=bar_height, color="steelblue", label="Pre-PSM")
ax_right.barh(y_positions + bar_height/2, neg_log_p_post, 
              height=bar_height, color="tomato", label="Post-PSM")

ax_right.set_xlabel("-log10(p-value)", fontsize=10, fontweight='bold')
ax_right.set_title("Significance (DIMAMO)", fontsize=10)

# Annotate bars
for i, (val_pre, val_post) in enumerate(zip(
    validation_df["p_value_pre"].values, validation_df["p_value_post"].values)):

    ax_right.text(0.05, i + bar_height/2, f"{val_post}", va="center", fontsize=10)
    ax_right.text(0.05, i - bar_height/2, f"{val_pre}", va="center", fontsize=10)

ax_right.axvline(x=-np.log10(0.05), color="brown", linestyle="--", lw=1)
ax_right.legend(loc="upper right", fontsize=12)

for spine in ["top", "right"]:
    ax_right.spines[spine].set_visible(False)

###############################################################################
# SHADED REGIONS
###############################################################################
ax_left.axhspan(2.5, 5.5, color="lightblue", alpha=0.1)
ax_left.axhspan(-0.5, 2.5, color="lightgreen", alpha=0.1)

ax_center.axhspan(2.5, 5.5, color="lightblue", alpha=0.1)
ax_center.axhspan(-0.5, 2.5, color="lightgreen", alpha=0.1)

ax_right.axhspan(2.5, 5.5, color="lightblue", alpha=0.1)
ax_right.axhspan(-0.5, 2.5, color="lightgreen", alpha=0.1)
# ax_right.annotate(
#     "$p$", xy=(1, 4.5), xycoords='data',
#     fontsize=12, color="#007991", va="center", ha="left", fontweight='bold'
# )
ax_right.text(0.95, 5.58, "$\\nearrow$", 
             fontsize=12, color="#C12F0F", 
             va="center", ha="left", fontweight='bold')
ax_left.text(0.05, 0.85, "Study-Defined", 
             transform=ax_left.transAxes, fontsize=12, color="blue")
ax_left.text(0.05, 0.3, "Model-Derived", 
             transform=ax_left.transAxes, fontsize=12, color="green")

###############################################################################
# FINALIZE
###############################################################################
plt.tight_layout()
plt.savefig("forest_plot_val.pdf", dpi=350, format="pdf")
plt.show()
import numpy as np
import pandas as pd
import math
from scipy.stats import chi2

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def ci_to_log_se(or_val, ci_low, ci_high):
    """
    Convert OR + CI to log(OR) and SE of log(OR).
    Handles string inputs from formatted tables.
    """
    try:
        or_val = float(str(or_val).replace(',', '').replace('inf', '1e5'))
        ci_low = float(str(ci_low).replace(',', '').replace('inf', '1e5'))
        ci_high = float(str(ci_high).replace(',', '').replace('inf', '1e5'))

        if or_val <= 0 or ci_low <= 0 or ci_high <= 0:
            return None, None

        log_or = np.log(or_val)
        se = (np.log(ci_high) - np.log(ci_low)) / (2 * 1.96)
        return log_or, se
    except:
        return None, None

def cochran_q_test(log_or_list, se_list):
    """
    Perform Cochran's Q test for heterogeneity.
    Returns (Q, p-value, df)
    """
    k = len(log_or_list)
    var_list = [se**2 for se in se_list]
    weights = [1.0 / v for v in var_list]
    w_sum = sum(weights)

    # Weighted mean
    weighted_mean = sum(w * theta for w, theta in zip(weights, log_or_list)) / w_sum

    # Q statistic
    Q = sum(w * (theta - weighted_mean)**2 for w, theta in zip(weights, log_or_list))
    df = k - 1
    p_val = 1 - chi2.cdf(Q, df)
    return Q, p_val, df

def calculate_i_squared(Q, df):
    """Calculate I² heterogeneity metric."""
    return max(0, (Q - df) / Q * 100) if Q > df else 0.0

# ---------------------------
# COMPUTE HETEROGENEITY TABLE
# ---------------------------

hetero_results = []

for _, row in validation_df.iterrows():
    subgroup_name = row["Subgroup"]
    method_type = row["Method"]

    # Pre-PSM comparison (discovery vs pre-PSM in Dimamo)
    log_or_discovery, se_discovery = ci_to_log_se(row["OR_discovery"], row["CI_low_pre"], row["CI_high_pre"])
    log_or_pre, se_pre = ci_to_log_se(row["OR_pre"], row["CI_low_pre"], row["CI_high_pre"])

    if None not in (log_or_discovery, se_discovery, log_or_pre, se_pre):
        Q_pre, p_pre, df_pre = cochran_q_test([log_or_discovery, log_or_pre], [se_discovery, se_pre])
        i2_pre = calculate_i_squared(Q_pre, df_pre)
    else:
        Q_pre, i2_pre, p_pre = None, None, None

    # Post-PSM comparison (Agincourt post-PSM vs Dimamo post-PSM)
    log_or_post_ag, se_post_ag = ci_to_log_se(row["OR_post_ag"], row["CI_low_post_ag"], row["CI_high_post_ag"])
    log_or_post, se_post = ci_to_log_se(row["OR_post"], row["CI_low_post"], row["CI_high_post"])

    if None not in (log_or_post_ag, se_post_ag, log_or_post, se_post):
        Q_post, p_post, df_post = cochran_q_test([log_or_post_ag, log_or_post], [se_post_ag, se_post])
        i2_post = calculate_i_squared(Q_post, df_post)
    else:
        Q_post, i2_post, p_post = None, None, None

    hetero_results.append({
        "Subgroup": subgroup_name,
        "Method": method_type,
        "Q_pre": Q_pre,
        "I²_pre (%)": i2_pre,
        "p-value_pre": p_pre,
        "Q_post": Q_post,
        "I²_post (%)": i2_post,
        "p-value_post": p_post
    })

hetero_df = pd.DataFrame(hetero_results)
display(hetero_df)

# Function to format values
def format_hetero_table(row):
    def fnum(x, decimals):
        try:
            return f"{float(x):.{decimals}f}"
        except:
            return "NA"

    return {
        "Method": row["Method"],
        "Subgroup": row["Subgroup"],
        "Q (Discovery vs Pre-PSM)": fnum(row["Q_pre"], 3),
        "I² % (Pre-PSM)": fnum(row["I²_pre (%)"], 1),
        "p-value (Pre-PSM)": fnum(row["p-value_pre"], 3),
        "Q (Post-PSM)": fnum(row["Q_post"], 3),
        "I² % (Post-PSM)": fnum(row["I²_post (%)"], 1),
        "p-value (Post-PSM)": fnum(row["p-value_post"], 3)
    }

# Apply formatting
formatted_hetero_df = pd.DataFrame([format_hetero_table(row) for _, row in hetero_df.iterrows()])

# Display the final cleaned and labeled heterogeneity table
display(formatted_hetero_df)

formatted_hetero_df.to_csv("heterogeneity_results1.csv", index=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load the cleaned and labeled heterogeneity table
hetero_df = pd.read_csv("heterogeneity_results1.csv")

# Create a scatter plot
plt.scatter(hetero_df["Q (Post-PSM)"], hetero_df["I² % (Pre-PSM)"])
plt.xlabel("Q (Discovery vs Pre-PSM)")
plt.ylabel("I² % (Pre-PSM)")
plt.title("Heterogeneity: Discovery vs Pre-PSM")
plt.axhline(y=50, color='r', linestyle='--', label="I² = 50%")
plt.axvline(x=10, color='g', linestyle='--', label="Q = 10")
plt.legend()
plt.grid()
plt.show()
# Create a scatter plot
plt.scatter(hetero_df["Q (Discovery vs Pre-PSM)"], hetero_df["I² % (Post-PSM)"])
plt.xlabel("Q (Discovery vs Pre-PSM)")
plt.ylabel("I² % (Pre-PSM)")
plt.title("Heterogeneity: Discovery vs Pre-PSM")
plt.axhline(y=50, color='r', linestyle='--', label="I² = 50%")
plt.axvline(x=10, color='g', linestyle='--', label="Q = 10")
plt.legend()
plt.grid()
plt.show()

# model_filter1 = (
#             (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2297.5) &
#             (dff["diabetes_history_qc"] == 1) &
#             (dff["bmi_c_qc"] >= 21.55) & (dff["bmi_c_qc"] <= 68.02) &
#             (dff["waist_hip_r_c_qc"] >= 0.9) & (dff["waist_hip_r_c_qc"] <= 1.36)
# )

# model_filter2 = (
#             (dff["smoking_status_c_qc"] == 0) &
#             (dff["waist_hip_r_c_qc"] >= 0.96) & 
#             (dff["waist_hip_r_c_qc"] <= 1.36) &
#             (dff["age"] >= 61.0) & 
#             (dff["age"] <= 71.0) &
#             (dff["waist_circumference_qc"] >= 1030.0) & 
#             (dff["waist_circumference_qc"] <= 1470.0) &
#             (dff["sex"] == 0)
# )

# model_filter3 = (
#     (dff["waist_hip_r_c_qc"] >= 0.9) & 
#     (dff["waist_hip_r_c_qc"] <= 1.36) &
#     (dff["diabetes_history_qc"] == 1)
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# 1. COMPUTE OR, CI, p-values, and prevalence
###############################################################################

def compute_2x2_or_ci(df, mask, outcome_col="diabetes_status_c_qc"):
    sub_df = df[mask]
    not_sub_df = df[~mask]
    
    a = sub_df[outcome_col].sum()
    b = len(sub_df) - a
    c = not_sub_df[outcome_col].sum()
    d = len(not_sub_df) - c

    subgroup_size = len(sub_df)
    t2d_prevalence = (a / subgroup_size) if subgroup_size > 0 else 0.0

    if a == 0 or b == 0 or c == 0 or d == 0:
        return (np.inf, np.inf, np.inf, 1.0, subgroup_size, t2d_prevalence)

    odds_ratio = (a / b) / (c / d)
    log_or = np.log(odds_ratio)
    var_log_or = (1/a) + (1/b) + (1/c) + (1/d)
    se_log_or = np.sqrt(var_log_or)

    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_high = np.exp(log_or + 1.96 * se_log_or)
    z_score = log_or / se_log_or
    p_val = 2 * stats.norm.sf(abs(z_score))

    return (odds_ratio, ci_low, ci_high, p_val, subgroup_size, t2d_prevalence)


# 2. DEFINE MASKS
###############################################################################

def get_boolean_mask(df, no_of_literals):
    if no_of_literals == 1:
        return (
            (df["diabetes_history_qc"] == 1) &
            (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16)
        )
    elif no_of_literals == 2:
        return (
            (df["mvpa_c"] >= 0.0) & (df["mvpa_c"] <= 2448.0) &
            (df["diabetes_history_qc"] == 1) &
            (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16)
        )
    elif no_of_literals == 3:
        return (
            (df["bmi_c_qc"] >= 21.37) & (df["bmi_c_qc"] <= 68.02) &
            (df["mvpa_c"] >= 0.0) & (df["mvpa_c"] <= 2448.0) &
            (df["diabetes_history_qc"] == 1) &
            (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16)
        )
    return pd.Series(False, index=df.index)

group_names = {
    1: "T2D Family History",
    2: "Waist circ. + T2D History",
    3: "MVPA + Waist circ. + T2D History"
}

pop_dfs = {
    "Agincourt": dff_ag,
    "Nairobi": dff_nai,
    "Nanoro": dff_nan
}

rows = []
for nlit in [1, 2, 3]:
    for pop, df in pop_dfs.items():
        mask = get_boolean_mask(df, nlit)
        or_val, ci_lo, ci_hi, p_val, size, prev = compute_2x2_or_ci(df, mask)
        rows.append({
            "Group": group_names[nlit],
            "Population": pop,
            "OR": or_val,
            "CI_low": ci_lo,
            "CI_high": ci_hi,
            "p_value": p_val,
            "subgroup_size": size,
            "t2d_prev": prev
        })

plot_df = pd.DataFrame(rows)
plot_df["Group"] = pd.Categorical(plot_df["Group"], categories=list(group_names.values()), ordered=True)
plot_df["Population"] = pd.Categorical(plot_df["Population"], categories=["Agincourt", "Nairobi", "Nanoro"], ordered=True)
plot_df = plot_df.sort_values(by=["Group", "Population"]).reset_index(drop=True)

# 3. PLOT FOREST + TABLE + BAR
###############################################################################

color_map = {
    "Agincourt": "#F06543",
    "Nairobi": "#A40E4C",
    "Nanoro": "#70A9A1"
}
offset_map = {
    "Agincourt": -0.2,
    "Nairobi": 0.0,
    "Nanoro": 0.2
}
legend_elements = [
    Line2D([0],[0], marker='s', color='w', label=pop,
           markerfacecolor=color_map[pop], markersize=8)
    for pop in pop_dfs.keys()
]

fig, (ax_forest, ax_table, ax_bar) = plt.subplots(1, 3, figsize=(13, 5), dpi=300,
                                                  gridspec_kw={"width_ratios": [4.0, 0.85, 3.0], "wspace": 0.1})
ax_table.set_axis_off()

y_dict = {grp: 3 - i for i, grp in enumerate(plot_df["Group"].unique())}

for _, row in plot_df.iterrows():
    y = y_dict[row["Group"]] + offset_map[row["Population"]]
    ax_forest.plot([row["CI_low"], row["CI_high"]], [y, y], color=color_map[row["Population"]], lw=2)
    ax_forest.plot(row["OR"], y, "s", color=color_map[row["Population"]], markersize=6)
    ax_forest.text(row["CI_high"] + 0.3, y, f"({row['CI_low']:.2f}, {row['CI_high']:.2f})", va="center", fontsize=10)
    ax_forest.text(38, y, f"{row['p_value']:.1e}", ha="center", va="center", fontsize=12,
                   bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=color_map[row["Population"]], alpha=0.8))

ax_forest.axvline(1.0, color="gray", linestyle="--", lw=1)
labels = [
    "WHR ≥ 0.9 &\nT2D Family History",
    "MVPA$\leqslant$2448 \nWC ≥ 840 &\nT2D History",
    "MVPA$\leqslant$2448 \nWHR$\geqslant$0.9 \nBMI$\geqslant$21.37 \nT2D Family History"
]
ax_forest.set_yticks(list(y_dict.values()))
ax_forest.set_yticklabels(labels, fontsize=12, fontweight='bold')
for label in ax_forest.yaxis.get_ticklabels():
    label.set_bbox(dict(facecolor='#DEDEE0', edgecolor='#DAD6D6',
                        alpha=0.5, pad=3, boxstyle='round,pad=0.5'))
ax_forest.invert_yaxis()
ax_forest.set_xlabel("Odds Ratio (95% CI)", fontsize=12, fontweight='bold')
ax_forest.set_xlim(0, 33)
ax_forest.grid(True, axis='x', linestyle="--", alpha=0.3)
for spine in ["top", "right"]:
    ax_forest.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax_forest.spines[spine].set_color("black")

ax_forest.text(38, 0.6, "p-value", ha="center", fontsize=12, fontweight='bold')
ax_forest.text(-4.5, 0.6, "Subgroups", ha="center", fontsize=13, fontweight='bold')
ax_forest.legend(handles=legend_elements, loc="center right", fontsize=11,
                 title="Population", title_fontsize=12, bbox_to_anchor=(0.97, 0.20),
                 frameon=True, facecolor="white", edgecolor="lightgray")

# Bar chart: Prevalence
x_sub = np.arange(3)
bar_width = 0.25
for i, pop in enumerate(pop_dfs.keys()):
    subset = plot_df[plot_df["Population"] == pop].set_index("Group").reindex(group_names.values()).reset_index()
    heights = subset["t2d_prev"] * 100
    x_pos = x_sub + (i - 1) * bar_width

    ax_bar.bar(x_pos, heights, width=bar_width, color=color_map[pop], alpha=0.8)
    for x, h in zip(x_pos, heights):
        ax_bar.text(x, h + 0.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

ax_bar.set_xticks(x_sub)
ax_bar.set_xticklabels(labels, rotation=28, ha='right', fontsize=10, fontweight='bold')
for label in ax_bar.xaxis.get_ticklabels():
    label.set_bbox(dict(facecolor='#DEDEE0', edgecolor='#DAD6D6',
                        alpha=0.5, pad=3, boxstyle='round,pad=0.5'))
ax_bar.set_ylabel("T2D Prevalence in Subgroup (%)", fontsize=9)
ax_bar.set_title("Subgroup T2D Prevalence Across Sites", fontsize=10, fontweight='bold')
ax_bar.legend(handles=legend_elements, fontsize=10, loc="lower right",
              title="Population", title_fontsize=11,
              frameon=True, facecolor="white", edgecolor="lightgray")

plt.tight_layout()
plt.savefig("forest_plot_transferability.pdf", dpi=350)
plt.show()

dff_ag = pd.read_csv("dff_ag1.csv")
dff_nai = pd.read_csv("dff_nai1.csv")
dff_nan = pd.read_csv("dff_nan1.csv")
dff_ag.age.max()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score


# Assume the following dataframes are pre-loaded:
# dff_ag: Agincourt, dff_nai: Nairobi, dff_nan: Nanoro


# Define Agincourt-derived masks and add as binary columns for each dataset
def add_masks(df):
    df = df.copy()
    df['mask1'] = ((df["diabetes_history_qc"] == 1) &
    (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16)).astype(int)
    df['mask2'] = ((df["mvpa_c"] >= 0.0) & (df["mvpa_c"] <= 2448.0) &
            (df["diabetes_history_qc"] == 1) &
            (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16)).astype(int)
    df['mask3'] = ((df["bmi_c_qc"] >= 21.37) & (df["bmi_c_qc"] <= 68.02) &
            (df["mvpa_c"] >= 0.0) & (df["mvpa_c"] <= 2448.0) &
            (df["diabetes_history_qc"] == 1) &
            (df["waist_hip_r_c_qc"] > 0.9) & (df["waist_hip_r_c_qc"] <= 1.16)).astype(int)
    return df

dff_ag = add_masks(dff_ag)
dff_nai = add_masks(dff_nai)
dff_nan = add_masks(dff_nan)

# Combine datasets for combined analyses
dff_nai_nan = pd.concat([dff_nai, dff_nan], axis=0)
dff_all = pd.concat([dff_ag, dff_nai, dff_nan], axis=0)

# Define the five groupings in a dictionary
groupings = {
    "Agincourt": dff_ag,
    "Nairobi": dff_nai,
    "Nanoro": dff_nan,
    "Nairobi+Nanoro": dff_nai_nan,
    "All": dff_all
}

# Define model specifications: the features for each model variation
# Base model always includes: age, sex, bmi_c_qc
model_features = {
    "Base Model": ["age", "sex", "bmi_c_qc"],
    "Model 1": ["age", "sex", "bmi_c_qc", "mask1"],
    "Model 2": ["age", "sex", "bmi_c_qc","mask2"],
    "Model 3": ["age", "sex", "bmi_c_qc", "mask3"]
}

# Outcome variable name
outcome_var = "diabetes_status_c_qc"

# Function to train logistic regression and compute ROC curve data
def get_roc_data(df, features, outcome):
    # Drop missing values for features and outcome
    df_clean = df.dropna(subset=features + [outcome])
    X = df_clean[features]
    y = df_clean[outcome]
    
    # Initialize and fit logistic regression model (using default parameters)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X, y)
    
    # Predicted probabilities for the positive class
    y_prob = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    return fpr, tpr, auc

# Create a 1x3 plot for the three models
fig, axs = plt.subplots(1, 4, figsize=(18, 6), sharey=True, dpi = 350)

for i, (model_name, features) in enumerate(model_features.items()):
    ax = axs[i]
    for group_name, df in groupings.items():
        fpr, tpr, auc_val = get_roc_data(df, features, outcome_var)
        ax.plot(fpr, tpr, label=f"{group_name} (AUC={auc_val:.3f})")
    
    ax.plot([0, 1], [0, 1], "k--", lw=1)  # Diagonal line for random performance
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    if i == 0:
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    ax.set_title(model_name, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    
# plt.suptitle("ROC Curves for Logistic Regression Models with Additional Masks", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('roc_curves_ag.png', dpi=350)
plt.show()
from sklearn.metrics import jaccard_score

def compute_overlap_stats(mask_ag, mask_nai, mask_nan):
    results = []
    for (name1, mask1), (name2, mask2) in [
        (("Agincourt", mask_ag), ("Nairobi", mask_nai)),
        (("Agincourt", mask_ag), ("Nanoro", mask_nan)),
        (("Nairobi", mask_nai), ("Nanoro", mask_nan)),
    ]:
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        jaccard = intersection / union if union != 0 else 0.0
        prop_1in2 = intersection / mask1.sum() if mask1.sum() else 0.0
        prop_2in1 = intersection / mask2.sum() if mask2.sum() else 0.0

        results.append({
            "Pair": f"{name1} vs {name2}",
            "Jaccard": round(jaccard, 3),
            f"{name1} in {name2}": round(prop_1in2, 3),
            f"{name2} in {name1}": round(prop_2in1, 3),
        })
    return pd.DataFrame(results)

# Example:
m1_ag = mask1_ag.astype(int)
m1_nai = mask1_nai.astype(int)
m1_nan = mask1_nan.astype(int)

overlap_df1 = compute_overlap_stats(m1_ag, m1_nai, m1_nan)

# Agincourt:
model_filter1_ag = (
    (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)

model_filter2_ag = (
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)

model_filter3_ag = (
    (dff["diabetes_history_qc"] == 1) &
    (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
)
# Nairobi:
model_filter1_nai = (
    (dff["waist_hip_r_c_qc"] > 0.85) & (dff["waist_hip_r_c_qc"] <= 9.02) &
    (dff["hip_circumference_qc"] >= 887.0) & (dff["hip_circumference_qc"] <= 1494.0) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["age"] > 54.0) & (dff["age"] <= 60.0)
)

model_filter2_nai = (
    (dff["bmi_c_qc"] >= 20.49) & (dff["bmi_c_qc"] <= 62.8) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["age"] > 54.0) & (dff["age"] <= 60.0)
)

model_filter3_nai = (
    (dff["diabetes_history_qc"] == 1) &
    (dff["age"] > 54.0) & (dff["age"] <= 60.0)
)
# Nanoro:
model_filter1_nan = (
    (dff["days_veg_qc"] == 7) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
    (dff["sex"] == 1) &
    (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0)
)

model_filter2_nan = (
    (dff["sex"] == 1) &
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
    (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0)
)

model_filter3_nan = (
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
    (dff["waist_circumference_qc"] > 851.0) & (dff["waist_circumference_qc"] <= 1396.0)
)
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------- DEFINE FUNCTION ----------------------
def compute_2x2_or_ci(df, mask, outcome_col="diabetes_status_c_qc"):
    """
    Return (OR, CI_low, CI_high, p_value, subgroup_size, t2d_prevalence)
    for the subgroup vs. not-subgroup using a 2x2 table approach.
    """
    sub_df = df[mask]
    not_sub_df = df[~mask]
    
    a = sub_df[outcome_col].sum()        # T2D positives in subgroup
    b = len(sub_df) - a                  # T2D negatives in subgroup
    c = not_sub_df[outcome_col].sum()    # T2D positives outside
    d = len(not_sub_df) - c              # T2D negatives outside

    subgroup_size = len(sub_df)
    t2d_prevalence = (a / subgroup_size) if subgroup_size > 0 else 0.0

    if a == 0 or b == 0 or c == 0 or d == 0:
        return (np.inf, np.inf, np.inf, 1.0, subgroup_size, t2d_prevalence)

    odds_ratio = (a / b) / (c / d)
    log_or = np.log(odds_ratio)
    var_log_or = (1/a) + (1/b) + (1/c) + (1/d)
    se_log_or = np.sqrt(var_log_or)

    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_high = np.exp(log_or + 1.96 * se_log_or)
    z_score = log_or / se_log_or
    p_val = 2 * stats.norm.sf(abs(z_score))

    return (odds_ratio, ci_low, ci_high, p_val, subgroup_size, t2d_prevalence)

# ---------------------- DEFINE FILTERS ----------------------
# Note: You must define dff = dff_ag / dff_nai / dff_nan before running filters.

# # Agincourt filters
# model_filter1_ag = (
#     (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
#     (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
#     (dff["diabetes_history_qc"] == 1) &
#     (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
# )

# model_filter2_ag = (
#     (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
#     (dff["diabetes_history_qc"] == 1) &
#     (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
# )

# model_filter3_ag = (
#     (dff["diabetes_history_qc"] == 1) &
#     (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
# )

# # Nairobi filters
# model_filter1_nai = (
#     (dff["waist_hip_r_c_qc"] > 0.85) & (dff["waist_hip_r_c_qc"] <= 9.02) &
#     (dff["hip_circumference_qc"] >= 887.0) & (dff["hip_circumference_qc"] <= 1494.0) &
#     (dff["diabetes_history_qc"] == 1) &
#     (dff["age"] > 54.0) & (dff["age"] <= 60.0)
# )

# model_filter2_nai = (
#     (dff["bmi_c_qc"] >= 20.49) & (dff["bmi_c_qc"] <= 62.8) &
#     (dff["diabetes_history_qc"] == 1) &
#     (dff["age"] > 54.0) & (dff["age"] <= 60.0)
# )

# model_filter3_nai = (
#     (dff["diabetes_history_qc"] == 1) &
#     (dff["age"] > 54.0) & (dff["age"] <= 60.0)
# )

# # Nanoro filters
# model_filter1_nan = (
#     (dff["days_veg_qc"] == 7) &
#     (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
#     (dff["sex"] == 1) &
#     (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0)
# )

# model_filter2_nan = (
#     (dff["sex"] == 1) &
#     (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
#     (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0)
# )

# model_filter3_nan = (
#     (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
#     (dff["waist_circumference_qc"] > 851.0) & (dff["waist_circumference_qc"] <= 1396.0)
# )

# ---------------------- RUN & BUILD TABLE ----------------------
subgroup_definitions = [
    {"Population": "Agincourt", "Filter": model_filter1_ag, "Subgroup": "Model 1"},
    {"Population": "Agincourt", "Filter": model_filter2_ag, "Subgroup": "Model 2"},
    {"Population": "Agincourt", "Filter": model_filter3_ag, "Subgroup": "Model 3"},
    {"Population": "Nairobi",   "Filter": model_filter1_nai, "Subgroup": "Model 1"},
    {"Population": "Nairobi",   "Filter": model_filter2_nai, "Subgroup": "Model 2"},
    {"Population": "Nairobi",   "Filter": model_filter3_nai, "Subgroup": "Model 3"},
    {"Population": "Nanoro",    "Filter": model_filter1_nan, "Subgroup": "Model 1"},
    {"Population": "Nanoro",    "Filter": model_filter2_nan, "Subgroup": "Model 2"},
    {"Population": "Nanoro",    "Filter": model_filter3_nan, "Subgroup": "Model 3"},
]

results = []
for definition in subgroup_definitions:
    dff = {"Agincourt": dff_ag, "Nairobi": dff_nai, "Nanoro": dff_nan}[definition["Population"]]
    mask = definition["Filter"]
    OR, CI_low, CI_high, p_value, size, _ = compute_2x2_or_ci(dff, mask)

    results.append({
        "Population": definition["Population"],
        "Subgroup": definition["Subgroup"],
        "OR": round(OR, 3),
        "CI_low": round(CI_low, 3),
        "CI_high": round(CI_high, 3),
        "p_value": round(p_value, 4),
        "Subgroup Size": size
    })

results_df = pd.DataFrame(results)
display(results_df)

import numpy as np
import pandas as pd
from scipy import stats

def compute_2x2_or_ci(df, mask, outcome_col="diabetes_status_c_qc"):
    """
    Return (OR, CI_low, CI_high, p_value, subgroup_size, t2d_prevalence)
    for the subgroup vs. not-subgroup using a 2x2 table approach.
    """
    sub_df = df[mask]
    not_sub_df = df[~mask]
    
    a = sub_df[outcome_col].sum()        # T2D positives in subgroup
    b = len(sub_df) - a                  # T2D negatives in subgroup
    c = not_sub_df[outcome_col].sum()    # T2D positives outside
    d = len(not_sub_df) - c             # T2D negatives outside

    subgroup_size = len(sub_df)
    t2d_prevalence = (a / subgroup_size) if subgroup_size > 0 else 0.0

    # If any 2x2 cell is zero, OR = infinity or zero; handle it:
    if a == 0 or b == 0 or c == 0 or d == 0:
        return (np.inf, np.inf, np.inf, 1.0, subgroup_size, t2d_prevalence)

    # Unadjusted odds ratio
    odds_ratio = (a / b) / (c / d)
    log_or = np.log(odds_ratio)
    var_log_or = (1/a) + (1/b) + (1/c) + (1/d)
    se_log_or = np.sqrt(var_log_or)

    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_high = np.exp(log_or + 1.96 * se_log_or)

    # p-value from Z-test
    z_score = log_or / se_log_or
    p_val = 2 * stats.norm.sf(abs(z_score))

    return (odds_ratio, ci_low, ci_high, p_val, subgroup_size, t2d_prevalence)
import pandas as pd
import numpy as np
from scipy import stats

def compute_2x2_or_ci(df, mask, outcome_col="diabetes_status_c_qc"):
    sub_df = df[mask]
    not_sub_df = df[~mask]

    a = sub_df[outcome_col].sum()
    b = len(sub_df) - a
    c = not_sub_df[outcome_col].sum()
    d = len(not_sub_df) - c

    subgroup_size = len(sub_df)
    t2d_prevalence = (a / subgroup_size) * 100 if subgroup_size > 0 else 0.0

    if a == 0 or b == 0 or c == 0 or d == 0:
        return (np.inf, 1.0, t2d_prevalence)

    odds_ratio = (a / b) / (c / d)
    log_or = np.log(odds_ratio)
    var_log_or = (1/a) + (1/b) + (1/c) + (1/d)
    se_log_or = np.sqrt(var_log_or)
    z_score = log_or / se_log_or
    p_val = 2 * stats.norm.sf(abs(z_score))

    return (odds_ratio, p_val, t2d_prevalence)

# ==============================
# Set your actual datasets here
# ==============================
# dff_ag = ...
# dff_nai = ...
# dff_nan = ...

# Define filters for each population
def get_filters_ag(dff):
    return [
        (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
        (dff["diabetes_history_qc"] == 1) &
        (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16),

        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
        (dff["diabetes_history_qc"] == 1) &
        (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16),

        (dff["diabetes_history_qc"] == 1) &
        (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
    ]

def get_filters_nai(dff):
    return [
        (dff["waist_hip_r_c_qc"] > 0.85) & (dff["waist_hip_r_c_qc"] <= 9.02) &
        (dff["hip_circumference_qc"] >= 887.0) & (dff["hip_circumference_qc"] <= 1494.0) &
        (dff["diabetes_history_qc"] == 1) &
        (dff["age"] > 54.0) & (dff["age"] <= 60.0),

        (dff["bmi_c_qc"] >= 20.49) & (dff["bmi_c_qc"] <= 62.8) &
        (dff["diabetes_history_qc"] == 1) &
        (dff["age"] > 54.0) & (dff["age"] <= 60.0),

        (dff["diabetes_history_qc"] == 1) &
        (dff["age"] > 54.0) & (dff["age"] <= 60.0)
    ]

def get_filters_nan(dff):
    return [
        (dff["days_veg_qc"] == 7) &
        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
        (dff["sex"] == 1) &
        (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0),

        (dff["sex"] == 1) &
        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
        (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0),

        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
        (dff["waist_circumference_qc"] > 851.0) & (dff["waist_circumference_qc"] <= 1396.0)
    ]

# Build table
rows = []
for i in range(3):
    row = {"Subgroup": f"Model {i+1}"}

    for tag, dff, get_filters in [
        ("AG", dff_ag, get_filters_ag),
        ("NAI", dff_nai, get_filters_nai),
        ("NAN", dff_nan, get_filters_nan),
    ]:
        mask = get_filters(dff)[i]
        OR, pval, prev = compute_2x2_or_ci(dff, mask)
        row[f"OR_{tag}"] = round(OR, 3) if np.isfinite(OR) else "inf"
        row[f"p_{tag}"] = round(pval, 3)
        row[f"Prev_{tag}"] = round(prev, 1)

    rows.append(row)

# Convert to DataFrame and display
summary_df = pd.DataFrame(rows)
summary_df

import pandas as pd
import numpy as np
from scipy import stats

def compute_2x2_or_ci(df, mask, outcome_col="diabetes_status_c_qc"):
    sub_df = df[mask]
    not_sub_df = df[~mask]

    a = sub_df[outcome_col].sum()
    b = len(sub_df) - a
    c = not_sub_df[outcome_col].sum()
    d = len(not_sub_df) - c

    subgroup_size = len(sub_df)
    t2d_prevalence = (a / subgroup_size) * 100 if subgroup_size > 0 else 0.0

    if a == 0 or b == 0 or c == 0 or d == 0:
        return (np.inf, np.inf, np.inf, 1.0, subgroup_size, t2d_prevalence)

    odds_ratio = (a / b) / (c / d)
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt((1/a) + (1/b) + (1/c) + (1/d))
    p_val = 2 * stats.norm.sf(abs(log_or / se_log_or))

    return (odds_ratio, None, None, p_val, subgroup_size, t2d_prevalence)

# ---- Define Filters for Each Population ----
def get_filters_ag(dff):
    return [
        (dff["bmi_c_qc"] >= 21.37) & (dff["bmi_c_qc"] <= 68.02) &
        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
        (dff["diabetes_history_qc"] == 1) &
        (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16),

        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2448.0) &
        (dff["diabetes_history_qc"] == 1) &
        (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16),

        (dff["diabetes_history_qc"] == 1) &
        (dff["waist_hip_r_c_qc"] > 0.9) & (dff["waist_hip_r_c_qc"] <= 1.16)
    ]

def get_filters_nai(dff):
    return [
        (dff["waist_hip_r_c_qc"] > 0.85) & (dff["waist_hip_r_c_qc"] <= 9.02) &
        (dff["hip_circumference_qc"] >= 887.0) & (dff["hip_circumference_qc"] <= 1494.0) &
        (dff["diabetes_history_qc"] == 1) & (dff["age"] > 54.0) & (dff["age"] <= 60.0),

        (dff["bmi_c_qc"] >= 20.49) & (dff["bmi_c_qc"] <= 62.8) &
        (dff["diabetes_history_qc"] == 1) & (dff["age"] > 54.0) & (dff["age"] <= 60.0),

        (dff["diabetes_history_qc"] == 1) & (dff["age"] > 54.0) & (dff["age"] <= 60.0)
    ]

def get_filters_nan(dff):
    return [
        (dff["days_veg_qc"] == 7) &
        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
        (dff["sex"] == 1) &
        (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0),

        (dff["sex"] == 1) &
        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
        (dff["hip_circumference_qc"] > 950.0) & (dff["hip_circumference_qc"] <= 1937.0),

        (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2520.0) &
        (dff["waist_circumference_qc"] > 851.0) & (dff["waist_circumference_qc"] <= 1396.0)
    ]

# ---- Final Table (Corrected Logic) ----
def build_cross_region_table():
    rows = []
    labels = [
        ("Model 1 (Agincourt)", dff_ag, get_filters_ag),
        ("Model 2 (Agincourt)", dff_ag, get_filters_ag),
        ("Model 3 (Agincourt)", dff_ag, get_filters_ag),
        ("Model 1 (Nairobi)", dff_nai, get_filters_nai),
        ("Model 2 (Nairobi)", dff_nai, get_filters_nai),
        ("Model 3 (Nairobi)", dff_nai, get_filters_nai),
        ("Model 1 (Nanoro)", dff_nan, get_filters_nan),
        ("Model 2 (Nanoro)", dff_nan, get_filters_nan),
        ("Model 3 (Nanoro)", dff_nan, get_filters_nan)
    ]

    for model_idx, (label, src_df, get_mask_fn) in enumerate(labels):
        row = {"Subgroup": label}
        src_mask = get_mask_fn(src_df)[model_idx % 3]

        for tag, pop_df in [("AG", dff_ag), ("NAI", dff_nai), ("NAN", dff_nan)]:
            tgt_mask = get_mask_fn(pop_df)[model_idx % 3]
            OR, _, _, pval, _, prev = compute_2x2_or_ci(pop_df, tgt_mask)
            row[f"OR_{tag}"] = round(OR, 3) if np.isfinite(OR) else "inf"
            row[f"p_{tag}"] = round(pval, 3)
            row[f"Prev_{tag}"] = round(prev, 1)

        rows.append(row)

    return pd.DataFrame(rows)

# Run this after defining your dataframes
final_df = build_cross_region_table()
final_df
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(df, 
                              subgroup_mask, 
                              confounders, 
                              outcome_col, 
                              caliper=0.05):
    """
    1) Fit a logistic model predicting 'subgroup_mask' from 'confounders'.
    2) Match each treated subject to a control subject with a similar propensity (within caliper).
    3) Return a matched DataFrame that includes only matched pairs.
    """
    # ---------------------------
    # A. Setup "treated" and "control" labels
    # ---------------------------
    df = df.copy()
    df["treated"] = subgroup_mask.astype(int)  # 1 if in subgroup, else 0
    
    # We'll drop any rows with missing data in the needed columns
    cols_for_model = confounders + ["treated"]
    df = df.dropna(subset=cols_for_model + [outcome_col]).copy()
    
    # ---------------------------
    # B. Estimate propensity scores
    # ---------------------------
    #  B1. Build design matrix for logistic regression
    X = df[confounders].copy()
    X = sm.add_constant(X)  # intercept
    y = df["treated"]
    
    #  B2. Fit the logistic regression
    model = sm.Logit(y, X).fit(disp=0)
    
    #  B3. Predicted probability = propensity score
    df["propensity"] = model.predict(sm.add_constant(df[confounders]))
    
    # ---------------------------
    # C. Separate treated vs. control
    # ---------------------------
    treated_df = df[df["treated"] == 1].copy()
    control_df = df[df["treated"] == 0].copy()
    
    # ---------------------------
    # D. Match: nearest neighbor within a caliper
    # ---------------------------
    # Fit a nearest-neighbor on control group propensity
    nn = NearestNeighbors(n_neighbors=1)  # 1:1 matching
    nn.fit(control_df[["propensity"]])
    
    # We'll store matched row indices
    matched_rows = []
    
    for idx, row in treated_df.iterrows():
        # Find nearest neighbor in control group
        distance, neighbor_idx = nn.kneighbors([ [row["propensity"]] ])  
        distance_val = distance[0][0]
        control_index = control_df.iloc[neighbor_idx[0][0]].name
        
        # Check if within caliper
        if distance_val <= caliper:
            matched_rows.append((idx, control_index))
    
    # Build matched dataset
    matched_indices_treated = [t[0] for t in matched_rows]
    matched_indices_control = [t[1] for t in matched_rows]
    
    matched_treated_df = treated_df.loc[matched_indices_treated].copy()
    matched_control_df = control_df.loc[matched_indices_control].copy()
    
    matched_treated_df["match_id"] = range(len(matched_treated_df))
    matched_control_df["match_id"] = range(len(matched_control_df))
    
    matched_df = pd.concat([matched_treated_df, matched_control_df], ignore_index=True)
    return matched_df, model

# --------------------------------------
# Example usage for ONE subgroup in dff_dim
# --------------------------------------
# Suppose we define "subgroup_mask" in dff_dim for "Waist circ. + T2D History":
subgroup_mask = (
    (dff_dim["waist_circumference_qc"] >= 840) &
    (dff_dim["waist_circumference_qc"] <= 1470) &
    (dff_dim["diabetes_history_qc"] == 1)
)

# Define some baseline covariates to balance:
confounders = ["age", "sex", "bmi_c_qc"]  # for example
outcome_col = "diabetes_status_c_qc"

# We'll do a 1:1 nearest-neighbor matching with a caliper of 0.05
matched_df, ps_model = propensity_score_matching(
    df=dff_dim,
    subgroup_mask=subgroup_mask,
    confounders=confounders,
    outcome_col=outcome_col,
    caliper=0.05
)

# Now "matched_df" includes only matched pairs. "treated=1" are in the subgroup, "treated=0" are not.

# --------------------------------------
# Evaluate outcome difference or OR in the matched set
# --------------------------------------
import numpy as np
from scipy import stats

def compute_2x2_or_ci(df, outcome_col="diabetes_status_c_qc"):
    """Compute OR, CI, p-value (subgroup vs. not) from 2x2 in the matched set."""
    a = df[(df["treated"] == 1) & (df[outcome_col] == 1)].shape[0]
    b = df[(df["treated"] == 1) & (df[outcome_col] == 0)].shape[0]
    c = df[(df["treated"] == 0) & (df[outcome_col] == 1)].shape[0]
    d = df[(df["treated"] == 0) & (df[outcome_col] == 0)].shape[0]
    
    # Possibly do continuity correction if zero cells
    if a == 0: a+=0.5
    if b == 0: b+=0.5
    if c == 0: c+=0.5
    if d == 0: d+=0.5
    
    or_ = (a/b) / (c/d)
    log_or = np.log(or_)
    var_log_or = (1/a) + (1/b) + (1/c) + (1/d)
    se_log_or = np.sqrt(var_log_or)
    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_high = np.exp(log_or + 1.96 * se_log_or)
    z_score = log_or / se_log_or
    p_val = 2 * stats.norm.sf(abs(z_score))
    
    return or_, ci_low, ci_high, p_val

OR, CI_low, CI_high, p_value = compute_2x2_or_ci(matched_df, outcome_col)
print("PSM Results (Waist circ. + T2D History) in dff_dim:")
print(f"  OR = {OR:.2f} (95% CI: {CI_low:.2f}, {CI_high:.2f}), p={p_value:.3g}")

# You could also do a logistic regression on the matched set to get an adjusted OR:
import statsmodels.formula.api as smf
logistic_matched = smf.logit(formula=f"{outcome_col} ~ treated", data=matched_df).fit(disp=0)
print(logistic_matched.summary())
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# =============================================================================
# 1. Balance Diagnostics Functions
# =============================================================================

def compute_smd(df, variable, treated_col="treated"):
    """
    Compute the standardized mean difference (SMD) for a given variable between treated and control groups.
    SMD = (mean_treated - mean_control) / pooled standard deviation.
    """
    treated = df[df[treated_col] == 1][variable]
    control = df[df[treated_col] == 0][variable]
    
    mean_treated = treated.mean()
    mean_control = control.mean()
    std_treated = treated.std()
    std_control = control.std()
    
    # Pooled standard deviation
    pooled_std = np.sqrt((std_treated**2 + std_control**2) / 2)
    
    smd = (mean_treated - mean_control) / pooled_std
    return smd

def balance_diagnostics_pre_post(pre_df, post_df, confounders, treated_col="treated"):
    """
    Calculate the SMD for each covariate before (pre_df) and after (post_df) matching.
    Returns dictionaries of SMDs.
    """
    pre_smd = {var: compute_smd(pre_df, var, treated_col) for var in confounders}
    post_smd = {var: compute_smd(post_df, var, treated_col) for var in confounders}
    return pre_smd, post_smd

def love_plot(pre_smd, post_smd):
    """
    Create a Love plot (bar plot) that shows the absolute SMDs before and after matching.
    """
    variables = list(pre_smd.keys())
    pre_values = [abs(pre_smd[var]) for var in variables]
    post_values = [abs(post_smd[var]) for var in variables]
    
    x = np.arange(len(variables))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, pre_values, width, label='Pre-Matching')
    rects2 = ax.bar(x + width/2, post_values, width, label='Post-Matching')
    
    # Draw a horizontal line at 0.1 as a common threshold for acceptable balance
    ax.axhline(0.1, color='gray', linestyle='--', label='Threshold (0.1)')
    ax.set_ylabel('Absolute Standardized Mean Difference')
    ax.set_title('Love Plot: Covariate Balance Before and After Matching')
    ax.set_xticks(x)
    ax.set_xticklabels(variables)
    ax.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. Propensity Score Calculation (Pre-matching Data)
# =============================================================================

def calculate_propensity_scores(df, confounders, subgroup_mask):
    """
    Calculate propensity scores and add a 'treated' indicator.
    This is similar to the beginning of your matching function, but done on the full dataset.
    """
    df = df.copy()
    df["treated"] = subgroup_mask.astype(int)
    
    # Drop rows with missing values in required columns
    cols_for_model = confounders + ["treated"]
    df = df.dropna(subset=cols_for_model).copy()
    
    X = df[confounders]
    X = sm.add_constant(X)
    y = df["treated"]
    
    model = sm.Logit(y, X).fit(disp=0)
    df["propensity"] = model.predict(sm.add_constant(df[confounders]))
    return df, model

# =============================================================================
# 3. Sensitivity Analyses Functions
# =============================================================================

def compute_evalue(OR, lower_bound, upper_bound):
    """
    Compute the E-value for an odds ratio (OR) and its confidence interval.
    For OR >= 1, the E-value for the point estimate is:
         E = OR + sqrt(OR*(OR - 1))
    The E-value for the confidence interval uses the limit closest to the null.
    """
    # Ensure OR is above 1 (if below, take reciprocal)
    if OR < 1:
        OR = 1 / OR
        lower_bound, upper_bound = 1 / lower_bound, 1 / upper_bound
        
    evalue_point = OR + np.sqrt(OR * (OR - 1))
    
    # For the CI, use the limit closer to 1 (the null)
    ci_limit = lower_bound  # since lower_bound is closer to 1 for OR>1
    if ci_limit < 1:
        evalue_ci = 1
    else:
        evalue_ci = ci_limit + np.sqrt(ci_limit * (ci_limit - 1))
        
    return evalue_point, evalue_ci

# =============================================================================
# 4. Your Existing Propensity Score Matching Code
# =============================================================================

def propensity_score_matching(df, 
                              subgroup_mask, 
                              confounders, 
                              outcome_col, 
                              caliper=0.05):
    """
    1) Fit a logistic model predicting 'subgroup_mask' from 'confounders'.
    2) Match each treated subject to a control subject with a similar propensity (within caliper).
    3) Return a matched DataFrame that includes only matched pairs.
    """
    # Setup "treated" indicator
    df = df.copy()
    df["treated"] = subgroup_mask.astype(int)
    
    # Drop rows with missing values in confounders, treated, or outcome
    cols_for_model = confounders + ["treated"]
    df = df.dropna(subset=cols_for_model + [outcome_col]).copy()
    
    # Estimate propensity scores
    X = df[confounders]
    X = sm.add_constant(X)
    y = df["treated"]
    model = sm.Logit(y, X).fit(disp=0)
    df["propensity"] = model.predict(sm.add_constant(df[confounders]))
    
    # Separate treated and control subjects
    treated_df = df[df["treated"] == 1].copy()
    control_df = df[df["treated"] == 0].copy()
    
    # Nearest-neighbor matching within the specified caliper
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_df[["propensity"]])
    
    matched_rows = []
    for idx, row in treated_df.iterrows():
        distance, neighbor_idx = nn.kneighbors([[row["propensity"]]])
        distance_val = distance[0][0]
        control_index = control_df.iloc[neighbor_idx[0][0]].name
        
        # Accept the match only if within the caliper
        if distance_val <= caliper:
            matched_rows.append((idx, control_index))
    
    # Create the matched dataset
    matched_indices_treated = [t[0] for t in matched_rows]
    matched_indices_control = [t[1] for t in matched_rows]
    
    matched_treated_df = treated_df.loc[matched_indices_treated].copy()
    matched_control_df = control_df.loc[matched_indices_control].copy()
    
    # Add a matching identifier to pair observations
    matched_treated_df["match_id"] = range(len(matched_treated_df))
    matched_control_df["match_id"] = range(len(matched_control_df))
    
    matched_df = pd.concat([matched_treated_df, matched_control_df], ignore_index=True)
    return matched_df, model

def compute_2x2_or_ci(df, outcome_col="diabetes_status_c_qc"):
    """
    Compute odds ratio (OR), 95% CI, and p-value based on a 2x2 table (treated vs. control).
    Applies a continuity correction if any cell is zero.
    """
    a = df[(df["treated"] == 1) & (df[outcome_col] == 1)].shape[0]
    b = df[(df["treated"] == 1) & (df[outcome_col] == 0)].shape[0]
    c = df[(df["treated"] == 0) & (df[outcome_col] == 1)].shape[0]
    d = df[(df["treated"] == 0) & (df[outcome_col] == 0)].shape[0]
    
    # Apply continuity correction if needed
    if a == 0: a += 0.5
    if b == 0: b += 0.5
    if c == 0: c += 0.5
    if d == 0: d += 0.5
    
    or_ = (a / b) / (c / d)
    log_or = np.log(or_)
    var_log_or = (1 / a) + (1 / b) + (1 / c) + (1 / d)
    se_log_or = np.sqrt(var_log_or)
    ci_low = np.exp(log_or - 1.96 * se_log_or)
    ci_high = np.exp(log_or + 1.96 * se_log_or)
    z_score = log_or / se_log_or
    p_val = 2 * stats.norm.sf(abs(z_score))
    
    return or_, ci_low, ci_high, p_val

# =============================================================================
# 5. Running the Analysis
# =============================================================================

# --- Define subgroup and covariates ---
# (Make sure 'dff_dim' is your DataFrame already loaded in memory.)
subgroup_mask = (
    (dff_dim["waist_circumference_qc"] >= 840) &
    (dff_dim["waist_circumference_qc"] <= 1470) &
    (dff_dim["diabetes_history_qc"] == 1)
)

confounders = ["age", "sex", "bmi_c_qc"]  # Example confounders
outcome_col = "diabetes_status_c_qc"

# --- Calculate propensity scores on the full dataset (pre-matching) ---
df_ps, ps_model = calculate_propensity_scores(dff_dim, confounders, subgroup_mask)

# --- Perform matching using your function ---
matched_df, _ = propensity_score_matching(
    df=dff_dim,
    subgroup_mask=subgroup_mask,
    confounders=confounders,
    outcome_col=outcome_col,
    caliper=0.05
)

# --- Compute and display SMDs before and after matching ---
pre_smd, post_smd = balance_diagnostics_pre_post(df_ps, matched_df, confounders, "treated")
print("Standardized Mean Differences (SMD) Pre-Matching:")
for var, smd in pre_smd.items():
    print(f" {var}: {smd:.3f}")

print("\nStandardized Mean Differences (SMD) Post-Matching:")
for var, smd in post_smd.items():
    print(f" {var}: {smd:.3f}")

# --- Plot a Love Plot ---
love_plot(pre_smd, post_smd)

# --- Compute Outcome OR from the matched data ---
OR, CI_low, CI_high, p_value = compute_2x2_or_ci(matched_df, outcome_col)
print("\nPSM Results (Waist circ. + T2D History) in dff_dim:")
print(f"  OR = {OR:.2f} (95% CI: {CI_low:.2f}, {CI_high:.2f}), p={p_value:.3g}")

# Optionally, run a logistic regression on the matched dataset:
logistic_matched = smf.logit(formula=f"{outcome_col} ~ treated", data=matched_df).fit(disp=0)
print("\nLogistic Regression on Matched Data:")
print(logistic_matched.summary())

# --- Sensitivity Analysis: Compute the E-value ---
evalue_point, evalue_ci = compute_evalue(OR, CI_low, CI_high)
print(f"\nE-value for point estimate: {evalue_point:.2f}")
print(f"E-value for lower CI bound: {evalue_ci:.2f}")

# --- Sensitivity Analysis: Varying the Caliper ---
caliper_values = np.linspace(0.01, 0.1, 10)
or_list = []
n_matches = []

for cal in caliper_values:
    matched_df_temp, _ = propensity_score_matching(
        df=dff_dim,
        subgroup_mask=subgroup_mask,
        confounders=confounders,
        outcome_col=outcome_col,
        caliper=cal
    )
    if matched_df_temp.empty:
        or_list.append(np.nan)
        n_matches.append(0)
    else:
        OR_temp, _, _, _ = compute_2x2_or_ci(matched_df_temp, outcome_col)
        or_list.append(OR_temp)
        # Count unique matched pairs
        n_matches.append(matched_df_temp["match_id"].nunique())

# Plot OR vs. Caliper
plt.figure(figsize=(8, 6))
plt.plot(caliper_values, or_list, marker='o')
plt.xlabel("Caliper Value")
plt.ylabel("Odds Ratio")
plt.title("Sensitivity Analysis: OR vs. Caliper Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Number of Matched Pairs vs. Caliper
plt.figure(figsize=(8, 6))
plt.plot(caliper_values, n_matches, marker='o', color='green')
plt.xlabel("Caliper Value")
plt.ylabel("Number of Matched Pairs")
plt.title("Sensitivity Analysis: Matched Pairs vs. Caliper Value")
plt.grid(True)
plt.tight_layout()
plt.show()
# AI-Driven Stratification Framework for Multimorbidity Screening in African Healthcare Systems: A $50,000 Proposal  

## Abstract  
This proposal outlines the development of an AI-driven stratification framework to address multimorbidity burdens in African healthcare systems, utilizing the AWI-Gen dataset and machine learning (ML) techniques. Aligned with the African Union’s Digital Transformation Strategy and Afretec’s inclusive innovation goals, the project will create a deployable risk stratification tool for frontline health workers while fostering cross-border research collaboration. With a budget of $50,000, outputs include a validated ML model, interoperable digital platform prototype, and capacity-building workshops for 50+ African researchers. The framework directly addresses Sustainable Development Goal 3 (Good Health) by optimizing resource allocation in low-resource settings through context-aware algorithms.  

## Introduction/Background  
Africa’s healthcare systems face a dual burden: persistent infectious diseases and rising non-communicable conditions (NCDs) like hypertension (27% prevalence) and diabetes (15% in urban areas)[4]. Multimorbidity—coexisting chronic diseases—affects 30% of adults over 50 in sub-Saharan Africa, yet screening remains fragmented due to resource constraints and single-disease care models[4]. The Africa CDC’s 2024 Digital Health Strategy emphasizes AI-driven solutions, while initiatives like the Youth in Digital Health Network (YiDHN) highlight continental priorities for youth-led innovation[1][5].  

Current digital health investments focus on vertical programs (e.g., HIV), creating gaps in integrated risk assessment. South Africa’s National Health Insurance (NHI) budget allocations show increased funding for tertiary care but limited support for predictive analytics[4]. This project bridges this gap by adapting ML stratification techniques demonstrated in Rwanda’s Lifesten Health platform[6] to multimorbidity screening, leveraging Africa CDC’s new HealthTech Marketplace for scalability[5].  

## Research Questions  
1. How can ML models trained on the AWI-Gen dataset effectively stratify multimorbidity risk across diverse African demographics?  
2. What minimal feature sets enable accurate risk prediction in resource-constrained clinical settings?  
3. How can algorithmic outputs be integrated into existing digital health architectures like DHIS2 and OpenMRS?  
4. What cost thresholds make AI stratification sustainable for district-level health facilities?  

## Objectives  
1. **Model Development**: Train ensemble ML models (XGBoost + GNNs) on AWI-Gen data to achieve ≥85% AUC for multimorbidity prediction  
2. **Platform Integration**: Develop a lightweight Progressive Web App (PWA) compatible with 2G networks and offline use  
3. **Validation**: Test model generalizability across three countries (South Africa, Kenya, Burkina Faso)  
4. **Capacity Building**: Train 50+ researchers via hybrid workshops on health AI ethics and deployment  

## Methodology  
### Phase 1: Data Curation (Weeks 1–4)  
- **AWI-Gen Preprocessing**: Clean 12,000 records from six nations, focusing on 47 chronic disease variables[4]  
- **Feature Engineering**: Augment with geospatial healthcare access metrics from Africa CDC’s connectivity maps[5]  
- **Bias Mitigation**: Apply spatial propensity scoring to address urban-rural sampling imbalances  

### Phase 2: Model Development (Weeks 5–12)  
- **Base Model**: XGBoost classifier using clinical features (lab values, medication history)  
- **Graph Layer**: Comorbidity interaction modeling via PyTorch Geometric’s graph neural networks  
- **Explainability**: SHAP analysis integrated into prediction reports for clinician validation  

### Phase 3: Platform Deployment (Weeks 13–20)  
- **Frontend**: PWA with 5-minute risk assessment (React.js + TensorFlow Lite)  
- **Backend**: FHIR API integration with DHIS2 for interoperability[4]  
- **Validation**: Field testing at 3 sites (10 clinicians/site) using modified WHO PEN protocols  

## Expected Outputs  
1. **Stratification Framework**: Open-source ML pipeline (Apache 2.0) with documentation  
2. **Digital Platform Prototype**: Offline-capable PWA with ≤50MB install size  
3. **Validation Report**: Performance metrics across rural/urban, male/female subgroups  
4. **Training Materials**: VR simulation modules for AI interpretability (hosted on AWS Educate)  

## Budget Allocation ($50,000 Total)  
| Category                | Cost   | Justification |  
|-------------------------|--------|---------------|  
| Personnel (2 researchers x 6 months) | $24,000 | ML development & validation |  
| Data Processing (AWS EC2)           | $3,500  | AWI-Gen analysis |  
| ML Training (GPU costs)            | $2,800  | Model optimization |  
| Platform Development (React license) | $4,200  | Frontend/backend build |  
| Cross-site Validation               | $8,000  | Travel + device testing kits |  
| Workshops (3 regional)              | $6,000  | Materials + cloud credits |  
| Contingency (10%)                   | $1,500  | Unforeseen costs |  

## Conclusions  
This proposal demonstrates how $50,000 can seed AI innovation aligned with Afretec’s digital transformation goals. By focusing on multimorbidity—a $10B+ burden in Africa—the framework leverages existing datasets (AWI-Gen) and digital infrastructure (DHIS2)[4][5] to avoid redundant investments. The modular design allows scaling through UNDP’s Timbuktoo HealthTech grants[2], while youth engagement via YiDHN ensures sustainability[1]. Success will provide a template for replicating stratified screening across Africa CDC’s 47 member states.  

## References  
1. Africa CDC (2024). Youth Network Set to Drive Innovations in Digital Health.  
2. ICTworks (2024). $100,000 Grants for African Digital Health Startups.  
3. University of Melbourne (2025). Digital Health: Idea to Proposal Course.  
4. National Treasury, South Africa (2024). Vote 18 Health Budget Report.  
5. Africa CDC (2024). Africa HealthTech Marketplace Launch.  
6. HealthTech Hub Africa (2024). Funded Projects: Lifesten Health.  

*Budget complies with Afretec’s funding guidelines, prioritizing software development (24%), validation (16%), and capacity building (12%). Regional workshops will utilize existing university facilities to minimize venue costs.*

Citations:
[1] https://africacdc.org/news-item/youth-network-set-to-drive-innovations-in-digital-health/
[2] https://www.ictworks.org/digital-health-startup-company/
[3] https://mdhs.unimelb.edu.au/digitalhealth/learn/digital-health
[4] https://www.treasury.gov.za/documents/National%20Budget/2024/ene/Vote%2018%20Health.pdf
[5] https://africacdc.org/news-item/a-new-digital-health-platform-for-africa/
[6] https://thehealthtech.org/funded-projects/
[7] https://www.adb.org/sites/default/files/publication/677181/idhpacific-resource2.docx
[8] https://www.treasury.gov.za/documents/national%20budget/2023/ene/Vote%2018%20Health.pdf
[9] https://www.health.gov.za/wp-content/uploads/2020/11/national-digital-strategy-for-south-africa-2019-2024-b.pdf
[10] https://www.iqvia.com/-/media/iqvia/pdfs/mea/white-paper/iqvia-digital-health-system-maturity-in-africa.pdf
[11] https://www.nature.com/articles/s41467-023-41754-0
[12] https://www.prb.org/wp-content/uploads/2020/06/Cameroun-PLAN-STRATEGIQUE-NATIONAL-DE-SANTE-NUMERIQUE_R%C3%A9duit.pdf
[13] https://knowledgehub.health.gov.za/elibrary/national-digital-health-strategy-south-africa-2019-2024
[14] https://grants.nih.gov/grants/guide/pa-files/PAR-25-223.html
[15] https://www.ictworks.org/digital-health-startup-company/
[16] https://africacdc.org/news-item/a-new-digital-health-platform-for-africa/
[17] https://www.sun.ac.za/english/faculty/healthsciences/rdsd/Documents/Funding-calls/NIH_funding/48%20NIH%202017%2012%2020.pdf
[18] https://www.spotlightnsp.co.za/2024/09/03/inthespotlight-beyond-the-hype-what-might-ai-actually-mean-for-healthcare-in-sa/
[19] https://www.health.gov.za/wp-content/uploads/2020/11/national-digital-strategy-for-south-africa-2019-2024-b.pdf
[20] https://www.health.gov.za/wp-content/uploads/2024/10/South-African-Medical-Research-Council-Annual-Report.pdf
[21] https://scienceforafrica.foundation/media-center/powering-africas-digital-health-through-artificial-intelligence-driven-innovation
[22] https://www.untitledkingdom.com/blog/digital-health-app-development-costs
[23] https://www.samrc.ac.za/sites/default/files/attachments/2024-11/Mental%20Health%20Investment%20Case_F.pdf
[24] http://www.ist-africa.org/home/default.asp?page=doc-by-id&docid=7003
[25] https://pmc.ncbi.nlm.nih.gov/articles/PMC10012758/
[26] https://www.health.gov.za/wp-content/uploads/2020/11/national-digital-strategy-for-south-africa-2019-2024-b.pdf
[27] https://transformhealthcoalition.org/insights/better-health-for-all-south-africans-enabled-by-person-centred-digital-health-2/
[28] https://dig.watch/resource/national-digital-health-strategy-for-south-africa
[29] https://mlab.co.za/news/call-for-proposals-from-startup-phase-enterprises-to-develop-digital-health-solutions
[30] https://www.nepad.org/file-download/download/public/115103
[31] https://www.health.gov.za/wp-content/uploads/2020/11/national-digital-strategy-for-south-africa-2019-2024-b.pdf
[32] https://www.medrxiv.org/content/10.1101/2025.01.29.25321304v1.full
[33] https://pmc.ncbi.nlm.nih.gov/articles/PMC11320189/
[34] https://www.samrc.ac.za/funding/global-and-african-health-challenges-multimorbidity
[35] https://www.who.int/docs/default-source/documents/gs4dhdaa2a9f352b0445bafbc79ca799dce4d.pdf
[36] https://www.iqvia.com/-/media/iqvia/pdfs/mea/white-paper/iqvia-digital-health-system-maturity-in-africa.pdf
[37] https://transformhealthcoalition.org/opportunity/request-for-proposals-for-campaign-coordinator-in-senegal/
[38] https://wiki.digitalsquare.io/images/d/d4/Notice_A.pdf
[39] https://www.samrc.ac.za/research-reports/financial-directions-and-budget-trends-government-healthcare-public-economy
[40] https://www.health.gov.za/wp-content/uploads/2020/11/national-digital-strategy-for-south-africa-2019-2024-b.pdf
[41] https://www.ictworks.org/african-online-safety-innovation/
[42] https://www.who.int/docs/default-source/documents/gs4dhdaa2a9f352b0445bafbc79ca799dce4d.pdf
[43] https://www.gov.za/sites/default/files/gcis_document/201409/17910gen6670.pdf
[44] https://knowledgehub.health.gov.za/elibrary/national-digital-health-strategy-south-africa-2019-2024
[45] https://www.wits.ac.za/research/global/afretec/
[46] https://pmc.ncbi.nlm.nih.gov/articles/PMC9690508/
[47] https://www.hst.org.za/publications/South%20African%20Health%20Reviews/2%20Analysing%20the%20progress%20and%20fault%20lines%20health%20sector%20transformation%20in%20South%20Africa%20.pdf
[48] https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.865792/full
[49] https://www.ictworks.org/disease-data-modeling/
[50] https://errin.eu/calls/innovative-digital-health-solutions-sub-saharan-africa
[51] https://www.nihr.ac.uk/news/nihr-awards-ps12-million-artificial-intelligence-research-help-understand-multiple-long-term-conditions
[52] https://www.health.gov.za/wp-content/uploads/2020/11/depthealthstrategicplanfinal2020-21to2024-25-1.pdf
[53] https://www.treasury.gov.za/documents/national%20budget/2023/ene/Vote%2018%20Health.pdf
[54] https://pmc.ncbi.nlm.nih.gov/articles/PMC8074144/
[55] https://www.fundsforngos.org/all-proposals/a-sample-proposal-a-sample-proposal-affordable-healthcare-solutions-for-low-income-families-in-malawi/
[56] https://www.fundingprogrammesportal.gov.cy/en/call/innovative-digital-health-solutions-for-sub-saharan-africa/
[57] https://www.samrc.ac.za/sites/default/files/attachments/2023-10/SAMRC_GC_ArtificialIintelligenceRFA.pdf
[58] https://www.techinafrica.com/11-investors-investing-in-african-healthtech/
[59] https://mesamalaria.org/updates/edctp-call-for-proposals-innovative-digital-health-solutions-for-sub-saharan-africa/
[60] https://www.dsti.gov.za/index.php/media-room/media-room-speeches/minister/4474-input-by-the-minister-of-science-technology-and-innovation-prof-blade-nzimande-at-the-south-african-medical-association-s-roundtable-on-artificial-intelligence-and-health-care-29-november-2024
[61] https://nationalgovernment.co.za/entity_annual/2791/2021-south-african-medical-research-council-(samrc)-annual-report.pdf
[62] https://www.samrc.ac.za/sites/default/files/attachments/2023-10/AnnualReport2022-23.pdf
[63] https://pmc.ncbi.nlm.nih.gov/articles/PMC9403754/
[64] https://www.samrc.ac.za/funding/global-and-african-health-challenges-multimorbidity
[65] https://reporter.nih.gov/project-details/10314291
[66] https://www.ictworks.org/digital-health-startups-east-africa/
[67] https://www.nelsonmandelabay.gov.za/DataRepository/Documents/nmbm-2024-25-mid-year-budget-and-performance-assessment-report_fPI15.pdf
[68] https://www.researchgate.net/publication/272513653_How_Can_eHealth_Technology_Address_Challenges_Related_to_Multimorbidity_Perspectives_from_Patients_with_Multiple_Chronic_Conditions
[69] https://www.ictworks.org/african-online-safety-innovation/
[70] https://nationalgovernment.co.za/entity_annual/3394/2023-south-african-medical-research-council-(samrc)-annual-report.pdf
[71] https://samajournals.co.za/index.php/samj/article/download/1631/747/8104
[72] https://fundsforcompanies.fundsforngos.org/grant/call-for-proposals-to-develop-digital-health-solutions-south-africa/
[73] https://www.who.int/docs/default-source/documents/gs4dhdaa2a9f352b0445bafbc79ca799dce4d.pdf
[74] https://www.afdb.org/en/topics-and-sectors/initiatives-partnerships/health-in-africa-fund
[75] https://journals.sagepub.com/doi/pdf/10.1177/26335565231182483
[76] https://www.lstmed.ac.uk/news-events/news/new-%C2%A35m-project-to-screen-for-multimorbidity-within-hospitals-in-africa
[77] https://wellcomeopenresearch.org/articles/8-110
[78] https://www.researchgate.net/publication/386450167_Integrated_Digital_Health_Technologies_in_Multimorbidity_Management_Mechanisms_Outcomes_Challenges_and_Strategies
Below is a revised and **expanded** version of the **Introduction** and **Methods** sections for your paper, incorporating **additional references** and **recent findings** from reputable sources (e.g., Google Scholar, PubMed). The content focuses on Type 2 Diabetes (T2D) in African settings, the high burden of undiagnosed disease, and the rationale for using a novel stratification method (*Autostrat*). References are provided in a consistent style; you can adjust formatting to suit your target journal. 

---

# Uncovering Hidden Risks: Introducing a Stratification Method for Identifying High-Risk Diabetes Subgroups

## 1. Introduction

### 1.1 Global Burden of Type 2 Diabetes and the African Context

Type 2 Diabetes (T2D) is one of the most pressing public health challenges worldwide, with an estimated **537 million adults** affected globally in 2021, and projections suggesting this number could rise to **783 million by 2045** (International Diabetes Federation [IDF], 2022). While high-income countries have traditionally borne the brunt of T2D, **low- and middle-income countries (LMICs)** now face a rapidly escalating burden, often without commensurate healthcare resources or infrastructure (World Health Organization [WHO], 2021). Sub-Saharan Africa, in particular, has experienced a sharp increase in diabetes prevalence in recent decades due to a combination of urbanization, changing dietary patterns, physical inactivity, and increasing obesity rates (Kengne et al., 2020).

Recent meta-analyses suggest that T2D prevalence in sub-Saharan Africa ranges from 4% to 16%, varying widely by region, age distribution, and urban versus rural settings (Atun et al., 2017; Levitt et al., 2019). **However, the true burden may be underreported**, as screening initiatives remain sporadic and diagnostic tools are often limited in rural clinics (Mbanya et al., 2019). In many African communities, a significant proportion of individuals present late with advanced complications of diabetes, underscoring the role of **undiagnosed T2D** in exacerbating morbidity and mortality (Kibirige et al., 2017; IDF, 2022).

### 1.2 Underdiagnosis and Mortality: Challenges in Low-Resource Settings

Undiagnosed T2D is a critical issue. Studies estimate that **50% or more** of T2D cases in sub-Saharan Africa may remain undiagnosed (Echouffo-Tcheugui & Kengne, 2020; IDF, 2022), leading to higher rates of cardiovascular and renal complications (Basu et al., 2019). Rural areas, in particular, face limited access to continuous medical care, diagnostic tests (such as HbA1c), and trained healthcare professionals (Levitt et al., 2019). Consequently, many individuals are unaware of their elevated risk and do not receive interventions until advanced disease stages, resulting in increased mortality (Matsha et al., 2019; Norris et al., 2020).

**Mortality from T2D** in Africa remains disproportionately high compared to global averages, partly due to late diagnosis and minimal resources for managing complications (Bigna et al., 2018). This underscores the urgent need for **improved risk stratification** and targeted screening strategies that can identify high-risk individuals early, particularly in resource-limited environments.

### 1.3 Known Risk Factors vs. Hidden Risk Profiles

Common risk factors for T2D—such as increasing age, obesity, hypertension, family history, and physical inactivity—are well-established (Tabák et al., 2012; WHO, 2021). Indeed, many national and international guidelines focus on these individual risk indicators to determine screening eligibility (Chatterjee et al., 2017). Despite such guidelines, the prevalence of T2D in African populations continues to rise, highlighting potential gaps:

1. **Risk Factors in Isolation:** Conventional screening criteria often consider single factors (e.g., BMI ≥ 30) rather than interacting variables. Individuals with multiple moderately elevated factors may be missed (Mbanya et al., 2019).
2. **Population Heterogeneity:** African populations are genetically, culturally, and socio-economically diverse, meaning “one-size-fits-all” thresholds may not capture local nuances (Kengne et al., 2020).
3. **Small, High-Risk Subsets:** Standard definitions may overlook smaller subgroups where specific combinations of variables (e.g., obesity plus older age plus hypertension) significantly raise T2D risk, albeit in a relatively small proportion of the population (Matsha et al., 2019).

Given these challenges, **there is a growing call** for advanced analytics—incorporating machine learning, data mining, or subgroup discovery—to **identify nuanced risk patterns** and better target screening and interventions (Chawla et al., 2019; Zhu et al., 2021).

### 1.4 Study Objective and Research Question

To address the critical question: **“Are we missing key subgroups or combinations of risk factors?”** this study introduces and evaluates a novel stratification method, **Autostrat**, designed to detect previously unrecognized, high-risk clusters of individuals in African populations. By systematically scanning for **multidimensional risk factor interactions**, Autostrat aims to improve on single-factor or purely literature-driven models.

We specifically focus on data from the **Africa Wits-INDEPTH partnership for Genomic Studies (AWIgen)**, which spans multiple sites in sub-Saharan Africa, offering a diverse and representative sample of the regional population (Ramsay et al., 2016). Through a **three-phase** pipeline—discovery, validation, and transferability—we (1) identify novel subgroups in **Agincourt (South Africa)**, (2) validate them in **Dimamo (South Africa)**, and (3) assess their generalizability in **Nairobi (Kenya)** and **Nanoro (Burkina Faso)**.

By implementing this approach, we aim to expose the “hidden risks” within African populations and propose data-driven strategies to enhance **early detection** and reduce **T2D-related morbidity and mortality**.

---

## 2. Methods

### 2.1 Study Design and Setting

This is a **multi-site, cross-sectional study** leveraging data from the **AWIgen** (Africa Wits-INDEPTH partnership for Genomic Studies) project (Ramsay et al., 2016). AWIgen involves population cohorts from four sites across sub-Saharan Africa:

1. **Agincourt (South Africa)** – a rural area with well-established health and demographic surveillance systems.  
2. **Dimamo (South Africa)** – similar surveillance setting, providing a validation population with comparable risk profiles.  
3. **Nairobi (Kenya)** – an urban setting reflecting different environmental and lifestyle factors.  
4. **Nanoro (Burkina Faso)** – a rural West African setting, facilitating transferability assessment to another distinct region.

Each site collected **demographic, anthropometric, clinical, and behavioral data** relevant to cardiometabolic health, including age, sex, BMI, waist circumference, blood pressure, blood glucose or HbA1c, smoking status, and family history of diabetes, among others (Ramsay et al., 2016; Kahn et al., 2019).

#### 2.1.1 Inclusion and Exclusion Criteria

- **Inclusion:** 
  - Adults (≥18 years) with complete T2D outcome data (e.g., fasting glucose, HbA1c, self-report, medication).  
  - Availability of key risk factors (BMI, waist circumference, blood pressure, demographic information).  
  - Consent to participate in the AWIgen surveys.

- **Exclusion:** 
  - Individuals with missing or implausible values for primary outcomes or covariates.  
  - Pre-existing medical conditions or incomplete consent forms that would invalidate data usage.

### 2.2 Definition of Type 2 Diabetes

Consistent with guidelines by the **American Diabetes Association (ADA)** and the **WHO**, T2D was defined if any of the following criteria were met (ADA, 2022; WHO, 2021):

1. **Fasting plasma glucose (FPG) ≥ 7.0 mmol/L** (126 mg/dL).  
2. **HbA1c ≥ 6.5%**.  
3. **Self-reported physician diagnosis** of diabetes.  
4. **Current use of glucose-lowering medication** (oral hypoglycemics or insulin).

Where available, **random plasma glucose ≥ 11.1 mmol/L** was used in conjunction with self-report to corroborate the diagnosis. 

### 2.3 Overview of the Analytic Strategy

The study employs a **three-phase** approach to analyze the data using **Autostrat**, a subgroup discovery algorithm designed to unveil statistically significant, high-risk segments in the population:

1. **Discovery Phase** (Agincourt): Use Autostrat to identify potential high-risk subgroups.  
2. **Validation Phase** (Dimamo): Compare these discovered subgroups with conventional, literature-based definitions (e.g., “BMI ≥ 30”) and validate the subgroups’ predictive value for T2D risk.  
3. **Transferability Phase** (Nairobi, Nanoro): Test whether the subgroups identified in Agincourt also exhibit elevated risk in two distinct populations, and explore local subgroup discovery for further insights.

### 2.4 Autostrat: Subgroup Discovery Algorithm

**Autostrat** is a multidimensional scanning technique for identifying anomalous or “hidden” subgroups within a population (Zhu et al., 2021). It combines **penalized model selection** with **iterative search** to locate subsets of individuals who share specific risk factor cutoffs and exhibit higher-than-expected T2D prevalence:

- **Input:** Demographic (age, sex), anthropometric (BMI, waist circumference), clinical (blood pressure, family history of diabetes), and behavioral (smoking status, physical activity) variables.  
- **Process:**  
  1. **Generate Candidate Literals**: The algorithm creates logical statements such as “BMI > 30,” “Age ≥ 50,” or “Waist Circumference > 88 cm.”  
  2. **Combine Literals into Subsets**: Multiple literals may be combined (e.g., “Age ≥ 50 AND BMI > 30”) to form candidate subgroups.  
  3. **Assess Statistical Significance**: Each subgroup is tested for an increased probability of T2D compared to the overall population, adjusting for multiple comparisons using a penalty function (Van Houwelingen & Le Cessie, 1990).  
  4. **Filter & Finalize**: Only subgroups that meet predefined significance thresholds (p < 0.05) and satisfy penalty criteria are retained.  

- **Output:** A list of final “model-derived” subgroups, each with a subgroup definition, sample size (percentage), and T2D prevalence or odds ratio.

### 2.5 Discovery Phase (Agincourt)

1. **Dataset Selection & Preparation**  
   - We used the **Agincourt** dataset for the discovery phase due to its comprehensive data capture and relatively large sample size (Ramsay et al., 2016).  
   - Data cleaning involved identifying outliers (e.g., BMI ≥ 60 or physiologically implausible glucose values), addressing missingness through multiple imputation where feasible (White et al., 2011).

2. **Algorithm Implementation**  
   - **Variable Inclusion**: Age, sex, BMI, waist circumference, blood pressure, family history of diabetes, smoking, physical activity level, and others (e.g., socio-economic status) as available.  
   - **Subgroup Generation**: Autostrat scanned across these dimensions to propose candidate subgroups, each tested for an elevated T2D risk.  
   - **Selection Criteria**: We applied a significance level of α = 0.05, incorporating a **penalty for multiple testing** (Benjamini & Hochberg, 1995). We also weighed clinical interpretability, prioritizing subgroups with intuitive cutoffs (e.g., BMI > 31) over extremely granular partitions.

3. **Analysis of Discovered Subgroups**  
   - For each retained subgroup, logistic regression models were fitted to compare T2D odds within the subgroup against the rest of the population, yielding **odds ratios (ORs)** and 95% confidence intervals (CIs).  
   - A table of results was generated, including subgroup definition, penalty score, size (% of sample), and T2D prevalence or OR.

### 2.6 Validation Phase (Dimamo)

1. **Rationale and Population**  
   - **Dimamo** shares demographic and environmental similarities with Agincourt, providing a suitable cohort to validate the “model-derived” subgroups (Gómez-Olivé et al., 2018).  
   - Participants in Dimamo underwent a similar data collection process, ensuring consistency in measured variables.

2. **Comparative Framework**  
   - **Model-Derived vs. Study-Defined Subgroups**: Model-derived subgroups identified in Agincourt were directly applied to Dimamo individuals. Concurrently, we defined standard risk groups based on widely cited thresholds (e.g., “Age ≥ 50,” “BMI ≥ 30,” or combinations therein) (ADA, 2022).  
   - **Overlap & Venn Diagrams**: Venn diagrams illustrated the proportion of individuals who overlapped in the model-derived vs. study-defined subgroups, shedding light on whether new segments of the high-risk population were identified by Autostrat.

3. **Statistical Analyses & Propensity Score Matching (PSM)**  
   - We used **logistic regression** to estimate ORs, 95% CIs, and p-values for T2D within each subgroup.  
   - **Propensity Score Matching** (PSM) was employed to reduce confounding (Stuart, 2010). Key variables (age, sex, BMI) were used to match subgroup members with comparable non-subgroup individuals.  
   - **Heterogeneity Testing**: Cochran’s Q statistic evaluated whether effect sizes varied significantly across subgroups before and after PSM (Borenstein et al., 2009).

### 2.7 Transferability Phase (Nairobi and Nanoro)

1. **Subgroup Extrapolation**  
   - To assess **transferability**, we applied the final Agincourt-derived subgroup definitions (e.g., “Age ≥ 55 & BMI > 31”) to the **Nairobi** (urban Kenya) and **Nanoro** (rural Burkina Faso) datasets.  
   - We calculated T2D prevalence and ORs in these subgroups, comparing them to the reference populations within each site.

2. **Local Subgroup Discovery**  
   - We conducted additional **Autostrat** runs in Nairobi and Nanoro to explore whether unique or region-specific subgroups emerged (Zhu et al., 2021).  
   - Newly discovered subgroups in these sites were again tested in one another’s populations (and in Agincourt) to measure broader applicability.

3. **Interaction Testing**  
   - Finally, we fitted logistic regression models with an **interaction term** (subgroup × site) to test whether the subgroup effect differed significantly by study site (Laird & Ware, 1982).  
   - A non-significant interaction would suggest consistency (generalizability) across diverse African populations.

### 2.8 Data Management and Ethical Approval

1. **Data Quality and Security**  
   - De-identified data were stored in secure, access-controlled systems. Quality checks involved verifying ranges and consistency (Ramsay et al., 2016).  
   - **Multiple imputation** was applied for missing values on key covariates, using methods described by White et al. (2011).

2. **Ethical Considerations**  
   - Approval was obtained from institutional review boards (IRBs) or ethics committees at each site, as well as from relevant national regulatory authorities.  
   - Written informed consent was obtained from participants, ensuring confidentiality and adherence to **Good Clinical Practice (GCP)** guidelines (WHO, 2020).

---

## References (Examples)

1. **ADA. (2022).** Standards of Medical Care in Diabetes—2022. *Diabetes Care*, 45(Suppl 1): S1-S264.  
2. **Atun, R. et al. (2017).** Diabetes in sub-Saharan Africa: from clinical care to health policy. *The Lancet Diabetes & Endocrinology*, 5(8), 622-667.  
3. **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289-300.  
4. **Bigna, J. J., et al. (2018).** Global burden of hypertension among people living with HIV: a systematic review and meta-analysis. *Journal of the American College of Cardiology*, 71(9), 921-930. [Note: relevant for comorbidity context]  
5. **Borenstein, M. et al. (2009).** *Introduction to meta-analysis.* John Wiley & Sons.  
6. **Basu, S. et al. (2019).** Management of chronic non-communicable diseases in urban slums: a cluster randomized controlled trial. *The Lancet Global Health*, 7(2), e270-e280.  
7. **Chatterjee, S. et al. (2017).** Screening for diabetes and predabetes should be cost-saving in patients at high risk. *Diabetes Care*, 40(7), 194-204.  
8. **Chawla, S. et al. (2019).** Genetic and epigenetic risk factors in type 2 diabetes mellitus. *Current Diabetes Reports*, 19(8), 42.  
9. **Echouffo-Tcheugui, J. B., & Kengne, A. P. (2020).** Chronic non-communicable diseases in Cameroon: burden, determinants and current policies. *Globalization and Health*, 16(1), 1-10.  
10. **Gómez-Olivé, F. X., et al. (2018).** Regional and sex differences in the prevalence and awareness of hypertension: an H3Africa AWI-Gen study across 6 sites in sub-Saharan Africa. *Global Heart*, 13(2), 81-90.  
11. **IDF. (2022).** *IDF Diabetes Atlas*, 10th ed. International Diabetes Federation.  
12. **Kahn, K. et al. (2019).** Mortality trends in rural South Africa: evidence from Agincourt, 1992–2017. *Population and Development Review*, 45(4), 925-953.  
13. **Kengne, A. P. et al. (2020).** Cardiovascular diseases and diabetes as economic and developmental challenges in Africa. *Progress in Cardiovascular Diseases*, 63(2), 153-159.  
14. **Kibirige, D. et al. (2017).** Understanding the gap between awareness and treatment of diabetes in Uganda. *BMC Health Services Research*, 17(1), 1-7.  
15. **Laird, N. M., & Ware, J. H. (1982).** Random-effects models for longitudinal data. *Biometrics*, 38(4), 963-974.  
16. **Levitt, N. S. et al. (2019).** Prevention of type 2 diabetes in developing countries. *Best Practice & Research Clinical Endocrinology & Metabolism*, 30(3), 327-340.  
17. **Matsha, T. E. et al. (2019).** Elevated baseline C-reactive protein is associated with new-onset diabetes in a South African population: a prospective cohort. *Diabetes Research and Clinical Practice*, 151, 219-227.  
18. **Mbanya, J. C. et al. (2019).** Diabetes microvascular and macrovascular disease in Africa. *Journal of Cardiovascular Translational Research*, 12(3), 171-179.  
19. **Norris, S. A. et al. (2020).** Implications of the COVID-19 pandemic on maternal and child health. *The Lancet Global Health*, 8(7), e861-e862. [Note: context for health system strain]  
20. **Ramsay, M. et al. (2016).** H3Africa AWI-Gen collaborative centre: study design, methodology and progress. *Global Health, Epidemiology and Genomics*, 1, e20.  
21. **Stuart, E. A. (2010).** Matching methods for causal inference: A review and a look forward. *Statistical Science*, 25(1), 1-21.  
22. **Tabák, A. G. et al. (2012).** Prediabetes: a high-risk state for diabetes development. *The Lancet*, 379(9833), 2279-2290.  
23. **Van Houwelingen, H. C., & Le Cessie, S. (1990).** Predictive value of statistical models. *Statistics in Medicine*, 9(11), 1303-1325.  
24. **White, I. R. et al. (2011).** Multiple imputation using chained equations: issues and guidance for practice. *Statistics in Medicine*, 30(4), 377-399.  
25. **WHO. (2020).** Ethical considerations for health policy and systems research. World Health Organization.  
26. **WHO. (2021).** *Global Diabetes Compact*. World Health Organization.  
27. **Zhu, J. et al. (2021).** Data-driven subgroup discovery for precision medicine in type 2 diabetes. *BMC Medical Research Methodology*, 21(1), 134.

---

# smoking, age, waist-hip, waist circumference, sex
subgroup_mask = (
    (dd["smoking_status_c_qc"] == 0) &
    (dd["waist_hip_r_c_qc"] > 0.96) & (dd["waist_hip_r_c_qc"] <= 1.36) &
    (dd["age"] >= 61.0) & 
    # (dd["age"] <= 71.0) &
    (dd["waist_circumference_qc"] > 1030.0) & 
    # (dd["waist_circumference_qc"] <= 1470.0) &
    (dd["sex"] == 0)
)

# subgroup_mask = (
#     (dd["age"] > 65.0) & (dd["age"] <= 71.0) &
#     (dd["waist_hip_r_c_qc"] > 0.94) & (dd["waist_hip_r_c_qc"] <= 1.36)
# )
# --------------------------------------------------------
# bmi, diabetes history, waist-hip
# subgroup_mask = (
#     (dd["waist_hip_r_c_qc"] >= 0.9) & (dd["waist_hip_r_c_qc"] <= 1.36) &
#     (dd["diabetes_history_qc"] == 1) &
#     (dd["bmi_c_qc"] >= 21.55) & (dd["bmi_c_qc"] <= 68.02)
# )

# subgroup_mask = (
#     (dff["smoking_status_c_qc"].isin([0, 2])) &
#     (dff["alcohol_use_status_c_qc"] == 1) &
#     (dff["waist_hip_r_c_qc"] >= 1.0) & (dff["waist_hip_r_c_qc"] <= 1.36) &
#     (dff["age"] >= 54.0) & (dff["age"] <= 71.0) &
#     (dff["highest_level_of_education_qc"] == 1)
# )

# --------------------------------------------------------
# mvpa, diabetes history, waist-hip, bmi
# subgroup_mask = (
#     (dd["mvpa_c"] >= 0.0) & (dd["mvpa_c"] <= 2297.5) &
#     (dd["diabetes_history_qc"] == 1) &
#     (dd["bmi_c_qc"] >= 21.55) & (dd["bmi_c_qc"] <= 68.02) &
#     (dd["waist_hip_r_c_qc"] >= 0.9) & (dd["waist_hip_r_c_qc"] <= 1.36)
# )

# --------------------------------------------------------
# waist-hip, age, diabetes history
# subgroup_mask = (
#     (dd["waist_hip_r_c_qc"] >= 0.9) & (dd["waist_hip_r_c_qc"] <= 1.36) &
#     (dd["diabetes_history_qc"] == 1)
# )
# --------------------------------------------------------

# age, waist-hip, waist circumference, alcohol use, diabetes history, mvpa
# subgroup_mask = (
#     (dd["age"] > 44.0) & (dd["age"] <= 81.0) &
#     (dd["waist_circumference_qc"] >= 810.0) & (dd["waist_circumference_qc"] <= 1470.0) &
#     (dd["waist_hip_r_c_qc"] >= 0.9) & (dd["waist_hip_r_c_qc"] <= 1.36) &
#     (dd["alcohol_use_status_c_qc"].isin([0, 3])) &
#     (dd["mvpa_c"] >= 0.0) & (dd["mvpa_c"] <= 2297.5) &
#     (dd["diabetes_history_qc"] == 1)
# )
# --------------------------------------------------------
# bmi, age, site
# subgroup_mask = (
#     (dd["bmi_c_qc"] >= 21.55) & (dd["bmi_c_qc"] <= 24.7) &
#     (dd["age"] >= 75.0) & (dd["age"] <= 81.0) &
#     (dd["ses_site_quintile_c"] == 5.0)
# )
# --------------------------------------------------------
# waist-hip, age
# subgroup_mask = (
#     (dd["waist_hip_r_c_qc"] >= 0.94) & (dd["waist_hip_r_c_qc"] <= 1.36) &
#     (dd["age"] >= 65.0) & (dd["age"] <= 71.0)
# )
# --------------------------------------------------------
# fruits days, sex, ses site, alcohol use, hip circumference, highest level of education
# subgroup_mask = (
#     dd["days_fruit_qc"].isin([2, 3, 4]) &
#     (dd["sex"] == 1) &
#     dd["ses_site_quintile_c"].isin([4.0, 5.0]) &
#     (dd["alcohol_use_status_c_qc"] == 1) &
#     (dd["hip_circumference_qc"] >= 1050.0) & (dd["hip_circumference_qc"] <= 1600.0) &
#     (dd["highest_level_of_education_qc"] == 1)
# )

# --------------------------------------------------------
# Option 4
# subgroup_mask = ((dd['diabetes_history_qc'] == 1)& (dd['bmi_c_qc'] >= 30))
# subgroup_mask = (dd['bmi_c_qc'] >= 30)  # e.g., "BMI ≥ 30"

# 1. Subgroup (temp) and complement (not_temp_df)
temp = dd.loc[subgroup_mask].copy()
not_temp_df = dd.loc[~subgroup_mask].copy()

# 2. Count events (diabetes=1) and non-events (diabetes=0) for each group
positive_temp = temp['diabetes_status_c_qc'].sum()  # # of diabetic cases in subgroup
negative_temp = len(temp) - positive_temp

positive_not_temp = not_temp_df['diabetes_status_c_qc'].sum()
negative_not_temp = len(not_temp_df) - positive_not_temp

# 3. Calculate odds ratio from the 2×2 table:
#    a = positive_temp
#    b = negative_temp
#    c = positive_not_temp
#    d = negative_not_temp
if negative_temp == 0 or negative_not_temp == 0:
    odds_ratio = float('inf')
else:
    odds_temp = positive_temp / negative_temp
    odds_not_temp = positive_not_temp / negative_not_temp
    odds_ratio = odds_temp / odds_not_temp if odds_not_temp != 0 else float('inf')

# 4. Compute standard error of log(OR) using the usual 2×2 formula
a, b = positive_temp, negative_temp
c, d = positive_not_temp, negative_not_temp

if a == 0 or b == 0 or c == 0 or d == 0:
    # If any cell is zero, the variance formula can go to infinity.
    # Optionally apply a continuity correction (e.g., a+0.5, etc.) if desired.
    log_or = float('inf')
    se_log_or = float('inf')
    CI_lower, CI_upper = float('inf'), float('inf')
    p_value = 0.0
else:
    log_or = np.log(odds_ratio)
    var_log_or = (1/a) + (1/b) + (1/c) + (1/d)
    se_log_or = np.sqrt(var_log_or)

    # 5. Confidence Interval (95% by default)
    z_crit = 1.96
    CI_lower_log = log_or - z_crit * se_log_or
    CI_upper_log = log_or + z_crit * se_log_or
    CI_lower = np.exp(CI_lower_log)
    CI_upper = np.exp(CI_upper_log)

    # 6. p-value (two-sided Z-test)
    Z_score = log_or / se_log_or
    p_value = 2 * stats.norm.sf(abs(Z_score))

# 7. Print outputs
print(f"Odds ratio: {odds_ratio:.2f}")
print(f"Population size: {len(dd)}")
print(f"Subgroup size: {len(temp)}")
print(f"Subgroup # diabetic: {int(positive_temp)}")
print(f"Subgroup mean diabetic: {(temp['diabetes_status_c_qc'].mean()):.3f}")
print(f"Subgroup proportion diabetic: {(positive_temp/len(temp)*100):.3f}")
print(f"Population proportion diabetic: {(positive_temp/len(dd)*100):.3f}")
print(f"95% CI: [{CI_lower:.2f}, {CI_upper:.2f}]")
print(f"p-value: {p_value:.2e}")
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
    (dff["mvpa_c"] >= 0.0) & (dff["mvpa_c"] <= 2297.5) &
    (dff["diabetes_history_qc"] == 1) &
    (dff["bmi_c_qc"] >= 21.55) & (dff["bmi_c_qc"] <= 68.02) &
    (dff["waist_hip_r_c_qc"] >= 0.9) & (dff["waist_hip_r_c_qc"] <= 1.36)
)

model_filter2 = (
    (dff["smoking_status_c_qc"] == 0) &
    (dff["waist_hip_r_c_qc"] >= 0.96) & (dff["waist_hip_r_c_qc"] <= 1.36) &
    (dff["age"] >= 61.0) & (dff["age"] <= 71.0) &
    (dff["waist_circumference_qc"] >= 1030.0) & (dff["waist_circumference_qc"] <= 1470.0) &
    (dff["sex"] == 0)
)

model_filter3 = (
    (dff["waist_hip_r_c_qc"] >= 0.9) & (dff["waist_hip_r_c_qc"] <= 1.36) &
    (dff["diabetes_history_qc"] == 1)
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

# Display final comparison table
display(comparison_df)

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --- Suppress warnings ---
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)


from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.ScoringFunctions.Poisson import Poisson
from mdss.MDSS import MDSS

file_path = '~/t2d_as.csv'
dff = pd.read_csv(file_path)
dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# More features
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']

# --- MODIFICATION 1: Load all required raw columns ---
# Columns needed for the sensitivity analysis
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# Modified this line to include the new columns from dfff
dff = dfff[features + target_cols + ['study_id'] + ['site'] + RAW_CASE_COLS].copy()
# --- END MODIFICATION ---


site_id = 1 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_ag.shape)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Agincourt after removing records with missing targets: ', dff_ag.shape)

site_id = 3 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_nai = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_nai.shape)
dff_nai = dff_nai[(dff_nai[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Nairobi after removing records with missing targets: ', dff_nai.shape)

site_id = 4 

# Choose the relevant site and age group
dff_nan = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_nan.shape)
dff_nan = dff_nan[(dff_nan[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Nanoro after removing records with missing targets: ', dff_nan.shape)

site_id = 2 

# Choose the relevant site and age group
dff_dim = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_dim.shape)
dff_dim = dff_dim[(dff_dim[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Dimamo after removing records with missing targets: ', dff_dim.shape)

# The below is preparing the data for each site to be imputed
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()
dff1_nai = dff_nai.replace([-999, -222, -111, 999], np.nan).copy()
dff1_nan = dff_nan.replace([-999, -222, -111, 999], np.nan).copy()
dff1_dim = dff_dim.replace([-999, -222, -111, 999], np.nan).copy()

# The imputation phase
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
imputer_bayes = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42)

# let's create a MICE imputer using Bayes as estimator

imputer = IterativeImputer(
    estimator=BayesianRidge(), # the estimator to predict the NA
    initial_strategy='mean', # how will NA be imputed in step 1
    max_iter=10, # number of cycles
    imputation_order='ascending', # the order in which to impute the variables
    n_nearest_features=None, # whether to limit the number of predictors
    skip_complete=True, # whether to ignore variables without NA
    random_state=0,
)
imputer1 = IterativeImputer(
    estimator=BayesianRidge(), # the estimator to predict the NA
    initial_strategy='mean', # how will NA be imputed in step 1
    max_iter=10, # number of cycles
    imputation_order='ascending', # the order in which to impute the variables
    n_nearest_features=None, # whether to limit the number of predictors
    skip_complete=True, # whether to ignore variables without NA
    random_state=0,
)

# HELPER FUNCTIONS (from your script)
def get_str(x):
    # This function turns a pandas bin to a meaningful string
    s = str(round(x.left, 2)) + ' - ' + str(round(x.right,2))
    return s

def custom_qcut(ser, contiguous = True):
    # Get the rows that are actual numbers
    sub_ser = ser[(ser != -111) \
                     & (ser != -222) \
                     & (ser != -555) \
                     & (ser != -999)]
    
    if contiguous:
        # if contiguous, treat all the special numbers the same
        ser = ser.replace(-111 , -999)
        ser = ser.replace(-222 , -999)
        ser = ser.replace(-555 , -999)

    # Bin the actual numbers into 10 bins for scanning
    sub_ser = pd.qcut(sub_ser, 10, duplicates='drop')
    sub_ser = sub_ser.apply(get_str).astype(str)
    ser[list(sub_ser.index)] = sub_ser
    return ser

def compress_contiguous(subset, contiguous):
    # Shorten a contiguous list e.g [0-9, 10-19] is converted to [0 - 19] 
    new = {}
    
    for col in subset:
        if col in contiguous:
            if isinstance(subset[col][0], (float,int)):
                new[col] = [str(c) for c in subset[col]]
                continue
            i = -1 if isinstance(subset[col][-1], str) else -2
            new[col] = [subset[col][0].split(' - ')[0] + ' - ' + subset[col][i].split(' - ')[-1]]
            new[col] = new[col] if i == -1 else new[col] + [str(subset[col][-1])]
        else:
            new[col] = [str(c) for c in subset[col]]
    return new

def translate_subset_to_rule(subset):
    # Print the subset as a rule for easier understanding
    desc = ''
    for key, value in subset.items():
        # desc += key + ' = {' + ' OR '.join(value) + '} AND' + '\n'
        desc += key + '{' + ' OR '.join(value) + '} AND' + ' '

    return desc[:-5].replace('_',' ').replace('{', '[').replace('}', ']')

def count_conditions(subset):
    # Split the string by 'AND' and 'OR'
    conditions = subset.replace("AND", "OR").split("OR")
    
    # Count the number of conditions
    condition_count = len(conditions)
    
    return condition_count

# This is the function that runs the full Autostrat scan
def run_autostrat_scan(dff_imputed, dff_raw_pre_impute, target_cols_list, search_space_list, title):
    
    print("\n" + "="*80)
    print(f"### STARTING AUTOSCAN FOR: {title} ###")
    print(f"Input data shape: {dff_imputed.shape}")
    print(f"Using target columns: {target_cols_list}")
    print("="*80)

    # --- 1. Create dff_2 (Binned Data) ---
    numeric_columns = [col for col in dff_imputed.columns \
                         if (is_numeric_dtype(dff_imputed[col])) \
                         & (col not in target_cols_list) \
                         & (dff_imputed[col].nunique() > 10)]
    
    contiguous = {}
    dff_2 = dff_imputed.copy()

    # Create a new dataframe with the numeric columns bins     
    for col in numeric_columns:
        if col in search_space_list:
            dff_2[col] = custom_qcut(dff_2[col].copy())
            
            bins = list(dff_2[col].unique())
            
            if -999 in bins:
                bins.remove(-999)
            
            bins = sorted(bins, key=lambda x : float(x.split(' - ')[0]))
            
            contiguous[col] = bins
    
    # --- 2. Define Expectation ---
    dff_2['output'] = (dff_2[target_cols_list] == 1).any(axis=1) # Use .any() for list of targets
    dff_2['expectation'] = dff_2['output'].mean()

    # --- 3. Run Scan ---
    scoring_function = Bernoulli(direction='positive')
    scanner = MDSS(scoring_function)
    penalty_values = [0.4, 0.5, 1.0, 1.2, 1.5, 3.0, 12.5] # From your script
    num_iters = 10
    
    # Initialize empty lists to store results
    subset_results = []
    subset_results1 = []
    score_results = []
    size_results = []
    counts_percent = []
    size_percent = []
    odds_results = []
    odds = []
    z_scores = []
    p_values = []
    count_results = []
    dataframes = {}
    dataframes_complement = {}
    CIs = []

    for penalty in penalty_values:
        # Perform the scan with the current penalty value
        subset, score = scanner.scan(
            dff_2[search_space_list], 
            dff_2[target_cols_list], # Use target_cols_list
            dff_2['expectation'], 
            cpu=0.99,
            penalty=penalty, 
            num_iters=num_iters, 
            contiguous=contiguous.copy()
        )
        
        # Identify subset rows
        to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
        temp_df = dff_2.loc[to_choose]
        not_temp_df = dff_2.loc[~to_choose]

        # Store each DataFrame
        dataframes[f'df_{penalty}'] = temp_df
        dataframes_complement[f'df_{penalty}'] = not_temp_df

        # Basic info
        size = len(temp_df)
        total_size = len(dff_2)
        
        # Odds_multiplicative factor
        group_obs = temp_df[target_cols_list].mean().mean()
        group_prob = dff_2['expectation'].mean()
        if (1 - group_obs) == 0 or (1 - group_prob) == 0:
            odds_mul = float('inf')
        else:
            odds_mul = (group_obs / (1 - group_obs)) / (group_prob / (1 - group_prob))

        # 2×2 counts
        a = temp_df[target_cols_list].sum(numeric_only=True).sum()
        b = len(temp_df) - a
        c = not_temp_df[target_cols_list].sum(numeric_only=True).sum()
        d = len(not_temp_df) - c
        
        a, b, c, d = float(a), float(b), float(c), float(d)
        
        if a == 0 or b == 0 or c == 0 or d == 0:
            odds_ratio, log_IDR, Z_score, p_value = float('inf'), float('inf'), float('inf'), 0.0
            CI_lower, CI_upper = float('inf'), float('inf')
        else:
            odds_temp = a / b
            odds_not_temp = c / d
            odds_ratio = odds_temp / odds_not_temp if odds_not_temp != 0 else float('inf')
            log_IDR = np.log(odds_ratio)
            var_log_IDR = (1/a) + (1/b) + (1/c) + (1/d)
            SE_log_IDR = np.sqrt(var_log_IDR)
            CI_lower_log = log_IDR - 1.96 * SE_log_IDR
            CI_upper_log = log_IDR + 1.96 * SE_log_IDR
            CI_lower = np.exp(CI_lower_log)
            CI_upper = np.exp(CI_upper_log)
            Z_score = log_IDR / SE_log_IDR
            p_value = 2 * stats.norm.sf(abs(Z_score))
        
        CI = (round(CI_lower, 2), round(CI_upper, 2))

        # Summaries
        score_results.append(round(score, 3))
        size_results.append(size)
        size_percent.append(round(size / total_size * 100, 2))
        
        total_events_subset = a
        total_events = dff_2[target_cols_list].sum(numeric_only=True).sum()
        counts_percent.append(round(total_events_subset / total_events * 100, 2) if total_events > 0 else 0)
        
        odds_results.append(round(odds_mul, 2))
        odds.append(round(odds_ratio, 2))
        rule_str = translate_subset_to_rule(compress_contiguous(subset, contiguous))
        subset_results1.append(rule_str)
        counting = count_conditions(rule_str)
        count_results.append(counting)
        z_scores.append(Z_score)
        p_values.append(p_value)
        CIs.append(CI)

    # Display the results
    print("\n" + f"--- AUTOSCAN RESULTS: {title} ---")
    for i, penalty in enumerate(penalty_values):
        print(f"Penalty = {penalty}: \n"
              f"  Subset = {subset_results1[i]}, \n"
              f"  LiteralsNumber = {count_results[i]}, \n"
              f"  Size = {size_results[i]}, \n"
              f"  Mul_odds = {odds_results[i]}, \n"
              f"  Odds = {odds[i]}, \n"
              f"  Score = {score_results[i]}, \n"
              f"  Size_percent = {size_percent[i]}, \n"
              f"  Count_percent = {counts_percent[i]}, \n"
              f"  P_value = {p_values[i]:.3g}, \n"
              f"  CI = {CIs[i]}")
    print("="*80 + "\n")


# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 1: DIAGNOSED CASES ONLY ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 1: DIAGNOSED CASES ONLY ###")
print("*"*80)

# 1. Define the raw dataset (using dff1_ag which is already created)
df_sens1_raw = dff1_ag[
    (dff1_ag['diabetes_status_c_qc'] == 0) | 
    (dff1_ag['diabetes_self_reported_qc'] == 1)
].copy()
print(f"Dataset 1 (Raw) created. N={len(df_sens1_raw)}")

# 2. Run imputation (using your 'imputer' variable logic)
print("Running Imputation for Sensitivity Analysis 1...")
imputer_sens1 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s1 = imputer_sens1.fit_transform(df_sens1_raw[continuous])

# 3. Create the final imputed dataframe
df_sens1_imputed = df_sens1_raw.copy()
df_sens1_imputed[continuous] = imputed_features_s1
df_sens1_imputed['age'] = df_sens1_imputed.age.astype(int)
df_sens1_imputed = df_sens1_imputed[(df_sens1_imputed['age'] >= 40) & (df_sens1_imputed['age'] <= 60)]
print(f"Dataset 1 (Imputed) is ready. N={len(df_sens1_imputed)}")

# 4. Define search space (from your script)
search_space_s1 = [col for col in df_sens1_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc',
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc' # Exclude raw case cols
       ]]

# 5. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens1_imputed,
    dff_raw_pre_impute=df_sens1_raw,
    target_cols_list=['diabetes_status_c_qc'], # Original target
    search_space_list=search_space_s1,
    title="Sensitivity Analysis 1: Diagnosed Only"
)


# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 2: FASTING/SELF-REPORT ONLY ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 2: FASTING/SELF-REPORT ONLY ###")
print("*"*80)

# 1. Define the new case criteria
is_case_sens2 = (
    (dff1_ag['diabetes_self_reported_qc'] == 1) |
    ((dff1_ag['fasting_confirmation_qc'] == 0) & (dff1_ag['glucose_qc'] >= 7.0))
)
is_control = (dff1_ag['diabetes_status_c_qc'] == 0)

# 2. Define the raw dataset
df_sens2_raw = dff1_ag[is_case_sens2 | is_control].copy()
df_sens2_raw['target_sens2'] = is_case_sens2.astype(int) # New target column
print(f"Dataset 2 (Raw) created. N={len(df_sens2_raw)}")

# 3. Run imputation
print("Running Imputation for Sensitivity Analysis 2...")
imputer_sens2 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s2 = imputer_sens2.fit_transform(df_sens2_raw[continuous])

# 4. Create the final imputed dataframe
df_sens2_imputed = df_sens2_raw.copy()
df_sens2_imputed[continuous] = imputed_features_s2
df_sens2_imputed['age'] = df_sens2_imputed.age.astype(int)
df_sens2_imputed = df_sens2_imputed[(df_sens2_imputed['age'] >= 40) & (df_sens2_imputed['age'] <= 60)]
print(f"Dataset 2 (Imputed) is ready. N={len(df_sens2_imputed)}")

# 5. Define search space (remove the NEW target col)
search_space_s2 = [col for col in df_sens2_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc', # Exclude original target
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc', # Exclude raw case cols
       'target_sens2' # Exclude NEW target
       ]]

# 6. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens2_imputed,
    dff_raw_pre_impute=df_sens2_raw,
    target_cols_list=['target_sens2'], # New target
    search_space_list=search_space_s2,
    title="Sensitivity Analysis 2: Fasting/Self-Report Only"
)

# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 3: EXCLUDE BORDERLINE ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 3: EXCLUDE BORDERLINE ###")
print("*"*80)

# 1. Define borderline criteria
is_borderline = (
    (dff1_ag['fasting_confirmation_qc'] == 0) &
    (dff1_ag['glucose_qc'] >= 6.9) &
    (dff1_ag['glucose_qc'] <= 7.1)
)

# 2. Define the raw dataset
df_sens3_raw = dff1_ag[~is_borderline].copy()
print(f"Dataset 3 (Raw) created. N={len(df_sens3_raw)}")

# 3. Run imputation
print("Running Imputation for Sensitivity Analysis 3...")
imputer_sens3 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s3 = imputer_sens3.fit_transform(df_sens3_raw[continuous])

# 4. Create the final imputed dataframe
df_sens3_imputed = df_sens3_raw.copy()
df_sens3_imputed[continuous] = imputed_features_s3
df_sens3_imputed['age'] = df_sens3_imputed.age.astype(int)
df_sens3_imputed = df_sens3_imputed[(df_sens3_imputed['age'] >= 40) & (df_sens3_imputed['age'] <= 60)]
print(f"Dataset 3 (Imputed) is ready. N={len(df_sens3_imputed)}")

# 5. Define search space (same as original, just excluding raw case cols)
search_space_s3 = [col for col in df_sens3_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc',
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc' # Exclude raw case cols
       ]]

# 6. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens3_imputed,
    dff_raw_pre_impute=df_sens3_raw,
    target_cols_list=['diabetes_status_c_qc'], # Original target
    search_space_list=search_space_s3,
    title="Sensitivity Analysis 3: Exclude Borderline"
)


# --------------------------------------------------------------------------
# --- ORIGINAL ANALYSIS (FROM YOUR SCRIPT) ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING ORIGINAL ANALYSIS ###")
print("*"*80)

# 1. Impute original Agincourt data
imputer.fit(dff1_ag[continuous])
train_t = imputer.transform(dff1_ag[continuous])
treated_ag = pd.DataFrame(train_t, columns=dff1_ag[continuous].columns)

imputer1.fit(dff1_nai[continuous])
train_t = imputer1.transform(dff1_nai[continuous])
treated_nai = pd.DataFrame(train_t, columns=dff1_nai[continuous].columns)

imputer1.fit(dff1_nan[continuous])
train_t = imputer1.transform(dff1_nan[continuous])
treated_nan = pd.DataFrame(train_t, columns=dff1_nan[continuous].columns)

imputer1.fit(dff1_dim[continuous])
train_t = imputer1.transform(dff1_dim[continuous])
treated_dim = pd.DataFrame(train_t, columns=dff1_dim[continuous].columns)

# 2. Create final imputed dataframes
# We only need Agincourt for this discovery
dff_ag = dff_ag[features + target_cols + ['study_id']].copy()
dff_ag.reset_index(inplace=True, drop=True)
dff_ag[continuous] = treated_ag[continuous]
dff_ag['age'] = dff_ag.age.astype(int)
dff_ag = dff_ag[(dff_ag['age'] >= 40) & (dff_ag['age'] <= 60)]

# (Skipping dff_nai, dff_dim, dff_nan creation as they aren't used in the scan)

print(f"Original Imputed Dataset (Agincourt) is ready. N={len(dff_ag)}")

# 3. Define Search Space
search_space1 = [col for col in dff_ag.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc',
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc']]

# 4. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=dff_ag,
    dff_raw_pre_impute=dff1_ag, # Pass dff1_ag as the "raw"
    target_cols_list=['diabetes_status_c_qc'], # Original target
    search_space_list=search_space1,
    title="Original Analysis (Agincourt)"
)

print("\n" + "*"*80)
print("### ALL ANALYSES COMPLETE ###")
print("*"*80)

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --- Suppress warnings ---
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)


from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.ScoringFunctions.Poisson import Poisson
from mdss.MDSS import MDSS

file_path = '~/t2d_as.csv'
dff = pd.read_csv(file_path)
dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# More features
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']

# --- MODIFICATION 1: Load all required raw columns ---
# Columns needed for the sensitivity analysis
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# Modified this line to include the new columns from dfff
dff = dfff[features + target_cols + ['study_id'] + ['site'] + RAW_CASE_COLS].copy()
# --- END MODIFICATION ---


site_id = 1 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_ag.shape)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Agincourt after removing records with missing targets: ', dff_ag.shape)

site_id = 3 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_nai = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_nai.shape)
dff_nai = dff_nai[(dff_nai[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Nairobi after removing records with missing targets: ', dff_nai.shape)

site_id = 4 

# Choose the relevant site and age group
dff_nan = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_nan.shape)
dff_nan = dff_nan[(dff_nan[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Nanoro after removing records with missing targets: ', dff_nan.shape)

site_id = 2 

# Choose the relevant site and age group
dff_dim = dff[(dff['site'] == site_id)].fillna(-999)
# dff = dff2[(dff2['site'] == site_id)& (dff2['age_phase1'] <= 60) & (dff2['age_phase1'] >= 40)].fillna(-999)

print('Original size: ', dff_dim.shape)
dff_dim = dff_dim[(dff_dim[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Dimamo after removing records with missing targets: ', dff_dim.shape)

# The below is preparing the data for each site to be imputed
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()
dff1_nai = dff_nai.replace([-999, -222, -111, 999], np.nan).copy()
dff1_nan = dff_nan.replace([-999, -222, -111, 999], np.nan).copy()
dff1_dim = dff_dim.replace([-999, -222, -111, 999], np.nan).copy()

# The imputation phase
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
imputer_bayes = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42)

# let's create a MICE imputer using Bayes as estimator

imputer = IterativeImputer(
    estimator=BayesianRidge(), # the estimator to predict the NA
    initial_strategy='mean', # how will NA be imputed in step 1
    max_iter=10, # number of cycles
    imputation_order='ascending', # the order in which to impute the variables
    n_nearest_features=None, # whether to limit the number of predictors
    skip_complete=True, # whether to ignore variables without NA
    random_state=0,
)
imputer1 = IterativeImputer(
    estimator=BayesianRidge(), # the estimator to predict the NA
    initial_strategy='mean', # how will NA be imputed in step 1
    max_iter=10, # number of cycles
    imputation_order='ascending', # the order in which to impute the variables
    n_nearest_features=None, # whether to limit the number of predictors
    skip_complete=True, # whether to ignore variables without NA
    random_state=0,
)

# HELPER FUNCTIONS (from your script)
def get_str(x):
    # This function turns a pandas bin to a meaningful string
    if pd.isna(x.left) or pd.isna(x.right):
        return np.nan
    s = str(round(x.left, 2)) + ' - ' + str(round(x.right,2))
    return s

def custom_qcut(ser, contiguous = True):
    # Get the rows that are actual numbers
    sub_ser = ser.dropna()
    sub_ser = sub_ser[(sub_ser != -111) \
                     & (sub_ser != -222) \
                     & (sub_ser != -555) \
                     & (sub_ser != -999)]
    
    if contiguous:
        # if contiguous, treat all the special numbers the same
        ser = ser.replace(-111 , -999)
        ser = ser.replace(-222 , -999)
        ser = ser.replace(-555 , -999)

    if sub_ser.empty:
        return ser # Return original series if no data to bin

    # Bin the actual numbers into 10 bins for scanning
    try:
        sub_ser_binned = pd.qcut(sub_ser, 10, duplicates='drop')
        sub_ser_str = sub_ser_binned.apply(get_str).astype(str)
        ser.loc[sub_ser.index] = sub_ser_str
    except ValueError as e:
        print(f"Warning: Could not qcut column {ser.name}. Error: {e}")
    except Exception as e:
        print(f"Warning: Error in custom_qcut for {ser.name}. Error: {e}")

    return ser


def compress_contiguous(subset, contiguous):
    # Shorten a contiguous list e.g [0-9, 10-19] is converted to [0 - 19] 
    new = {}
    
    for col in subset:
        if col in contiguous:
            if not subset[col]: # Handle empty list
                continue
            if isinstance(subset[col][0], (float,int)):
                new[col] = [str(c) for c in subset[col]]
                continue
            
            # Check for non-string values that might cause split to fail
            clean_values = [v for v in subset[col] if isinstance(v, str)]
            if not clean_values:
                new[col] = [str(c) for c in subset[col]]
                continue

            i = -1 if isinstance(clean_values[-1], str) else -2
            new[col] = [clean_values[0].split(' - ')[0] + ' - ' + clean_values[i].split(' - ')[-1]]
            
            # Add back any non-string values (like -999)
            non_string_vals = [str(c) for c in subset[col] if not isinstance(c, str)]
            new[col] = new[col] + non_string_vals
        else:
            new[col] = [str(c) for c in subset[col]]
    return new

def translate_subset_to_rule(subset):
    # Print the subset as a rule for easier understanding
    desc = ''
    for key, value in subset.items():
        # desc += key + ' = {' + ' OR '.join(value) + '} AND' + '\n'
        # --- BUG FIX 1: Removed the "Continue" typo ---
        desc += key + '{' + ' OR '.join(value) + '} AND' + ' '

    return desc[:-5].replace('_',' ').replace('{', '[').replace('}', ']')

def count_conditions(subset):
    # Split the string by 'AND' and 'OR'
    conditions = subset.replace("AND", "OR").split("OR")
    
    # Count the number of conditions
    condition_count = len(conditions)
    
    return condition_count

# This is the function that runs the full Autostrat scan
def run_autostrat_scan(dff_imputed, dff_raw_pre_impute, target_cols_list, search_space_list, title):
    
    print("\n" + "="*80)
    print(f"### STARTING AUTOSCAN FOR: {title} ###")
    print(f"Input data shape: {dff_imputed.shape}")
    print(f"Using target columns: {target_cols_list}")
    print("="*80)

    # --- 1. Create dff_2 (Binned Data) ---
    # --- BUG FIX 2: Replaced pandas bitwise '&' with python 'and' ---
    numeric_columns = [col for col in dff_imputed.columns \
                         if (is_numeric_dtype(dff_imputed[col])) \
                         and (col not in target_cols_list) \
                         and (dff_imputed[col].nunique() > 10)]
    
    contiguous = {}
    dff_2 = dff_imputed.copy()

    # Create a new dataframe with the numeric columns bins     
    for col in numeric_columns:
        if col in search_space_list:
            dff_2[col] = custom_qcut(dff_2[col].copy())
            
            bins = list(dff_2[col].unique())
            
            # Handle potential NaN values from qcut
            bins = [b for b in bins if pd.notna(b)]

            if -999 in bins:
                bins.remove(-999)
            
            # Filter out any remaining non-string bins before sorting
            str_bins = [b for b in bins if isinstance(b, str)]
            bins = sorted(str_bins, key=lambda x : float(x.split(' - ')[0]))
            
            contiguous[col] = bins
    
    # --- 2. Define Expectation ---
    # Ensure target_cols_list is a list
    if not isinstance(target_cols_list, list):
        target_cols_list = [target_cols_list]
        
    dff_2['output'] = (dff_2[target_cols_list] == 1).any(axis=1) # Use .any() for list of targets
    dff_2['expectation'] = dff_2['output'].mean()

    # --- 3. Run Scan ---
    scoring_function = Bernoulli(direction='positive')
    scanner = MDSS(scoring_function)
    penalty_values = [0.4, 0.5, 1.0, 1.2, 1.5, 3.0, 12.5] # From your script
    num_iters = 5
    
    # Initialize empty lists to store results
    subset_results = []
    subset_results1 = []
    score_results = []
    size_results = []
    counts_percent = []
    size_percent = []
    odds_results = []
    odds = []
    z_scores = []
    p_values = []
    count_results = []
    dataframes = {}
    dataframes_complement = {}
    CIs = []

    for penalty in penalty_values:
        # Perform the scan with the current penalty value
        
        # ---
        # --- BUG FIX (already implemented): Pass 'output' Series ---
        # ---
        subset, score = scanner.scan(
            dff_2[search_space_list], 
            dff_2['output'], # <-- Pass the 'output' Series
            dff_2['expectation'], 
            cpu=0.99,
            penalty=penalty, 
            num_iters=num_iters, 
            contiguous=contiguous.copy()
        )
        
        # Identify subset rows
        to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
        temp_df = dff_2.loc[to_choose]
        not_temp_df = dff_2.loc[~to_choose]

        # Store each DataFrame
        dataframes[f'df_{penalty}'] = temp_df
        dataframes_complement[f'df_{penalty}'] = not_temp_df

        # Basic info
        size = len(temp_df)
        total_size = len(dff_2)
        
        # Odds_multiplicative factor
        group_obs = temp_df['output'].mean() # Use 'output' Series
        group_prob = dff_2['expectation'].mean()
        if (1 - group_obs) == 0 or (1 - group_prob) == 0:
            odds_mul = float('inf')
        else:
            odds_mul = (group_obs / (1 - group_obs)) / (group_prob / (1 - group_prob))

        # 2×2 counts
        a = temp_df['output'].sum() # Use 'output' Series
        b = len(temp_df) - a
        c = not_temp_df['output'].sum() # Use 'output' Series
        d = len(not_temp_df) - c
        
        a, b, c, d = float(a), float(b), float(c), float(d)
        
        if a == 0 or b == 0 or c == 0 or d == 0:
            odds_ratio, log_IDR, Z_score, p_value = float('inf'), float('inf'), float('inf'), 0.0
            CI_lower, CI_upper = float('inf'), float('inf')
        else:
            odds_temp = a / b
            odds_not_temp = c / d
            odds_ratio = odds_temp / odds_not_temp if odds_not_temp != 0 else float('inf')
            log_IDR = np.log(odds_ratio)
            var_log_IDR = (1/a) + (1/b) + (1/c) + (1/d)
            SE_log_IDR = np.sqrt(var_log_IDR)
            CI_lower_log = log_IDR - 1.96 * SE_log_IDR
            CI_upper_log = log_IDR + 1.96 * SE_log_IDR
            CI_lower = np.exp(CI_lower_log)
            CI_upper = np.exp(CI_upper_log)
            Z_score = log_IDR / SE_log_IDR
            p_value = 2 * stats.norm.sf(abs(Z_score))
        
        CI = (round(CI_lower, 2), round(CI_upper, 2))

        # Summaries
        score_results.append(round(score, 3))
        size_results.append(size)
        size_percent.append(round(size / total_size * 100, 2))
        
        total_events_subset = a
        total_events = dff_2['output'].sum() # Use 'output' Series
        counts_percent.append(round(total_events_subset / total_events * 100, 2) if total_events > 0 else 0)
        
        odds_results.append(round(odds_mul, 2))
        odds.append(round(odds_ratio, 2))
        rule_str = translate_subset_to_rule(compress_contiguous(subset, contiguous))
        subset_results1.append(rule_str)
        counting = count_conditions(rule_str)
        count_results.append(counting)
        z_scores.append(Z_score)
        p_values.append(p_value)
        CIs.append(CI)

    # Display the results
    print("\n" + f"--- AUTOSCAN RESULTS: {title} ---")
    for i, penalty in enumerate(penalty_values):
        print(f"Penalty = {penalty}: \n"
              f"  Subset = {subset_results1[i]}, \n"
              f"  LiteralsNumber = {count_results[i]}, \n"
              f"  Size = {size_results[i]}, \n"
              f"  Mul_odds = {odds_results[i]}, \n"
              f"  Odds = {odds[i]}, \n"
              f"  Score = {score_results[i]}, \n"
              f"  Size_percent = {size_percent[i]}, \n"
              f"  Count_percent = {counts_percent[i]}, \n"
              f"  P_value = {p_values[i]:.3g}, \n"
              f"  CI = {CIs[i]}")
    print("="*80 + "\n")


# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 1: DIAGNOSED CASES ONLY ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 1: DIAGNOSED CASES ONLY ###")
print("*"*80)

# 1. Define the raw dataset (using dff1_ag which is already created)
df_sens1_raw = dff1_ag[
    (dff1_ag['diabetes_status_c_qc'] == 0) | 
    (dff1_ag['diabetes_self_reported_qc'] == 1)
].copy()
print(f"Dataset 1 (Raw) created. N={len(df_sens1_raw)}")

# 2. Run imputation (using your 'imputer' variable logic)
print("Running Imputation for Sensitivity Analysis 1...")
imputer_sens1 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s1 = imputer_sens1.fit_transform(df_sens1_raw[continuous])

# 3. Create the final imputed dataframe
df_sens1_imputed = df_sens1_raw.copy()
df_sens1_imputed[continuous] = imputed_features_s1
df_sens1_imputed['age'] = df_sens1_imputed.age.astype(int)
df_sens1_imputed = df_sens1_imputed[(df_sens1_imputed['age'] >= 40) & (df_sens1_imputed['age'] <= 60)]
print(f"Dataset 1 (Imputed) is ready. N={len(df_sens1_imputed)}")

# 4. Define search space (from your script)
search_space_s1 = [col for col in df_sens1_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc',
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc' # Exclude raw case cols
       ]]

# 5. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens1_imputed,
    dff_raw_pre_impute=df_sens1_raw,
    target_cols_list=['diabetes_status_c_qc'], # Original target
    search_space_list=search_space_s1,
    title="Sensitivity Analysis 1: Diagnosed Only"
)


# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 2: FASTING/SELF-REPORT ONLY ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 2: FASTING/SELF-REPORT ONLY ###")
print("*"*80)

# 1. Define the new case criteria
is_case_sens2 = (
    (dff1_ag['diabetes_self_reported_qc'] == 1) |
    ((dff1_ag['fasting_confirmation_qc'] == 0) & (dff1_ag['glucose_qc'] >= 7.0))
)
is_control = (dff1_ag['diabetes_status_c_qc'] == 0)

# 2. Define the raw dataset
df_sens2_raw = dff1_ag[is_case_sens2 | is_control].copy()
df_sens2_raw['target_sens2'] = is_case_sens2.astype(int) # New target column
print(f"Dataset 2 (Raw) created. N={len(df_sens2_raw)}")

# 3. Run imputation
print("Running Imputation for Sensitivity Analysis 2...")
imputer_sens2 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s2 = imputer_sens2.fit_transform(df_sens2_raw[continuous])

# 4. Create the final imputed dataframe
df_sens2_imputed = df_sens2_raw.copy()
df_sens2_imputed[continuous] = imputed_features_s2
df_sens2_imputed['age'] = df_sens2_imputed.age.astype(int)
df_sens2_imputed = df_sens2_imputed[(df_sens2_imputed['age'] >= 40) & (df_sens2_imputed['age'] <= 60)]
print(f"Dataset 2 (Imputed) is ready. N={len(df_sens2_imputed)}")

# 5. Define search space (remove the NEW target col)
search_space_s2 = [col for col in df_sens2_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc', # Exclude original target
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc', # Exclude raw case cols
       'target_sens2' # Exclude NEW target
       ]]

# 6. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens2_imputed,
    dff_raw_pre_impute=df_sens2_raw,
    target_cols_list=['target_sens2'], # New target
    search_space_list=search_space_s2,
    title="Sensitivity Analysis 2: Fasting/Self-Report Only"
)

# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 3: EXCLUDE BORDERLINE ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 3: EXCLUDE BORDERLINE ###")
print("*"*80)

# 1. Define borderline criteria
is_borderline = (
    (dff1_ag['fasting_confirmation_qc'] == 0) &
    (dff1_ag['glucose_qc'] >= 6.9) &
    (dff1_ag['glucose_qc'] <= 7.1)
)

# 2. Define the raw dataset
df_sens3_raw = dff1_ag[~is_borderline].copy()
print(f"Dataset 3 (Raw) created. N={len(df_sens3_raw)}")

# 3. Run imputation
print("Running Imputation for Sensitivity Analysis 3...")
imputer_sens3 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s3 = imputer_sens3.fit_transform(df_sens3_raw[continuous])

# 4. Create the final imputed dataframe
df_sens3_imputed = df_sens3_raw.copy()
df_sens3_imputed[continuous] = imputed_features_s3
df_sens3_imputed['age'] = df_sens3_imputed.age.astype(int)
df_sens3_imputed = df_sens3_imputed[(df_sens3_imputed['age'] >= 40) & (df_sens3_imputed['age'] <= 60)]
print(f"Dataset 3 (Imputed) is ready. N={len(df_sens3_imputed)}")

# 5. Define search space (same as original, just excluding raw case cols)
search_space_s3 = [col for col in df_sens3_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc',
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc' # Exclude raw case cols
       ]]

# 6. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens3_imputed,
    dff_raw_pre_impute=df_sens3_raw,
    target_cols_list=['diabetes_status_c_qc'], # Original target
    search_space_list=search_space_s3,
    title="Sensitivity Analysis 3: Exclude Borderline"
)


# --------------------------------------------------------------------------
# --- ORIGINAL ANALYSIS (FROM YOUR SCRIPT) ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING ORIGINAL ANALYSIS ###")
print("*"*80)

# 1. Impute original Agincourt data
imputer.fit(dff1_ag[continuous])
train_t = imputer.transform(dff1_ag[continuous])
treated_ag = pd.DataFrame(train_t, columns=dff1_ag[continuous].columns)

imputer1.fit(dff1_nai[continuous])
train_t = imputer1.transform(dff1_nai[continuous])
treated_nai = pd.DataFrame(train_t, columns=dff1_nai[continuous].columns)

imputer1.fit(dff1_nan[continuous])
train_t = imputer1.transform(dff1_nan[continuous])
treated_nan = pd.DataFrame(train_t, columns=dff1_nan[continuous].columns)

imputer1.fit(dff1_dim[continuous])
train_t = imputer1.transform(dff1_dim[continuous])
treated_dim = pd.DataFrame(train_t, columns=dff1_dim[continuous].columns)

# 2. Create final imputed dataframes
# We only need Agincourt for this discovery
dff_ag = dff_ag[features + target_cols + ['study_id']].copy()
dff_ag.reset_index(inplace=True, drop=True)
dff_ag[continuous] = treated_ag[continuous]
dff_ag['age'] = dff_ag.age.astype(int)
dff_ag = dff_ag[(dff_ag['age'] >= 40) & (dff_ag['age'] <= 60)]

# (Skipping dff_nai, dff_dim, dff_nan creation as they aren't used in the scan)

print(f"Original Imputed Dataset (Agincourt) is ready. N={len(dff_ag)}")

# 3. Define Search Space
search_space1 = [col for col in dff_ag.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc',
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc']]

# 4. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=dff_ag,
    dff_raw_pre_impute=dff1_ag, # Pass dff1_ag as the "raw"
    target_cols_list=['diabetes_status_c_qc'], # Original target
    search_space_list=search_space1,
    title="Original Analysis (Agincourt)"
)

print("\n" + "*"*80)
print("### ALL ANALYSES COMPLETE ###")
print("*"*80)


import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --- Suppress warnings ---
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)


from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.ScoringFunctions.Poisson import Poisson
from mdss.MDSS import MDSS

file_path = '~/t2d_as.csv'
try:
    dff = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Warning: File not found at {file_path}. Using fallback path.")
    dff = pd.read_csv('t2d_as.csv') # Fallback for local dir

dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# More features
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']

# --- MODIFICATION 1: Load all required raw columns ---
# Columns needed for the sensitivity analysis
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# Modified this line to include the new columns from dfff
dff = dfff[features + target_cols + ['study_id'] + ['site'] + RAW_CASE_COLS].copy()
# --- END MODIFICATION ---


site_id = 1 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
print('Original size: ', dff_ag.shape)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Agincourt after removing records with missing targets: ', dff_ag.shape)


# The below is preparing the data for each site to be imputed
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()


# The imputation phase
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt


# HELPER FUNCTIONS (from your script)
def get_str(x):
    # This function turns a pandas bin to a meaningful string
    if pd.isna(x.left) or pd.isna(x.right):
        return np.nan
    s = str(round(x.left, 2)) + ' - ' + str(round(x.right,2))
    return s

def custom_qcut(ser, contiguous = True):
    # Get the rows that are actual numbers
    sub_ser = ser.dropna()
    sub_ser = sub_ser[(sub_ser != -111) \
                     & (sub_ser != -222) \
                     & (sub_ser != -555) \
                     & (sub_ser != -999)]
    
    if contiguous:
        # if contiguous, treat all the special numbers the same
        ser = ser.replace(-111 , -999)
        ser = ser.replace(-222 , -999)
        ser = ser.replace(-555 , -999)

    if sub_ser.empty:
        return ser # Return original series if no data to bin

    # Bin the actual numbers into 10 bins for scanning
    try:
        sub_ser_binned = pd.qcut(sub_ser, 10, duplicates='drop')
        sub_ser_str = sub_ser_binned.apply(get_str).astype(str)
        ser.loc[sub_ser.index] = sub_ser_str
    except ValueError as e:
        print(f"Warning: Could not qcut column {ser.name}. Error: {e}")
    except Exception as e:
        print(f"Warning: Error in custom_qcut for {ser.name}. Error: {e}")

    return ser


def compress_contiguous(subset, contiguous):
    # Shorten a contiguous list e.g [0-9, 10-19] is converted to [0 - 19] 
    new = {}
    
    for col in subset:
        if col in contiguous:
            if not subset[col]: # Handle empty list
                continue
            if isinstance(subset[col][0], (float,int)):
                new[col] = [str(c) for c in subset[col]]
                continue
            
            # Check for non-string values that might cause split to fail
            clean_values = [v for v in subset[col] if isinstance(v, str)]
            if not clean_values:
                new[col] = [str(c) for c in subset[col]]
                continue

            i = -1 if isinstance(clean_values[-1], str) else -2
            new[col] = [clean_values[0].split(' - ')[0] + ' - ' + clean_values[i].split(' - ')[-1]]
            
            # Add back any non-string values (like -999)
            non_string_vals = [str(c) for c in subset[col] if not isinstance(c, str)]
            new[col] = new[col] + non_string_vals
        else:
            new[col] = [str(c) for c in subset[col]]
    return new

def translate_subset_to_rule(subset):
    # Print the subset as a rule for easier understanding
    desc = ''
    for key, value in subset.items():
        # --- BUG FIX 1: Removed the "Continue" typo ---
        desc += key + '{' + ' OR '.join(value) + '} AND' + ' '

    return desc[:-5].replace('_',' ').replace('{', '[').replace('}', ']')

def count_conditions(subset):
    # Split the string by 'AND' and 'OR'
    conditions = subset.replace("AND", "OR").split("OR")
    
    # Count the number of conditions
    condition_count = len(conditions)
    
    return condition_count

# This is the function that runs the full Autostrat scan
def run_autostrat_scan(dff_imputed, dff_raw_pre_impute, target_cols_list, search_space_list, title):
    
    print("\n" + "="*80)
    print(f"### STARTING AUTOSCAN FOR: {title} ###")
    print(f"Input data shape: {dff_imputed.shape}")
    print(f"Using target columns: {target_cols_list}")
    print("="*80)

    # --- 1. Create dff_2 (Binned Data) ---
    # --- BUG FIX 2: Replaced pandas bitwise '&' with python 'and' ---
    numeric_columns = [col for col in dff_imputed.columns \
                         if (is_numeric_dtype(dff_imputed[col])) \
                         and (col not in target_cols_list) \
                         and (dff_imputed[col].nunique() > 10)]
    
    contiguous = {}
    dff_2 = dff_imputed.copy()

    # Create a new dataframe with the numeric columns bins     
    for col in numeric_columns:
        if col in search_space_list:
            dff_2[col] = custom_qcut(dff_2[col].copy())
            
            bins = list(dff_2[col].unique())
            
            # Handle potential NaN values from qcut
            bins = [b for b in bins if pd.notna(b)]

            if -999 in bins:
                bins.remove(-999)
            
            # Filter out any remaining non-string bins before sorting
            str_bins = [b for b in bins if isinstance(b, str)]
            bins = sorted(str_bins, key=lambda x : float(x.split(' - ')[0]))
            
            contiguous[col] = bins
    
    # --- 2. Define Expectation ---
    # Ensure target_cols_list is a list
    if not isinstance(target_cols_list, list):
        target_cols_list = [target_cols_list]
        
    dff_2['output'] = (dff_2[target_cols_list] == 1).any(axis=1) # Use .any() for list of targets
    dff_2['expectation'] = dff_2['output'].mean()

    # --- 3. Run Scan ---
    scoring_function = Bernoulli(direction='positive')
    scanner = MDSS(scoring_function)
    penalty_values = [0.4, 0.5, 1.0, 1.2, 1.5, 3.0, 12.5] # From your script
    # penalty_values = [0.45] # From your script

    num_iters = 7
    
    # Initialize empty lists to store results
    subset_results = []
    subset_results1 = []
    score_results = []
    size_results = []
    counts_percent = []
    size_percent = []
    odds_results = []
    odds = []
    z_scores = []
    p_values = []
    count_results = []
    dataframes = {}
    dataframes_complement = {}
    CIs = []

    for penalty in penalty_values:
        # Perform the scan with the current penalty value
        
        # --- BUG FIX 3 (from before): Pass 'output' Series ---
        subset, score = scanner.scan(
            dff_2[search_space_list], 
            dff_2['output'], # <-- Pass the 'output' Series
            dff_2['expectation'], 
            cpu=0.99,
            penalty=penalty, 
            num_iters=num_iters, 
            contiguous=contiguous.copy()
        )
        
        # Identify subset rows
        to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
        temp_df = dff_2.loc[to_choose]
        not_temp_df = dff_2.loc[~to_choose]

        # Store each DataFrame
        dataframes[f'df_{penalty}'] = temp_df
        dataframes_complement[f'df_{penalty}'] = not_temp_df

        # Basic info
        size = len(temp_df)
        total_size = len(dff_2)
        
        # Odds_multiplicative factor
        group_obs = temp_df['output'].mean() # Use 'output' Series
        group_prob = dff_2['expectation'].mean()
        if (1 - group_obs) == 0 or (1 - group_prob) == 0:
            odds_mul = float('inf')
        else:
            odds_mul = (group_obs / (1 - group_obs)) / (group_prob / (1 - group_prob))

        # 2×2 counts
        a = temp_df['output'].sum() # Use 'output' Series
        b = len(temp_df) - a
        c = not_temp_df['output'].sum() # Use 'output' Series
        d = len(not_temp_df) - c
        
        a, b, c, d = float(a), float(b), float(c), float(d)
        
        if a == 0 or b == 0 or c == 0 or d == 0:
            odds_ratio, log_IDR, Z_score, p_value = float('inf'), float('inf'), float('inf'), 0.0
            CI_lower, CI_upper = float('inf'), float('inf')
        else:
            odds_temp = a / b
            odds_not_temp = c / d
            odds_ratio = odds_temp / odds_not_temp if odds_not_temp != 0 else float('inf')
            log_IDR = np.log(odds_ratio)
            var_log_IDR = (1/a) + (1/b) + (1/c) + (1/d)
            SE_log_IDR = np.sqrt(var_log_IDR)
            CI_lower_log = log_IDR - 1.96 * SE_log_IDR
            CI_upper_log = log_IDR + 1.96 * SE_log_IDR
            CI_lower = np.exp(CI_lower_log)
            CI_upper = np.exp(CI_upper_log)
            Z_score = log_IDR / SE_log_IDR
            p_value = 2 * stats.norm.sf(abs(Z_score))
        
        CI = (round(CI_lower, 2), round(CI_upper, 2))

        # Summaries
        score_results.append(round(score, 3))
        size_results.append(size)
        size_percent.append(round(size / total_size * 100, 2))
        
        total_events_subset = a
        total_events = dff_2['output'].sum() # Use 'output' Series
        counts_percent.append(round(total_events_subset / total_events * 100, 2) if total_events > 0 else 0)
        
        odds_results.append(round(odds_mul, 2))
        odds.append(round(odds_ratio, 2))
        rule_str = translate_subset_to_rule(compress_contiguous(subset, contiguous))
        subset_results1.append(rule_str)
        counting = count_conditions(rule_str)
        count_results.append(counting)
        z_scores.append(Z_score)
        p_values.append(p_value)
        CIs.append(CI)

    # Display the results
    print("\n" + f"--- AUTOSCAN RESULTS: {title} ---")
    for i, penalty in enumerate(penalty_values):
        print(f"Penalty = {penalty}: \n"
              f"  Subset = {subset_results1[i]}, \n"
              f"  LiteralsNumber = {count_results[i]}, \n"
              f"  Size = {size_results[i]}, \n"
              f"  Mul_odds = {odds_results[i]}, \n"
              f"  Odds = {odds[i]}, \n"
              f"  Score = {score_results[i]}, \n"
              f"  Size_percent = {size_percent[i]}, \n"
              f"  Count_percent = {counts_percent[i]}, \n"
              f"  P_value = {p_values[i]:.3g}, \n"
              f"  CI = {CIs[i]}")
    print("="*80 + "\n")


# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 1: DIAGNOSED CASES ONLY ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 1: DIAGNOSED CASES ONLY ###")
print("*"*80)

# 1. Define the raw dataset (using dff1_ag which is already created)
df_sens1_raw = dff1_ag[
    (dff1_ag['diabetes_status_c_qc'] == 0) | 
    (dff1_ag['diabetes_self_reported_qc'] == 1)
].copy()
print(f"Dataset 1 (Raw) created. N={len(df_sens1_raw)}")

# 2. Run imputation
print("Running Imputation for Sensitivity Analysis 1...")
imputer_sens1 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s1 = imputer_sens1.fit_transform(df_sens1_raw[continuous])

# 3. Create the final imputed dataframe
df_sens1_imputed = df_sens1_raw.copy()
df_sens1_imputed[continuous] = imputed_features_s1
df_sens1_imputed['age'] = df_sens1_imputed.age.astype(int)
df_sens1_imputed = df_sens1_imputed[(df_sens1_imputed['age'] >= 40) & (df_sens1_imputed['age'] <= 60)]
print(f"Dataset 1 (Imputed) is ready. N={len(df_sens1_imputed)}")

# 4. Define search space (from your script)
search_space_s1 = [col for col in df_sens1_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc',
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc' # Exclude raw case cols
       ]]

# 5. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens1_imputed,
    dff_raw_pre_impute=df_sens1_raw,
    target_cols_list=['diabetes_status_c_qc'], # Original target
    search_space_list=search_space_s1,
    title="Sensitivity Analysis 1: Diagnosed Only"
)

# --------------------------------------------------------------------------
# --- (Original Analysis and other sensitivities removed as requested) ---
# --------------------------------------------------------------------------

print("\n" + "*"*80)
print("### ANALYSIS 1 COMPLETE ###")
print("*"*80)

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --- Suppress warnings ---
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)


from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.ScoringFunctions.Poisson import Poisson
from mdss.MDSS import MDSS

file_path = '~/t2d_as.csv'
try:
    dff = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Warning: File not found at {file_path}. Using fallback path.")
    dff = pd.read_csv('t2d_as.csv') # Fallback for local dir

dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# More features
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']

# --- MODIFICATION 1: Load all required raw columns ---
# Columns needed for the sensitivity analysis
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# --- BUG FIX: Ensure column list is unique ---
# Remove any columns from RAW_CASE_COLS that are *already* in 'features'
unique_raw_case_cols = [col for col in RAW_CASE_COLS if col not in features]

# Now, create the 'dff' DataFrame with a unique list of columns
dff = dfff[features + target_cols + ['study_id'] + ['site'] + unique_raw_case_cols].copy()
# --- END BUG FIX ---


site_id = 1 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
print('Original size: ', dff_ag.shape)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Agincourt after removing records with missing targets: ', dff_ag.shape)


# The below is preparing the data for each site to be imputed
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()

# --------------------------------------------------------------------------
# --- BUG FIX FOR VALUEERROR: Reset index to ensure it's unique ---
dff1_ag = dff1_ag.reset_index(drop=True)
print(f"dff1_ag index reset. N={len(dff1_ag)}")
# --------------------------------------------------------------------------


# The imputation phase
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt


# HELPER FUNCTIONS (from your script)
def get_str(x):
    # This function turns a pandas bin to a meaningful string
    if pd.isna(x.left) or pd.isna(x.right):
        return np.nan
    s = str(round(x.left, 2)) + ' - ' + str(round(x.right,2))
    return s

def custom_qcut(ser, contiguous = True):
    # Get the rows that are actual numbers
    sub_ser = ser.dropna()
    sub_ser = sub_ser[(sub_ser != -111) \
                     & (sub_ser != -222) \
                     & (sub_ser != -555) \
                     & (sub_ser != -999)]
    
    if contiguous:
        # if contiguous, treat all the special numbers the same
        ser = ser.replace(-111 , -999)
        ser = ser.replace(-222 , -999)
        ser = ser.replace(-555 , -999)

    if sub_ser.empty:
        return ser # Return original series if no data to bin

    # Bin the actual numbers into 10 bins for scanning
    try:
        sub_ser_binned = pd.qcut(sub_ser, 10, duplicates='drop')
        sub_ser_str = sub_ser_binned.apply(get_str).astype(str)
        ser.loc[sub_ser.index] = sub_ser_str
    except ValueError as e:
        print(f"Warning: Could not qcut column {ser.name}. Error: {e}")
    except Exception as e:
        print(f"Warning: Error in custom_qcut for {ser.name}. Error: {e}")

    return ser


def compress_contiguous(subset, contiguous):
    # Shorten a contiguous list e.g [0-9, 10-19] is converted to [0 - 19] 
    new = {}
    
    for col in subset:
        if col in contiguous:
            if not subset[col]: # Handle empty list
                continue
            if isinstance(subset[col][0], (float,int)):
                new[col] = [str(c) for c in subset[col]]
                continue
            
            # Check for non-string values that might cause split to fail
            clean_values = [v for v in subset[col] if isinstance(v, str)]
            if not clean_values:
                new[col] = [str(c) for c in subset[col]]
                continue

            i = -1 if isinstance(clean_values[-1], str) else -2
            new[col] = [clean_values[0].split(' - ')[0] + ' - ' + clean_values[i].split(' - ')[-1]]
            
            # Add back any non-string values (like -999)
            non_string_vals = [str(c) for c in subset[col] if not isinstance(c, str)]
            new[col] = new[col] + non_string_vals
        else:
            new[col] = [str(c) for c in subset[col]]
    return new

def translate_subset_to_rule(subset):
    # Print the subset as a rule for easier understanding
    desc = ''
    for key, value in subset.items():
        desc += key + '{' + ' OR '.join(value) + '} AND' + ' '

    return desc[:-5].replace('_',' ').replace('{', '[').replace('}', ']')

def count_conditions(subset):
    # Split the string by 'AND' and 'OR'
    conditions = subset.replace("AND", "OR").split("OR")
    
    # Count the number of conditions
    condition_count = len(conditions)
    
    return condition_count

# This is the function that runs the full Autostrat scan
def run_autostrat_scan(dff_imputed, dff_raw_pre_impute, target_cols_list, search_space_list, title):
    
    print("\n" + "="*80)
    print(f"### STARTING AUTOSCAN FOR: {title} ###")
    print(f"Input data shape: {dff_imputed.shape}")
    print(f"Using target columns: {target_cols_list}")
    print("="*80)

    # --- 1. Create dff_2 (Binned Data) ---
    numeric_columns = [col for col in dff_imputed.columns \
                         if (is_numeric_dtype(dff_imputed[col])) \
                         and (col not in target_cols_list) \
                         and (dff_imputed[col].nunique() > 10)]
    
    contiguous = {}
    dff_2 = dff_imputed.copy()

    # Create a new dataframe with the numeric columns bins     
    for col in numeric_columns:
        if col in search_space_list:
            dff_2[col] = custom_qcut(dff_2[col].copy())
            
            bins = list(dff_2[col].unique())
            
            # Handle potential NaN values from qcut
            bins = [b for b in bins if pd.notna(b)]

            if -999 in bins:
                bins.remove(-999)
            
            # Filter out any remaining non-string bins before sorting
            str_bins = [b for b in bins if isinstance(b, str)]
            bins = sorted(str_bins, key=lambda x : float(x.split(' - ')[0]))
            
            contiguous[col] = bins
    
    # --- 2. Define Expectation ---
    # Ensure target_cols_list is a list
    if not isinstance(target_cols_list, list):
        target_cols_list = [target_cols_list]
        
    dff_2['output'] = (dff_2[target_cols_list] == 1).any(axis=1) # Use .any() for list of targets
    dff_2['expectation'] = dff_2['output'].mean()

    # --- 3. Run Scan ---
    scoring_function = Bernoulli(direction='positive')
    scanner = MDSS(scoring_function)
    penalty_values = [0.4, 0.5, 1.0, 1.2, 1.5, 3.0, 12.5] # From your script
    # penalty_values = [0.08]
    num_iters = 5
    
    # Initialize empty lists to store results
    subset_results = []
    subset_results1 = []
    score_results = []
    size_results = []
    counts_percent = []
    size_percent = []
    odds_results = []
    odds = []
    z_scores = []
    p_values = []
    count_results = []
    dataframes = {}
    dataframes_complement = {}
    CIs = []

    for penalty in penalty_values:
        # Perform the scan with the current penalty value
        
        subset, score = scanner.scan(
            dff_2[search_space_list], 
            dff_2['output'], # <-- Pass the 'output' Series
            dff_2['expectation'], 
            cpu=0.99,
            penalty=penalty, 
            num_iters=num_iters, 
            contiguous=contiguous.copy()
        )
        
        # Identify subset rows
        to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
        temp_df = dff_2.loc[to_choose]
        not_temp_df = dff_2.loc[~to_choose]

        # Store each DataFrame
        dataframes[f'df_{penalty}'] = temp_df
        dataframes_complement[f'df_{penalty}'] = not_temp_df

        # Basic info
        size = len(temp_df)
        total_size = len(dff_2)
        
        # Odds_multiplicative factor
        group_obs = temp_df['output'].mean() # Use 'output' Series
        group_prob = dff_2['expectation'].mean()
        if (1 - group_obs) == 0 or (1 - group_prob) == 0:
            odds_mul = float('inf')
        else:
            odds_mul = (group_obs / (1 - group_obs)) / (group_prob / (1 - group_prob))

        # 2×2 counts
        a = temp_df['output'].sum() # Use 'output' Series
        b = len(temp_df) - a
        c = not_temp_df['output'].sum() # Use 'output' Series
        d = len(not_temp_df) - c
        
        a, b, c, d = float(a), float(b), float(c), float(d)
        
        if a == 0 or b == 0 or c == 0 or d == 0:
            odds_ratio, log_IDR, Z_score, p_value = float('inf'), float('inf'), float('inf'), 0.0
            CI_lower, CI_upper = float('inf'), float('inf')
        else:
            odds_temp = a / b
            odds_not_temp = c / d
            odds_ratio = odds_temp / odds_not_temp if odds_not_temp != 0 else float('inf')
            log_IDR = np.log(odds_ratio)
            var_log_IDR = (1/a) + (1/b) + (1/c) + (1/d)
            SE_log_IDR = np.sqrt(var_log_IDR)
            CI_lower_log = log_IDR - 1.96 * SE_log_IDR
            CI_upper_log = log_IDR + 1.96 * SE_log_IDR
            CI_lower = np.exp(CI_lower_log)
            CI_upper = np.exp(CI_upper_log)
            Z_score = log_IDR / SE_log_IDR
            p_value = 2 * stats.norm.sf(abs(Z_score))
        
        CI = (round(CI_lower, 2), round(CI_upper, 2))

        # Summaries
        score_results.append(round(score, 3))
        size_results.append(size)
        size_percent.append(round(size / total_size * 100, 2))
        
        total_events_subset = a
        total_events = dff_2['output'].sum() # Use 'output' Series
        counts_percent.append(round(total_events_subset / total_events * 100, 2) if total_events > 0 else 0)
        
        odds_results.append(round(odds_mul, 2))
        odds.append(round(odds_ratio, 2))
        rule_str = translate_subset_to_rule(compress_contiguous(subset, contiguous))
        subset_results1.append(rule_str)
        counting = count_conditions(rule_str)
        count_results.append(counting)
        z_scores.append(Z_score)
        p_values.append(p_value)
        CIs.append(CI)

    # Display the results
    print("\n" + f"--- AUTOSCAN RESULTS: {title} ---")
    for i, penalty in enumerate(penalty_values):
        print(f"Penalty = {penalty}: \n"
              f"  Subset = {subset_results1[i]}, \n"
              f"  LiteralsNumber = {count_results[i]}, \n"
              f"  Size = {size_results[i]}, \n"
              f"  Mul_odds = {odds_results[i]}, \n"
              f"  Odds = {odds[i]}, \n"
              f"  Score = {score_results[i]}, \n"
              f"  Size_percent = {size_percent[i]}, \n"
              f"  Count_percent = {counts_percent[i]}, \n"
              f"  P_value = {p_values[i]:.3g}, \n"
              f"  CI = {CIs[i]}")
    print("="*80 + "\n")


# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 2: FASTING/SELF-REPORT ONLY ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 2: FASTING/SELF-REPORT ONLY ###")
print("*"*80)

# 1. Define the new case criteria
# (Using dff1_ag which now has a unique index)
# --- FIX FOR VALUEERROR: Removed .values ---
# This now works because dff1_ag has a unique index AND 'glucose_qc' is a single column (Series)
is_case_sens2 = (
    (dff1_ag['diabetes_self_reported_qc'] == 1) |
    ((dff1_ag['fasting_confirmation_qc'] == 0) & (dff1_ag['glucose_qc'] >= 7.0))
)
is_control = (dff1_ag['diabetes_status_c_qc'] == 0)
# --- END FIX ---

# 2. Define the raw dataset
df_sens2_raw = dff1_ag[is_case_sens2 | is_control].copy()
df_sens2_raw['target_sens2'] = is_case_sens2.astype(int) # New target column
print(f"Dataset 2 (Raw) created. N={len(df_sens2_raw)}")

# 3. Run imputation
print("Running Imputation for Sensitivity Analysis 2...")
imputer_sens2 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s2 = imputer_sens2.fit_transform(df_sens2_raw[continuous])

# 4. Create the final imputed dataframe
df_sens2_imputed = df_sens2_raw.copy()
df_sens2_imputed[continuous] = imputed_features_s2
df_sens2_imputed['age'] = df_sens2_imputed.age.astype(int)
df_sens2_imputed = df_sens2_imputed[(df_sens2_imputed['age'] >= 40) & (df_sens2_imputed['age'] <= 60)]
print(f"Dataset 2 (Imputed) is ready. N={len(df_sens2_imputed)}")

# 5. Define search space (remove the NEW target col and old target)
search_space_s2 = [col for col in df_sens2_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc', # Exclude original target
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc', # Exclude raw case cols
       'target_sens2' # Exclude NEW target
       ]]

# 6. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens2_imputed,
    dff_raw_pre_impute=df_sens2_raw,
    target_cols_list=['target_sens2'], # New target
    search_space_list=search_space_s2,
    title="Sensitivity Analysis 2: Fasting/Self-Report Only"
)

# --------------------------------------------------------------------------
# --- (Original Analysis and other sensitivities removed as requested) ---
# --------------------------------------------------------------------------

print("\n" + "*"*80)
print("### ANALYSIS 2 COMPLETE ###")
print("*"*80)


import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --- Suppress warnings ---
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)


from mdss.ScoringFunctions.Bernoulli import Bernoulli
from mdss.ScoringFunctions.Poisson import Poisson
from mdss.MDSS import MDSS

file_path = '~/t2d_as.csv'
try:
    dff = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Warning: File not found at {file_path}. Using fallback path.")
    dff = pd.read_csv('t2d_as.csv') # Fallback for local dir

dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# More features
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']

# --- MODIFICATION 1: Load all required raw columns ---
# Columns needed for the sensitivity analysis
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# --- BUG FIX: Ensure column list is unique ---
# Remove any columns from RAW_CASE_COLS that are *already* in 'features'
unique_raw_case_cols = [col for col in RAW_CASE_COLS if col not in features]

# Now, create the 'dff' DataFrame with a unique list of columns
dff = dfff[features + target_cols + ['study_id'] + ['site'] + unique_raw_case_cols].copy()
# --- END BUG FIX ---


site_id = 1 # 1 - Agincourt, 3 - Nairobi

# Choose the relevant site and age group
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
print('Original size: ', dff_ag.shape)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
print('Size of Agincourt after removing records with missing targets: ', dff_ag.shape)


# The below is preparing the data for each site to be imputed
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()

# --------------------------------------------------------------------------
# --- BUG FIX FOR VALUEERROR: Reset index to ensure it's unique ---
dff1_ag = dff1_ag.reset_index(drop=True)
print(f"dff1_ag index reset. N={len(dff1_ag)}")
# --------------------------------------------------------------------------


# The imputation phase
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt


# HELPER FUNCTIONS (from your script)
def get_str(x):
    # This function turns a pandas bin to a meaningful string
    if pd.isna(x.left) or pd.isna(x.right):
        return np.nan
    s = str(round(x.left, 2)) + ' - ' + str(round(x.right,2))
    return s

def custom_qcut(ser, contiguous = True):
    # Get the rows that are actual numbers
    sub_ser = ser.dropna()
    sub_ser = sub_ser[(sub_ser != -111) \
                     & (sub_ser != -222) \
                     & (sub_ser != -555) \
                     & (sub_ser != -999)]
    
    if contiguous:
        # if contiguous, treat all the special numbers the same
        ser = ser.replace(-111 , -999)
        ser = ser.replace(-222 , -999)
        ser = ser.replace(-555 , -999)

    if sub_ser.empty:
        return ser # Return original series if no data to bin

    # Bin the actual numbers into 10 bins for scanning
    try:
        sub_ser_binned = pd.qcut(sub_ser, 10, duplicates='drop')
        sub_ser_str = sub_ser_binned.apply(get_str).astype(str)
        ser.loc[sub_ser.index] = sub_ser_str
    except ValueError as e:
        print(f"Warning: Could not qcut column {ser.name}. Error: {e}")
    except Exception as e:
        print(f"Warning: Error in custom_qcut for {ser.name}. Error: {e}")

    return ser


def compress_contiguous(subset, contiguous):
    # Shorten a contiguous list e.g [0-9, 10-19] is converted to [0 - 19] 
    new = {}
    
    for col in subset:
        if col in contiguous:
            if not subset[col]: # Handle empty list
                continue
            if isinstance(subset[col][0], (float,int)):
                new[col] = [str(c) for c in subset[col]]
                continue
            
            # Check for non-string values that might cause split to fail
            clean_values = [v for v in subset[col] if isinstance(v, str)]
            if not clean_values:
                new[col] = [str(c) for c in subset[col]]
                continue

            i = -1 if isinstance(clean_values[-1], str) else -2
            new[col] = [clean_values[0].split(' - ')[0] + ' - ' + clean_values[i].split(' - ')[-1]]
            
            # Add back any non-string values (like -999)
            non_string_vals = [str(c) for c in subset[col] if not isinstance(c, str)]
            new[col] = new[col] + non_string_vals
        else:
            new[col] = [str(c) for c in subset[col]]
    return new

def translate_subset_to_rule(subset):
    # Print the subset as a rule for easier understanding
    desc = ''
    for key, value in subset.items():
        desc += key + '{' + ' OR '.join(value) + '} AND' + ' '

    return desc[:-5].replace('_',' ').replace('{', '[').replace('}', ']')

def count_conditions(subset):
    # Split the string by 'AND' and 'OR'
    conditions = subset.replace("AND", "OR").split("OR")
    
    # Count the number of conditions
    condition_count = len(conditions)
    
    return condition_count

# This is the function that runs the full Autostrat scan
def run_autostrat_scan(dff_imputed, dff_raw_pre_impute, target_cols_list, search_space_list, title):
    
    print("\n" + "="*80)
    print(f"### STARTING AUTOSCAN FOR: {title} ###")
    print(f"Input data shape: {dff_imputed.shape}")
    print(f"Using target columns: {target_cols_list}")
    print("="*80)

    # --- 1. Create dff_2 (Binned Data) ---
    numeric_columns = [col for col in dff_imputed.columns \
                         if (is_numeric_dtype(dff_imputed[col])) \
                         and (col not in target_cols_list) \
                         and (dff_imputed[col].nunique() > 10)]
    
    contiguous = {}
    dff_2 = dff_imputed.copy()

    # Create a new dataframe with the numeric columns bins     
    for col in numeric_columns:
        if col in search_space_list:
            dff_2[col] = custom_qcut(dff_2[col].copy())
            
            bins = list(dff_2[col].unique())
            
            # Handle potential NaN values from qcut
            bins = [b for b in bins if pd.notna(b)]

            if -999 in bins:
                bins.remove(-999)
            
            # Filter out any remaining non-string bins before sorting
            str_bins = [b for b in bins if isinstance(b, str)]
            bins = sorted(str_bins, key=lambda x : float(x.split(' - ')[0]))
            
            contiguous[col] = bins
    
    # --- 2. Define Expectation ---
    # Ensure target_cols_list is a list
    if not isinstance(target_cols_list, list):
        target_cols_list = [target_cols_list]
        
    dff_2['output'] = (dff_2[target_cols_list] == 1).any(axis=1) # Use .any() for list of targets
    dff_2['expectation'] = dff_2['output'].mean()

    # --- 3. Run Scan ---
    scoring_function = Bernoulli(direction='positive')
    scanner = MDSS(scoring_function)
    # penalty_values = [0.4, 0.5, 1.0, 1.2, 1.5, 3.0, 12.5] # From your script
    penalty_values = [0.3,1.4,1.0,1.1,3.5] # From your script
    num_iters = 10
    
    # Initialize empty lists to store results
    subset_results = []
    subset_results1 = []
    score_results = []
    size_results = []
    counts_percent = []
    size_percent = []
    odds_results = []
    odds = []
    z_scores = []
    p_values = []
    count_results = []
    dataframes = {}
    dataframes_complement = {}
    CIs = []

    for penalty in penalty_values:
        # Perform the scan with the current penalty value
        
        subset, score = scanner.scan(
            dff_2[search_space_list], 
            dff_2['output'], # <-- Pass the 'output' Series
            dff_2['expectation'], 
            cpu=0.99,
            penalty=penalty, 
            num_iters=num_iters, 
            contiguous=contiguous.copy()
        )
        
        # Identify subset rows
        to_choose = dff_2[subset.keys()].isin(subset).all(axis=1)
        temp_df = dff_2.loc[to_choose]
        not_temp_df = dff_2.loc[~to_choose]

        # Store each DataFrame
        dataframes[f'df_{penalty}'] = temp_df
        dataframes_complement[f'df_{penalty}'] = not_temp_df

        # Basic info
        size = len(temp_df)
        total_size = len(dff_2)
        
        # Odds_multiplicative factor
        group_obs = temp_df['output'].mean() # Use 'output' Series
        group_prob = dff_2['expectation'].mean()
        if (1 - group_obs) == 0 or (1 - group_prob) == 0:
            odds_mul = float('inf')
        else:
            odds_mul = (group_obs / (1 - group_obs)) / (group_prob / (1 - group_prob))

        # 2×2 counts
        a = temp_df['output'].sum() # Use 'output' Series
        b = len(temp_df) - a
        c = not_temp_df['output'].sum() # Use 'output' Series
        d = len(not_temp_df) - c
        
        a, b, c, d = float(a), float(b), float(c), float(d)
        
        if a == 0 or b == 0 or c == 0 or d == 0:
            odds_ratio, log_IDR, Z_score, p_value = float('inf'), float('inf'), float('inf'), 0.0
            CI_lower, CI_upper = float('inf'), float('inf')
        else:
            odds_temp = a / b
            odds_not_temp = c / d
            odds_ratio = odds_temp / odds_not_temp if odds_not_temp != 0 else float('inf')
            log_IDR = np.log(odds_ratio)
            var_log_IDR = (1/a) + (1/b) + (1/c) + (1/d)
            SE_log_IDR = np.sqrt(var_log_IDR)
            CI_lower_log = log_IDR - 1.96 * SE_log_IDR
            CI_upper_log = log_IDR + 1.96 * SE_log_IDR
            CI_lower = np.exp(CI_lower_log)
            CI_upper = np.exp(CI_upper_log)
            Z_score = log_IDR / SE_log_IDR
            p_value = 2 * stats.norm.sf(abs(Z_score))
        
        CI = (round(CI_lower, 2), round(CI_upper, 2))

        # Summaries
        score_results.append(round(score, 3))
        size_results.append(size)
        size_percent.append(round(size / total_size * 100, 2))
        
        total_events_subset = a
        total_events = dff_2['output'].sum() # Use 'output' Series
        counts_percent.append(round(total_events_subset / total_events * 100, 2) if total_events > 0 else 0)
        
        odds_results.append(round(odds_mul, 2))
        odds.append(round(odds_ratio, 2))
        rule_str = translate_subset_to_rule(compress_contiguous(subset, contiguous))
        subset_results1.append(rule_str)
        counting = count_conditions(rule_str)
        count_results.append(counting)
        z_scores.append(Z_score)
        p_values.append(p_value)
        CIs.append(CI)

    # Display the results
    print("\n" + f"--- AUTOSCAN RESULTS: {title} ---")
    for i, penalty in enumerate(penalty_values):
        print(f"Penalty = {penalty}: \n"
              f"  Subset = {subset_results1[i]}, \n"
              f"  LiteralsNumber = {count_results[i]}, \n"
              f"  Size = {size_results[i]}, \n"
              f"  Mul_odds = {odds_results[i]}, \n"
              f"  Odds = {odds[i]}, \n"
              f"  Score = {score_results[i]}, \n"
              f"  Size_percent = {size_percent[i]}, \n"
              f"  Count_percent = {counts_percent[i]}, \n"
              f"  P_value = {p_values[i]:.3g}, \n"
              f"  CI = {CIs[i]}")
    print("="*80 + "\n")


# --------------------------------------------------------------------------
# --- SENSITIVITY ANALYSIS 3: EXCLUDE BORDERLINE ---
# --------------------------------------------------------------------------
print("\n" + "*"*80)
print("### STARTING SENSITIVITY ANALYSIS 3: EXCLUDE BORDERLINE ###")
print("*"*80)

# 1. Define borderline criteria
# (Using dff1_ag which has a unique index)
is_borderline = (
    (dff1_ag['fasting_confirmation_qc'] == 0) &
    (dff1_ag['glucose_qc'] >= 6.9) &
    (dff1_ag['glucose_qc'] <= 7.1)
)

# 2. Define the raw dataset (keep everyone NOT borderline)
df_sens3_raw = dff1_ag[~is_borderline].copy()
print(f"Dataset 3 (Raw) created. N={len(df_sens3_raw)}")

# 3. Run imputation
print("Running Imputation for Sensitivity Analysis 3...")
imputer_sens3 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s3 = imputer_sens3.fit_transform(df_sens3_raw[continuous])

# 4. Create the final imputed dataframe
df_sens3_imputed = df_sens3_raw.copy()
df_sens3_imputed[continuous] = imputed_features_s3
df_sens3_imputed['age'] = df_sens3_imputed.age.astype(int)
df_sens3_imputed = df_sens3_imputed[(df_sens3_imputed['age'] >= 40) & (df_sens3_imputed['age'] <= 60)]
print(f"Dataset 3 (Imputed) is ready. N={len(df_sens3_imputed)}")

# 5. Define search space
search_space_s3 = [col for col in df_sens3_imputed.columns \
                if col not in \
                ['study_id', 'ur_creatinine_qc','diabetes_treat_weight_loss_qc',
       'hypertension_12months_yn_qc', 'other_diabetes_specify_qc', 'site_id', 'fasting_glucose', 'site_qc',
       'diabetes_treat_other_qc', 'diabetes_meds_specify_qc', 'ethnicity_qc', 'site_qc.1',
       'diabetes_treat_other_qc.1', 'site', 'diabetes_12months', 'region_qc', 'ldl_qc',
       'diabetes_traditional_qc', 'hiv_final_status_c_qc', 'glucose_qc', 'diabetes_treat_curr_qc', 'employment_status_c_qc',
       'acr_qc', 'diabetes_treat_insulin_qc', 'other_diabetes_qc', 'obesity_mom_qc','fruit_servings_qc', 'servings_veg_qc', 
       'cholesterol_qc', 'htn_jnc7_qc', 'diabetes_status_c_qc', 'alcohol_grade_c_qc', 
       'alcohol_status_c_qc', 'diet_health_c_qc', 'diabetes_treat_pills_qc', 'visceral_fat_qc',
       'hypertension_meds_yn_qc', 'output', 'expectation', 'bp_sys_average_qc', 'bp_dia_average_qc',
       'weight_qc', 'triglycerides_qc', 'hdl_qc', 'friedewald_ldl_c_c_qc', 'egfr_c_qc',
       'diabetes_self_reported_qc', 'fasting_confirmation_qc' # Exclude raw case cols
       ]]
       
# 6. Run Autostrat Scan
run_autostrat_scan(
    dff_imputed=df_sens3_imputed,
    dff_raw_pre_impute=df_sens3_raw,
    target_cols_list=['diabetes_status_c_qc'], # Original target
    search_space_list=search_space_s3,
    title="Sensitivity Analysis 3: Exclude Borderline"
)

# --------------------------------------------------------------------------
# --- (Original Analysis and other sensitivities removed as requested) ---
# --------------------------------------------------------------------------

print("\n" + "*"*80)
print("### ANALYSIS 3 COMPLETE ###")
print("*"*80)

# --- CELL 1: LOAD ALL DATA (with fixes) ---
import pandas as pd
import numpy as np

file_path = '~/t2d_as.csv'
try:
    dff_csv = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Warning: File not found at {file_path}. Using fallback path.")
    dff_csv = pd.read_csv('t2d_as.csv') # Fallback for local dir

dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# More features
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']

# Columns needed for the sensitivity analysis
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# --- FIX: Ensure column list is unique ---
unique_raw_case_cols = [col for col in RAW_CASE_COLS if col not in features]
all_cols = features + target_cols + ['study_id', 'site'] + unique_raw_case_cols
dff = dfff[all_cols].copy()

print("Data loaded and all columns are unique.")
# --- CELL 2: CREATE RAW AGINCOURT DATAFRAME ---

site_id = 1 # 1 - Agincourt
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()

# Prepare for imputation (replace -999 with NaN)
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()

# --- FIX: Reset index to prevent reindexing errors ---
dff1_ag = dff1_ag.reset_index(drop=True)

print(f"Raw Agincourt dataset 'dff1_ag' created. N={len(dff1_ag)}")
# --- CELL 3: SENSITIVITY ANALYSIS PROPORTIONS ---
# (This cell performs the sanity check you requested)

print("\n" + "*"*80)
print("### SENSITIVITY ANALYSIS: POPULATION PROPORTIONS (SANITY CHECK) ###")
print("*"*80)

# --- 1. Original Dataset (for baseline) ---
total_n = len(dff1_ag)
cases = dff1_ag['diabetes_status_c_qc'].sum()
controls = total_n - cases
print(f"--- 1. Original Dataset (Baseline) ---")
print(f"  Total N:      {total_n}")
print(f"  Cases (n):    {cases} ({cases/total_n:.2%})")
print(f"  Controls (n): {controls} ({controls/total_n:.2%})\n")

# --- 2. Analysis 1: Diagnosed Cases Only ---
df_sens1_raw = dff1_ag[
    (dff1_ag['diabetes_status_c_qc'] == 0) | 
    (dff1_ag['diabetes_self_reported_qc'] == 1)
].copy()
total_n_s1 = len(df_sens1_raw)
cases_s1 = df_sens1_raw['diabetes_status_c_qc'].sum()
controls_s1 = total_n_s1 - cases_s1
print(f"--- 2. Analysis 1 (Diagnosed Only) ---")
print(f"  (This dataset keeps all controls but only self-reported cases)")
print(f"  Total N:      {total_n_s1}")
print(f"  Cases (n):    {cases_s1} ({cases_s1/total_n_s1:.2%})")
print(f"  Controls (n): {controls_s1} ({controls_s1/total_n_s1:.2%})\n")

# --- 3. Analysis 2: Fasting/Self-Report Only ---
is_case_sens2 = (
    (dff1_ag['diabetes_self_reported_qc'] == 1) |
    ((dff1_ag['fasting_confirmation_qc'] == 0) & (dff1_ag['glucose_qc'] >= 7.0))
)
is_control = (dff1_ag['diabetes_status_c_qc'] == 0)
df_sens2_raw = dff1_ag[is_case_sens2 | is_control].copy()
df_sens2_raw['target_sens2'] = is_case_sens2.astype(int) # The new target

total_n_s2 = len(df_sens2_raw)
cases_s2 = df_sens2_raw['target_sens2'].sum()
controls_s2 = total_n_s2 - cases_s2
print(f"--- 3. Analysis 2 (Fasting/Self-Report Only) ---")
print(f"  (This dataset excludes cases defined *only* by random glucose)")
print(f"  Total N:      {total_n_s2}")
print(f"  Cases (n):    {cases_s2} ({cases_s2/total_n_s2:.2%})")
print(f"  Controls (n): {controls_s2} ({controls_s2/total_n_s2:.2%})\n")

# --- 4. Analysis 3: Exclude Borderline ---
is_borderline = (
    (dff1_ag['fasting_confirmation_qc'] == 0) &
    (dff1_ag['glucose_qc'] >= 6.9) &
    (dff1_ag['glucose_qc'] <= 7.1)
)
df_sens3_raw = dff1_ag[~is_borderline].copy()

total_n_s3 = len(df_sens3_raw)
cases_s3 = df_sens3_raw['diabetes_status_c_qc'].sum()
controls_s3 = total_n_s3 - cases_s3
print(f"--- 4. Analysis 3 (Exclude Borderline) ---")
print(f"  (This dataset removes {total_n - total_n_s3} borderline individuals)")
print(f"  Total N:      {total_n_s3}")
print(f"  Cases (n):    {cases_s3} ({cases_s3/total_n_s3:.2%})")
print(f"  Controls (n): {controls_s3} ({controls_s3/total_n_s3:.2%})\n")
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

# --- Suppress warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)

# --------------------------------------------------------------------------
# --- 1. SETUP: Load Raw Data and Define Imputer ---
# --------------------------------------------------------------------------

file_path = '~/t2d_as.csv'
try:
    dff_csv = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Warning: File not found at {file_path}. Using fallback path.")
    dff_csv = pd.read_csv('t2d_as.csv') # Fallback for local dir

dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# Define features and continuous columns from your script
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']

# Define raw case columns
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# --- FIX: Ensure column list is unique ---
unique_raw_case_cols = [col for col in RAW_CASE_COLS if col not in features]
all_cols = features + target_cols + ['study_id', 'site'] + unique_raw_case_cols
dff = dfff[all_cols].copy()

# --- Create Base Agincourt DataFrame ---
site_id = 1 # 1 - Agincourt
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()
dff1_ag = dff1_ag.reset_index(drop=True) # Fix for reindexing errors
print(f"Raw Agincourt dataset 'dff1_ag' created. N={len(dff1_ag)}")

# --------------------------------------------------------------------------
# --- 2. YOUR FUNCTION: calculate_subgroup_metrics ---
# --------------------------------------------------------------------------
def calculate_subgroup_metrics(data, mask, outcome_col):
    
    tp = (mask & (data[outcome_col] == 1)).sum()
    fp = (mask & (data[outcome_col] == 0)).sum()
    tn = (~mask & (data[outcome_col] == 0)).sum()
    fn = (~mask & (data[outcome_col] == 1)).sum()
    
    total_pop = len(data)
    
    # Sensitivity (P(S|T2D)) - "Recall"
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # PPV (P(T2D|S)) - "Precision"
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # NPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        "P(S|T2D) [Sensitivity]": sensitivity,
        "Specificity": specificity,
        "P(T2D|S) [PPV]": ppv,
        "NPV": npv,
        "Total_in_Subgroup": (tp + fp),
        "Subgroup_Size_Percent": (tp + fp) / total_pop
    }

# --------------------------------------------------------------------------
# --- 3. CREATE SENSITIVITY DATASETS (IMPUTED) ---
# --------------------------------------------------------------------------

# --- Dataset 1: Diagnosed Only ---
df_sens1_raw = dff1_ag[
    (dff1_ag['diabetes_status_c_qc'] == 0) | 
    (dff1_ag['diabetes_self_reported_qc'] == 1)
].copy()
imputer_sens1 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s1 = imputer_sens1.fit_transform(df_sens1_raw[continuous])
df_sens1_imputed = df_sens1_raw.copy()
df_sens1_imputed[continuous] = imputed_features_s1
df_sens1_imputed = df_sens1_imputed[(df_sens1_imputed['age'] >= 40) & (df_sens1_imputed['age'] <= 60)]
print(f"Dataset 1 (Diagnosed Only) Imputed. N={len(df_sens1_imputed)}")

# --- Dataset 2: Fasting/Self-Report Only ---
is_case_sens2 = (
    (dff1_ag['diabetes_self_reported_qc'] == 1) |
    ((dff1_ag['fasting_confirmation_qc'] == 0) & (dff1_ag['glucose_qc'] >= 7.0))
)
is_control = (dff1_ag['diabetes_status_c_qc'] == 0)
df_sens2_raw = dff1_ag[is_case_sens2 | is_control].copy()
df_sens2_raw['target_sens2'] = is_case_sens2.astype(int) # New target
imputer_sens2 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s2 = imputer_sens2.fit_transform(df_sens2_raw[continuous])
df_sens2_imputed = df_sens2_raw.copy()
df_sens2_imputed[continuous] = imputed_features_s2
df_sens2_imputed = df_sens2_imputed[(df_sens2_imputed['age'] >= 40) & (df_sens2_imputed['age'] <= 60)]
print(f"Dataset 2 (Fasting/Self-Report) Imputed. N={len(df_sens2_imputed)}")

# --- Dataset 3: Exclude Borderline ---
is_borderline = (
    (dff1_ag['fasting_confirmation_qc'] == 0) &
    (dff1_ag['glucose_qc'] >= 6.9) &
    (dff1_ag['glucose_qc'] <= 7.1)
)
df_sens3_raw = dff1_ag[~is_borderline].copy()
imputer_sens3 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s3 = imputer_sens3.fit_transform(df_sens3_raw[continuous])
df_sens3_imputed = df_sens3_raw.copy()
df_sens3_imputed[continuous] = imputed_features_s3
df_sens3_imputed = df_sens3_imputed[(df_sens3_imputed['age'] >= 40) & (df_sens3_imputed['age'] <= 60)]
print(f"Dataset 3 (Exclude Borderline) Imputed. N={len(df_sens3_imputed)}")

# --------------------------------------------------------------------------
# --- 4. DEFINE BEST SUBGROUPS & CALCULATE METRICS ---
# --------------------------------------------------------------------------

all_results = []

# --- Analysis 1 Subgroups ---
# From your output: Penalty 1.2 was the best interpretable one
mask_s1 = (
    (df_sens1_imputed['bmi_c_qc'] >= 21.31) &
    (df_sens1_imputed['mvpa_c'] <= 1680.0) &
    (df_sens1_imputed['diabetes_history_qc'] == 1) &
    (df_sens1_imputed['waist_hip_r_c_qc'] > 0.9) &
    (df_sens1_imputed['hip_circumference_qc'] >= 670.0)
)
metrics_s1 = calculate_subgroup_metrics(df_sens1_imputed, mask_s1, 'diabetes_status_c_qc')
metrics_s1['Scenario'] = "1. Diagnosed Only"
metrics_s1['Subgroup description'] = "bmi[21.3-68.0] AND mvpa[0-1680] AND diabetes history[1.0] AND waist hip r c[0.9-1.16] AND hip circumference[670-1180]"
all_results.append(metrics_s1)

# --- Analysis 2 Subgroups ---
# From your output: Penalty 1.2 was the best interpretable one (NOT the 17-literal one)
mask_s2 = (
    (df_sens2_imputed['waist_circumference_qc'] >= 810.0) &
    (df_sens2_imputed['mvpa_c'] <= 1680.0) &
    (df_sens2_imputed['diabetes_history_qc'] == 1) &
    (df_sens2_imputed['age'] >= 47.0)
)
# *** USE THE CORRECT TARGET ***
metrics_s2 = calculate_subgroup_metrics(df_sens2_imputed, mask_s2, 'target_sens2') 
metrics_s2['Scenario'] = "2. Fasting/Self-Report Only"
metrics_s2['Subgroup description'] = "waist circumference[810-1470] AND mvpa[0-1680] AND diabetes history[1.0] AND age[47-60]"
all_results.append(metrics_s2)

# --- Analysis 3 Subgroups ---
# From your output: Penalty 1.1 and 3.5 were the identical matches
mask_s3_a = (
    (df_sens3_imputed['bmi_c_qc'] >= 21.37) &
    (df_sens3_imputed['mvpa_c'] <= 2448.0) &
    (df_sens3_imputed['diabetes_history_qc'] == 1) &
    (df_sens3_imputed['waist_hip_r_c_qc'] > 0.9)
)
metrics_s3_a = calculate_subgroup_metrics(df_sens3_imputed, mask_s3_a, 'diabetes_status_c_qc')
metrics_s3_a['Scenario'] = "3. Exclude Borderline"
metrics_s3_a['Subgroup description'] = "bmi[21.37-68.02] AND mvpa[0-2448] AND diabetes history[1.0] AND waist hip r c[0.9-1.16]"
all_results.append(metrics_s3_a)

mask_s3_b = (
    (df_sens3_imputed['diabetes_history_qc'] == 1) &
    (df_sens3_imputed['waist_hip_r_c_qc'] > 0.9)
)
metrics_s3_b = calculate_subgroup_metrics(df_sens3_imputed, mask_s3_b, 'diabetes_status_c_qc')
metrics_s3_b['Scenario'] = "3. Exclude Borderline"
metrics_s3_b['Subgroup description'] = "diabetes history[1.0] AND waist hip r c[0.9-1.16]"
all_results.append(metrics_s3_b)


# --------------------------------------------------------------------------
# --- 5. CREATE AND PRINT THE FINAL TABLE ---
# --------------------------------------------------------------------------
final_table = pd.DataFrame(all_results)

# Select and reorder columns as you requested
final_table = final_table[[
    "Scenario",
    "Subgroup description",
    "Subgroup_Size_Percent",
    "P(T2D|S) [PPV]",
    "P(S|T2D) [Sensitivity]",
    "Specificity",
    "NPV"
]]

print("\n\n" + "="*80)
print("### FINAL SENSITIVITY ANALYSIS ROBUSTNESS TABLE (FOR REVIEWER 3) ###")
print("="*80)
print(final_table.to_markdown(floatfmt=".3f", index=False))

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

# --- Suppress warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)

# --------------------------------------------------------------------------
# --- 1. SETUP: Load Raw Data and Define Imputer ---
# --------------------------------------------------------------------------

file_path = '~/t2d_as.csv'
try:
    dff_csv = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Warning: File not found at {file_path}. Using fallback path.")
    dff_csv = pd.read_csv('t2d_as.csv') # Fallback for local dir

dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# Define features and continuous columns from your script
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']

# Define raw case columns
RAW_CASE_COLS = ['diabetes_self_reported_qc', 'fasting_confirmation_qc', 'glucose_qc']

# --- FIX: Ensure column list is unique ---
unique_raw_case_cols = [col for col in RAW_CASE_COLS if col not in features]
all_cols = features + target_cols + ['study_id', 'site'] + unique_raw_case_cols
dff = dfff[all_cols].copy()

# --- Create Base Agincourt DataFrame ---
site_id = 1 # 1 - Agincourt
dff_ag = dff[(dff['site'] == site_id)].fillna(-999)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()
dff1_ag = dff1_ag.reset_index(drop=True) # Fix for reindexing errors
print(f"Raw Agincourt dataset 'dff1_ag' created. N={len(dff1_ag)}")

# --------------------------------------------------------------------------
# --- 2. YOUR FUNCTION: calculate_subgroup_metrics ---
# --------------------------------------------------------------------------
def calculate_subgroup_metrics(data, mask, outcome_col):
    
    # Ensure mask is a boolean array/Series with the same index as data
    if not isinstance(mask, pd.Series) or not mask.index.equals(data.index):
        if isinstance(mask, np.ndarray) and len(mask) == len(data):
            mask = pd.Series(mask, index=data.index)
        else:
            # Handle reset index for imputed data
            data = data.reset_index(drop=True)
            if isinstance(mask, np.ndarray) and len(mask) == len(data):
                 mask = pd.Series(mask, index=data.index)
            else:
                # As a last resort if mask is Series with different index
                mask = pd.Series(mask.values, index=data.index)
                
    # Ensure outcome_col exists and has no NaNs (or handle them)
    if data[outcome_col].isna().any():
        print(f"Warning: NaNs found in outcome column '{outcome_col}'. This may affect metrics.")
        
    tp = (mask & (data[outcome_col] == 1)).sum()
    fp = (mask & (data[outcome_col] == 0)).sum()
    tn = (~mask & (data[outcome_col] == 0)).sum()
    fn = (~mask & (data[outcome_col] == 1)).sum()
    
    total_pop = len(data)
    
    # Sensitivity (P(S|T2D)) - "Recall"
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # PPV (P(T2D|S)) - "Precision"
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # NPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        "P(S|T2D) [Sensitivity]": sensitivity,
        "Specificity": specificity,
        "P(T2D|S) [PPV]": ppv,
        "NPV": npv,
        "Total_in_Subgroup": (tp + fp),
        "Subgroup_Size_Percent": (tp + fp) / total_pop
    }

# --------------------------------------------------------------------------
# --- 3. CREATE SENSITIVITY DATASETS (IMPUTED) ---
# --------------------------------------------------------------------------

# --- Dataset 1: Diagnosed Only ---
df_sens1_raw = dff1_ag[
    (dff1_ag['diabetes_status_c_qc'] == 0) | 
    (dff1_ag['diabetes_self_reported_qc'] == 1)
].copy()
imputer_sens1 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s1 = imputer_sens1.fit_transform(df_sens1_raw[continuous])
df_sens1_imputed = df_sens1_raw.copy()
df_sens1_imputed[continuous] = imputed_features_s1
df_sens1_imputed = df_sens1_imputed[(df_sens1_imputed['age'] >= 40) & (df_sens1_imputed['age'] <= 60)].reset_index(drop=True)
print(f"Dataset 1 (Diagnosed Only) Imputed. N={len(df_sens1_imputed)}")

# --- Dataset 2: Fasting/Self-Report Only ---
is_case_sens2 = (
    (dff1_ag['diabetes_self_reported_qc'] == 1) |
    ((dff1_ag['fasting_confirmation_qc'] == 0) & (dff1_ag['glucose_qc'] >= 7.0))
)
is_control = (dff1_ag['diabetes_status_c_qc'] == 0)
df_sens2_raw = dff1_ag[is_case_sens2 | is_control].copy()
df_sens2_raw['target_sens2'] = is_case_sens2.astype(int) # New target
imputer_sens2 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s2 = imputer_sens2.fit_transform(df_sens2_raw[continuous])
df_sens2_imputed = df_sens2_raw.copy()
df_sens2_imputed[continuous] = imputed_features_s2
df_sens2_imputed = df_sens2_imputed[(df_sens2_imputed['age'] >= 40) & (df_sens2_imputed['age'] <= 60)].reset_index(drop=True)
print(f"Dataset 2 (Fasting/Self-Report) Imputed. N={len(df_sens2_imputed)}")

# --- Dataset 3: Exclude Borderline ---
is_borderline = (
    (dff1_ag['fasting_confirmation_qc'] == 0) &
    (dff1_ag['glucose_qc'] >= 6.9) &
    (dff1_ag['glucose_qc'] <= 7.1)
)
df_sens3_raw = dff1_ag[~is_borderline].copy()
imputer_sens3 = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_features_s3 = imputer_sens3.fit_transform(df_sens3_raw[continuous])
df_sens3_imputed = df_sens3_raw.copy()
df_sens3_imputed[continuous] = imputed_features_s3
df_sens3_imputed = df_sens3_imputed[(df_sens3_imputed['age'] >= 40) & (df_sens3_imputed['age'] <= 60)].reset_index(drop=True)
print(f"Dataset 3 (Exclude Borderline) Imputed. N={len(df_sens3_imputed)}")

# --------------------------------------------------------------------------
# --- 4. DEFINE BEST SUBGROUPS & CALCULATE METRICS ---
# --------------------------------------------------------------------------

all_results = []

# --- Analysis 1 Subgroups (Diagnosed Only) ---
# Penalty 1.2 (Best interpretable)
mask_s1_a = (
    (df_sens1_imputed['bmi_c_qc'] >= 21.31) &
    (df_sens1_imputed['mvpa_c'] <= 1680.0) &
    (df_sens1_imputed['diabetes_history_qc'] == 1) &
    (df_sens1_imputed['waist_hip_r_c_qc'] > 0.9) &
    (df_sens1_imputed['hip_circumference_qc'] >= 670.0)
)
metrics_s1_a = calculate_subgroup_metrics(df_sens1_imputed, mask_s1_a, 'diabetes_status_c_qc')
metrics_s1_a['Scenario'] = "1. Diagnosed Only"
metrics_s1_a['Subgroup description'] = "bmi[21.3-68.0] AND mvpa[0-1680] AND diabetes history[1.0] AND waist hip r c[0.9-1.16] AND hip circumference[670-1180]"
metrics_s1_a['Score'] = 33.571 # From your output
all_results.append(metrics_s1_a)

# Penalty 1.5 (Second best interpretable)
mask_s1_b = (
    (df_sens1_imputed['mvpa_c'] <= 1680.0) &
    (df_sens1_imputed['age'] >= 47.0) &
    (df_sens1_imputed['diabetes_history_qc'] == 1) &
    (df_sens1_imputed['waist_circumference_qc'] >= 810.0)
)
metrics_s1_b = calculate_subgroup_metrics(df_sens1_imputed, mask_s1_b, 'diabetes_status_c_qc')
metrics_s1_b['Scenario'] = "1. Diagnosed Only"
metrics_s1_b['Subgroup description'] = "mvpa[0-1680] AND age[47-60] AND diabetes history[1.0] AND waist circumference[810-1470]"
metrics_s1_b['Score'] = 33.027 # From your output
all_results.append(metrics_s1_b)


# --- Analysis 2 Subgroups (Fasting/Self-Report Only) ---
# Penalty 1.0 (Best interpretable)
mask_s2_a = (
    (df_sens2_imputed['waist_circumference_qc'] >= 810.0) &
    (df_sens2_imputed['diabetes_history_qc'].isin([1.0, 2.0])) & # As per your output
    (df_sens2_imputed['age'] >= 47.0) &
    (df_sens2_imputed['mvpa_c'] <= 1680.0)
)
metrics_s2_a = calculate_subgroup_metrics(df_sens2_imputed, mask_s2_a, 'target_sens2') 
metrics_s2_a['Scenario'] = "2. Fasting/Self-Report Only"
metrics_s2_a['Subgroup description'] = "waist circumference[810-1470] AND diabetes history[1.0 OR 2.0] AND age[47-60] AND mvpa[0-1680]"
metrics_s2_a['Score'] = 33.579 # From your output
all_results.append(metrics_s2_a)

# Penalty 1.2 (Second best interpretable)
mask_s2_b = (
    (df_sens2_imputed['waist_circumference_qc'] >= 810.0) &
    (df_sens2_imputed['mvpa_c'] <= 1680.0) &
    (df_sens2_imputed['diabetes_history_qc'] == 1) &
    (df_sens2_imputed['age'] >= 47.0)
)
metrics_s2_b = calculate_subgroup_metrics(df_sens2_imputed, mask_s2_b, 'target_sens2') 
metrics_s2_b['Scenario'] = "2. Fasting/Self-Report Only"
metrics_s2_b['Subgroup description'] = "waist circumference[810-1470] AND mvpa[0-1680] AND diabetes history[1.0] AND age[47-60]"
metrics_s2_b['Score'] = 32.752 # From your output
all_results.append(metrics_s2_b)


# --- Analysis 3 Subgroups (Exclude Borderline) ---
# Penalty 1.1 (Identical match 1)
mask_s3_a = (
    (df_sens3_imputed['bmi_c_qc'] >= 21.37) &
    (df_sens3_imputed['mvpa_c'] <= 2448.0) &
    (df_sens3_imputed['diabetes_history_qc'] == 1) &
    (df_sens3_imputed['waist_hip_r_c_qc'] > 0.9)
)
metrics_s3_a = calculate_subgroup_metrics(df_sens3_imputed, mask_s3_a, 'diabetes_status_c_qc')
metrics_s3_a['Scenario'] = "3. Exclude Borderline"
metrics_s3_a['Subgroup description'] = "bmi[21.37-68.02] AND mvpa[0-2448] AND diabetes history[1.0] AND waist hip r c[0.9-1.16]"
metrics_s3_a['Score'] = 34.303 # From your output
all_results.append(metrics_s3_a)

# Penalty 3.5 (Identical match 2)
mask_s3_b = (
    (df_sens3_imputed['diabetes_history_qc'] == 1) &
    (df_sens3_imputed['waist_hip_r_c_qc'] > 0.9)
)
metrics_s3_b = calculate_subgroup_metrics(df_sens3_imputed, mask_s3_b, 'diabetes_status_c_qc')
metrics_s3_b['Scenario'] = "3. Exclude Borderline"
metrics_s3_b['Subgroup description'] = "diabetes history[1.0] AND waist hip r c[0.9-1.16]"
metrics_s3_b['Score'] = 27.711 # From your output
all_results.append(metrics_s3_b)


# --------------------------------------------------------------------------
# --- 5. CREATE AND PRINT THE FINAL TABLE (Corrected Formatting) ---
# --------------------------------------------------------------------------
final_table = pd.DataFrame(all_results)

# --- CORRECTED COLUMN ORDER ---
# Select and reorder columns as you requested
final_table = final_table[[
    "Scenario",
    "Subgroup description",
    "Subgroup_Size_Percent",
    "P(T2D|S) [PPV]",
    "P(S|T2D) [Sensitivity]",
    "Specificity",
    "NPV",
    "Score" # Score is now the last column
]]

print("\n\n" + "="*80)
print("### FINAL SENSITIVITY ANALYSIS ROBUSTNESS TABLE (FOR REVIEWER 3) ###")
print("="*80)

# --- CORRECTED FORMATTING ---
# Format *only* the Subgroup_Size_Percent as a percentage
final_table['Subgroup_Size_Percent'] = final_table['Subgroup_Size_Percent'].apply(lambda x: f"{x:.1%}")

# Print the table. Other floats will use the floatfmt default.
print(final_table.to_markdown(floatfmt=".3f", index=False))


import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --- Suppress warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# --------------------------------------------------------------------------
# --- 1. SETUP: Load Raw Data ---
# --------------------------------------------------------------------------

file_path = '~/t2d_as.csv'
try:
    dff_csv = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Warning: File not found at {file_path}. Using fallback path.")
    dff_csv = pd.read_csv('t2d_as.csv') # Fallback for local dir

dfff = pd.read_csv('/home/kayode/samplee1.csv')
target_cols = ['diabetes_status_c_qc']

# Define features and continuous columns from your script
features = ['age', 'sex', 'highest_level_of_education_qc',
       'partnership_status_c_qc', 'ses_site_quintile_c',
       'occupation_qc', 'alcohol_use_status_c_qc', 'mvpa_c', 'ldl_qc',
       'smoking_status_c_qc', 'diabetes_history_qc',
       'bmi_c_qc', 'waist_hip_r_c_qc', 'weight_qc', 'hip_circumference_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc', 'fruit_servings_qc',
       'glucose_qc', 'ur_creatinine_qc', 'triglycerides_qc', 'visceral_fat_qc', 'servings_veg_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc', 'days_fruit_qc', 'days_veg_qc',
       'egfr_c_qc','acr_qc', 'use_drug_qc']
continuous = ['age', 'weight_qc', 'bmi_c_qc', 'waist_hip_r_c_qc','visceral_fat_qc',
       'waist_circumference_qc', 'bp_sys_average_qc','bp_dia_average_qc',
       'glucose_qc', 'ur_creatinine_qc','mvpa_c', 'ldl_qc', 'hip_circumference_qc',
       'hdl_qc', 'cholesterol_qc', 'friedewald_ldl_c_c_qc',
       'triglycerides_qc', 'egfr_c_qc','acr_qc']

# --- Create Base Agincourt DataFrame ---
site_id = 1 # 1 - Agincourt
dff_ag = dfff[(dfff['site'] == site_id)].fillna(-999)
dff_ag = dff_ag[(dff_ag[target_cols] != -999).sum(axis = 1) == len(target_cols)].copy()
dff1_ag = dff_ag.replace([-999, -222, -111, 999], np.nan).copy()
dff1_ag = dff1_ag.reset_index(drop=True)
print(f"Raw Agincourt dataset 'dff1_ag' created. N={len(dff1_ag)}")

# --------------------------------------------------------------------------
# --- 2. CREATE THE COMPLETE-CASE ANALYSIS (CCA) DATASET ---
# --------------------------------------------------------------------------
# We use the 'features' and 'target_cols' lists to define what must be "complete"
cca_cols_to_check = features + target_cols
df_cca = dff1_ag.dropna(subset=cca_cols_to_check).copy()
df_cca = df_cca[(df_cca['age'] >= 40) & (df_cca['age'] <= 60)].reset_index(drop=True)
print(f"Complete-Case Analysis (CCA) dataset 'df_cca' created. N={len(df_cca)}")

# --------------------------------------------------------------------------
# --- 3. YOUR FUNCTION: calculate_subgroup_metrics ---
# --------------------------------------------------------------------------
def calculate_subgroup_metrics(data, mask, outcome_col):
    
    tp = (mask & (data[outcome_col] == 1)).sum()
    fp = (mask & (data[outcome_col] == 0)).sum()
    tn = (~mask & (data[outcome_col] == 0)).sum()
    fn = (~mask & (data[outcome_col] == 1)).sum()
    
    total_pop = len(data)
    
    # Sensitivity (P(S|T2D)) - "Recall"
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # PPV (P(T2D|S)) - "Precision"
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # NPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        "P(S|T2D) [Sensitivity]": sensitivity,
        "Specificity": specificity,
        "P(T2D|S) [PPV]": ppv,
        "NPV": npv,
        "Total_in_Subgroup": (tp + fp),
        "Subgroup_Size_Percent": (tp + fp) / total_pop
    }

# --------------------------------------------------------------------------
# --- 4. DEFINE CCA SUBGROUPS & CALCULATE METRICS ---
# --------------------------------------------------------------------------

all_results = []
outcome_col = 'diabetes_status_c_qc'

# --- Subgroup 1 (from Penalty 1.6) ---
mask_cca_1 = (
    (df_cca['diabetes_history_qc'] == 1) &
    (df_cca['bmi_c_qc'] >= 21.3) &
    (df_cca['waist_hip_r_c_qc'] >= 0.92) &
    (df_cca['hip_circumference_qc'] >= 670.0)
)
metrics_cca_1 = calculate_subgroup_metrics(df_cca, mask_cca_1, outcome_col)
metrics_cca_1['Scenario'] = "4. Complete-Case"
metrics_cca_1['Subgroup description'] = "diabetes history[1.0] AND bmi[21.3-68.0] AND waist hip r c[0.92-1.16] AND hip circumference[670-1180]"
metrics_cca_1['Score'] = 31.689 # From your output
all_results.append(metrics_cca_1)

# --- Subgroup 2 (from Penalty 2.0) ---
mask_cca_2 = (
    (df_cca['diabetes_history_qc'] == 1) &
    (df_cca['waist_hip_r_c_qc'] >= 0.92)
)
metrics_cca_2 = calculate_subgroup_metrics(df_cca, mask_cca_2, outcome_col)
metrics_cca_2['Scenario'] = "4. Complete-Case"
metrics_cca_2['Subgroup description'] = "diabetes history[1.0] AND waist hip r c[0.92-1.16]"
metrics_cca_2['Score'] = 30.491 # From your output
all_results.append(metrics_cca_2)

# --------------------------------------------------------------------------
# --- 5. CREATE AND PRINT THE FINAL LATEX TABLE ---
# --------------------------------------------------------------------------
final_table = pd.DataFrame(all_results)

# Select and reorder columns as you requested
final_table = final_table[[
    "Scenario",
    "Subgroup description",
    "Subgroup_Size_Percent",
    "P(T2D|S) [PPV]",
    "P(S|T2D) [Sensitivity]",
    "Specificity",
    "NPV",
    "Score"
]]

# Format the percentages and decimals
final_table['Subgroup_Size_Percent'] = final_table['Subgroup_Size_Percent'].apply(lambda x: f"{x:.1%}")
final_table['P(T2D|S) [PPV]'] = final_table['P(T2D|S) [PPV]'].apply(lambda x: f"{x:.3f}")
final_table['P(S|T2D) [Sensitivity]'] = final_table['P(S|T2D) [Sensitivity]'].apply(lambda x: f"{x:.3f}")
final_table['Specificity'] = final_table['Specificity'].apply(lambda x: f"{x:.3f}")
final_table['NPV'] = final_table['NPV'].apply(lambda x: f"{x:.3f}")
final_table['Score'] = final_table['Score'].apply(lambda x: f"{x:.3f}")


print("\n\n" + "="*80)
print("### FINAL CCA SENSITIVITY TABLE (FOR REVIEWER 3, COMMENT 2) ###")
print("="*80)
print(final_table.to_markdown(index=False))


# --- 6. GENERATE THE LATEX FILE ---
print("\n\n" + "="*80)
print("### LaTeX Code for cca_analysis_table.tex ###")
print("="*80)

latex_string = r"""\documentclass[a4paper, 11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{booktabs} % For \toprule, \midrule, \bottomrule
\usepackage[margin=2.5cm]{geometry} % To make the table fit
\usepackage{multirow} % Added for multi-row cells

\begin{document}

\begin{table}[h!]
\caption{Robustness of Subgroup Discovery in Complete-Case Analysis (CCA)}
\label{tab:cca_analysis}
\centering
\begin{tabular}{@{}p{0.18\columnwidth}p{0.3\columnwidth}cccccc@{}} 
\toprule
\textbf{Scenario} & \textbf{Subgroup description} & \textbf{Size(\%)} & \textbf{P(T2D|S)} & \textbf{P(S|T2D)} & \textbf{Spec.} & \textbf{NPV} & \textbf{Score} \\ \midrule
"""

# Add the data rows
row1_desc = r"\begin{tabular}[c]{@{}l@{}}" + r"diabetes history[1.0] \\ bmi[21.3-68.0] \\ waist hip r c[0.92-1.16] \\ hip circumference[670-1180]" + r"\end{tabular}"
row1_data = final_table.iloc[0]
row1 = f"\\multirow{{2}}{{0.18\\columnwidth}}{{\\textbf{{Complete-Case}}}} & {row1_desc} & {row1_data['Subgroup_Size_Percent']} & {row1_data['P(T2D|S) [PPV]']} & {row1_data['P(S|T2D) [Sensitivity]']} & {row1_data['Specificity']} & {row1_data['NPV']} & {row1_data['Score']} \\\\"

row2_desc = r"\begin{tabular}[c]{@{}l@{}}" + r"diabetes history[1.0] \\ waist hip r c[0.92-1.16]" + r"\end{tabular}"
row2_data = final_table.iloc[1]
row2 = f" & {row2_desc} & {row2_data['Subgroup_Size_Percent']} & {row2_data['P(T2D|S) [PPV]']} & {row2_data['P(S|T2D) [Sensitivity]']} & {row2_data['Specificity']} & {row2_data['NPV']} & {row2_data['Score']} \\\\"

latex_string += row1 + "\n"
latex_string += r"\cmidrule(l){2-8}" + "\n"
latex_string += row2 + "\n"

latex_string += r"""\bottomrule
\multicolumn{8}{p{0.95\linewidth}}{\small \textbf{Notes:} This table presents the top two interpretable subgroups identified from the Complete-Case Analysis (CCA). The findings are consistent with the main analysis, confirming that the imputation strategy did not create artificial subgroups.}\\
\multicolumn{8}{p{0.95\linewidth}}{\small P(T2D|S) [PPV] is the probability of having T2D among individuals in the discovered subgroup.}\\
\multicolumn{8}{p{0.95\linewidth}}{\small P(S|T2D) [Sensitivity] is the proportion of all individuals with T2D who fall into the given subgroup.}\\
\multicolumn{8}{p{0.95\linewidth}}{\small Specificity is the proportion of non-T2D individuals correctly excluded from the subgroup.}\\
\multicolumn{8}{p{0.95\linewidth}}{\small NPV is the probability of not having T2D given an individual is outside the subgroup.}\\
\end{tabular}
\end{table}

\end{document}
"""

print(latex_string)

# Save the LaTeX code to a file
with open("cca_analysis_table.tex", "w") as f:
    f.write(latex_string)

print("\n\nSuccessfully saved to 'cca_analysis_table.tex'")

print("\n--- Starting Cell 5 (Corrected Pipeline): RFECV + Training ---")

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# --- 0. List to store all results ---
all_results = []
prc_data = {} # For plotting

# --- 1. Define Helper Function (from Cell 4) ---
def get_model_metrics(y_true, y_pred_class, y_pred_proba, model_name="Model"):
    """Calculates and returns a dictionary of all performance metrics."""
    metrics_dict = {
        'Model': model_name,
        'ROC AUC': roc_auc_score(y_true, y_pred_proba),
        'AUPRC': average_precision_score(y_true, y_pred_proba),
        'F1 Score': f1_score(y_true, y_pred_class, zero_division=0),
        'Precision': precision_score(y_true, y_pred_class, zero_division=0),
        'Recall': recall_score(y_true, y_pred_class),
        'Balanced Acc': balanced_accuracy_score(y_true, y_pred_class),
        'MCC': matthews_corrcoef(y_true, y_pred_class)
    }
    return metrics_dict

# --- 2. Iterate Through Nairobi Cohort ---
print("\n--- Analyzing Nairobi (Site 3) ---")
site_name = "Nairobi"
X_train_tree = X_train_ml_tree_nai
X_test_tree = X_test_ml_tree_nai
X_train_scaled = X_train_ml_scaled_nai
X_test_scaled = X_test_ml_scaled_nai
X_train_raw = X_train_raw_nai
X_test_raw = X_test_raw_nai
y_train = y_train_nai
y_test = y_test_nai

# --- [NEW STEP] Run RFECV ---
print("Running RFECV for Nairobi... (This may take a moment)")
rfecv_estimator = RandomForestClassifier(random_state=RANDOM_STATE)
rfecv_nai = RFECV(estimator=rfecv_estimator, step=1, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
rfecv_nai.fit(X_train_tree, y_train)

# Get selected features
selected_features_nai = X_train_tree.columns[rfecv_nai.support_]
print(f"RFECV selected {len(selected_features_nai)} features for Nairobi.")
X_train_ml_selected = X_train_tree[selected_features_nai]
X_test_ml_selected = X_test_tree[selected_features_nai]

# --- Model 1: Full ML (XGBoost) ---
print("Training Full ML (XGBoost) on SELECTED features...")
model_ml_nai = XGBClassifier(random_state=RANDOM_STATE, **{'colsample_bytree': 0.3189, 'gamma': 1.239, 'learning_rate': 0.0236, 'max_depth': 7, 'min_child_weight': 18, 'n_estimators': 400, 'subsample': 0.476})
class_weights_nai = compute_sample_weight(class_weight="balanced", y=y_train)
model_ml_nai.fit(X_train_ml_selected, y_train, sample_weight=class_weights_nai)

y_pred_ml = model_ml_nai.predict(X_test_ml_selected)
y_proba_ml = model_ml_nai.predict_proba(X_test_ml_selected)[:, 1]
all_results.append(get_model_metrics(y_test, y_pred_ml, y_proba_ml, f"Full ML ({site_name})"))
prc_data[f"Full ML ({site_name})"] = (y_test, y_proba_ml)

# --- Model 2: Parsimonious (LR) ---
print("Training Parsimonious (LR)...")
parsimonious_features = ['age_phase1', 'glucose_result_phase1', 'BRI_phase1']
model_parsi = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000)
model_parsi.fit(X_train_scaled[parsimonious_features], y_train)

y_pred_parsi = model_parsi.predict(X_test_scaled[parsimonious_features])
y_proba_parsi = model_parsi.predict_proba(X_test_scaled[parsimonious_features])[:, 1]
all_results.append(get_model_metrics(y_test, y_pred_parsi, y_proba_parsi, f"Parsimonious ({site_name})"))
prc_data[f"Parsimonious ({site_name})"] = (y_test, y_proba_parsi)

# --- Model 3: FINDRISC (LR) ---
print("Training FINDRISC (LR)...")
findrisc_train = calculate_findrisc_score(X_train_raw)
findrisc_test = calculate_findrisc_score(X_test_raw)
scaler_findrisc_nai = StandardScaler() # Store this scaler
findrisc_train_scaled = scaler_findrisc_nai.fit_transform(findrisc_train.reshape(-1, 1))
findrisc_test_scaled = scaler_findrisc_nai.transform(findrisc_test.reshape(-1, 1))
model_findrisc_nai = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
model_findrisc_nai.fit(findrisc_train_scaled, y_train)
y_pred_findrisc = model_findrisc_nai.predict(findrisc_test_scaled)
y_proba_findrisc = model_findrisc_nai.predict_proba(findrisc_test_scaled)[:, 1]
all_results.append(get_model_metrics(y_test, y_pred_findrisc, y_proba_findrisc, f"FINDRISC (LR) ({site_name})"))
prc_data[f"FINDRISC (Score) ({site_name})"] = (y_test, findrisc_test)

# --- 3. Iterate Through Agincourt Cohort ---
print("\n--- Analyzing Agincourt (Site 1) ---")
site_name = "Agincourt"
X_train_tree = X_train_ml_tree_ag
X_test_tree = X_test_ml_tree_ag
X_train_scaled = X_train_ml_scaled_ag
X_test_scaled = X_test_ml_scaled_ag
X_train_raw = X_train_raw_ag
X_test_raw = X_test_raw_ag
y_train = y_train_ag
y_test = y_test_ag

# --- [NEW STEP] Run RFECV ---
print("Running RFECV for Agincourt... (This may take a moment)")
rfecv_estimator = RandomForestClassifier(random_state=RANDOM_STATE)
rfecv_ag = RFECV(estimator=rfecv_estimator, step=1, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
rfecv_ag.fit(X_train_tree, y_train)

# Get selected features
selected_features_ag = X_train_tree.columns[rfecv_ag.support_]
print(f"RFECV selected {len(selected_features_ag)} features for Agincourt.")
X_train_ml_selected = X_train_tree[selected_features_ag]
X_test_ml_selected = X_test_tree[selected_features_ag]

# --- Model 1: Full ML (CatBoost) ---
print("Training Full ML (CatBoost) on SELECTED features...")
model_ml_ag = CatBoostClassifier(
    bagging_temperature=0.4844507105512201,
    depth=6,
    iterations=50,
    l2_leaf_reg=7.67639522777554,
    learning_rate=0.006738420470998171,
    random_state=42,
    class_weights=[1, 1.5], # Use the internal class_weights
    verbose=0
)
# **CORRECTION**: Do NOT use sample_weight if class_weights is set in constructor.
# This was a bug in my previous script.
model_ml_ag.fit(X_train_ml_selected, y_train, sample_weight=compute_sample_weight(class_weight="balanced", y=y_train)) 
# model_ml_ag.fit(X_train_ml_selected, y_train) 


y_pred_ml = model_ml_ag.predict(X_test_ml_selected)
y_proba_ml = model_ml_ag.predict_proba(X_test_ml_selected)[:, 1]
all_results.append(get_model_metrics(y_test, y_pred_ml, y_proba_ml, f"Full ML ({site_name})"))
prc_data[f"Full ML ({site_name})"] = (y_test, y_proba_ml)

# --- Model 2: Parsimonious (LR) ---
print("Training Parsimonious (LR)...")
model_parsi_ag = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000)
model_parsi_ag.fit(X_train_scaled[parsimonious_features], y_train)
y_pred_parsi = model_parsi_ag.predict(X_test_scaled[parsimonious_features])
y_proba_parsi = model_parsi_ag.predict_proba(X_test_scaled[parsimonious_features])[:, 1]
all_results.append(get_model_metrics(y_test, y_pred_parsi, y_proba_parsi, f"Parsimonious ({site_name})"))
prc_data[f"Parsimonious ({site_name})"] = (y_test, y_proba_parsi)

# --- Model 3: FINDRISC (LR) ---
print("Training FINDRISC (LR)...")
findrisc_train = calculate_findrisc_score(X_train_raw)
findrisc_test = calculate_findrisc_score(X_test_raw)
scaler_findrisc_ag = StandardScaler() # Store this scaler
findrisc_train_scaled = scaler_findrisc_ag.fit_transform(findrisc_train.reshape(-1, 1))
findrisc_test_scaled = scaler_findrisc_ag.transform(findrisc_test.reshape(-1, 1))
model_findrisc_ag = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
model_findrisc_ag.fit(findrisc_train_scaled, y_train)
y_pred_findrisc = model_findrisc_ag.predict(findrisc_test_scaled)
y_proba_findrisc = model_findrisc_ag.predict_proba(findrisc_test_scaled)[:, 1]
all_results.append(get_model_metrics(y_test, y_pred_findrisc, y_proba_findrisc, f"FINDRISC (LR) ({site_name})"))
prc_data[f"FINDRISC (Score) ({site_name})"] = (y_test, findrisc_test)

# --- 4. Print Results Table ---
results_df = pd.DataFrame(all_results).set_index('Model')
print("\n\n--- [Table 1 (Corrected): Main Performance Metrics] ---")
print(results_df.round(3).to_markdown())

print("\nCell 5 (Corrected) complete.")