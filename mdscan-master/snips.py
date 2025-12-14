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

dff_ag = dff_ag[(dff_ag['age'] >= 40) & (dff_ag['age'] <= 60)]
dff_nai = dff_nai[(dff_nai['age'] >= 40) & (dff_nai['age'] <= 60)]
dff_nan = dff_nan[(dff_nan['age'] >= 40) & (dff_nan['age'] <= 60)]
dff_dim = dff_dim[(dff_dim['age'] >= 40) & (dff_dim['age'] <= 60)]

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
dff_2 = dff_ag.copy()

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
penalty_values = [0.4, 0.5, 1.0, 1.2, 1.5, 3.0,12.5]
# penalty_values = [1.0, 1.5, 2, 2.5, 3.0,3.5]
# For complete case
# penalty_values = [0.25, 1.0, 1.6, 1.8, 2.0,3.5]
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