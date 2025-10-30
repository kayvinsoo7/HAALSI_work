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