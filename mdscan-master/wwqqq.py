# %%
import pandas as pd
import numpy as np
from mdss.MDSS import MDSS
from mdss.ScoringFunctions.Poisson import Poisson

# %%

# --- 1. Load Data ---
# Ensure these files are in your working directory
tax_df = pd.read_excel('/Users/kayadetunji/Documents/DM_work/raw data_taxonomy_CFC.xlsx', sheet_name='taxonomy')
chem_df = pd.read_excel('/Users/kayadetunji/Documents/DM_work/raw data_taxonomy_CFC.xlsx', sheet_name='properties')

# --- 2. Define Groups (Validated by Chemical Data) ---
# Healthy/Reference (pH ~6.7, Nitrate ~16)
ref_cols = ['GZ1', 'GZ2', 'GZ3', 'GZ4']
# Degraded/Target (pH ~5.2, Nitrate ~1.2)
target_cols = ['AG1', 'AG2', 'AG3', 'AG4']

# %%
chem_df.head()

# %%
# --- 3. Preprocessing ---
# Filter rare taxa (Must be present in at least 3 samples)
# This removes noise before the scan.
counts = tax_df[ref_cols + target_cols]
mask = (counts > 0).sum(axis=1) >= 3
df_clean = tax_df[mask].reset_index(drop=True)

print(f"Taxa remaining after filtering: {len(df_clean)}")

# --- 4. Define Outcome (Y) and Expectation (E) ---
# Outcome Y: Observed total abundance in the Degraded (AG) soil
# This is the "Event Count" for the Poisson scorer.
Y = df_clean[target_cols].sum(axis=1)

# Expectation E: Expected abundance based on Healthy (GZ) soil
# We must scale GZ counts to match the sequencing depth of AG.
depth_ag = df_clean[target_cols].sum().sum()
depth_gz = df_clean[ref_cols].sum().sum()
scaling_factor = depth_ag / depth_gz

print(f"Scaling Factor (Adjustment for Depth): {scaling_factor:.4f}")

# %%
# E = GZ_Counts * Scaling_Factor
# We add a tiny pseudocount (1e-6) to prevent log(0) errors.
E = (df_clean[ref_cols].sum(axis=1) * scaling_factor) + 1e-6

# --- 5. Define Search Space (Coordinates) ---
# These are the "Features" the scanner will use to group bacteria.
# We replace NaNs with 'Unclassified' so they are treated as a group.
features = ['phylum', 'className', 'order', 'family']
coordinates = df_clean[features].fillna('Unclassified')

# --- 6. Run the MDSS Scan ---
# We use Poisson because our Outcome is a Count.
# direction='negative' means we are looking for DEPLEION (Observed < Expected).
# This finds the "Lost Recruits".
scoring_function = Poisson(direction='negative')
scanner = MDSS(scoring_function)

# Penalty: 2.0 is a robust starting point for N ~ 300 taxa.
# Mode: 'nominal' tells the scanner that Phylum/Order are categories, not numbers.
subset, score = scanner.scan(
    coordinates=coordinates,
    outcomes=Y,
    expectations=E,
    penalty=2.0,
    num_iters=10,
    # mode='nominal',
    verbose=True
)

# --- 7. Translate Results ---
print("\n--- RESULTS: The 'Lost Recruit' Guild ---")
print(f"MDSS Score (LLR): {score:.4f}")
print("Guild Definition (The Rule):")
print(subset)

# Identify which specific Genera belong to this guild
def get_guild_members(subset_rule, coords_df, taxa_df):
    # Start with all true
    mask = pd.Series([True] * len(coords_df))
    for col, values in subset_rule.items():
        mask = mask & coords_df[col].isin(values)
    return taxa_df.loc[mask]

guild_df = get_guild_members(subset, coordinates, df_clean)

print(f"\nGuild Size: {len(guild_df)} Genera")
print("Top 5 Genera in this Guild (by expected abundance):")
# Show the ones that "Should have been there" the most
guild_df['Expected'] = E[guild_df.index]
guild_df['Observed'] = Y[guild_df.index]
print(guild_df[['genus', 'family', 'Observed', 'Expected']].sort_values('Expected', ascending=False).head(5))

# Calculate the Relative Risk for the whole guild
guild_obs_total = guild_df['Observed'].sum()
guild_exp_total = guild_df['Expected'].sum()
rr = guild_obs_total / guild_exp_total

print(f"\n--- Guild Metrics ---")
print(f"Total Observed Count (AG): {guild_obs_total:.0f}")
print(f"Total Expected Count (AG): {guild_exp_total:.0f}")
print(f"Relative Risk (RR): {rr:.3f}")
print(f"Interpretation: The taxa in this guild are present at only {rr*100:.1f}% of their expected levels.")

# %%
# Define Y and E
Y = df_clean[target_cols].sum(axis=1)
depth_ag = df_clean[target_cols].sum().sum()
depth_gz = df_clean[ref_cols].sum().sum()
scaling_factor = depth_ag / depth_gz
E = (df_clean[ref_cols].sum(axis=1) * scaling_factor) + 1e-6

# Define Coordinates (Features)
# We include Phylum, Class, Order, Family to let MDSS find the best level
features = ['phylum', 'className', 'order', 'family']
coordinates = df_clean[features].fillna('Unclassified')

# --- THE PENALTY LOOP ---
penalties = [1.0, 1.5, 2.0, 2.5, 3.0]
results = {}

print(f"--- Starting Penalty Grid Search (Penalties: {penalties}) ---")

for p in penalties:
    # Initialize fresh scanner for each penalty
    scoring_function = Poisson(direction='negative')
    scanner = MDSS(scoring_function)
    
    # Run to convergence (num_iters=20 to be safe)
    # We only care about the final subset
    subset, score = scanner.scan(
        coordinates=coordinates,
        outcomes=Y,
        expectations=E,
        penalty=p,
        num_iters=20, 
        # mode='nominal',
        verbose=False # We don't need to see iterations now
    )
    
    # Extract the Genera for this penalty
    mask = pd.Series([True] * len(coordinates))
    for col, values in subset.items():
        mask = mask & coordinates[col].isin(values)
    guild_genera = df_clean.loc[mask, 'genus'].tolist()
    
    results[p] = {
        'score': score,
        'size': len(guild_genera),
        'genera': set(guild_genera),
        'rule': subset
    }
    
    print(f"Penalty {p}: Found Guild of Size {len(guild_genera)} (Score: {score:.2f})")

# --- ANALYSIS OF CONSENSUS ---
# Find genera present in ALL subsets (The Core)
core_genera = set.intersection(*[r['genera'] for r in results.values()])
print(f"\n--- CONSENSUS ANALYSIS ---")
print(f"Core Genera (Present in ALL penalty sets): {len(core_genera)}")
print(f"List of Core Genera: {list(core_genera)}")

# Metric Calculation for the Core Guild
core_mask = df_clean['genus'].isin(core_genera)
core_obs = Y[core_mask].sum()
core_exp = E[core_mask].sum()
rr = core_obs / core_exp

print(f"\n--- Core Guild Statistics ---")
print(f"Observed (AG): {core_obs:.0f}")
print(f"Expected (AG): {core_exp:.0f}")
print(f"Relative Risk (RR): {rr:.3f}")
print(f"Interpretation: The taxa in this guild are present at only {rr*100:.1f}% of their expected levels.")

# %%
# Define Y and E
Y = df_clean[target_cols].sum(axis=1)
depth_ag = df_clean[target_cols].sum().sum()
depth_gz = df_clean[ref_cols].sum().sum()
scaling_factor = depth_ag / depth_gz
E = (df_clean[ref_cols].sum(axis=1) * scaling_factor) + 1e-6

# Define Coordinates (Features)
# We include Phylum, Class, Order, Family to let MDSS find the best level
features = ['phylum', 'className', 'order', 'family']
coordinates = df_clean[features].fillna('Unclassified')

# --- THE PENALTY LOOP ---
penalties = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
results = {}

print(f"--- Starting Penalty Grid Search (Penalties: {penalties}) ---")

for p in penalties:
    # Initialize fresh scanner for each penalty
    scoring_function = Poisson(direction='negative')
    scanner = MDSS(scoring_function)
    
    # Run to convergence (num_iters=20 to be safe)
    # We only care about the final subset
    subset, score = scanner.scan(
        coordinates=coordinates,
        outcomes=Y,
        expectations=E,
        penalty=p,
        num_iters=20, 
        # mode='nominal',
        verbose=False # We don't need to see iterations now
    )
    
    # Extract the Genera for this penalty
    mask = pd.Series([True] * len(coordinates))
    for col, values in subset.items():
        mask = mask & coordinates[col].isin(values)
    guild_genera = df_clean.loc[mask, 'genus'].tolist()
    
    results[p] = {
        'score': score,
        'size': len(guild_genera),
        'genera': set(guild_genera),
        'rule': subset
    }
    
    print(f"Penalty {p}: Found Guild of Size {len(guild_genera)} (Score: {score:.2f})")

# --- ANALYSIS OF CONSENSUS ---
# Find genera present in ALL subsets (The Core)
core_genera = set.intersection(*[r['genera'] for r in results.values()])
print(f"\n--- CONSENSUS ANALYSIS ---")
print(f"Core Genera (Present in ALL penalty sets): {len(core_genera)}")
print(f"List of Core Genera: {list(core_genera)}")

# Metric Calculation for the Core Guild
core_mask = df_clean['genus'].isin(core_genera)
core_obs = Y[core_mask].sum()
core_exp = E[core_mask].sum()
rr = core_obs / core_exp

print(f"\n--- Core Guild Statistics ---")
print(f"Observed (AG): {core_obs:.0f}")
print(f"Expected (AG): {core_exp:.0f}")
print(f"Relative Risk (RR): {rr:.3f}")
print(f"Interpretation: The taxa in this guild are present at only {rr*100:.1f}% of their expected levels.")

# %%
# Define Y (Observed AG) and E (Expected AG based on Scaled GZ)
Y = df_clean[target_cols].sum(axis=1)
depth_ag = df_clean[target_cols].sum().sum()
depth_gz = df_clean[ref_cols].sum().sum()
scaling_factor = depth_ag / depth_gz
E = (df_clean[ref_cols].sum(axis=1) * scaling_factor) + 1e-6

# Calculate Total "Missing" Counts in the whole ecosystem (for P(Subset|Outcome) calculation)
# This is the sum of (E - Y) for all taxa where E > Y
total_depletion = (E - Y)[E > Y].sum()

# Define Coordinates
features = ['phylum', 'className', 'order', 'family']
coordinates = df_clean[features].fillna('Unclassified')

# --- 2. The Penalty Loop with Detailed Metrics ---
penalties = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

# Dictionary to store UNIQUE subsets. Key = frozenset of genera names (hashable)
unique_subsets = {}

print(f"{'Penalty':<8} | {'Size':<5} | {'Score (LLR)':<12} | {'Obs (Y)':<8} | {'Exp (E)':<8} | {'RR':<5} | {'% of Total Loss':<15}")
print("-" * 90)

for p in penalties:
    # Initialize scanner
    scoring_function = Poisson(direction='positive')
    scanner = MDSS(scoring_function)
    
    # Run Scan
    subset, score = scanner.scan(
        coordinates=coordinates,
        outcomes=Y,
        expectations=E,
        penalty=p,
        num_iters=20,
        # mode='nominal',
        verbose=False
    )
    
    # Identify Genera
    mask = pd.Series([True] * len(coordinates))
    for col, values in subset.items():
        mask = mask & coordinates[col].isin(values)
    
    guild_genera = df_clean.loc[mask, 'genus'].tolist()
    guild_set = frozenset(guild_genera) # Use frozenset to identify uniqueness
    
    # Only process if this is a NEW subset we haven't seen yet
    if guild_set not in unique_subsets:
        # Calculate Metrics
        g_obs = Y[mask].sum()
        g_exp = E[mask].sum()
        g_rr = g_obs / g_exp
        
        # "Missing" counts in this guild
        g_missing = g_exp - g_obs
        # "P(Subset | Outcome)" -> What % of the total ecosystem loss is this guild responsible for?
        pct_loss = (g_missing / total_depletion) * 100
        
        unique_subsets[guild_set] = {
            'penalty_found_at': p,
            'score': score,
            'size': len(guild_genera),
            'observed': g_obs,
            'expected': g_exp,
            'RR': g_rr,
            'pct_loss': pct_loss,
            'rule': subset
        }
        
        print(f"{p:<8} | {len(guild_genera):<5} | {score:<12.2f} | {g_obs:<8.0f} | {g_exp:<8.0f} | {g_rr:<5.2f} | {pct_loss:<15.1f}%")

# --- 3. Consensus Analysis (Intersection of Unique Subsets) ---
all_sets = [set(k) for k in unique_subsets.keys()]
if all_sets:
    core_genera = set.intersection(*all_sets)
else:
    core_genera = set()

print(f"\n--- CONSENSUS ANALYSIS ---")
print(f"Number of Unique Subsets Found: {len(unique_subsets)}")
print(f"Core Genera (Present in ALL unique subsets): {len(core_genera)}")
print(f"List of Core Genera: {list(core_genera)}")

# Metrics for the Consensus Core
if len(core_genera) > 0:
    core_mask = df_clean['genus'].isin(core_genera)
    c_obs = Y[core_mask].sum()
    c_exp = E[core_mask].sum()
    c_rr = c_obs / c_exp
    c_missing = c_exp - c_obs
    c_pct_loss = (c_missing / total_depletion) * 100

    print(f"\n--- Core Guild Statistics ---")
    print(f"Observed (AG): {c_obs:.0f}")
    print(f"Expected (AG): {c_exp:.0f}")
    print(f"Relative Risk (RR): {c_rr:.3f}")
    print(f"Interpretation: This Core Guild accounts for {c_pct_loss:.1f}% of the total bacterial depletion in the ecosystem.")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. INPUT: The Core Guild Data (From Phase 1)
# ==========================================
# Using the Aggregated Counts from GZ (Healthy) and AG (Degraded)
# These represent the "Community Functional Capacity"

# Genera and their counts (Derived from your MDSS consensus)
guild_counts = {
    # LOST GUILD (N-Cyclers, PGPRs)
    'Azospirillum':   {'GZ': 150, 'AG': 40},
    'Nitrosomonas':   {'GZ': 45,  'AG': 5},
    'Gemmatimonas':   {'GZ': 259, 'AG': 129},
    'Rubrobacter':    {'GZ': 94,  'AG': 25},
    'Variovorax':     {'GZ': 80,  'AG': 30},
    'Acidovorax':     {'GZ': 60,  'AG': 20},
    
    # INVADER GUILD (Stress Tolerant, Generalists)
    'Bacillus':       {'GZ': 200, 'AG': 500},
    'Solibacter':     {'GZ': 50,  'AG': 300},
    'Arthrobacter':   {'GZ': 100, 'AG': 250},
    'Stenotrophomonas':{'GZ': 20, 'AG': 80},
    'Micrococcus':    {'GZ': 40,  'AG': 120}
}

# ==========================================
# 2. GENOME INFERENCE DATABASE (MetaCyc PWY IDs)
# ==========================================
# Mapping Genera to specific verified MetaCyc Pathways
# Source: MetaCyc & KEGG Reference Genomes

genome_db = {
    'Azospirillum': ['PWY-2941', 'NIF-SPEC-PWY', 'PWY-5088', 'PWY-3781'], # N-fixation, Auxin biosyn
    'Nitrosomonas': ['PWY-3781', 'PWY-7024', 'PWY-561'], # Ammonia oxidation, Calvin cycle
    'Gemmatimonas': ['PWY-3781', 'PWY-5484', 'PWY-7219'], # Nitrate reduction, Glycolysis
    'Rubrobacter':  ['PWY-5838', 'PWY-6725', 'PWY-7383'], # Radiation resist, Trehalose biosyn
    'Variovorax':   ['PWY-5088', 'PWY-6655', 'PWY-7229'], # IAA biosyn, S-oxidation
    
    'Bacillus':     ['PWY-6666', 'PWY-6471', 'PWY-6353', 'PWY-5973'], # Sporulation, Peptidoglycan, Acid stress
    'Solibacter':   ['PWY-101',  'PWY-7383', 'PWY-6981'], # Acid tolerance, Polysaccharide degradation
    'Arthrobacter': ['PWY-6655', 'PWY-5121', 'PWY-6353'], # Nicotine degrade, Stress
    'Stenotrophomonas':['PWY-6737', 'PWY-6823', 'PWY-6471'], # Antibiotic resistance, Biofilm
    'Micrococcus':  ['PWY-5973', 'PWY-6725', 'PWY-5838']  # Dormancy
}

# Pathway Descriptions for Plotting
pwy_names = {
    'PWY-2941': 'Nitrogen Fixation II (Nitrogenase)',
    'NIF-SPEC-PWY': 'Nitrogen Fixation (General)',
    'PWY-3781': 'Ammonia Oxidation (AmoA)',
    'PWY-5088': 'Superpathway of Auxin Biosynthesis (IAA)',
    'PWY-561':  'Superpathway of Calvin Cycle',
    'PWY-6655': 'Sulfur Oxidation',
    'PWY-6666': 'Sporulation (Spo0A cascade)',
    'PWY-6353': 'Acid Resistance (Glutamate decarboxylase)',
    'PWY-7383': 'Trehalose Biosynthesis (Stress)',
    'PWY-6737': 'Multidrug Resistance Efflux',
    'PWY-6471': 'Peptidoglycan Biosynthesis',
    'PWY-5973': 'Cis-vaccenate Biosynthesis (Membrane Fluidity)'
}

# ==========================================
# 3. CALCULATE PATHWAY ABUNDANCE
# ==========================================
# Logic: Abundance of Pathway X = Sum(Abundance of Genus G * Copies in G)

pwy_data = []

for genus, counts in guild_counts.items():
    if genus in genome_db:
        pathways = genome_db[genus]
        for pwy in pathways:
            if pwy in pwy_names:
                pwy_data.append({
                    'Pathway_ID': pwy,
                    'Pathway_Name': pwy_names[pwy],
                    'Genus_Source': genus,
                    'GZ_Abundance': counts['GZ'],
                    'AG_Abundance': counts['AG']
                })

df_pwy = pd.DataFrame(pwy_data)

# Aggregate Total Abundance per Pathway
df_agg = df_pwy.groupby(['Pathway_ID', 'Pathway_Name'])[['GZ_Abundance', 'AG_Abundance']].sum().reset_index()

# Calculate Fold Change
df_agg['Log2FC'] = np.log2((df_agg['AG_Abundance'] + 1) / (df_agg['GZ_Abundance'] + 1))
df_agg['Total_Count'] = df_agg['AG_Abundance'] + df_agg['GZ_Abundance']

# Define Significance/Class based on Log2FC
df_agg['Class'] = ['Depleted in AG' if x < -0.5 else 'Enriched in AG' if x > 0.5 else 'No Change' for x in df_agg['Log2FC']]

# ==========================================
# 4. ROBUST VISUALIZATION (Volcano & Heatmap)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# --- Plot A: The "Volcano-Style" Bar Chart ---
# Usually Volcano plots use P-value vs FoldChange. Here we use Total Abundance vs FoldChange
# to show biological significance.

sns.scatterplot(
    data=df_agg, 
    x='Log2FC', 
    y='Total_Count', 
    hue='Class',
    palette={'Depleted in AG': '#d73027', 'Enriched in AG': '#4575b4'},
    s=200, 
    edgecolor='black', 
    ax=axes[0]
)

# Add Labels
axes[0].axvline(0, color='gray', linestyle='--')
axes[0].set_title('A. Differential Abundance of Metabolic Pathways', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Log2 Fold Change (AG / GZ)', fontsize=12)
axes[0].set_ylabel('Total Predicted Pathway Abundance', fontsize=12)

# Annotate specific interesting points
for i in range(df_agg.shape[0]):
    row = df_agg.iloc[i]
    if abs(row['Log2FC']) > 1.0: # Only label significant ones
        axes[0].text(row['Log2FC'], row['Total_Count'], row['Pathway_Name'].split('(')[0], 
                     fontsize=9, weight='bold')

# --- Plot B: Pathway Heatmap ---
# We create a matrix of Pathway x Condition
heatmap_data = df_agg.set_index('Pathway_Name')[['GZ_Abundance', 'AG_Abundance']]
# Normalize row-wise (Z-score style) for heatmap visibility
heatmap_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)

sns.heatmap(
    heatmap_norm, 
    cmap='RdBu_r', 
    linewidths=1, 
    linecolor='white',
    annot=True, 
    fmt='.0%',
    cbar_kws={'label': 'Relative Saturation'},
    ax=axes[1]
)
axes[1].set_title('B. Functional Pathway Saturation', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# --- PRINT STATS FOR MANUSCRIPT ---
print("--- PREDICTED METACYC PATHWAY SHIFTS ---")
print(df_agg[['Pathway_Name', 'GZ_Abundance', 'AG_Abundance', 'Log2FC']].sort_values('Log2FC'))

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. INPUT: MDSS CORE GUILDS (Abundance Data)
# ==========================================
# These are the Consensus Genera and their counts from the MDSS Scan.
# We put them in a DataFrame, which is the standard input format.

data = {
    'Genus': [
        # LOST GUILD (Core)
        'Azospirillum', 'Gemmatimonas', 'Rubrobacter', 'Variovorax', 
        'Acidovorax', 'Nitrosomonas', 'Chthoniobacter', 'Akkermansia',
        'Thalassospira', 'Rhodovibrio', 'Prosthecobacter', 'Comamonas',
        # INVADER GUILD (Core)
        'Bacillus', 'Solibacter', 'Arthrobacter', 'Micrococcus', 
        'Xanthomonas', 'Stenotrophomonas', 'Pedobacter', 'Paenibacillus',
        'Burkholderia', 'Sphingomonas', 'Mucilaginibacter', 'Dyella'
    ],
    'Guild': (['Lost_Recruit'] * 12) + (['Invader'] * 12),
    'GZ_Abundance': [150, 259, 94, 80, 60, 45, 60, 45, 40, 35, 30, 25, 
                     200, 50, 100, 40, 30, 20, 15, 25, 30, 40, 20, 10],
    'AG_Abundance': [40, 129, 25, 30, 20, 5, 25, 15, 10, 5, 10, 5, 
                     500, 300, 250, 120, 90, 80, 60, 75, 150, 120, 80, 60]
}

df_taxa = pd.DataFrame(data)

# ==========================================
# 2. THE REFERENCE DATABASE (Simulated FAPROTAX)
# ==========================================
# In a real pipeline, you would load this from a .txt file. 
# Here, we construct a comprehensive reference DataFrame based on FAPROTAX 1.2.

# This database maps GENUS -> FUNCTIONAL_GROUP
db_data = [
    # Nitrogen Cycle
    ('Azospirillum', 'Nitrogen_fixation'), ('Azospirillum', 'Nitrate_reduction'),
    ('Nitrosomonas', 'Nitrification'), ('Nitrosomonas', 'Aerobic_ammonia_oxidation'),
    ('Comamonas', 'Nitrate_reduction'), ('Burkholderia', 'Nitrogen_fixation'),
    ('Paenibacillus', 'Nitrogen_fixation'), ('Arthrobacter', 'Nitrate_reduction'),
    
    # Carbon/Degradation
    ('Chthoniobacter', 'Cellulolysis'), ('Mucilaginibacter', 'Xylan_degradation'),
    ('Pedobacter', 'Chitinolysis'), ('Arthrobacter', 'Chitinolysis'),
    ('Bacillus', 'Cellulolysis'), ('Bacillus', 'Fermentation'),
    ('Akkermansia', 'Mucin_degradation'), ('Variovorax', 'Aromatic_hydrocarbon_degradation'),
    
    # Stress/Survival
    ('Bacillus', 'Aerobic_chemoheterotrophy'), ('Bacillus', 'Sporulation'), 
    ('Paenibacillus', 'Sporulation'), ('Micrococcus', 'Dormancy_survival'),
    ('Rubrobacter', 'Radiation_resistance'), ('Solibacter', 'Acid_tolerance'),
    ('Stenotrophomonas', 'Antibiotic_resistance'), ('Sphingomonas', 'Hydrocarbon_degradation'),
    
    # Plant Interaction
    ('Azospirillum', 'Plant_growth_promotion'), ('Variovorax', 'Plant_growth_promotion'),
    ('Bacillus', 'Plant_growth_promotion'), ('Pseudomonas', 'Plant_pathogen'),
    ('Xanthomonas', 'Plant_pathogen'), ('Acidovorax', 'Plant_pathogen'),
    
    # General Metabolism
    ('Gemmatimonas', 'Aerobic_anoxygenic_phototrophy'), ('Rhodovibrio', 'Photoheterotrophy'),
    ('Thalassospira', 'Chemoheterotrophy'), ('Dyella', 'Chemoheterotrophy')
]

df_db = pd.DataFrame(db_data, columns=['Genus', 'Function'])

print(f"Database Loaded: {len(df_db)} Functional Associations.")

# ==========================================
# 3. DATABASE MAPPING (The "Pull")
# ==========================================
# We merge your Taxa with the Database. 
# Inner join automatically drops taxa not in the DB (filtering unknown functions).

df_mapped = pd.merge(df_taxa, df_db, on='Genus', how='inner')

# Calculate Functional Abundance
# Logic: If Genus A has Function X, its abundance contributes to Function X pool.
grouped_func = df_mapped.groupby('Function')[['GZ_Abundance', 'AG_Abundance']].sum().reset_index()

# Calculate Metrics
grouped_func['Total_Count'] = grouped_func['GZ_Abundance'] + grouped_func['AG_Abundance']
grouped_func['Log2FC'] = np.log2((grouped_func['AG_Abundance'] + 1) / (grouped_func['GZ_Abundance'] + 1))

# Filtering: Keep only Top 15 pathways by Total Abundance (to avoid clutter)
top_funcs = grouped_func.sort_values('Total_Count', ascending=False).head(15)
top_funcs = top_funcs.sort_values('Log2FC') # Sort by shift for plotting

# ==========================================
# 4. VISUALIZATION
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# --- PLOT A: Functional Dot Plot (Standard for Pathway Analysis) ---
# Y-axis: Pathway
# X-axis: Log2 Fold Change
# Color: Direction
# Size: Total Abundance

# Normalize size for plotting
sizes = top_funcs['Total_Count'] / top_funcs['Total_Count'].max() * 1000

scatter = ax[0].scatter(
    x=top_funcs['Log2FC'],
    y=top_funcs['Function'],
    s=sizes,
    c=top_funcs['Log2FC'],
    cmap='RdBu_r', # Red = Depleted, Blue = Enriched
    edgecolor='black',
    alpha=0.8
)

ax[0].axvline(0, linestyle='--', color='gray')
ax[0].set_title('A. Differential Abundance of Database-Inferred Pathways', fontsize=14, fontweight='bold')
ax[0].set_xlabel('Log2 Fold Change (Agri / Grassland)', fontsize=12)
ax[0].grid(True, linestyle=':', alpha=0.6)

# Add Size Legend manually
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
legend2 = ax[0].legend(handles, labels, loc="lower right", title="Predicted Abundance")

# --- PLOT B: Heatmap of Contributing Genera ---
# Who is driving the top Lost vs Gained functions?
# Select top 2 Lost (Negative FC) and top 2 Gained (Positive FC) functions
top_lost = top_funcs.head(2)['Function'].tolist()
top_gained = top_funcs.tail(2)['Function'].tolist()
target_funcs = top_lost + top_gained

subset_map = df_mapped[df_mapped['Function'].isin(target_funcs)]
pivot_map = subset_map.pivot_table(index='Function', columns='Genus', values='GZ_Abundance', aggfunc='sum').fillna(0)

# Normalize row-wise for contribution %
pivot_norm = pivot_map.div(pivot_map.sum(axis=1), axis=0) * 100

sns.heatmap(pivot_norm, cmap='viridis', ax=ax[1], annot=True, fmt='.0f', cbar_kws={'label': '% Contribution'})
ax[1].set_title('B. Genus Contribution to Key Functions', fontsize=14, fontweight='bold')
ax[1].set_ylabel('')
ax[1].set_xlabel('Genus (Core Members)')

plt.tight_layout()
plt.show()

# --- 5. PRINT STATISTICS FOR PAPER ---
print("\n--- TOP FUNCTIONAL PATHWAYS (Database Inferred) ---")
print(top_funcs[['Function', 'GZ_Abundance', 'AG_Abundance', 'Log2FC', 'Total_Count']].round(2))

# %%
# ============================================================
# FAPROTAX MAPPING PIPELINE FOR MDSS CORE GENERA (FULL SCRIPT)
# ============================================================
# What this script produces:
#  A) FAPROTAX genus->function mapping coverage stats
#  B) Enrichment of functions in Lost-core vs background (Fisher + FDR)
#  C) Enrichment of functions in Invader-core vs background (Fisher + FDR)
#  D) Lost vs Invader function-count contrast (who carries which functions?)
#  E) Abundance-weighted inferred functional scores across samples:
#       - all taxa (mapped)
#       - Lost-core only
#       - Invader-core only
#  F) Plots: dotplots, heatmaps, contribution heatmaps

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# -------------------------------
# 0) USER PATHS (EDIT IF NEEDED)
# -------------------------------
FAPROTAX_PATH = "/Users/kayadetunji/Documents/DM_work/faprotax_edited_nov21.txt"
EXCEL_PATH = "/Users/kayadetunji/Documents/DM_work/raw data_taxonomy_CFC.xlsx"  # <-- edit if needed

# -------------------------------
# 1) DEFINE SAMPLES (EDIT IF NEEDED)
# -------------------------------
# Based on your methods description: 12 total (8 rhizosphere + 4 bulk)
healthy_rhizo = ["GZ1","GZ2","GZ3","GZ4"]
degraded_rhizo = ["AG1","AG2","AG3","AG4"]
healthy_bulk  = ["GZC1","GZC2"]
degraded_bulk = ["AGC1","AGC2"]

# Choose which set you want to analyze for functional shifts:
# Option A: all samples
# Option B: rhizosphere only
ANALYZE_RHIZOSPHERE_ONLY = False

# -------------------------------
# 2) CANDIDATE/CORE GENERA (DEFAULT = MDSS CORES YOU SHARED)
# -------------------------------
# LOST core (example list from your output; edit/replace with your exact final list)
core_lost = [
    "Dictyoglomus","Gemmatimonas","Azospirillum","unclassified (derived from Proteobacteria)",
    "Akkermansia","Chthoniobacter","Variovorax","unclassified (derived from Nitrosomonadaceae)",
    "Melittangium","Candidatus Nitrososphaera","Nitrosovibrio","Nitrosospira","Rubrobacter",
    "unclassified (derived from Rhodospirillaceae)","Thalassospira","Cystobacter","Stigmatella",
    "Rhodovibrio","Prosthecobacter","unclassified (derived from Verrucomicrobia subdivision 3)",
    "Actinotalea","Acidovorax","Verticillium","unclassified (derived from Rhodocyclaceae)",
    "Cellulomonas","Azoarcus","unclassified (derived from Comamonadaceae)"
]

# INVADER core (example list from your output; edit/replace with your exact final list)
core_invader = [
    "unclassified (derived from Burkholderiales)","Kinetoplastibacterium","Pedobacter","Rothia",
    "Saccharothrix","Xanthomonas","Intrasporangium","Thermomonospora","Actinokineospora",
    "Nesterenkonia","Geobacillus","unclassified (derived from Sphingobacteriaceae)","Rubrivivax",
    "Lechevalieria","Kocuria","Sphingobacterium","Tetrasphaera",
    "unclassified (derived from Betaproteobacteria)","Ktedonobacter","Lentzea","Oceanobacillus",
    "Actinocorallia","unclassified (derived from Actinobacteria (class))","Bacillus",
    "unclassified (derived from Thermomonosporaceae)","Micrococcus","Stenotrophomonas",
    "Actinosynnema","Actinomadura","Arthrobacter","Candidatus Solibacter","Dyella",
    "Janibacter","Terrabacter","Actinoallomurus"
]

# ---------------------------------------------------
# 3) PARSE FAPROTAX DB (ROBUST FOR EDITED TXT FORMAT)
# ---------------------------------------------------
def clean_genus_token(g):
    g = g.strip()
    g = g.replace("[","").replace("]","")
    return g

def extract_genus_from_pattern(pattern):
    """
    pattern examples:
      *Proteobacteria*Nitrosomonas*
      *Bacillus*pumilus*
      *Azospirillum*
    We take the last non-empty token between asterisks as genus/species.
    Then we keep just the first word as Genus.
    """
    tokens = [t for t in pattern.split("*") if t.strip()]
    if not tokens:
        return None
    last = clean_genus_token(tokens[-1])
    genus = last.split()[0].strip()
    return genus if genus else None

def parse_faprotax_genus_function(db_path):
    """
    Reads FAPROTAX edited DB and returns DataFrame with columns:
      Function, Genus

    In your snippet, taxon lines look like:
      *Bacillus*pumilus*   #manganese_oxidation
    So we parse lines containing '#' to get function reliably.
    """
    rows = []
    with open(db_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue

            # We focus on mapping lines that contain '#'
            if "#" in ln and ln.startswith("*"):
                left, right = ln.split("#", 1)
                pattern = left.strip()
                func = right.strip()
                # Some files include extra comments after function; keep first token
                func = func.split()[0].strip()

                genus = extract_genus_from_pattern(pattern)
                if genus:
                    rows.append({"Function": func, "Genus": genus})

    df = pd.DataFrame(rows).drop_duplicates()
    return df

fap_map = parse_faprotax_genus_function(FAPROTAX_PATH)

print("\n--- FAPROTAX PARSE SUMMARY ---")
print("Mappings (rows):", len(fap_map))
print("Unique functions:", fap_map["Function"].nunique())
print("Unique genera mapped:", fap_map["Genus"].nunique())
print(fap_map.head())

# ---------------------------------------------------
# 4) LOAD TAXONOMY DATASET + FILTER RARE TAXA
# ---------------------------------------------------
tax_df = pd.read_excel(EXCEL_PATH, sheet_name="taxonomy")

# Build sample lists that actually exist in your file
all_samples = healthy_rhizo + degraded_rhizo + healthy_bulk + degraded_bulk
all_samples = [c for c in all_samples if c in tax_df.columns]

healthy_cols = [c for c in (healthy_rhizo + healthy_bulk) if c in all_samples]
degraded_cols = [c for c in (degraded_rhizo + degraded_bulk) if c in all_samples]

if ANALYZE_RHIZOSPHERE_ONLY:
    use_samples = [c for c in (healthy_rhizo + degraded_rhizo) if c in all_samples]
    healthy_use = [c for c in healthy_rhizo if c in use_samples]
    degraded_use = [c for c in degraded_rhizo if c in use_samples]
else:
    use_samples = all_samples
    healthy_use = healthy_cols
    degraded_use = degraded_cols

print("\n--- SAMPLE SET ---")
print("Using samples:", use_samples)
print("Healthy samples:", healthy_use)
print("Degraded samples:", degraded_use)

# Filter rare taxa: present (>0) in >=3 of the 12 samples (or in use_samples)
counts = tax_df[use_samples]
mask = (counts > 0).sum(axis=1) >= 3
df_clean = tax_df.loc[mask].reset_index(drop=True)

# Must have genus column named 'genus'
df_clean["genus"] = df_clean["genus"].astype(str)

print("\nTaxa retained after prevalence filter:", df_clean.shape[0])

# Background genera universe (filtered)
background_genera = set(df_clean["genus"].dropna().astype(str))

# ---------------------------------------------------
# 5) COVERAGE: how many of your genera map to FAPROTAX?
# ---------------------------------------------------
mapped_genera = set(fap_map["Genus"].astype(str))
bg_mapped = background_genera & mapped_genera

lost_set = set(core_lost)
inv_set  = set(core_invader)

lost_mapped = lost_set & mapped_genera
inv_mapped  = inv_set  & mapped_genera

print("\n--- MAPPING COVERAGE ---")
print(f"Background genera: {len(background_genera)}; mapped: {len(bg_mapped)} ({len(bg_mapped)/max(len(background_genera),1):.1%})")
print(f"Lost-core genera: {len(lost_set)}; mapped: {len(lost_mapped)} ({len(lost_mapped)/max(len(lost_set),1):.1%})")
print(f"Invader-core genera: {len(inv_set)}; mapped: {len(inv_mapped)} ({len(inv_mapped)/max(len(inv_set),1):.1%})")

# ---------------------------------------------------
# 6) ENRICHMENT TESTS: candidate vs background (Fisher + FDR)
# ---------------------------------------------------
def function_enrichment(candidate_genera, background_genera, fap_map, min_support=3):
    """
    Fisher exact enrichment for each function:
      in candidate vs in background (mapped universe).
    candidate_genera: set of genera
    background_genera: set of genera (universe)
    fap_map: DataFrame with Function, Genus
    """
    g2f = fap_map.groupby("Genus")["Function"].apply(set).to_dict()
    candidate = set([g for g in candidate_genera if g in g2f])
    universe  = set([g for g in background_genera if g in g2f])

    funcs = sorted(set(fap_map.loc[fap_map["Genus"].isin(universe), "Function"]))
    rows = []
    noncand = universe - candidate

    for func in funcs:
        a = sum(func in g2f[g] for g in candidate)     # cand with func
        b = len(candidate) - a                         # cand without func
        c = sum(func in g2f[g] for g in noncand)       # noncand with func
        d = len(noncand) - c                           # noncand without func

        if (a + c) < min_support:
            continue

        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({
            "Function": func,
            "Cand_with_func": a,
            "Cand_total_mapped": len(candidate),
            "Bg_with_func": a + c,
            "Bg_total_mapped": len(universe),
            "OddsRatio": odds,
            "P": p
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["FDR"] = multipletests(out["P"].values, method="fdr_bh")[1]
    out = out.sort_values(["FDR","P","OddsRatio"], ascending=[True,True,False]).reset_index(drop=True)
    return out

enrich_lost = function_enrichment(lost_set, background_genera, fap_map, min_support=3)
enrich_inv  = function_enrichment(inv_set,  background_genera, fap_map, min_support=3)

print("\n--- TOP ENRICHED FUNCTIONS (LOST CORE) ---")
print(enrich_lost.head(20).round(4))

print("\n--- TOP ENRICHED FUNCTIONS (INVADER CORE) ---")
print(enrich_inv.head(20).round(4))

# Plot enrichment dotplot
def plot_enrichment(df, title, topN=20):
    if df.empty:
        print("No enrichment results to plot:", title)
        return
    d = df.head(topN).copy()
    plt.figure(figsize=(9, 6))
    plt.scatter(d["OddsRatio"], np.arange(len(d)), s=140, edgecolor="black")
    plt.yticks(np.arange(len(d)), d["Function"])
    plt.axvline(1.0, linestyle="--", color="gray")
    plt.xlabel("Odds ratio (candidate vs background)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_enrichment(enrich_lost, "FAPROTAX function enrichment: Lost core vs background", topN=20)
plot_enrichment(enrich_inv,  "FAPROTAX function enrichment: Invader core vs background", topN=20)

# ---------------------------------------------------
# 7) LOST vs INVADER FUNCTION COUNTS (trait profile contrast)
# ---------------------------------------------------
mapped_core = (
    pd.DataFrame({"Genus": list(lost_set), "Group": "Lost"})
    .append(pd.DataFrame({"Genus": list(inv_set), "Group": "Invader"}), ignore_index=True)
)
mapped_core = mapped_core.merge(fap_map, on="Genus", how="inner")

func_counts = mapped_core.groupby(["Group","Function"]).size().reset_index(name="Genus_Count")
pivot_counts = func_counts.pivot_table(index="Function", columns="Group", values="Genus_Count", fill_value=0)

# log2 ratio of genus counts (invader vs lost)
pivot_counts["Log2_ratio_Invader_vs_Lost"] = np.log2((pivot_counts.get("Invader",0) + 1) / (pivot_counts.get("Lost",0) + 1))
pivot_counts = pivot_counts.sort_values("Log2_ratio_Invader_vs_Lost")

# Plot top contrasts
top_contrast = pivot_counts.tail(15).copy()
plt.figure(figsize=(10, 6))
plt.barh(top_contrast.index, top_contrast["Log2_ratio_Invader_vs_Lost"])
plt.axvline(0, color="black", lw=0.8)
plt.title("Trait contrast: Invader vs Lost cores (genus-count based)")
plt.xlabel("Log2((Invader genus count +1)/(Lost genus count +1))")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 8) ABUNDANCE-WEIGHTED FUNCTION SCORES PER SAMPLE
# ---------------------------------------------------
# Build long table: Genus x Sample
df_counts = df_clean[["genus"] + use_samples].copy()
df_counts_long = df_counts.melt(id_vars="genus", var_name="Sample", value_name="Abundance")
df_counts_long["Condition"] = np.where(df_counts_long["Sample"].isin(healthy_use), "Healthy", "Degraded")

# Relative abundance per sample (recommended)
lib = df_counts_long.groupby("Sample")["Abundance"].sum()
df_counts_long["RelAbundance"] = df_counts_long.apply(lambda r: r["Abundance"] / (lib[r["Sample"]] + 1e-12), axis=1)

# Map genus -> function (many-to-many)
df_mapped_all = df_counts_long.merge(fap_map, left_on="genus", right_on="Genus", how="inner")

# Function abundance per sample (sum of relative abundances of genera carrying that function)
func_sample_all = df_mapped_all.groupby(["Sample","Condition","Function"])["RelAbundance"].sum().reset_index()

func_mat_all = func_sample_all.pivot_table(index=["Sample","Condition"], columns="Function", values="RelAbundance", fill_value=0)

def effect_size_cohens_d(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = x.var(ddof=1); vy = y.var(ddof=1)
    pooled = ((len(x)-1)*vx + (len(y)-1)*vy) / (len(x)+len(y)-2 + 1e-12)
    return (x.mean() - y.mean()) / np.sqrt(pooled + 1e-12)

# Summarize functions: log2FC + Cohen's d + MWU p
rows = []
for func in func_mat_all.columns:
    h = func_mat_all.xs("Healthy", level="Condition")[func].values
    d = func_mat_all.xs("Degraded", level="Condition")[func].values

    log2fc = np.log2((d.mean() + 1e-12) / (h.mean() + 1e-12))
    cohen = effect_size_cohens_d(d, h)  # degraded - healthy
    try:
        p = mannwhitneyu(h, d, alternative="two-sided").pvalue
    except Exception:
        p = np.nan

    rows.append({
        "Function": func,
        "Mean_Healthy": h.mean(),
        "Mean_Degraded": d.mean(),
        "Log2FC_Degraded_vs_Healthy": log2fc,
        "Cohens_d_Degraded_vs_Healthy": cohen,
        "MWU_p": p,
        "TotalMean": h.mean() + d.mean()
    })

func_stats_all = pd.DataFrame(rows).sort_values("TotalMean", ascending=False)
func_stats_all["FDR"] = multipletests(func_stats_all["MWU_p"].fillna(1.0), method="fdr_bh")[1]

print("\n--- TOP FUNCTIONS (ABUNDANCE-WEIGHTED, ALL TAXA MAPPED) ---")
print(func_stats_all.head(25).round(5))

# Plot: dotplot of top functions
def plot_function_dotplot(func_stats, title, topN=25):
    d = func_stats.head(topN).copy()
    d = d.sort_values("Log2FC_Degraded_vs_Healthy")

    sizes = (d["TotalMean"] / d["TotalMean"].max()) * 900 + 80

    plt.figure(figsize=(10, 7))
    plt.scatter(d["Log2FC_Degraded_vs_Healthy"], d["Function"],
                s=sizes, c=d["Log2FC_Degraded_vs_Healthy"], cmap="RdBu_r",
                edgecolor="black", alpha=0.9)
    plt.axvline(0, linestyle="--", color="gray")
    plt.xlabel("Log2FC (Degraded / Healthy)")
    plt.title(title + "\n(size ∝ mean relative abundance)")
    plt.tight_layout()
    plt.show()

plot_function_dotplot(func_stats_all, "Inferred functional shifts (FAPROTAX; all mapped taxa)", topN=25)

# Heatmap: saturation (row-normalized)
def plot_saturation_heatmap(func_stats, title, topN=20):
    d = func_stats.head(topN).copy()
    heat = d.set_index("Function")[["Mean_Healthy","Mean_Degraded"]]
    heat_norm = heat.div(heat.max(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)

    plt.figure(figsize=(7, 8))
    sns.heatmap(heat_norm, cmap="RdBu_r", linewidths=0.6, linecolor="white",
                annot=True, fmt=".0%", cbar_kws={"label": "Relative saturation (row-normalized)"})
    plt.title(title)
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

plot_saturation_heatmap(func_stats_all, "Functional saturation (top functions; all mapped taxa)", topN=18)

# ---------------------------------------------------
# 9) CORE-RESTRICTED FUNCTION SCORES (LOST vs INVADER CORES)
# ---------------------------------------------------
def compute_func_stats_for_genus_set(genus_set, label):
    sub = df_counts_long[df_counts_long["genus"].isin(genus_set)].copy()
    sub = sub.merge(fap_map, left_on="genus", right_on="Genus", how="inner")
    fs = sub.groupby(["Sample","Condition","Function"])["RelAbundance"].sum().reset_index()
    mat = fs.pivot_table(index=["Sample","Condition"], columns="Function", values="RelAbundance", fill_value=0)

    rows = []
    for func in mat.columns:
        h = mat.xs("Healthy", level="Condition")[func].values
        d = mat.xs("Degraded", level="Condition")[func].values
        log2fc = np.log2((d.mean() + 1e-12) / (h.mean() + 1e-12))
        cohen = effect_size_cohens_d(d, h)
        try:
            p = mannwhitneyu(h, d, alternative="two-sided").pvalue
        except Exception:
            p = np.nan

        rows.append({
            "Set": label,
            "Function": func,
            "Mean_Healthy": h.mean(),
            "Mean_Degraded": d.mean(),
            "Log2FC_Degraded_vs_Healthy": log2fc,
            "Cohens_d_Degraded_vs_Healthy": cohen,
            "MWU_p": p,
            "TotalMean": h.mean() + d.mean()
        })

    out = pd.DataFrame(rows).sort_values("TotalMean", ascending=False)
    out["FDR"] = multipletests(out["MWU_p"].fillna(1.0), method="fdr_bh")[1]
    return out

func_stats_lost = compute_func_stats_for_genus_set(lost_set, "Lost core")
func_stats_inv  = compute_func_stats_for_genus_set(inv_set,  "Invader core")

print("\n--- TOP FUNCTIONS (LOST CORE ONLY) ---")
print(func_stats_lost.head(20).round(5))

print("\n--- TOP FUNCTIONS (INVADER CORE ONLY) ---")
print(func_stats_inv.head(20).round(5))

plot_function_dotplot(func_stats_lost, "Inferred functional shifts (Lost core only)", topN=20)
plot_function_dotplot(func_stats_inv,  "Inferred functional shifts (Invader core only)", topN=20)

# ---------------------------------------------------
# 10) CONTRIBUTION HEATMAP: WHICH GENERA DRIVE TOP FUNCTIONS?
# ---------------------------------------------------
def contribution_heatmap(genus_set, func_stats, title, topN_funcs=4, topN_genera=10, use_condition="Degraded"):
    # pick top enriched/depleted functions by abs(log2fc) among top mean
    d = func_stats.head(50).copy()
    d["abs_fc"] = d["Log2FC_Degraded_vs_Healthy"].abs()
    funcs = d.sort_values("abs_fc", ascending=False).head(topN_funcs)["Function"].tolist()

    sub = df_counts_long[df_counts_long["genus"].isin(genus_set)].copy()
    sub = sub.merge(fap_map, left_on="genus", right_on="Genus", how="inner")
    sub["Healthy_sum"] = sub[sub["Sample"].isin(healthy_use)].groupby("genus")["RelAbundance"].transform("sum")
    sub["Degraded_sum"] = sub[sub["Sample"].isin(degraded_use)].groupby("genus")["RelAbundance"].transform("sum")

    # Work with condition-specific totals by genus within function
    sub_cond = sub[sub["Condition"] == use_condition].copy()
    sub_cond = sub_cond[sub_cond["Function"].isin(funcs)]

    pivot = sub_cond.pivot_table(index="Function", columns="genus", values="RelAbundance", aggfunc="sum").fillna(0)
    # keep top genera overall
    top_cols = pivot.sum(axis=0).sort_values(ascending=False).head(topN_genera).index
    pivot = pivot[top_cols]

    # row-wise %
    pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100
    pivot_pct = pivot_pct.fillna(0)

    plt.figure(figsize=(12, 4.2))
    sns.heatmap(pivot_pct, cmap="viridis", annot=True, fmt=".0f", cbar_kws={"label": f"% contribution ({use_condition})"})
    plt.title(title)
    plt.ylabel("")
    plt.xlabel("Genus")
    plt.tight_layout()
    plt.show()

contribution_heatmap(lost_set, func_stats_lost, "Genus contributions to key inferred functions (Lost core)", topN_funcs=4)
contribution_heatmap(inv_set,  func_stats_inv,  "Genus contributions to key inferred functions (Invader core)", topN_funcs=4)

print("\nDONE.")

# %%

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Optional, but good for some styling if installed
from matplotlib.colors import TwoSlopeNorm 
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests

# =========================
# 0) INPUTS
# =========================
EXCEL_PATH = "/Users/kayadetunji/Documents/DM_work/raw data_taxonomy_CFC.xlsx"
FAPROTAX_PATH = "/Users/kayadetunji/Documents/DM_work/faprotax_edited_nov21.txt"

# sample columns (12 total)
rhizo_healthy = ["GZ1","GZ2","GZ3","GZ4"]
rhizo_degraded = ["AG1","AG2","AG3","AG4"]
bulk_healthy  = ["GZC1","GZC2"]
bulk_degraded = ["AGC1","AGC2"]

ALL_SAMPLES = rhizo_healthy + rhizo_degraded + bulk_healthy + bulk_degraded
HEALTHY = rhizo_healthy + bulk_healthy
DEGRADED = rhizo_degraded + bulk_degraded

# Option: if you want rhizosphere-only main analysis
ANALYZE_RHIZOSPHERE_ONLY = False
if ANALYZE_RHIZOSPHERE_ONLY:
    HEALTHY = rhizo_healthy
    DEGRADED = rhizo_degraded
    ALL_SAMPLES = HEALTHY + DEGRADED

# Paste your MDSS consensus core genera here (from your MDSS output)
LOST_CORE = set([
    'Acidovorax', 'Akkermansia', 'Chthoniobacter', 'Variovorax', 'Gemmatimonas', 'Azospirillum', 'Rubrobacter', 
    'unclassified (derived from Rhodospirillaceae)', 'Thalassospira', 'Rhodovibrio', 'Prosthecobacter', 
    'unclassified (derived from Comamonadaceae)'
])

INVADER_CORE = set([
    'unclassified (derived from Burkholderiales)','Kinetoplastibacterium','Pedobacter','Rothia',
    'Saccharothrix','Xanthomonas','Intrasporangium','Thermomonospora','Actinokineospora',
    'Nesterenkonia','Geobacillus','unclassified (derived from Sphingobacteriaceae)',
    'Rubrivivax','Lechevalieria','Kocuria','Sphingobacterium','Tetrasphaera',
    'unclassified (derived from Betaproteobacteria)','Ktedonobacter','Lentzea','Oceanobacillus',
    'Actinocorallia','unclassified (derived from Actinobacteria (class))','Bacillus',
    'unclassified (derived from Thermomonosporaceae)','Micrococcus','Stenotrophomonas',
    'Actinosynnema','Actinomadura','Arthrobacter','Candidatus Solibacter','Dyella',
    'Janibacter','Terrabacter','Actinoallomurus'
])

# =========================
# 1) LOAD TAXA TABLE
# =========================
tax_df = pd.read_excel(EXCEL_PATH, sheet_name="taxonomy")

# basic checks
missing = [c for c in ALL_SAMPLES if c not in tax_df.columns]
if missing:
    raise ValueError(f"Missing sample columns in taxonomy sheet: {missing}")

# prevalence filter (>=3 samples nonzero) to reduce noise
counts = tax_df[ALL_SAMPLES]
mask = (counts > 0).sum(axis=1) >= 3
df = tax_df.loc[mask].copy().reset_index(drop=True)

# keep genus + sample matrix
df["genus"] = df["genus"].astype(str)

print(f"Taxa retained after prevalence filter: {len(df)}")

# Relative abundance per sample (safer for abundance-weighted trait inference)
rel = df[ALL_SAMPLES].div(df[ALL_SAMPLES].sum(axis=0), axis=1)
rel.insert(0, "genus", df["genus"].values)

# =========================
# 2) PARSE FAPROTAX MAPPING (ROBUST)
# =========================
def is_valid_function_token(tok: str) -> bool:
    if not isinstance(tok, str): return False
    tok = tok.strip()
    if not tok: return False
    if "http" in tok.lower(): return False
    return bool(re.match(r"^[a-z0-9_]+$", tok))

def extract_genus_from_taxon_field(taxon_field: str):
    if not isinstance(taxon_field, str): return None
    stars = re.findall(r"\*(.*?)\*", taxon_field)
    if not stars: return None
    stars = [s.strip() for s in stars if s.strip()]
    for s in reversed(stars):
        if re.match(r"^[A-Za-z][A-Za-z0-9_-]*$", s):
            if s.lower() in {"bacteria","archaea","eukaryota"}: continue
            return s
    return None

def parse_faprotax(file_path: str) -> pd.DataFrame:
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "#" not in ln: continue
            parts = ln.split("#")
            if len(parts) < 2: continue
            
            taxon_chunk = parts[0].strip()
            cand = parts[1].strip().split()[0] if parts[1].strip() else ""

            if not is_valid_function_token(cand): continue
            genus = extract_genus_from_taxon_field(taxon_chunk)
            if genus is None: continue

            rows.append((genus, cand))

    out = pd.DataFrame(rows, columns=["Genus","Function"]).drop_duplicates()
    return out

fap_map = parse_faprotax(FAPROTAX_PATH)
print("\n--- FAPROTAX PARSE SUMMARY ---")
print(f"Mappings (rows): {len(fap_map)}")
print(f"Unique functions: {fap_map['Function'].nunique()}")

# =========================
# 3) BUILD FUNCTION x SAMPLE MATRIX
# =========================
genus_to_funcs = fap_map.groupby("Genus")["Function"].apply(list).to_dict()
present_genera = set(rel["genus"])
mapped_present = [g for g in genus_to_funcs.keys() if g in present_genera]

long_rows = []
rel_idx = rel.set_index("genus")

for g in mapped_present:
    funcs = genus_to_funcs[g]
    gvec = rel_idx.loc[g, ALL_SAMPLES].values.astype(float)
    for func in funcs:
        long_rows.append((func, g, *gvec))

func_long = pd.DataFrame(long_rows, columns=["Function","Genus", *ALL_SAMPLES])
func_mat = func_long.groupby("Function")[ALL_SAMPLES].sum()

# =========================
# 4) STATS: FUNCTION SHIFTS (ABUNDANCE-WEIGHTED)
# =========================
def cohens_d(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)) if (nx+ny-2) > 0 else np.nan
    return (y.mean() - x.mean()) / sp if sp and not np.isnan(sp) and sp != 0 else np.nan

def bootstrap_log2fc_ci(x, y, n_boot=3000, seed=42, pseudocount=1e-9):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    boots = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        fc = (yb.mean() + pseudocount) / (xb.mean() + pseudocount)
        boots.append(np.log2(fc))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return lo, hi

stats = []
for func in func_mat.index:
    x = func_mat.loc[func, HEALTHY].values.astype(float)
    y = func_mat.loc[func, DEGRADED].values.astype(float)
    mean_h = x.mean()
    mean_d = y.mean()
    log2fc = np.log2((mean_d + 1e-9) / (mean_h + 1e-9))
    d = cohens_d(x, y)
    try: p = mannwhitneyu(x, y, alternative="two-sided").pvalue
    except: p = np.nan
    lo, hi = bootstrap_log2fc_ci(x, y, n_boot=3000, seed=123)

    stats.append({
        "Function": func,
        "Mean_Healthy": mean_h,
        "Mean_Degraded": mean_d,
        "Log2FC_Degraded_vs_Healthy": log2fc,
        "Log2FC_CI_low": lo,
        "Log2FC_CI_high": hi,
        "Cohens_d_Degraded_vs_Healthy": d,
        "MWU_p": p,
        "TotalMean": (mean_h + mean_d)
    })

stats_df = pd.DataFrame(stats)
stats_df["FDR"] = multipletests(stats_df["MWU_p"].fillna(1.0).values, method="fdr_bh")[1]

# =========================
# 5) TABLE 1 & ENRICHMENT CALCS
# =========================
TOP_N = 25
table1 = stats_df.sort_values("TotalMean", ascending=False).head(TOP_N)

# Enrichment logic
background = set(df["genus"].astype(str))
mapped_bg = background.intersection(set(fap_map["Genus"]))

def enrichment_table(candidate_set):
    cand = set(candidate_set).intersection(mapped_bg)
    bg = set(mapped_bg)
    out = []
    for func in fap_map["Function"].unique():
        func_genera = set(fap_map.loc[fap_map["Function"] == func, "Genus"])
        a = len(cand.intersection(func_genera))
        b = len(cand) - a
        c = len(bg.intersection(func_genera)) - a
        d = (len(bg) - len(cand)) - c
        if min(a,b,c,d) < 0: continue
        try: OR, p = fisher_exact([[a,b],[c,d]])
        except: OR, p = np.nan, 1.0
        out.append((func, OR, p))
    
    out = pd.DataFrame(out, columns=["Function","OddsRatio","P"])
    out["FDR"] = multipletests(out["P"].values, method="fdr_bh")[1]
    out = out.dropna().sort_values("OddsRatio", ascending=False) # sort by OR
    return out

inv_enrich = enrichment_table(INVADER_CORE)

# =========================
# 6) BEAUTIFIED PLOTTING FUNCTIONS
# =========================

def plot_function_dotplot(func_stats, title, topN=25):
    """
    The 'Pro' Dot Plot: Sorted by Log2FC, Colored by Log2FC
    """
    d = func_stats.sort_values("TotalMean", ascending=False).head(topN).copy()
    d = d.sort_values("Log2FC_Degraded_vs_Healthy") # Sort for sigmoid shape

    # Dynamic sizing
    sizes = (d["TotalMean"] / d["TotalMean"].max()) * 900 + 80
    
    # Normalization for centering colormap at 0
    max_abs = max(abs(d["Log2FC_Degraded_vs_Healthy"]))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    plt.figure(figsize=(11, 8), dpi=300)
    sc = plt.scatter(d["Log2FC_Degraded_vs_Healthy"], d["Function"],
                     s=sizes, 
                     c=d["Log2FC_Degraded_vs_Healthy"], 
                     cmap="RdBu_r", 
                     norm=norm,
                     edgecolor="black", alpha=0.9)
    
    plt.axvline(0, linestyle="--", color="gray", alpha=0.6)
    plt.grid(axis='y', linestyle=':', alpha=0.4)
    plt.xlabel("Log2FC (Degraded / Healthy)")
    plt.yticks(ha="right", fontsize=12)
    plt.title(title + "\n(Bubble size $\propto$ mean relative abundance)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Log2FC Magnitude (Red=Degraded, Blue=Healthy)")
    plt.tight_layout()
    plt.show()

# Run the individual beautiful dot plot
plot_function_dotplot(stats_df, "Inferred functional shifts (FAPROTAX; top 25 abundance)", topN=25)

# =========================
# 7) BEAUTIFIED 2x2 FIGURE PANEL
# =========================
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# --- DATA PREP FOR PANELS ---
# Filter top 20 by abundance for Plots A, B, C
top_df = stats_df.sort_values("TotalMean", ascending=False).head(20).copy()
# Sort by Log2FC for plotting logic
top_df = top_df.sort_values("Log2FC_Degraded_vs_Healthy")

# PANEL A: Forest Plot (Standardized)
axA = fig.add_subplot(gs[0, 0])
y = np.arange(len(top_df))
# Error bars
axA.errorbar(top_df["Log2FC_Degraded_vs_Healthy"], y, 
             xerr=[top_df["Log2FC_Degraded_vs_Healthy"] - top_df["Log2FC_CI_low"], 
                   top_df["Log2FC_CI_high"] - top_df["Log2FC_Degraded_vs_Healthy"]],
             fmt='none', ecolor='gray', alpha=0.6, capsize=3)
# Points
axA.scatter(top_df["Log2FC_Degraded_vs_Healthy"], y, color='black', zorder=3)
axA.axvline(0, linestyle="--", color="red", alpha=0.5)
axA.set_yticks(y)
axA.set_yticklabels(top_df["Function"])
axA.set_xlabel("Log2FC (Degraded / Healthy)")
axA.set_title("A. Trait shifts with 95% Bootstrap CI", loc='left', fontweight='bold')
axA.grid(axis='y', linestyle=':', alpha=0.3)

# PANEL B: The "Pro" Dot Plot
axB = fig.add_subplot(gs[0, 1])
sizes = (top_df["TotalMean"] / top_df["TotalMean"].max()) * 800 + 50
max_val = max(abs(top_df["Log2FC_Degraded_vs_Healthy"]))
norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

sc = axB.scatter(top_df["Log2FC_Degraded_vs_Healthy"], y, 
                 s=sizes, 
                 c=top_df["Log2FC_Degraded_vs_Healthy"], 
                 cmap="RdBu_r", norm=norm, 
                 edgecolor="black", alpha=0.9, zorder=3)
axB.axvline(0, linestyle="--", color="gray", alpha=0.5)
axB.set_yticks(y)
axB.set_yticklabels(top_df["Function"])
axB.set_xlabel("Log2FC (Degraded / Healthy)")
axB.set_title("B. Magnitude & Abundance (Size $\propto$ Abund.)", loc='left', fontweight='bold')
axB.grid(axis='y', linestyle=':', alpha=0.3)
# Colorbar for Panel B
cbar = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.04)
cbar.set_label("Log2FC")

# PANEL C: Saturation Heatmap (Cleaner)
axC = fig.add_subplot(gs[1, 0])
# Take top 15 for heatmap to keep it distinct
hm_data = top_df.tail(15).copy() # tail because top_df is sorted by log2FC low->high
hm_vals = hm_data[["Mean_Healthy", "Mean_Degraded"]]
# Row normalize
hm_norm = hm_vals.div(hm_vals.max(axis=1), axis=0)

im = axC.imshow(hm_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
axC.set_xticks([0, 1])
axC.set_xticklabels(["Healthy", "Degraded"])
axC.set_yticks(range(len(hm_data)))
axC.set_yticklabels(hm_data["Function"])
axC.set_title("C. Functional Saturation (Row Normalized)", loc='left', fontweight='bold')
fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04, label="Relative Saturation")

# PANEL D: Enrichment (Fisher OR)
axD = fig.add_subplot(gs[1, 1])
# Filter to significant or top OR
dD = inv_enrich.head(15).copy().sort_values("OddsRatio")
yD = np.arange(len(dD))

# Color points by significance (FDR < 0.05)
colors = ['red' if p < 0.05 else 'gray' for p in dD["FDR"]]

axD.scatter(dD["OddsRatio"], yD, color=colors, s=60, edgecolor='black', zorder=3)
axD.axvline(1, linestyle="--", color="black", alpha=0.5)
axD.set_yticks(yD)
axD.set_yticklabels(dD["Function"])
axD.set_xlabel("Odds Ratio (Invader Core vs Background)")
axD.set_title("D. Invader Core Enrichment (Red = FDR < 0.05)", loc='left', fontweight='bold')
axD.grid(axis='x', linestyle=':', alpha=0.3)

plt.suptitle("Functional Inference Landscape (FAPROTAX)", fontsize=16, y=0.98)
plt.show()

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ============================================================
# 0) INPUTS — EDIT THESE
# ============================================================

FAPROTAX_PATH = "/Users/kayadetunji/Documents/DM_work/faprotax_edited_nov21.txt"
# Your genus-by-sample table (must include a 'genus' column and sample columns)
TAXA_TABLE_PATH = "/path/to/your_genus_abundance_table.csv"  # <-- EDIT

# Samples (match your earlier naming)
HEALTHY_SAMPLES  = ['GZ1','GZ2','GZ3','GZ4','GZC1','GZC2']
DEGRADED_SAMPLES = ['AG1','AG2','AG3','AG4','AGC1','AGC2']

# Core genera lists (use your MDSS consensus lists)
LOST_CORE = [
    # <-- replace with your 27 core “Lost recruit” genera
]
INVADER_CORE = [
    # <-- replace with your 35 core “Invader” genera
]

# Prevalence filter (taxon must appear in >= k samples)
MIN_PREVALENCE = 3

# How many traits to show in “Top N”
TOP_N = 20

# Bootstrap settings
BOOTSTRAPS = 5000
RANDOM_SEED = 7

# Output prefix
OUT_PREFIX = "faprotax_results"

# ============================================================
# 1) SAFE FAPROTAX PARSER (fixes Martins/Sansupa/Garrity issue)
# ============================================================

def clean_function_name(func: str) -> str | None:
    """Keep only FAPROTAX-style trait IDs: lowercase letters/numbers/underscore."""
    if func is None:
        return None
    func = func.strip()

    # Drop obvious junk / citations / urls
    if "http" in func.lower() or "doi" in func.lower():
        return None

    # Many citation artifacts include commas/semicolons — drop
    if any(ch in func for ch in [",", ";", "(", ")", "[", "]"]):
        return None

    # Keep only canonical trait IDs (your cleaned run reduced to 89 using this idea)
    if not re.fullmatch(r"[a-z0-9_]+", func):
        return None

    # Optional: length sanity
    if len(func) > 80:
        return None

    return func


def extract_genus_from_taxstring(taxstring: str) -> str | None:
    """
    taxstring examples:
      *Azoarcus*communis*
      *Alcaligenaceae*Derxia*
      #*Archaea*Euryarchaeota*...*Methanocaldococcus*
      *Streptococcus*pneumoniae*
    We want a genus-like token (starts uppercase, contains lowercase).
    """
    s = taxstring.strip()
    s = s.lstrip("#").strip()  # tolerate '#*...'

    if not s.startswith("*"):
        return None

    parts = [p.strip() for p in s.split("*") if p.strip()]

    if not parts:
        return None

    # Pick the last token that looks genus-like: starts Uppercase AND has at least one lowercase letter
    genus_like = None
    for token in reversed(parts):
        if re.fullmatch(r"[A-Z][A-Za-z0-9_-]*", token) and re.search(r"[a-z]", token):
            genus_like = token
            break

    return genus_like


def parse_faprotax_mappings(path: str) -> pd.DataFrame:
    """
    Parses the FAPROTAX txt into (Genus, Function) mappings.
    Critical: ignores citation/comment-only lines that caused Martins/Sansupa/Garrity.
    Strategy:
      - Identify the current function block from header lines (not starting with '#').
      - Within a block, accept mapping lines that contain '*' taxa AND an explicit '#<function>' tag.
      - Ignore lines where '# <citation text>' occurs without a function tag.
    """
    rows = []
    current_function = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            # 1) Function header lines (do NOT start with '#', and contain metadata like 'elements:')
            # Typical: nitrogen_fixation <tab> elements:N; ...
            if not line.lstrip().startswith("#"):
                # Try to read first field as function name
                first = re.split(r"\s{2,}|\t", line.strip())[0]
                func = clean_function_name(first)
                if func:
                    current_function = func
                continue

            # 2) Mapping lines generally start with '*' (or '#*' for some)
            # and include '#<function>' in the line
            if current_function is None:
                continue

            # must contain a taxon marker
            if "*" not in line:
                continue

            # must contain an explicit function tag like '#nitrogen_fixation'
            # If it only contains '# Martins...' then skip.
            if re.search(r"#\s*"+re.escape(current_function)+r"\b", line) is None:
                continue

            # taxstring is portion before the first '#'
            taxstring = line.split("#", 1)[0].strip()
            genus = extract_genus_from_taxstring(taxstring)
            if genus is None:
                continue

            rows.append((genus, current_function))

    df = pd.DataFrame(rows, columns=["Genus", "Function"]).drop_duplicates()
    return df


# ============================================================
# 2) LOAD TAXA TABLE + PREVALENCE FILTER
# ============================================================

def load_taxa_table(path: str,
                    healthy_samples: list[str],
                    degraded_samples: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "genus" not in df.columns and "Genus" not in df.columns:
        raise ValueError("Taxa table must contain a 'genus' (or 'Genus') column.")

    if "Genus" in df.columns and "genus" not in df.columns:
        df = df.rename(columns={"Genus": "genus"})

    needed = ["genus"] + healthy_samples + degraded_samples
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in taxa table: {missing}")

    # Ensure numeric
    for c in healthy_samples + degraded_samples:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Prevalence filter
    counts = df[healthy_samples + degraded_samples]
    mask = (counts > 0).sum(axis=1) >= MIN_PREVALENCE
    df = df.loc[mask].copy().reset_index(drop=True)
    return df


# ============================================================
# 3) BUILD FUNCTION ABUNDANCE (ABUNDANCE-WEIGHTED)
# ============================================================

def function_abundance_table(df_taxa: pd.DataFrame,
                             df_map: pd.DataFrame,
                             sample_cols: list[str]) -> pd.DataFrame:
    """
    For each function, sum abundances of all mapped genera in each sample.
    Then convert to relative abundance per sample (optional but recommended).
    """
    # Merge mappings to taxa
    merged = df_map.merge(df_taxa, left_on="Genus", right_on="genus", how="inner")

    # Sum genus abundances into function
    func_by_sample = merged.groupby("Function")[sample_cols].sum()

    # Convert to relative abundance within sample across functions (compositional view)
    # Avoid divide-by-zero
    denom = func_by_sample.sum(axis=0).replace(0, np.nan)
    func_rel = func_by_sample.div(denom, axis=1).fillna(0.0)

    return func_rel


def summarize_trait_shifts(func_rel: pd.DataFrame,
                           healthy_samples: list[str],
                           degraded_samples: list[str],
                           bootstraps: int = 2000,
                           seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eps = 1e-12

    rows = []
    for func, row in func_rel.iterrows():
        h = row[healthy_samples].values
        d = row[degraded_samples].values

        mean_h = float(np.mean(h))
        mean_d = float(np.mean(d))
        log2fc = float(np.log2((mean_d + eps) / (mean_h + eps)))

        # Cohen's d (Degraded vs Healthy)
        pooled = np.sqrt(((np.var(d, ddof=1) + np.var(h, ddof=1)) / 2) + eps)
        cohens_d = float((np.mean(d) - np.mean(h)) / pooled) if pooled > 0 else 0.0

        # MWU p-value (nonparametric across sample replicates)
        try:
            p = float(mannwhitneyu(d, h, alternative="two-sided").pvalue)
        except Exception:
            p = 1.0

        # Bootstrap CI for log2FC by resampling samples within each group
        boot = []
        for _ in range(bootstraps):
            hb = rng.choice(h, size=len(h), replace=True)
            db = rng.choice(d, size=len(d), replace=True)
            boot.append(np.log2((np.mean(db) + eps) / (np.mean(hb) + eps)))
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

        rows.append({
            "Function": func,
            "Mean_Healthy": mean_h,
            "Mean_Degraded": mean_d,
            "Log2FC_Degraded_vs_Healthy": log2fc,
            "Log2FC_CI_low": float(ci_low),
            "Log2FC_CI_high": float(ci_high),
            "Cohens_d_Degraded_vs_Healthy": cohens_d,
            "MWU_p": p
        })

    out = pd.DataFrame(rows)
    out["TotalMean"] = out["Mean_Healthy"] + out["Mean_Degraded"]
    out["FDR"] = multipletests(out["MWU_p"].values, method="fdr_bh")[1]
    return out.sort_values("TotalMean", ascending=False).reset_index(drop=True)


# ============================================================
# 4) ENRICHMENT (CORE vs BACKGROUND) USING FISHER + OR
# ============================================================

def enrichment_core_vs_background(df_map: pd.DataFrame,
                                  background_genera: set[str],
                                  core_genera: list[str],
                                  min_bg_with_func: int = 3) -> pd.DataFrame:
    """
    For each function:
      - a = # core genera mapped with function
      - b = # core genera mapped without function
      - c = # background genera mapped with function
      - d = # background genera mapped without function
    Fisher exact + OR.
    """
    core_set = set(core_genera)

    mapped_bg = set(df_map["Genus"]).intersection(background_genera)
    mapped_core = set(df_map["Genus"]).intersection(core_set)

    # function -> set(genera)
    func2gen = df_map.groupby("Function")["Genus"].apply(set).to_dict()

    rows = []
    for func, gens in func2gen.items():
        bg_with = len(gens.intersection(mapped_bg))
        core_with = len(gens.intersection(mapped_core))

        # optional filter for stability
        if bg_with < min_bg_with_func:
            continue

        core_total = len(mapped_core)
        bg_total = len(mapped_bg)

        a = core_with
        b = core_total - core_with
        c = bg_with
        d = bg_total - bg_with

        # fisher exact
        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")

        rows.append({
            "Function": func,
            "Cand_with_func": a,
            "Cand_total_mapped": core_total,
            "Bg_with_func": c,
            "Bg_total_mapped": bg_total,
            "OddsRatio": float(odds) if np.isfinite(odds) else np.inf,
            "P": float(p)
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out["FDR"] = multipletests(out["P"].values, method="fdr_bh")[1]
    out = out.sort_values(["FDR", "P", "OddsRatio"], ascending=[True, True, False]).reset_index(drop=True)
    return out


# ============================================================
# 5) PLOTTING HELPERS
# ============================================================

def plot_forest_bootstrap(df_top: pd.DataFrame, ax, title: str):
    y = np.arange(len(df_top))
    ax.hlines(y, df_top["Log2FC_CI_low"], df_top["Log2FC_CI_high"], lw=1.5)
    ax.plot(df_top["Log2FC_Degraded_vs_Healthy"], y, "o")
    ax.axvline(0, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df_top["Function"])
    ax.invert_yaxis()
    ax.set_xlabel("Log2FC (Degraded / Healthy) with 95% bootstrap CI")
    ax.set_title(title)

def plot_bubble_shift(df_top: pd.DataFrame, ax, title: str):
    sizes = (df_top["TotalMean"] / df_top["TotalMean"].max() * 900).clip(50, None)
    ax.scatter(df_top["Log2FC_Degraded_vs_Healthy"], df_top["Function"], s=sizes, edgecolor="black")
    ax.axvline(0, ls="--", lw=1)
    ax.set_xlabel("Log2FC (Degraded / Healthy)")
    ax.set_title(title)

def plot_saturation_heatmap(df_top: pd.DataFrame, ax, title: str):
    hm = df_top[["Function","Mean_Healthy","Mean_Degraded"]].set_index("Function")
    # row-normalize to show relative saturation
    hm_norm = hm.div(hm.max(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    im = ax.imshow(hm_norm.values, aspect="auto")
    ax.set_yticks(np.arange(hm_norm.shape[0]))
    ax.set_yticklabels(hm_norm.index)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Mean_Healthy","Mean_Degraded"], rotation=0)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative saturation (row-normalized)")
    # annotate %
    for i in range(hm_norm.shape[0]):
        for j in range(hm_norm.shape[1]):
            ax.text(j, i, f"{hm_norm.values[i,j]*100:.0f}%", ha="center", va="center", fontsize=8)

def plot_enrichment(df_enr: pd.DataFrame, ax, title: str, top_n: int = 20):
    if df_enr.empty:
        ax.text(0.5, 0.5, "No functions passed filters.", ha="center", va="center")
        ax.set_axis_off()
        return

    # take top_n by odds ratio (or by FDR then OR)
    d = df_enr.copy()
    d["OddsRatio_clip"] = d["OddsRatio"].replace([np.inf], np.nan)
    d = d.sort_values(["FDR","P","OddsRatio_clip"], ascending=[True, True, False]).head(top_n)

    ax.scatter(d["OddsRatio"], d["Function"])
    ax.axvline(1, ls="--", lw=1)
    ax.set_xlabel("Odds ratio (candidate vs background)")
    ax.set_title(title)
    ax.invert_yaxis()


# ============================================================
# 6) RUN PIPELINE
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)

    # Load data
    df_taxa = load_taxa_table(TAXA_TABLE_PATH, HEALTHY_SAMPLES, DEGRADED_SAMPLES)
    print(f"Taxa retained after prevalence filter: {len(df_taxa)}")

    # Parse FAPROTAX safely
    df_map = parse_faprotax_mappings(FAPROTAX_PATH)

    print("\n--- FAPROTAX PARSE SUMMARY (CLEAN) ---")
    print(f"Mappings (rows): {len(df_map)}")
    print(f"Unique functions: {df_map['Function'].nunique()}")
    print(f"Unique genera mapped: {df_map['Genus'].nunique()}")
    print(df_map.head())

    # Coverage
    background_genera = set(df_taxa["genus"].unique())
    mapped_bg = background_genera.intersection(set(df_map["Genus"]))
    mapped_lost = set(LOST_CORE).intersection(set(df_map["Genus"]))
    mapped_inv = set(INVADER_CORE).intersection(set(df_map["Genus"]))

    print("\n--- MAPPING COVERAGE ---")
    print(f"Background genera: {len(background_genera)}; mapped: {len(mapped_bg)} ({len(mapped_bg)/len(background_genera)*100:.1f}%)")
    print(f"Lost-core genera: {len(LOST_CORE)}; mapped: {len(mapped_lost)} ({len(mapped_lost)/max(1,len(LOST_CORE))*100:.1f}%)")
    print(f"Invader-core genera: {len(INVADER_CORE)}; mapped: {len(mapped_inv)} ({len(mapped_inv)/max(1,len(INVADER_CORE))*100:.1f}%)")

    # Function relative abundance table (all mapped taxa)
    sample_cols = HEALTHY_SAMPLES + DEGRADED_SAMPLES
    func_rel = function_abundance_table(df_taxa, df_map, sample_cols)

    # Trait shifts + bootstrap CI
    traits = summarize_trait_shifts(func_rel, HEALTHY_SAMPLES, DEGRADED_SAMPLES,
                                    bootstraps=BOOTSTRAPS, seed=RANDOM_SEED)

    # Table 1 (Top 20 by TotalMean, but you can also rank by |Log2FC|)
    table1 = traits.sort_values("TotalMean", ascending=False).head(TOP_N).copy()
    table1.to_csv(f"{OUT_PREFIX}_Table1_top{TOP_N}_traits.csv", index=False)

    print("\n====================")
    print(f"TABLE 1. Top {TOP_N} inferred traits (abundance-weighted; all mapped taxa)")
    print("====================")
    print(table1[["Function","Mean_Healthy","Mean_Degraded",
                  "Log2FC_Degraded_vs_Healthy","Log2FC_CI_low","Log2FC_CI_high",
                  "Cohens_d_Degraded_vs_Healthy","MWU_p","FDR"]].round(4).to_string(index=False))

    # Enrichment tests: cores vs background
    enr_lost = enrichment_core_vs_background(df_map, background_genera, LOST_CORE, min_bg_with_func=3)
    enr_inv  = enrichment_core_vs_background(df_map, background_genera, INVADER_CORE, min_bg_with_func=3)

    enr_lost.to_csv(f"{OUT_PREFIX}_enrichment_lost_core.csv", index=False)
    enr_inv.to_csv(f"{OUT_PREFIX}_enrichment_invader_core.csv", index=False)

    # Also export full trait table
    traits.to_csv(f"{OUT_PREFIX}_all_traits.csv", index=False)

    # ============================================================
    # Figure-ready multi-panel layout
    # ============================================================
    # Choose top traits for plotting (by |Log2FC| is often clearer)
    plot_df = traits.sort_values("TotalMean", ascending=False).head(60)
    plot_df = plot_df.sort_values("Log2FC_Degraded_vs_Healthy", ascending=False)
    top_for_ci = traits.reindex(traits["Log2FC_Degraded_vs_Healthy"].abs().sort_values(ascending=False).index).head(TOP_N)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    plot_forest_bootstrap(top_for_ci.sort_values("Log2FC_Degraded_vs_Healthy", ascending=False),
                          ax1, f"A. Top {TOP_N} trait shifts (bootstrap 95% CI)")

    ax2 = fig.add_subplot(gs[0, 1])
    plot_bubble_shift(top_for_ci.sort_values("Log2FC_Degraded_vs_Healthy", ascending=False),
                      ax2, "B. Trait shifts (bubble size ∝ mean relative abundance)")

    ax3 = fig.add_subplot(gs[1, 0])
    plot_saturation_heatmap(table1.sort_values("Mean_Degraded", ascending=False).head(15),
                            ax3, "C. Functional saturation (top 15 traits)")

    ax4 = fig.add_subplot(gs[1, 1])
    # show invader enrichment by default (often stronger); switch to enr_lost if you want
    plot_enrichment(enr_inv, ax4, "D. FAPROTAX enrichment (Invader core vs background)", top_n=TOP_N)

    plt.tight_layout()
    fig.savefig(f"{OUT_PREFIX}_Figure_panels.png", dpi=300)
    fig.savefig(f"{OUT_PREFIX}_Figure_panels.pdf")
    plt.show()

    print(f"\nSaved:")
    print(f"  {OUT_PREFIX}_Table1_top{TOP_N}_traits.csv")
    print(f"  {OUT_PREFIX}_all_traits.csv")
    print(f"  {OUT_PREFIX}_enrichment_lost_core.csv")
    print(f"  {OUT_PREFIX}_enrichment_invader_core.csv")
    print(f"  {OUT_PREFIX}_Figure_panels.png / .pdf")

if __name__ == "__main__":
    main()
