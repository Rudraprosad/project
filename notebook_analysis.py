# CELL 1
# ============================================================
# 📦 CELL 1: Import All Required Libraries
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_curve, auc)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Plotting configuration
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style('whitegrid')
sns.set_palette('husl')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  PyTorch Version: {torch.__version__}")
print(f"🔧 Device: {DEVICE}")
print(f"✅ All libraries loaded successfully!")

---
# CELL 3
# ============================================================
# 📂 CELL 2: Load All Datasets
# ============================================================

# --- Dataset 1: IoT Sensor Data (archive/) ---
print("=" * 70)
print("📡 DATASET 1: Distributed IoT Sensor Data")
print("=" * 70)

df_voltage_current = pd.read_csv('data/archive/CurrentVoltage.csv')
df_overview        = pd.read_csv('data/archive/Overview.csv')
df_power           = pd.read_csv('data/archive/Power.csv')
df_power_factor    = pd.read_csv('data/archive/PowerFactor.csv')
df_total_power     = pd.read_csv('data/archive/TotalPower.csv')

print(f"  CurrentVoltage : {df_voltage_current.shape} → Columns: {list(df_voltage_current.columns)}")
print(f"  Overview       : {df_overview.shape} → Columns: {list(df_overview.columns)}")
print(f"  Power          : {df_power.shape} → Columns: {list(df_power.columns)}")
print(f"  PowerFactor    : {df_power_factor.shape} → Columns: {list(df_power_factor.columns)}")
print(f"  TotalPower     : {df_total_power.shape} → Columns: {list(df_total_power.columns)}")

# --- Dataset 2: Dissolved Gas Analysis (archive1/) ---
print("\n" + "=" * 70)
print("🧪 DATASET 2: Dissolved Gas Analysis (DGA) - Health Index")
print("=" * 70)

df_dga = pd.read_csv('data/archive1/Health index2.csv')
print(f"  Health Index   : {df_dga.shape} → Columns: {list(df_dga.columns)}")

# --- Dataset 3: Electrical Fault Classification (archive2/) ---
print("\n" + "=" * 70)
print("⚡ DATASET 3: Electrical Fault Classification")
print("=" * 70)

df_fault_class = pd.read_csv('data/archive2/classData.csv')
df_fault_detect = pd.read_csv('data/archive2/detect_dataset.csv')

print(f"  Fault Class    : {df_fault_class.shape} → Columns: {list(df_fault_class.columns)}")
print(f"  Fault Detection: {df_fault_detect.shape} → Columns: {list(df_fault_detect.columns)}")

# --- Dataset 4: Periodic Inspection Data (archive3/) ---
print("\n" + "=" * 70)
print("🔍 DATASET 4: Periodic Inspection Data")
print("=" * 70)

df_inspection = pd.read_excel('data/archive3/OH Transformer 25KVA.xlsx')
print(f"  Inspection     : {df_inspection.shape} → Columns: {list(df_inspection.columns)}")

print("\n✅ All datasets loaded successfully!")

---
# CELL 4
# ============================================================
# 🔍 CELL 3: Preview Each Dataset
# ============================================================

print("=" * 70)
print("📡 IoT Sensor Data - Current & Voltage (First 5 Rows)")
print("=" * 70)
display(df_voltage_current.head())

print("\n" + "=" * 70)
print("📡 IoT Sensor Data - Overview (Oil Temp, Winding Temp)")  
print("=" * 70)
display(df_overview.head())

print("\n" + "=" * 70)
print("🧪 Dissolved Gas Analysis - Health Index")
print("=" * 70)
display(df_dga.head())

print("\n" + "=" * 70)
print("⚡ Electrical Fault Classification")
print("=" * 70)
display(df_fault_class.head())

print("\n" + "=" * 70)
print("⚡ Fault Detection Dataset")
print("=" * 70)
display(df_fault_detect.head())

print("\n" + "=" * 70)
print("🔍 Periodic Inspection Data")
print("=" * 70)
display(df_inspection.head())

---
# CELL 5
# ============================================================
# 📊 CELL 4: Statistical Summary & Missing Values
# ============================================================

datasets = {
    'CurrentVoltage': df_voltage_current,
    'Overview (Thermal)': df_overview,
    'Power': df_power,
    'PowerFactor': df_power_factor,
    'TotalPower': df_total_power,
    'DGA Health Index': df_dga,
    'Fault Classification': df_fault_class,
    'Fault Detection': df_fault_detect,
    'Inspection': df_inspection
}

print("=" * 70)
print("📊 Dataset Summary Report")
print("=" * 70)

summary_data = []
for name, df in datasets.items():
    missing = df.isnull().sum().sum()
    missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
    summary_data.append({
        'Dataset': name,
        'Rows': df.shape[0],
        'Columns': df.shape[1],
        'Missing Values': missing,
        'Missing %': f'{missing_pct:.2f}%',
        'Memory (KB)': f'{df.memory_usage(deep=True).sum() / 1024:.1f}'
    })

summary_df = pd.DataFrame(summary_data)
display(summary_df)

print("\n✅ Summary report generated!")

---
# CELL 7
# ============================================================
# 📈 CELL 5: EDA - IoT Sensor Data Visualization
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('📡 IoT Sensor Data - Voltage & Current Distribution', fontsize=16, fontweight='bold')

# Voltage distributions
for i, col in enumerate(['VL1', 'VL2', 'VL3']):
    axes[0, i].hist(df_voltage_current[col].replace(0, np.nan).dropna(), 
                     bins=50, color=sns.color_palette('husl', 6)[i], alpha=0.7, edgecolor='black')
    axes[0, i].set_title(f'Distribution of {col}')
    axes[0, i].set_xlabel('Voltage (V)')
    axes[0, i].set_ylabel('Frequency')

# Current distributions
for i, col in enumerate(['IL1', 'IL2', 'IL3']):
    axes[1, i].hist(df_voltage_current[col].replace(0, np.nan).dropna(), 
                     bins=50, color=sns.color_palette('husl', 6)[i+3], alpha=0.7, edgecolor='black')
    axes[1, i].set_title(f'Distribution of {col}')
    axes[1, i].set_xlabel('Current (A)')
    axes[1, i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/iot_sensor_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ IoT Sensor Distribution plots saved!")

---
# CELL 8
# ============================================================
# 🌡️ CELL 6: EDA - Thermal & Oil Level Analysis
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('🌡️ Thermal Monitoring Data Distribution', fontsize=16, fontweight='bold')

thermal_cols = ['OTI', 'WTI', 'ATI', 'OLI']
thermal_labels = ['Oil Temperature\nIndicator', 'Winding Temperature\nIndicator', 
                  'Ambient Temperature\nIndicator', 'Oil Level\nIndicator']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, (col, label) in enumerate(zip(thermal_cols, thermal_labels)):
    data = df_overview[col].replace(0, np.nan).dropna()
    if len(data) > 0:
        axes[i].hist(data, bins=40, color=colors[i], alpha=0.8, edgecolor='black')
        axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
        axes[i].legend()
    axes[i].set_title(label)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/thermal_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Thermal Distribution plots saved!")

---
# CELL 9
# ============================================================
# 🧪 CELL 7: EDA - Dissolved Gas Analysis (DGA) Visualization
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('🧪 Dissolved Gas Analysis (DGA)', fontsize=16, fontweight='bold')

# Health Index Distribution
axes[0].hist(df_dga['Health index'], bins=30, color='#6C5CE7', alpha=0.8, edgecolor='black')
axes[0].axvline(df_dga['Health index'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df_dga['Health index'].mean():.1f}")
axes[0].set_title('Health Index Distribution')
axes[0].set_xlabel('Health Index')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Gas Concentrations (Top gases)
gas_cols = ['Hydrogen', 'Methane', 'CO', 'CO2', 'Ethylene', 'Ethane', 'Acethylene']
gas_means = df_dga[gas_cols].mean().sort_values(ascending=True)
axes[1].barh(gas_means.index, gas_means.values, color=sns.color_palette('viridis', len(gas_cols)))
axes[1].set_title('Mean Gas Concentrations')
axes[1].set_xlabel('Concentration (ppm)')

# Correlation heatmap of key DGA features
corr_cols = ['Hydrogen', 'Methane', 'CO', 'Ethylene', 'Acethylene', 'Health index']
corr = df_dga[corr_cols].corr()
im = axes[2].imshow(corr, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
axes[2].set_xticks(range(len(corr_cols)))
axes[2].set_yticks(range(len(corr_cols)))
axes[2].set_xticklabels([c[:6] for c in corr_cols], rotation=45, ha='right')
axes[2].set_yticklabels([c[:6] for c in corr_cols])
axes[2].set_title('DGA Feature Correlation')
plt.colorbar(im, ax=axes[2], shrink=0.8)

plt.tight_layout()
plt.savefig('outputs/dga_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ DGA analysis plots saved!")

---
# CELL 10
# ============================================================
# ⚡ CELL 8: EDA - Electrical Fault Classification
# ============================================================

# Create fault type labels from G, C, B, A columns
def get_fault_label(row):
    """
    Fault encoding: G=Ground, C=Phase-C, B=Phase-B, A=Phase-A
    G=1,C=0,B=0,A=0 → Ground Fault
    G=0,C=1,B=0,A=0 → Phase-C Fault  
    G=0,C=0,B=1,A=0 → Phase-B Fault
    G=0,C=0,B=0,A=1 → Phase-A Fault
    Multiple 1s → Multi-phase Fault
    All 0s → Normal
    """
    faults = []
    if row['G'] == 1: faults.append('G')
    if row['C'] == 1: faults.append('C')
    if row['B'] == 1: faults.append('B')
    if row['A'] == 1: faults.append('A')
    if len(faults) == 0:
        return 'Normal'
    return '-'.join(faults) + ' Fault'

df_fault_class['fault_label'] = df_fault_class.apply(get_fault_label, axis=1)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('⚡ Electrical Fault Classification Analysis', fontsize=16, fontweight='bold')

# Fault type distribution
fault_counts = df_fault_class['fault_label'].value_counts()
colors = sns.color_palette('Set2', len(fault_counts))
axes[0].pie(fault_counts.values, labels=fault_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 9})
axes[0].set_title('Fault Type Distribution')

# Phase currents by fault type (box plot)
fault_types_unique = df_fault_class['fault_label'].unique()[:5]
data_for_box = df_fault_class[df_fault_class['fault_label'].isin(fault_types_unique)]
data_melted = data_for_box.melt(id_vars=['fault_label'], value_vars=['Ia', 'Ib', 'Ic'],
                                 var_name='Phase Current', value_name='Value')
sns.boxplot(data=data_melted, x='fault_label', y='Value', hue='Phase Current', ax=axes[1])
axes[1].set_title('Phase Currents by Fault Type')
axes[1].set_xlabel('Fault Type')
axes[1].set_ylabel('Current (A)')
axes[1].tick_params(axis='x', rotation=30)

# Fault detection binary distribution
detect_counts = df_fault_detect['Output (S)'].value_counts()
axes[2].bar(['Normal (0)', 'Fault (1)'], detect_counts.values, 
            color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.8)
axes[2].set_title('Fault Detection Distribution')
axes[2].set_ylabel('Count')
for i, v in enumerate(detect_counts.values):
    axes[2].text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fault_classification_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Fault classification EDA saved!")

---
# CELL 11
# ============================================================
# 🔍 CELL 9: EDA - Periodic Inspection Data
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('🔍 Periodic Inspection Data Analysis', fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(df_inspection['Age '], bins=30, color='#3498db', alpha=0.8, edgecolor='black')
axes[0, 0].set_title('Transformer Age Distribution')
axes[0, 0].set_xlabel('Age (Years)')

# Health Index distribution
axes[0, 1].hist(df_inspection['Health Index'], bins=30, color='#e74c3c', alpha=0.8, edgecolor='black')
axes[0, 1].set_title('Health Index Distribution')
axes[0, 1].set_xlabel('Health Index')

# Oil Leak categories
oil_counts = df_inspection['Oil Leak'].value_counts()
axes[0, 2].bar(oil_counts.index, oil_counts.values, color=sns.color_palette('Paired', len(oil_counts)),
               edgecolor='black')
axes[0, 2].set_title('Oil Leak Categories')
axes[0, 2].tick_params(axis='x', rotation=30)

# Visual Conditions
vis_counts = df_inspection['Visual Conditions'].value_counts()
axes[1, 0].barh(vis_counts.index, vis_counts.values, color=sns.color_palette('Spectral', len(vis_counts)))
axes[1, 0].set_title('Visual Condition Categories')

# Age vs Health Index scatter
scatter = axes[1, 1].scatter(df_inspection['Age '], df_inspection['Health Index'], 
                              c=df_inspection['Loading'], cmap='RdYlGn_r', alpha=0.5, s=10)
axes[1, 1].set_title('Age vs Health Index (Colored by Loading)')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Health Index')
plt.colorbar(scatter, ax=axes[1, 1], label='Loading %')

# Loading vs Health Index
axes[1, 2].scatter(df_inspection['Loading'], df_inspection['Health Index'], 
                    c='#8e44ad', alpha=0.3, s=10)
axes[1, 2].set_title('Loading vs Health Index')
axes[1, 2].set_xlabel('Loading %')
axes[1, 2].set_ylabel('Health Index')

plt.tight_layout()
plt.savefig('outputs/inspection_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Inspection data EDA saved!")

---
# CELL 13
# ============================================================
# 🔗 CELL 10: Merge IoT Sensor Data (Time-Synced)
# ============================================================

print("=" * 70)
print("🔗 Step 1: Merging IoT Sensor Tables by Timestamp")
print("=" * 70)

# Convert timestamps to datetime for all IoT tables
iot_dfs = {
    'CurrentVoltage': df_voltage_current,
    'Overview': df_overview,
    'Power': df_power,
    'PowerFactor': df_power_factor,
    'TotalPower': df_total_power
}

for name, df in iot_dfs.items():
    df['DeviceTimeStamp'] = pd.to_datetime(df['DeviceTimeStamp'])
    df.set_index('DeviceTimeStamp', inplace=True)
    print(f"  ✓ {name}: Time range {df.index.min()} → {df.index.max()}")

# Resample all to 1-minute intervals using forward fill + mean
print("\n⏱️ Resampling all IoT data to 1-minute intervals...")

resampled_dfs = {}
for name, df in iot_dfs.items():
    # Remove duplicate indices, then resample
    df_clean = df[~df.index.duplicated(keep='first')]
    resampled = df_clean.resample('1min').mean()
    resampled_dfs[name] = resampled
    print(f"  ✓ {name}: {df.shape} → {resampled.shape} (resampled)")

# Merge all IoT tables on their time index
print("\n🔗 Merging all IoT tables...")
df_iot_merged = resampled_dfs['CurrentVoltage']
for name in ['Overview', 'Power', 'PowerFactor', 'TotalPower']:
    df_iot_merged = df_iot_merged.join(resampled_dfs[name], how='outer', rsuffix=f'_{name}')

# Forward fill and backward fill for small gaps
df_iot_merged = df_iot_merged.ffill(limit=5).bfill(limit=5)

print(f"\n📊 Merged IoT DataFrame Shape: {df_iot_merged.shape}")
print(f"   Columns: {list(df_iot_merged.columns)}")
print(f"   Time Range: {df_iot_merged.index.min()} → {df_iot_merged.index.max()}")
print(f"   Missing Values: {df_iot_merged.isnull().sum().sum()}")

# Drop columns with too many missing values (>50%)
threshold = len(df_iot_merged) * 0.5
df_iot_merged = df_iot_merged.dropna(axis=1, thresh=int(threshold))
df_iot_merged = df_iot_merged.dropna()

print(f"\n✅ Cleaned IoT DataFrame: {df_iot_merged.shape}")
display(df_iot_merged.head())

---
# CELL 14
# ============================================================
# 🛠️ CELL 11: Feature Engineering - Create Unified Feature Set
# ============================================================

print("=" * 70)
print("🛠️ Step 2: Feature Engineering & Fusion")
print("=" * 70)

# --- A) Create fault labels for IoT data based on thresholds ---
# Since IoT data doesn't have explicit fault labels, we create synthetic labels
# using domain knowledge thresholds for transformer monitoring

def create_iot_fault_labels(df):
    """
    Create fault labels based on transformer monitoring thresholds:
    - Overvoltage: V > 260V or V < 200V (for 240V nominal)
    - Overcurrent: IL > threshold
    - Thermal fault: OTI > 80°C or WTI > 90°C  
    - Oil Level fault: OLI < 25% or OLI > 95%
    """
    labels = pd.Series('Normal', index=df.index)
    
    # Check voltage thresholds
    for col in ['VL1', 'VL2', 'VL3']:
        if col in df.columns:
            labels[(df[col] > 260) | ((df[col] < 200) & (df[col] > 0))] = 'Voltage_Fault'
    
    # Check thermal thresholds
    if 'OTI' in df.columns:
        labels[df['OTI'] > 80] = 'Thermal_Fault'
    if 'WTI' in df.columns:
        labels[df['WTI'] > 90] = 'Thermal_Fault'
    
    # Check oil level
    if 'OLI' in df.columns:
        labels[(df['OLI'] < 25) & (df['OLI'] > 0)] = 'Oil_Level_Fault'
    
    return labels

df_iot_merged['fault_label'] = create_iot_fault_labels(df_iot_merged)
print("\n📊 IoT Fault Label Distribution:")
print(df_iot_merged['fault_label'].value_counts())

# --- B) DGA Health Index Categories ---
print("\n" + "=" * 70)
print("🧪 DGA Health Index Feature Engineering")
print("=" * 70)

# Categorize health index
def categorize_health(hi):
    if hi >= 85:
        return 'Good'
    elif hi >= 50:
        return 'Fair'
    elif hi >= 25:
        return 'Poor'
    else:
        return 'Critical'

df_dga['health_category'] = df_dga['Health index'].apply(categorize_health)
print(f"DGA Health Categories:\n{df_dga['health_category'].value_counts()}")

# Key DGA ratios (standard industry ratios for fault diagnosis)
df_dga['rogers_ratio_1'] = df_dga['Methane'] / (df_dga['Hydrogen'] + 1)  # CH4/H2
df_dga['rogers_ratio_2'] = df_dga['Ethylene'] / (df_dga['Ethane'] + 1)   # C2H4/C2H6
df_dga['rogers_ratio_3'] = df_dga['Acethylene'] / (df_dga['Ethylene'] + 1) # C2H2/C2H4

# --- C) Inspection Data Encoding ---
print("\n" + "=" * 70)
print("🔍 Inspection Data Feature Engineering")
print("=" * 70)

le_oil = LabelEncoder()
le_visual = LabelEncoder()
df_inspection['Oil_Leak_Encoded'] = le_oil.fit_transform(df_inspection['Oil Leak'])
df_inspection['Visual_Cond_Encoded'] = le_visual.fit_transform(df_inspection['Visual Conditions'])

print(f"Oil Leak Encoding: {dict(zip(le_oil.classes_, le_oil.transform(le_oil.classes_)))}")
print(f"Visual Conditions Encoding: {dict(zip(le_visual.classes_, le_visual.transform(le_visual.classes_)))}")

# Health Index categories from inspection
df_inspection['inspection_health_cat'] = df_inspection['Health Index'].apply(
    lambda x: 'Critical' if x <= 2 else ('Poor' if x <= 3 else ('Fair' if x <= 4 else 'Good'))
)
print(f"\nInspection Health Categories:\n{df_inspection['inspection_health_cat'].value_counts()}")

print("\n✅ Feature Engineering Complete!")

---
# CELL 16
# ============================================================
# 🔧 CELL 12: Prepare Primary Dataset for LSTM Training
# ============================================================

print("=" * 70)
print("🔧 Preparing Combined Dataset for LSTM Fault Detection")
print("=" * 70)

# ---------------------------------------------------------------
# STRATEGY: Use the Electrical Fault Classification dataset (classData)
# as the primary training data since it has labeled fault categories.
# Augment with features from other datasets.
# ---------------------------------------------------------------

# Step 1: Prepare fault classification data
print("\n📋 Step 1: Preparing Fault Classification Features...")

df_class = df_fault_class.copy()

# Create multi-class fault label
def encode_fault_type(row):
    """Convert G,C,B,A binary flags to a single fault class."""
    code = f"{int(row['G'])}{int(row['C'])}{int(row['B'])}{int(row['A'])}"
    fault_map = {
        '0000': 'Normal',
        '1000': 'Ground_Fault',
        '0100': 'Phase_C_Fault',
        '0010': 'Phase_B_Fault',
        '0001': 'Phase_A_Fault',
        '1001': 'AG_Fault',
        '0011': 'AB_Fault',
        '0110': 'BC_Fault',
        '1010': 'BG_Fault',
        '0101': 'AC_Fault',
        '1100': 'CG_Fault',
        '1110': 'BCG_Fault',
        '1011': 'ABG_Fault',
        '0111': 'ABC_Fault',
        '1101': 'ACG_Fault',
        '1111': 'ABCG_Fault'
    }
    return fault_map.get(code, 'Unknown')

df_class['fault_type'] = df_class.apply(encode_fault_type, axis=1)
feature_cols_class = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

print(f"Fault Type Distribution:")
print(df_class['fault_type'].value_counts())

# Step 2: Prepare detection data
print("\n📋 Step 2: Preparing Fault Detection Features...")

df_detect = df_fault_detect.copy()
df_detect = df_detect.drop(columns=['Unnamed: 7', 'Unnamed: 8'], errors='ignore')
df_detect.rename(columns={'Output (S)': 'is_fault'}, inplace=True)
feature_cols_detect = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

print(f"Detection Labels: {df_detect['is_fault'].value_counts().to_dict()}")

# Step 3: Add engineered features to both datasets
print("\n📋 Step 3: Feature Engineering (Electrical Features)...")

for df in [df_class, df_detect]:
    # Power-related features
    df['power_a'] = df['Va'] * df['Ia']  # Instantaneous power Phase A
    df['power_b'] = df['Vb'] * df['Ib']  # Instantaneous power Phase B  
    df['power_c'] = df['Vc'] * df['Ic']  # Instantaneous power Phase C
    
    # Current/Voltage magnitudes
    df['I_magnitude'] = np.sqrt(df['Ia']**2 + df['Ib']**2 + df['Ic']**2)
    df['V_magnitude'] = np.sqrt(df['Va']**2 + df['Vb']**2 + df['Vc']**2)
    
    # Imbalance features (key indicator of faults)
    df['I_imbalance'] = df[['Ia', 'Ib', 'Ic']].std(axis=1)
    df['V_imbalance'] = df[['Va', 'Vb', 'Vc']].std(axis=1)
    
    # Phase angle approximation
    df['I_ratio_ab'] = df['Ia'] / (df['Ib'] + 1e-8)
    df['V_ratio_ab'] = df['Va'] / (df['Vb'] + 1e-8)

# Updated feature columns
feature_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc',
                'power_a', 'power_b', 'power_c',
                'I_magnitude', 'V_magnitude',
                'I_imbalance', 'V_imbalance',
                'I_ratio_ab', 'V_ratio_ab']

print(f"\n✅ Total Features: {len(feature_cols)}")
print(f"   Features: {feature_cols}")

---
# CELL 17
# ============================================================
# 🧪 CELL 13: Prepare DGA & Inspection Supplementary Features
# ============================================================

print("=" * 70)
print("🧪 Preparing DGA Features for Health Prediction Model")
print("=" * 70)

# DGA Features
dga_features = ['Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2', 
                'Ethylene', 'Ethane', 'Acethylene', 'DBDS', 'Power factor',
                'Interfacial V', 'Dielectric rigidity', 'Water content',
                'rogers_ratio_1', 'rogers_ratio_2', 'rogers_ratio_3']

# Scale DGA data
scaler_dga = StandardScaler()
X_dga = scaler_dga.fit_transform(df_dga[dga_features])
y_dga = df_dga['Health index'].values

# Create health categories for classification
le_health = LabelEncoder()
y_dga_class = le_health.fit_transform(df_dga['health_category'])

print(f"DGA Features Shape: {X_dga.shape}")
print(f"Health Categories: {dict(zip(le_health.classes_, range(len(le_health.classes_))))}")

# Inspection Features
print("\n" + "=" * 70)
print("🔍 Preparing Inspection Features")
print("=" * 70)

inspection_features = ['Age ', 'Infrared Scan Results', 'Loading', 
                       'Oil_Leak_Encoded', 'Visual_Cond_Encoded']

scaler_insp = StandardScaler()
X_insp = scaler_insp.fit_transform(df_inspection[inspection_features])
y_insp = df_inspection['Health Index'].values

print(f"Inspection Features Shape: {X_insp.shape}")
print(f"Health Index Range: [{y_insp.min():.1f}, {y_insp.max():.1f}]")

print("\n✅ All supplementary features prepared!")

---
# CELL 18
# ============================================================
# 🧠 CELL 14: Create LSTM Sequences (Sliding Window)
# ============================================================

print("=" * 70)
print("🧠 Creating LSTM Input Sequences")
print("=" * 70)

# ---------------------------------------------------------------
# LSTM requires 3D input: (samples, timesteps, features)
# We use a sliding window approach to create temporal sequences
# Window size = 10 (represents 10 consecutive readings)
# ---------------------------------------------------------------

SEQUENCE_LENGTH = 10  # Look-back window (10 timesteps)

def create_sequences(X, y, seq_length):
    """Create sliding window sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])  # Predict next step
    return np.array(X_seq), np.array(y_seq)

# --- Prepare Binary Fault Detection Dataset ---
print("\n📋 Preparing BINARY Fault Detection (Normal vs Fault)...")

# Scale features
scaler_detect = StandardScaler()
X_detect_scaled = scaler_detect.fit_transform(df_detect[feature_cols].values)
y_detect = df_detect['is_fault'].values

# Create sequences
X_seq_detect, y_seq_detect = create_sequences(X_detect_scaled, y_detect, SEQUENCE_LENGTH)
print(f"  Binary Detection - X shape: {X_seq_detect.shape}, y shape: {y_seq_detect.shape}")
print(f"  Class distribution: {np.unique(y_seq_detect, return_counts=True)}")

# --- Prepare Multi-class Fault Classification Dataset ---
print("\n📋 Preparing MULTI-CLASS Fault Classification...")

# Encode fault types
le_fault = LabelEncoder()
y_fault_encoded = le_fault.fit_transform(df_class['fault_type'])
num_classes = len(le_fault.classes_)

scaler_class = StandardScaler()
X_class_scaled = scaler_class.fit_transform(df_class[feature_cols].values)

# Create sequences
X_seq_class, y_seq_class = create_sequences(X_class_scaled, y_fault_encoded, SEQUENCE_LENGTH)
print(f"  Multi-class - X shape: {X_seq_class.shape}, y shape: {y_seq_class.shape}")
print(f"  Number of fault classes: {num_classes}")
print(f"  Classes: {list(le_fault.classes_)}")

# --- Train/Val/Test Split ---
print("\n📋 Splitting data into Train/Val/Test (70/15/15)...")

# Binary detection split
X_train_det, X_temp_det, y_train_det, y_temp_det = train_test_split(
    X_seq_detect, y_seq_detect, test_size=0.3, random_state=42, stratify=y_seq_detect)
X_val_det, X_test_det, y_val_det, y_test_det = train_test_split(
    X_temp_det, y_temp_det, test_size=0.5, random_state=42, stratify=y_temp_det)

print(f"  Binary - Train: {X_train_det.shape}, Val: {X_val_det.shape}, Test: {X_test_det.shape}")

# Multi-class split
X_train_cls, X_temp_cls, y_train_cls, y_temp_cls = train_test_split(
    X_seq_class, y_seq_class, test_size=0.3, random_state=42, stratify=y_seq_class)
X_val_cls, X_test_cls, y_val_cls, y_test_cls = train_test_split(
    X_temp_cls, y_temp_cls, test_size=0.5, random_state=42, stratify=y_temp_cls)

print(f"  Multi-class - Train: {X_train_cls.shape}, Val: {X_val_cls.shape}, Test: {X_test_cls.shape}")

# --- Convert to PyTorch Tensors ---
print("\n📋 Converting to PyTorch tensors...")

# Binary detection tensors
X_train_det_t = torch.FloatTensor(X_train_det).to(DEVICE)
X_val_det_t = torch.FloatTensor(X_val_det).to(DEVICE)
X_test_det_t = torch.FloatTensor(X_test_det).to(DEVICE)
y_train_det_t = torch.LongTensor(y_train_det).to(DEVICE)
y_val_det_t = torch.LongTensor(y_val_det).to(DEVICE)
y_test_det_t = torch.LongTensor(y_test_det).to(DEVICE)

# Multi-class tensors
X_train_cls_t = torch.FloatTensor(X_train_cls).to(DEVICE)
X_val_cls_t = torch.FloatTensor(X_val_cls).to(DEVICE)
X_test_cls_t = torch.FloatTensor(X_test_cls).to(DEVICE)
y_train_cls_t = torch.LongTensor(y_train_cls).to(DEVICE)
y_val_cls_t = torch.LongTensor(y_val_cls).to(DEVICE)
y_test_cls_t = torch.LongTensor(y_test_cls).to(DEVICE)

# Create DataLoaders
BATCH_SIZE = 64

train_det_dataset = TensorDataset(X_train_det_t, y_train_det_t)
val_det_dataset = TensorDataset(X_val_det_t, y_val_det_t)
test_det_dataset = TensorDataset(X_test_det_t, y_test_det_t)

train_det_loader = DataLoader(train_det_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_det_loader = DataLoader(val_det_dataset, batch_size=BATCH_SIZE)
test_det_loader = DataLoader(test_det_dataset, batch_size=BATCH_SIZE)

train_cls_dataset = TensorDataset(X_train_cls_t, y_train_cls_t)
val_cls_dataset = TensorDataset(X_val_cls_t, y_val_cls_t)
test_cls_dataset = TensorDataset(X_test_cls_t, y_test_cls_t)

train_cls_loader = DataLoader(train_cls_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_cls_loader = DataLoader(val_cls_dataset, batch_size=BATCH_SIZE)
test_cls_loader = DataLoader(test_cls_dataset, batch_size=BATCH_SIZE)

print("\n✅ All data prepared and loaded into PyTorch DataLoaders!")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Sequence Length: {SEQUENCE_LENGTH}")
print(f"   Number of Features: {len(feature_cols)}")

---
# CELL 20
# ============================================================
# 🧠 CELL 15: Define LSTM Model Architecture
# ============================================================

class TransformerFaultLSTM(nn.Module):
    """
    LSTM-based Deep Learning Model for Industrial Transformer Fault Detection.
    
    Architecture:
        Input(seq_len, n_features) → LSTM(64, return_sequences=True) → 
        Dropout(0.2) → LSTM(32) → Dropout(0.2) → 
        BatchNorm → Dense(16, ReLU) → Dropout(0.1) → Dense(n_classes, Softmax)
    
    Parameters:
        input_size: Number of input features per timestep
        hidden_size_1: Units in first LSTM layer (default: 64)
        hidden_size_2: Units in second LSTM layer (default: 32)
        num_classes: Number of output classes
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(self, input_size, num_classes, hidden_size_1=64, hidden_size_2=32, dropout=0.2):
        super(TransformerFaultLSTM, self).__init__()
        
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        
        # LSTM Layer 1: Captures temporal patterns in sensor readings
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM Layer 2: Refines temporal features
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout2 = nn.Dropout(dropout)
        
        # Batch normalization for training stability
        self.batch_norm = nn.BatchNorm1d(hidden_size_2)
        
        # Fully connected classification layers
        self.fc1 = nn.Linear(hidden_size_2, 16)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # LSTM Layer 1 (return sequences for stacking)
        lstm1_out, _ = self.lstm1(x)  # (batch, seq, 64)
        lstm1_out = self.dropout1(lstm1_out)
        
        # LSTM Layer 2 (only take last timestep output)
        lstm2_out, (h_n, c_n) = self.lstm2(lstm1_out)  # (batch, seq, 32)
        lstm2_out = lstm2_out[:, -1, :]  # Take last timestep: (batch, 32)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Batch normalization
        lstm2_out = self.batch_norm(lstm2_out)
        
        # Dense layers
        out = self.fc1(lstm2_out)  # (batch, 16)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)  # (batch, num_classes)
        
        return out

# --- Instantiate Models ---
n_features = len(feature_cols)

# Model 1: Binary Fault Detection (Normal vs Fault)
model_binary = TransformerFaultLSTM(
    input_size=n_features,
    num_classes=2,  # Normal / Fault
    hidden_size_1=64,
    hidden_size_2=32,
    dropout=0.2
).to(DEVICE)

# Model 2: Multi-class Fault Classification
model_multiclass = TransformerFaultLSTM(
    input_size=n_features,
    num_classes=num_classes,
    hidden_size_1=64,
    hidden_size_2=32,
    dropout=0.2
).to(DEVICE)

print("=" * 70)
print("🧠 Model Architecture Summary")
print("=" * 70)

print("\n📋 Model 1: Binary Fault Detection")
print(model_binary)
total_params = sum(p.numel() for p in model_binary.parameters())
trainable_params = sum(p.numel() for p in model_binary.parameters() if p.requires_grad)
print(f"\n  Total Parameters: {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,}")

print("\n" + "-" * 40)
print("\n📋 Model 2: Multi-class Fault Classification")
print(model_multiclass)
total_params2 = sum(p.numel() for p in model_multiclass.parameters())
print(f"\n  Total Parameters: {total_params2:,}")

---
# CELL 21
# ============================================================
# 🏋️ CELL 16: Training & Validation Function
# ============================================================

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                model_name="model", patience=10):
    """
    Train the LSTM model with early stopping and learning rate scheduling.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Maximum training epochs
        learning_rate: Initial learning rate
        model_name: Name for saving the model
        patience: Early stopping patience
    
    Returns:
        history: Dictionary containing training metrics
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print(f"🏋️ Training {model_name}")
    print(f"{'='*70}")
    print(f"  Epochs: {num_epochs} | LR: {learning_rate} | Patience: {patience}")
    print(f"  Optimizer: Adam | Loss: CrossEntropyLoss")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = " ✅ Best"
        else:
            patience_counter += 1
            marker = ""
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
            print(f"  Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{marker}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  ⏹️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    model_path = f'models/{model_name}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': n_features,
            'num_classes': model.fc2.out_features,
            'hidden_size_1': model.hidden_size_1,
            'hidden_size_2': model.hidden_size_2,
        }
    }, model_path)
    
    print(f"\n  💾 Best model saved to '{model_path}'")
    print(f"  📊 Best Validation Loss: {best_val_loss:.4f}")
    
    return history

print("✅ Training function defined!")

---
# CELL 22
# ============================================================
# 🏋️ CELL 17: Train Model 1 - Binary Fault Detection
# ============================================================

history_binary = train_model(
    model=model_binary,
    train_loader=train_det_loader,
    val_loader=val_det_loader,
    num_epochs=50,
    learning_rate=0.001,
    model_name="binary_fault_detector",
    patience=10
)

---
# CELL 23
# ============================================================
# 🏋️ CELL 18: Train Model 2 - Multi-class Fault Classification
# ============================================================

history_multiclass = train_model(
    model=model_multiclass,
    train_loader=train_cls_loader,
    val_loader=val_cls_loader,
    num_epochs=50,
    learning_rate=0.001,
    model_name="multiclass_fault_classifier",
    patience=10
)

---
# CELL 25
# ============================================================
# 📈 CELL 19: Plot Training Curves
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('📈 LSTM Model Training History', fontsize=18, fontweight='bold')

# --- Binary Model Training Curves ---
# Loss
axes[0, 0].plot(history_binary['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
axes[0, 0].plot(history_binary['val_loss'], label='Val Loss', color='#3498db', linewidth=2, linestyle='--')
axes[0, 0].set_title('Binary Detection - Loss', fontsize=14)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history_binary['train_acc'], label='Train Accuracy', color='#2ecc71', linewidth=2)
axes[0, 1].plot(history_binary['val_acc'], label='Val Accuracy', color='#9b59b6', linewidth=2, linestyle='--')
axes[0, 1].set_title('Binary Detection - Accuracy', fontsize=14)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# --- Multi-class Model Training Curves ---
# Loss
axes[1, 0].plot(history_multiclass['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
axes[1, 0].plot(history_multiclass['val_loss'], label='Val Loss', color='#3498db', linewidth=2, linestyle='--')
axes[1, 0].set_title('Multi-class Classification - Loss', fontsize=14)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Accuracy
axes[1, 1].plot(history_multiclass['train_acc'], label='Train Accuracy', color='#2ecc71', linewidth=2)
axes[1, 1].plot(history_multiclass['val_acc'], label='Val Accuracy', color='#9b59b6', linewidth=2, linestyle='--')
axes[1, 1].set_title('Multi-class Classification - Accuracy', fontsize=14)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Training curves saved!")

---
# CELL 27
# ============================================================
# 📋 CELL 20: Evaluate Models on Test Set
# ============================================================

def evaluate_model(model, test_loader, class_names, model_name="Model"):
    """Evaluate model and return predictions, metrics, and classification report."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\n{'='*70}")
    print(f"📋 {model_name} - Test Set Evaluation")
    print(f"{'='*70}")
    print(f"\n  📊 Overall Metrics:")
    print(f"     Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"     F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"     F1-Score (Macro):    {f1_macro:.4f}")
    
    print(f"\n  📋 Classification Report:")
    print("  " + "-" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    for line in report.split('\n'):
        print(f"  {line}")
    
    return all_preds, all_labels, all_probs, accuracy, f1_weighted

# --- Evaluate Binary Detection Model ---
binary_class_names = ['Normal', 'Fault']
preds_bin, labels_bin, probs_bin, acc_bin, f1_bin = evaluate_model(
    model_binary, test_det_loader, binary_class_names, "Binary Fault Detection"
)

# --- Evaluate Multi-class Classification Model ---
multiclass_names = list(le_fault.classes_)
preds_cls, labels_cls, probs_cls, acc_cls, f1_cls = evaluate_model(
    model_multiclass, test_cls_loader, multiclass_names, "Multi-class Fault Classification"
)

---
# CELL 28
# ============================================================
# 📊 CELL 21: Confusion Matrix Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('📊 Confusion Matrices', fontsize=18, fontweight='bold')

# Binary confusion matrix
cm_bin = confusion_matrix(labels_bin, preds_bin)
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', 
            xticklabels=binary_class_names, yticklabels=binary_class_names,
            ax=axes[0], cbar_kws={'shrink': 0.8})
axes[0].set_title(f'Binary Detection\nAccuracy: {acc_bin:.4f}', fontsize=14)
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)

# Multi-class confusion matrix
cm_cls = confusion_matrix(labels_cls, preds_cls)
# For readability, use short names if too many classes
short_names = [n[:8] for n in multiclass_names]
sns.heatmap(cm_cls, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=short_names, yticklabels=short_names,
            ax=axes[1], cbar_kws={'shrink': 0.8})
axes[1].set_title(f'Multi-class Classification\nAccuracy: {acc_cls:.4f}', fontsize=14)
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved!")

---
# CELL 29
# ============================================================
# 📈 CELL 22: ROC Curve - Binary Fault Detection
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('📈 ROC Curve Analysis', fontsize=16, fontweight='bold')

# Binary ROC
fpr, tpr, thresholds = roc_curve(labels_bin, probs_bin[:, 1])
roc_auc = auc(fpr, tpr)

axes[0].plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Classifier')
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
axes[0].set_title('Binary Fault Detection ROC')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Per-class accuracy bar chart
per_class_acc = cm_cls.diagonal() / cm_cls.sum(axis=1)
colors = sns.color_palette('viridis', len(multiclass_names))
bars = axes[1].barh(short_names, per_class_acc, color=colors, edgecolor='black', alpha=0.8)
axes[1].set_title('Per-Class Accuracy (Multi-class)')
axes[1].set_xlabel('Accuracy')
for bar, acc in zip(bars, per_class_acc):
    axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{acc:.2f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/roc_and_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ROC curve and per-class accuracy saved!")

---
# CELL 31
# ============================================================
# 🚨 CELL 23: Real-Time Fault Detection & Alert Function
# ============================================================

class TransformerMonitor:
    """
    Real-time Smart Transformer Monitoring System.
    
    Maintains a sliding window buffer of sensor readings and uses 
    the trained LSTM models for continuous fault detection.
    
    Usage:
        monitor = TransformerMonitor(model_binary, model_multiclass, ...)
        alert = monitor.predict(sensor_reading_dict)
    """
    
    def __init__(self, binary_model, multiclass_model, scaler, label_encoder,
                 feature_cols, seq_length=10, device='cpu'):
        self.binary_model = binary_model.eval()
        self.multiclass_model = multiclass_model.eval()
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_cols = feature_cols
        self.seq_length = seq_length
        self.device = device
        
        # Sliding window buffer for temporal context
        self.buffer = []
        
        # Alert history
        self.alert_history = []
        
    def _engineer_features(self, reading):
        """Add engineered features to raw sensor reading."""
        reading['power_a'] = reading['Va'] * reading['Ia']
        reading['power_b'] = reading['Vb'] * reading['Ib']
        reading['power_c'] = reading['Vc'] * reading['Ic']
        reading['I_magnitude'] = np.sqrt(reading['Ia']**2 + reading['Ib']**2 + reading['Ic']**2)
        reading['V_magnitude'] = np.sqrt(reading['Va']**2 + reading['Vb']**2 + reading['Vc']**2)
        reading['I_imbalance'] = np.std([reading['Ia'], reading['Ib'], reading['Ic']])
        reading['V_imbalance'] = np.std([reading['Va'], reading['Vb'], reading['Vc']])
        reading['I_ratio_ab'] = reading['Ia'] / (reading['Ib'] + 1e-8)
        reading['V_ratio_ab'] = reading['Va'] / (reading['Vb'] + 1e-8)
        return reading
    
    def predict(self, sensor_reading):
        """
        Predict fault status from a real-time sensor reading.
        
        Args:
            sensor_reading (dict): Dictionary with keys:
                'Ia', 'Ib', 'Ic' - Phase currents
                'Va', 'Vb', 'Vc' - Phase voltages
        
        Returns:
            dict: Alert information with status, confidence, and fault type
        """
        # Add engineered features
        reading = self._engineer_features(sensor_reading.copy())
        
        # Extract feature vector
        feature_vector = np.array([reading[col] for col in self.feature_cols])
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_vector.reshape(1, -1))[0]
        
        # Add to buffer
        self.buffer.append(feature_scaled)
        
        # Keep only last seq_length readings
        if len(self.buffer) > self.seq_length:
            self.buffer = self.buffer[-self.seq_length:]
        
        # If buffer is not full, pad with zeros and predict
        if len(self.buffer) < self.seq_length:
            padded = np.zeros((self.seq_length, len(self.feature_cols)))
            padded[-len(self.buffer):] = np.array(self.buffer)
            sequence = padded
            buffer_status = f"Buffering... ({len(self.buffer)}/{self.seq_length})"
        else:
            sequence = np.array(self.buffer)
            buffer_status = "Ready"
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # --- Binary Detection ---
        with torch.no_grad():
            binary_output = self.binary_model(input_tensor)
            binary_probs = torch.softmax(binary_output, dim=1).cpu().numpy()[0]
            binary_pred = np.argmax(binary_probs)
            binary_conf = binary_probs[binary_pred]
        
        # --- Multi-class Classification ---
        with torch.no_grad():
            class_output = self.multiclass_model(input_tensor)
            class_probs = torch.softmax(class_output, dim=1).cpu().numpy()[0]
            class_pred = np.argmax(class_probs)
            class_conf = class_probs[class_pred]
            fault_type = self.label_encoder.inverse_transform([class_pred])[0]
        
        # --- Generate Alert ---
        if binary_pred == 0:  # Normal
            status = "✅ NORMAL"
            severity = "None"
            color = "green"
        else:  # Fault
            if binary_conf > 0.9:
                status = "🚨 CRITICAL FAULT"
                severity = "Critical"
                color = "red"
            elif binary_conf > 0.7:
                status = "⚠️ FAULT DETECTED"
                severity = "Warning"
                color = "orange"
            else:
                status = "⚡ POSSIBLE FAULT"
                severity = "Low"
                color = "yellow"
        
        alert = {
            'status': status,
            'is_fault': bool(binary_pred),
            'fault_confidence': float(binary_conf),
            'fault_type': fault_type,
            'fault_type_confidence': float(class_conf),
            'severity': severity,
            'buffer_status': buffer_status,
            'raw_reading': sensor_reading,
            'probabilities': {
                'normal': float(binary_probs[0]),
                'fault': float(binary_probs[1])
            }
        }
        
        self.alert_history.append(alert)
        return alert
    
    def display_alert(self, alert):
        """Pretty-print the alert information."""
        print(f"\n{'='*60}")
        print(f"  {alert['status']}")
        print(f"{'='*60}")
        print(f"  📊 Fault Confidence:  {alert['fault_confidence']:.2%}")
        print(f"  🏷️ Fault Type:        {alert['fault_type']} ({alert['fault_type_confidence']:.2%})")
        print(f"  ⚡ Severity:          {alert['severity']}")
        print(f"  📡 Buffer:            {alert['buffer_status']}")
        print(f"  📊 P(Normal):         {alert['probabilities']['normal']:.4f}")
        print(f"  📊 P(Fault):          {alert['probabilities']['fault']:.4f}")
        print(f"{'='*60}")

# --- Instantiate the Monitor ---
monitor = TransformerMonitor(
    binary_model=model_binary,
    multiclass_model=model_multiclass,
    scaler=scaler_detect,
    label_encoder=le_fault,
    feature_cols=feature_cols,
    seq_length=SEQUENCE_LENGTH,
    device=DEVICE
)

print("✅ TransformerMonitor initialized and ready for real-time predictions!")

---
# CELL 32
# ============================================================
# 🚨 CELL 24: Demo - Real-Time Fault Detection
# ============================================================

print("=" * 70)
print("🚨 REAL-TIME TRANSFORMER FAULT DETECTION DEMO")
print("=" * 70)

# --- Test Case 1: Normal Operation ---
print("\n📡 Test Case 1: Normal Operating Conditions")
normal_reading = {
    'Ia': -170.47, 'Ib': 9.22, 'Ic': 161.25,
    'Va': 0.054, 'Vb': -0.660, 'Vc': 0.605
}
alert1 = monitor.predict(normal_reading)
monitor.display_alert(alert1)

# --- Test Case 2: Phase A-G Fault ---
print("\n📡 Test Case 2: Suspected Phase A-Ground Fault")
fault_reading_ag = {
    'Ia': -593.94, 'Ib': -217.70, 'Ic': -124.89,
    'Va': 0.236, 'Vb': -0.105, 'Vc': -0.131
}
alert2 = monitor.predict(fault_reading_ag)
monitor.display_alert(alert2)

# --- Test Case 3: Severe Multi-phase Fault ---
print("\n📡 Test Case 3: Severe Imbalanced Current (Multi-phase Fault)")
fault_reading_severe = {
    'Ia': -900.0, 'Ib': 50.0, 'Ic': 850.0,
    'Va': 0.9, 'Vb': -0.01, 'Vc': -0.89
}
alert3 = monitor.predict(fault_reading_severe)
monitor.display_alert(alert3)

# --- Feed more normal readings to fill buffer ---
print("\n📡 Feeding additional readings to fill LSTM buffer...")
for i in range(8):
    test_reading = {
        'Ia': -170.0 + np.random.normal(0, 5),
        'Ib': 9.0 + np.random.normal(0, 2),
        'Ic': 161.0 + np.random.normal(0, 5),
        'Va': 0.054 + np.random.normal(0, 0.01),
        'Vb': -0.660 + np.random.normal(0, 0.01),
        'Vc': 0.605 + np.random.normal(0, 0.01)
    }
    alert = monitor.predict(test_reading)

print("  Buffer filled! Making full-context predictions...")

# --- Test Case 4: Normal with full buffer ---
print("\n📡 Test Case 4: Normal Reading (Full Buffer Context)")
alert4 = monitor.predict(normal_reading)
monitor.display_alert(alert4)

# --- Test Case 5: Fault after normal readings (trend detection!) ---
print("\n📡 Test Case 5: Fault After Normal Trend (LSTM Trend Detection)")
alert5 = monitor.predict(fault_reading_ag)
monitor.display_alert(alert5)

print("\n✅ Real-time demonstration complete!")

---
# CELL 33
# ============================================================
# 🔧 CELL 25: Standalone predict_fault() Function
# ============================================================

def predict_fault(sensor_reading):
    # Clear the buffer to prevent past test cases from skewing the sequence
    monitor.buffer.clear()
    """
    Standalone function for real-time fault prediction.
    
    Takes a sensor reading dictionary and returns 'Normal' or 'Fault' alert.
    
    Args:
        sensor_reading (dict): Must contain keys:
            'Ia', 'Ib', 'Ic' - Phase currents (Amperes)
            'Va', 'Vb', 'Vc' - Phase voltages (per-unit or Volts)
    
    Returns:
        str: 'Normal' if system is healthy, 'Fault' with details if fault detected.
    
    Example:
        >>> reading = {'Ia': -150.5, 'Ib': -9.7, 'Ic': 85.8,
        ...            'Va': 0.40, 'Vb': -0.13, 'Vc': -0.27}
        >>> result = predict_fault(reading)
        >>> print(result)  # 'Normal' or '⚠️ FAULT: AG_Fault (Confidence: 95.2%)'
    """
    alert = monitor.predict(sensor_reading)
    
    if not alert['is_fault']:
        return f"Normal (Confidence: {alert['probabilities']['normal']:.1%})"
    else:
        return (f"⚠️ FAULT: {alert['fault_type']} "
                f"(Confidence: {alert['fault_confidence']:.1%}, "
                f"Severity: {alert['severity']})")

# --- Quick Test ---
print("=" * 70)
print("🔧 predict_fault() - Standalone Function Test")
print("=" * 70)

test_readings = [
    {'Ia': -170.47, 'Ib': 9.22, 'Ic': 161.25, 'Va': 0.054, 'Vb': -0.660, 'Vc': 0.605},
    {'Ia': -593.94, 'Ib': -217.70, 'Ic': -124.89, 'Va': 0.236, 'Vb': -0.105, 'Vc': -0.131},
    {'Ia': -900.0, 'Ib': 50.0, 'Ic': 850.0, 'Va': 0.9, 'Vb': -0.01, 'Vc': -0.89},
]

labels = ['Normal Operation', 'AG Fault Condition', 'Severe Imbalance']

for reading, label in zip(test_readings, labels):
    result = predict_fault(reading)
    print(f"\n  📡 {label}:")
    print(f"     Input: Ia={reading['Ia']:.1f}, Ib={reading['Ib']:.1f}, Ic={reading['Ic']:.1f}")
    print(f"     Result: {result}")

---
# CELL 35
# ============================================================
# 🏥 CELL 26: DGA Health Index Analysis & Prediction
# ============================================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 70)
print("🏥 DGA-Based Transformer Health Prediction")
print("=" * 70)

# --- Regression Model: Predict Health Index ---
X_train_dga, X_test_dga, y_train_dga, y_test_dga = train_test_split(
    X_dga, y_dga, test_size=0.2, random_state=42
)

gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
gb_model.fit(X_train_dga, y_train_dga)

y_pred_dga = gb_model.predict(X_test_dga)
mae_dga = mean_absolute_error(y_test_dga, y_pred_dga)
r2_dga = r2_score(y_test_dga, y_pred_dga)

print(f"\n📊 Health Index Regression Results:")
print(f"   MAE:  {mae_dga:.4f}")
print(f"   R²:   {r2_dga:.4f}")

# --- Classification Model: Health Category ---
X_train_dga_c, X_test_dga_c, y_train_dga_c, y_test_dga_c = train_test_split(
    X_dga, y_dga_class, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train_dga_c, y_train_dga_c)

y_pred_dga_c = rf_model.predict(X_test_dga_c)
print(f"\n📊 Health Category Classification:")
print(classification_report(y_test_dga_c, y_pred_dga_c, 
                             target_names=le_health.classes_, digits=4))

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('🏥 DGA Health Index Analysis', fontsize=16, fontweight='bold')

# Actual vs Predicted
axes[0].scatter(y_test_dga, y_pred_dga, alpha=0.5, color='#6C5CE7', s=30)
axes[0].plot([y_test_dga.min(), y_test_dga.max()], [y_test_dga.min(), y_test_dga.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_title(f'Actual vs Predicted Health Index\nR² = {r2_dga:.4f}')
axes[0].set_xlabel('Actual Health Index')
axes[0].set_ylabel('Predicted Health Index')
axes[0].legend()

# Feature importance
importances = gb_model.feature_importances_
feat_imp = pd.Series(importances, index=dga_features).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[1], color=sns.color_palette('viridis', len(dga_features)))
axes[1].set_title('Feature Importance (DGA)')
axes[1].set_xlabel('Importance')

# Health category confusion matrix
cm_dga = confusion_matrix(y_test_dga_c, y_pred_dga_c)
sns.heatmap(cm_dga, annot=True, fmt='d', cmap='Greens',
            xticklabels=le_health.classes_, yticklabels=le_health.classes_, ax=axes[2])
axes[2].set_title('Health Category Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/dga_health_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ DGA Health prediction complete!")

---
# CELL 36
# ============================================================
# 📡 CELL 27: IoT Sensor Time-Series Analysis
# ============================================================

print("=" * 70)
print("📡 IoT Sensor Time-Series Monitoring Dashboard")
print("=" * 70)

# Select key columns for visualization
key_cols_plot = [col for col in ['VL1', 'VL2', 'VL3', 'IL1', 'IL2', 'IL3', 'OTI', 'WTI', 'OLI'] 
                 if col in df_iot_merged.columns]

if len(key_cols_plot) > 0 and len(df_iot_merged) > 0:
    # Take a representative sample for plotting
    sample_size = min(500, len(df_iot_merged))
    df_plot = df_iot_merged[key_cols_plot].iloc[:sample_size]
    
    n_plots = min(len(key_cols_plot), 6)
    fig, axes = plt.subplots(n_plots, 1, figsize=(18, 3*n_plots), sharex=True)
    fig.suptitle('📡 IoT Sensor Time-Series (Sample)', fontsize=16, fontweight='bold')
    
    if n_plots == 1:
        axes = [axes]
    
    colors = sns.color_palette('husl', n_plots)
    
    for i, col in enumerate(key_cols_plot[:n_plots]):
        axes[i].plot(df_plot.index, df_plot[col], color=colors[i], linewidth=0.8, alpha=0.8)
        axes[i].fill_between(df_plot.index, df_plot[col], alpha=0.1, color=colors[i])
        axes[i].set_ylabel(col, fontsize=11)
        axes[i].grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = df_plot[col].mean()
        axes[i].axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[i].text(df_plot.index[0], mean_val, f' Mean: {mean_val:.1f}', 
                     color='red', fontsize=9, va='bottom')
    
    axes[-1].set_xlabel('Time', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/iot_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ IoT time-series visualization saved!")
else:
    print("⚠️ Not enough IoT data for time-series plot (this is normal if data was mostly zeros)")
    print(f"   Available columns: {key_cols_plot}")
    print(f"   DataFrame length: {len(df_iot_merged)}")

---
# CELL 37
# ============================================================
# 🔍 CELL 28: Inspection Data - Health Prediction Model
# ============================================================

print("=" * 70)
print("🔍 Inspection-Based Health Prediction")
print("=" * 70)

# Train a simple model to predict health index from inspection features
X_train_insp, X_test_insp, y_train_insp, y_test_insp = train_test_split(
    X_insp, y_insp, test_size=0.2, random_state=42
)

gb_insp = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
gb_insp.fit(X_train_insp, y_train_insp)

y_pred_insp = gb_insp.predict(X_test_insp)
mae_insp = mean_absolute_error(y_test_insp, y_pred_insp)
r2_insp = r2_score(y_test_insp, y_pred_insp)

print(f"\nInspection Health Prediction Results:")
print(f"  MAE:  {mae_insp:.4f}")
print(f"  R²:   {r2_insp:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('🔍 Inspection-Based Health Prediction', fontsize=16, fontweight='bold')

axes[0].scatter(y_test_insp, y_pred_insp, alpha=0.3, color='#e67e22', s=15)
axes[0].plot([y_test_insp.min(), y_test_insp.max()], [y_test_insp.min(), y_test_insp.max()], 
             'r--', linewidth=2)
axes[0].set_title(f'Actual vs Predicted Health Index\nR² = {r2_insp:.4f}')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')

feat_imp_insp = pd.Series(gb_insp.feature_importances_, index=inspection_features).sort_values(ascending=True)
feat_imp_insp.plot(kind='barh', ax=axes[1], color=sns.color_palette('rocket', len(inspection_features)))
axes[1].set_title('Feature Importance (Inspection)')

plt.tight_layout()
plt.savefig('outputs/inspection_health_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Inspection health prediction complete!")

---
# CELL 39
# ============================================================
# 💾 CELL 29: Export Models for Deployment
# ============================================================

print("=" * 70)
print("💾 Exporting Models for Low-Cost Deployment")
print("=" * 70)

# --- Export as TorchScript (for Raspberry Pi / Edge deployment) ---
print("\n📦 Exporting as TorchScript...")

model_binary.eval()
model_multiclass.eval()

# Create example input
example_input = torch.randn(1, SEQUENCE_LENGTH, len(feature_cols)).to(DEVICE)

# Trace models
traced_binary = torch.jit.trace(model_binary, example_input)
traced_multiclass = torch.jit.trace(model_multiclass, example_input)

# Save TorchScript models
traced_binary.save('models/binary_detector_torchscript.pt')
traced_multiclass.save('models/multiclass_classifier_torchscript.pt')
print("  ✓ TorchScript models saved!")

# --- Export as ONNX (for mobile/web deployment) ---
print("\n📦 Exporting as ONNX...")
try:
    torch.onnx.export(
        model_binary,
        example_input,
        'models/binary_detector.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['sensor_sequence'],
        output_names=['fault_prediction'],
        dynamic_axes={'sensor_sequence': {0: 'batch_size'},
                      'fault_prediction': {0: 'batch_size'}}
    )
    print("  ✓ ONNX binary model saved!")
    
    torch.onnx.export(
        model_multiclass,
        example_input,
        'models/multiclass_classifier.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['sensor_sequence'],
        output_names=['fault_classification'],
        dynamic_axes={'sensor_sequence': {0: 'batch_size'},
                      'fault_classification': {0: 'batch_size'}}
    )
    print("  ✓ ONNX multi-class model saved!")
except Exception as e:
    print(f"  ⚠️ ONNX export skipped (optional): {e}")

# --- Save Scaler and Label Encoder ---
import pickle

with open('models/scaler_detect.pkl', 'wb') as f:
    pickle.dump(scaler_detect, f)
    
with open('models/label_encoder_fault.pkl', 'wb') as f:
    pickle.dump(le_fault, f)
    
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("\n  ✓ Scaler and Label Encoder saved!")

# --- Print model sizes ---
print("\n📊 Model File Sizes:")
for f in os.listdir('models'):
    fpath = os.path.join('models', f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f}: {size_kb:.1f} KB")

print("\n✅ All models exported successfully!")
print("   Ready for deployment on Raspberry Pi, Mobile, or Cloud!")

---
# CELL 41
# ============================================================
# 📊 CELL 30: Final Results Summary Dashboard
# ============================================================

print("=" * 70)
print("📊 FINAL RESULTS SUMMARY")
print("   Smart Industrial Transformer Monitoring System")
print("=" * 70)

# Summary table
summary_results = pd.DataFrame({
    'Model': [
        'LSTM Binary Fault Detector',
        'LSTM Multi-class Fault Classifier',
        'DGA Health Regressor (GBR)',
        'DGA Health Classifier (RF)',
        'Inspection Health Regressor (GBR)'
    ],
    'Task': [
        'Normal vs Fault Detection',
        'Specific Fault Type Classification',
        'Health Index Prediction',
        'Health Category Classification',
        'Inspection Health Prediction'
    ],
    'Metric': [
        f'Accuracy: {acc_bin:.4f}, F1: {f1_bin:.4f}',
        f'Accuracy: {acc_cls:.4f}, F1: {f1_cls:.4f}',
        f'MAE: {mae_dga:.4f}, R²: {r2_dga:.4f}',
        f'Accuracy: {accuracy_score(y_test_dga_c, y_pred_dga_c):.4f}',
        f'MAE: {mae_insp:.4f}, R²: {r2_insp:.4f}'
    ],
    'Architecture': [
        'Input→LSTM(64)→Drop→LSTM(32)→Dense(16)→Softmax(2)',
        'Input→LSTM(64)→Drop→LSTM(32)→Dense(16)→Softmax(N)',
        'Gradient Boosting (200 trees)',
        'Random Forest (200 trees)',
        'Gradient Boosting (150 trees)'
    ]
})

display(summary_results)

# --- Final visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('📊 System Performance Overview', fontsize=16, fontweight='bold')

# Model accuracies comparison
model_names = ['Binary\nDetection', 'Multi-class\nClassification', 
               'DGA Health\n(R²)', 'Inspection\nHealth (R²)']
scores = [acc_bin, acc_cls, r2_dga, r2_insp]
colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = axes[0].bar(model_names, scores, color=colors_bar, edgecolor='black', alpha=0.85)
axes[0].set_title('Model Performance Comparison')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1.1)
for bar, score in zip(bars, scores):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{score:.3f}', ha='center', fontweight='bold')

# Dataset sizes
ds_names = ['IoT\nSensor', 'DGA', 'Fault\nClass', 'Fault\nDetect', 'Inspection']
ds_sizes = [len(df_iot_merged), len(df_dga), len(df_fault_class), len(df_fault_detect), len(df_inspection)]
axes[1].bar(ds_names, ds_sizes, color=sns.color_palette('pastel', len(ds_names)), edgecolor='black')
axes[1].set_title('Dataset Sizes')
axes[1].set_ylabel('Samples')
for i, v in enumerate(ds_sizes):
    axes[1].text(i, v + max(ds_sizes)*0.02, str(v), ha='center', fontweight='bold')

# Feature counts
feat_names = ['Electrical\n(V,I)', 'Engineered\n(Power,Mag)', 'DGA\nGases', 'Inspection']
feat_counts = [6, 9, 17, 5]
axes[2].barh(feat_names, feat_counts, color=sns.color_palette('muted', len(feat_names)), edgecolor='black')
axes[2].set_title('Feature Engineering Summary')
axes[2].set_xlabel('Number of Features')
for i, v in enumerate(feat_counts):
    axes[2].text(v + 0.3, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/final_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("🎉 PROJECT COMPLETE!")
print("=" * 70)
print("""
📁 Project File Structure:
├── 📓 Transformer_Monitoring_System.ipynb  (This Notebook)
├── 📂 data/
│   ├── archive/          ← IoT Sensor Data (V, I, Power, Temp)
│   ├── archive1/         ← DGA Health Index Data
│   ├── archive2/         ← Electrical Fault Classification
│   └── archive3/         ← Periodic Inspection Data
├── 📂 models/
│   ├── binary_fault_detector.pth
│   ├── multiclass_fault_classifier.pth
│   ├── binary_detector_torchscript.pt
│   ├── multiclass_classifier_torchscript.pt
│   ├── *.onnx            ← For mobile/edge deployment
│   ├── scaler_detect.pkl
│   ├── label_encoder_fault.pkl
│   └── feature_cols.pkl
├── 📂 outputs/
│   ├── iot_sensor_distributions.png
│   ├── thermal_distributions.png
│   ├── dga_analysis.png
│   ├── fault_classification_eda.png
│   ├── inspection_eda.png
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   ├── roc_and_class_accuracy.png
│   ├── dga_health_prediction.png
│   ├── inspection_health_prediction.png
│   ├── iot_timeseries.png
│   └── final_summary.png
└── 📂 utils/
    └── (utility scripts)

🚀 Deployment Options:
  • Raspberry Pi  → Use TorchScript (.pt) models
  • Mobile App    → Use ONNX models
  • Cloud/AWS     → Use PyTorch (.pth) models
  • Edge Device   → Use ONNX + quantization
""")
