import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# Download from: https://www.kaggle.com/datasets/mirichoi0218/insurance
# Save as insurance.csv in the same folder as this script
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv('insurance.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample:")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nBasic stats:\n", df.describe().round(2))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: EDA — WHAT DRIVES HIGH CLAIMS?
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Health Insurance — Risk Factor Analysis', fontsize=15, fontweight='bold')

# Plot 1: Charges by smoker status
smoker_charges = df.groupby('smoker')['charges'].mean().reset_index()
axes[0, 0].bar(smoker_charges['smoker'], smoker_charges['charges'],
               color=['steelblue', 'tomato'], edgecolor='white', width=0.5)
axes[0, 0].set_title('Average Charges: Smoker vs Non-Smoker')
axes[0, 0].set_xlabel('Smoker Status')
axes[0, 0].set_ylabel('Average Charges ($)')
for i, v in enumerate(smoker_charges['charges']):
    axes[0, 0].text(i, v + 200, f'${v:,.0f}', ha='center', fontweight='bold')

# Plot 2: Charges by age band
df['age_band'] = pd.cut(df['age'], bins=[17, 30, 40, 50, 65],
                         labels=['18-30', '31-40', '41-50', '51+'])
age_charges = df.groupby('age_band', observed=True)['charges'].mean().reset_index()
axes[0, 1].bar(age_charges['age_band'].astype(str), age_charges['charges'],
               color='steelblue', edgecolor='white')
axes[0, 1].set_title('Average Charges by Age Band')
axes[0, 1].set_xlabel('Age Band')
axes[0, 1].set_ylabel('Average Charges ($)')
for i, v in enumerate(age_charges['charges']):
    axes[0, 1].text(i, v + 200, f'${v:,.0f}', ha='center', fontsize=9)

# Plot 3: Charges by BMI band
df['bmi_band'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                         labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
bmi_charges = df.groupby('bmi_band', observed=True)['charges'].mean().reset_index()
axes[1, 0].bar(bmi_charges['bmi_band'].astype(str), bmi_charges['charges'],
               color=['lightgreen', 'steelblue', 'orange', 'tomato'], edgecolor='white')
axes[1, 0].set_title('Average Charges by BMI Band')
axes[1, 0].set_xlabel('BMI Band')
axes[1, 0].set_ylabel('Average Charges ($)')
for i, v in enumerate(bmi_charges['charges']):
    axes[1, 0].text(i, v + 200, f'${v:,.0f}', ha='center', fontsize=9)

# Plot 4: Charge distribution
axes[1, 1].hist(df['charges'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[1, 1].axvline(df['charges'].mean(), color='tomato', linestyle='--',
                    label=f"Mean: ${df['charges'].mean():,.0f}")
axes[1, 1].axvline(df['charges'].median(), color='orange', linestyle='--',
                    label=f"Median: ${df['charges'].median():,.0f}")
axes[1, 1].set_title('Distribution of Insurance Charges')
axes[1, 1].set_xlabel('Charges ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('eda_risk_factors.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: eda_risk_factors.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: ACTUARIAL RATING FACTORS
# Rating factors show how much each risk characteristic multiplies the base rate
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ACTUARIAL RATING FACTORS")
print("="*60)

base_rate = df['charges'].mean()
print(f"\nBase Rate (Overall Average Charge): ${base_rate:,.2f}")

# Smoker relativity
smoker_relativity = df.groupby('smoker')['charges'].mean() / base_rate
print("\nSmoker Relativity:")
print(smoker_relativity.round(3))

# Age band relativity
age_relativity = df.groupby('age_band', observed=True)['charges'].mean() / base_rate
print("\nAge Band Relativity:")
print(age_relativity.round(3))

# BMI band relativity
bmi_relativity = df.groupby('bmi_band', observed=True)['charges'].mean() / base_rate
print("\nBMI Band Relativity:")
print(bmi_relativity.round(3))

# Save rating factor tables
rating_factors = pd.DataFrame({
    'Factor': ['Non-Smoker', 'Smoker',
               'Age 18-30', 'Age 31-40', 'Age 41-50', 'Age 51+',
               'BMI Underweight', 'BMI Normal', 'BMI Overweight', 'BMI Obese'],
    'Relativity': [
        smoker_relativity['no'], smoker_relativity['yes'],
        age_relativity.get('18-30', 1.0), age_relativity.get('31-40', 1.0),
        age_relativity.get('41-50', 1.0), age_relativity.get('51+', 1.0),
        bmi_relativity.get('Underweight', 1.0), bmi_relativity.get('Normal', 1.0),
        bmi_relativity.get('Overweight', 1.0), bmi_relativity.get('Obese', 1.0)
    ]
}).round(3)

print("\nFull Rating Factor Table:")
print(rating_factors.to_string(index=False))

# Premium calculation function using rating factors
def calculate_premium(age, bmi, smoker, base=base_rate,
                       s_rel=smoker_relativity, a_rel=age_relativity, b_rel=bmi_relativity):
    # Age band
    if age <= 30:
        ab = '18-30'
    elif age <= 40:
        ab = '31-40'
    elif age <= 50:
        ab = '41-50'
    else:
        ab = '51+'

    # BMI band
    if bmi < 18.5:
        bb = 'Underweight'
    elif bmi < 25:
        bb = 'Normal'
    elif bmi < 30:
        bb = 'Overweight'
    else:
        bb = 'Obese'

    smoker_key = 'yes' if smoker else 'no'
    premium = base * s_rel[smoker_key] * a_rel[ab] * b_rel[bb]
    return round(premium, 2)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: RISK SEGMENTATION — K-MEANS CLUSTERING
# Segment policyholders into risk tiers based on key features
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RISK SEGMENTATION — K-MEANS CLUSTERING")
print("="*60)

# Encode smoker for clustering
df['smoker_enc'] = (df['smoker'] == 'yes').astype(int)

# Features for clustering
cluster_features = ['age', 'bmi', 'smoker_enc', 'charges']
X_cluster = df[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Elbow method to confirm 4 clusters
inertia = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Fit with 4 clusters (Low / Medium / High / Very High)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Label clusters by average charges
cluster_means = df.groupby('cluster')['charges'].mean().sort_values()
cluster_label_map = {
    cluster_means.index[0]: 'Low Risk',
    cluster_means.index[1]: 'Medium Risk',
    cluster_means.index[2]: 'High Risk',
    cluster_means.index[3]: 'Very High Risk'
}
df['risk_tier'] = df['cluster'].map(cluster_label_map)

# Cluster summary
cluster_summary = df.groupby('risk_tier').agg(
    Count=('charges', 'count'),
    Avg_Age=('age', 'mean'),
    Avg_BMI=('bmi', 'mean'),
    Pct_Smoker=('smoker_enc', 'mean'),
    Avg_Charges=('charges', 'mean'),
    Min_Charges=('charges', 'min'),
    Max_Charges=('charges', 'max')
).round(2)

print("\nRisk Tier Summary:")
print(cluster_summary.to_string())

# Plot clusters
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Risk Segmentation — K-Means Clustering', fontsize=13, fontweight='bold')

colors = {'Low Risk': 'steelblue', 'Medium Risk': 'orange',
          'High Risk': 'tomato', 'Very High Risk': 'darkred'}

for tier, color in colors.items():
    subset = df[df['risk_tier'] == tier]
    axes[0].scatter(subset['age'], subset['charges'], c=color,
                    label=tier, alpha=0.5, s=20)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Charges ($)')
axes[0].set_title('Risk Tiers: Age vs Charges')
axes[0].legend()

for tier, color in colors.items():
    subset = df[df['risk_tier'] == tier]
    axes[1].scatter(subset['bmi'], subset['charges'], c=color,
                    label=tier, alpha=0.5, s=20)
axes[1].set_xlabel('BMI')
axes[1].set_ylabel('Charges ($)')
axes[1].set_title('Risk Tiers: BMI vs Charges')
axes[1].legend()

plt.tight_layout()
plt.savefig('risk_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: risk_segmentation.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: UNDERWRITING DECISION ENGINE — DECISION TREE
# Learns underwriting rules from the data
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("UNDERWRITING DECISION ENGINE — DECISION TREE")
print("="*60)

# Define underwriting label from risk tier
tier_to_decision = {
    'Low Risk': 'ACCEPT',
    'Medium Risk': 'ACCEPT WITH LOADING',
    'High Risk': 'REFER',
    'Very High Risk': 'DECLINE'
}
df['uw_decision'] = df['risk_tier'].map(tier_to_decision)

# Encode categoricals for model
le_sex = LabelEncoder()
le_region = LabelEncoder()
df['sex_enc'] = le_sex.fit_transform(df['sex'])
df['region_enc'] = le_region.fit_transform(df['region'])

features = ['age', 'bmi', 'children', 'smoker_enc', 'sex_enc', 'region_enc']
X = df[features]
y = df['uw_decision']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree — max_depth=4 keeps it readable and interpretable
dt = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=20)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(22, 10))
plot_tree(dt, feature_names=features,
          class_names=dt.classes_,
          filled=True, rounded=True, fontsize=9)
plt.title('Underwriting Decision Tree', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: decision_tree.png")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Decision Tree):")
print(importance_df.round(4).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: FULL UNDERWRITING OUTPUT TABLE
# For every policyholder: risk tier, decision, premium loading, final premium
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("UNDERWRITING OUTPUT")
print("="*60)

loading_map = {
    'ACCEPT': 0.00,
    'ACCEPT WITH LOADING': 0.20,
    'REFER': 0.40,
    'DECLINE': None
}

df['uw_decision_pred'] = dt.predict(X)
df['loading_pct'] = df['uw_decision_pred'].map(loading_map)
df['base_premium'] = df.apply(
    lambda row: calculate_premium(row['age'], row['bmi'], row['smoker'] == 'yes'), axis=1)
df['final_premium'] = df.apply(
    lambda row: round(row['base_premium'] * (1 + row['loading_pct']), 2)
    if row['loading_pct'] is not None else None, axis=1)

output_cols = ['age', 'sex', 'bmi', 'smoker', 'region',
               'risk_tier', 'uw_decision_pred', 'loading_pct',
               'base_premium', 'final_premium', 'charges']

print("\nSample Underwriting Output (first 15 rows):")
print(df[output_cols].head(15).to_string(index=False))

# Decision breakdown
print("\nUnderwriting Decision Distribution:")
print(df['uw_decision_pred'].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: LOSS RATIO BY RISK SEGMENT
# Loss Ratio = Expected Claims / Premium Collected
# A healthy insurer targets loss ratio of 70-85%
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("LOSS RATIO ANALYSIS BY RISK SEGMENT")
print("="*60)

accepted = df[df['uw_decision_pred'] != 'DECLINE'].copy()

loss_ratio = accepted.groupby('risk_tier').agg(
    Policies=('charges', 'count'),
    Total_Claims=('charges', 'sum'),
    Total_Premium=('final_premium', 'sum'),
    Avg_Claim=('charges', 'mean'),
    Avg_Premium=('final_premium', 'mean')
).round(2)

loss_ratio['Loss_Ratio_%'] = (loss_ratio['Total_Claims'] / loss_ratio['Total_Premium'] * 100).round(1)
loss_ratio['Profitable'] = loss_ratio['Loss_Ratio_%'].apply(
    lambda x: 'Yes' if x < 85 else 'No')

print("\nLoss Ratio by Risk Tier:")
print(loss_ratio.to_string())

overall_lr = accepted['charges'].sum() / accepted['final_premium'].sum() * 100
print(f"\nOverall Portfolio Loss Ratio: {overall_lr:.1f}%")
print("(Target: 70-85% for a healthy insurance portfolio)")

# Plot loss ratio
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Portfolio Loss Ratio Analysis', fontsize=13, fontweight='bold')

tiers = loss_ratio.index.tolist()
lr_vals = loss_ratio['Loss_Ratio_%'].tolist()
bar_colors = ['steelblue' if v < 85 else 'tomato' for v in lr_vals]

axes[0].bar(tiers, lr_vals, color=bar_colors, edgecolor='white')
axes[0].axhline(85, color='red', linestyle='--', label='85% threshold')
axes[0].axhline(70, color='green', linestyle='--', label='70% threshold')
axes[0].set_title('Loss Ratio by Risk Tier')
axes[0].set_xlabel('Risk Tier')
axes[0].set_ylabel('Loss Ratio (%)')
axes[0].legend()
for i, v in enumerate(lr_vals):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

# Decision distribution pie
decision_counts = df['uw_decision_pred'].value_counts()
pie_colors = ['steelblue', 'orange', 'gold', 'tomato']
axes[1].pie(decision_counts.values, labels=decision_counts.index,
            autopct='%1.1f%%', colors=pie_colors, startangle=90)
axes[1].set_title('Underwriting Decision Distribution')

plt.tight_layout()
plt.savefig('loss_ratio_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: loss_ratio_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

df[output_cols].to_csv('underwriting_output.csv', index=False)
rating_factors.to_csv('rating_factors.csv', index=False)
loss_ratio.to_csv('loss_ratio_summary.csv')

print("\n" + "="*60)
print("ALL OUTPUTS SAVED")
print("="*60)
print("underwriting_output.csv  — full applicant-level decisions")
print("rating_factors.csv       — actuarial rating factor table")
print("loss_ratio_summary.csv   — loss ratio by risk segment")
print("eda_risk_factors.png     — EDA plots")
print("risk_segmentation.png    — cluster plots")
print("decision_tree.png        — decision tree visual")
print("loss_ratio_analysis.png  — loss ratio and decision mix")
print("\nProject complete.")
