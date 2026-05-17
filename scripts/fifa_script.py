# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Model persistence
import joblib

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', 50)
%matplotlib inline

print("All libraries imported successfully!")

# Load the FIFA 22 player dataset
fifa = pd.read_csv('players_22.csv', low_memory=False)

print(f"Dataset shape: {fifa.shape}")
print(f"Number of players: {fifa.shape[0]:,}")
print(f"Number of features: {fifa.shape[1]}")

# First few rows
fifa.head()

# Data types and non-null counts
fifa.info()

# Statistical summary
fifa.describe()

# Target variable: wage_eur
print("Target variable 'wage_eur' statistics:")
print(fifa['wage_eur'].describe())
print(f"\nRange: EUR {fifa['wage_eur'].min():,.0f} - EUR {fifa['wage_eur'].max():,.0f}")
print(f"Players with wage data: {fifa['wage_eur'].notna().sum():,}")
print(f"Players with zero wage: {(fifa['wage_eur'] == 0).sum():,}")

# Define columns to exclude
exclude_cols = [
    # IDs and URLs
    'sofifa_id', 'player_url', 'player_face_url', 'club_logo_url', 
    'club_flag_url', 'nation_logo_url', 'nation_flag_url',
    # Names
    'short_name', 'long_name',
    # Target variable
    'wage_eur',
    # Leaky features (directly related to wage/value)
    'value_eur', 'release_clause_eur',
    # Position-specific ratings (calculated/derived)
    'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw',
    'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm',
    'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk',
    # Other IDs
    'club_team_id', 'nationality_id', 'nation_team_id',
    # Date columns
    'dob', 'club_joined',
    # Complex text columns
    'player_tags', 'player_traits',
    # Mostly null
    'club_loaned_from', 'nation_position', 'nation_jersey_number'
]

# Select features
feature_cols = [col for col in fifa.columns if col not in exclude_cols]
print(f"Selected {len(feature_cols)} features")

# Create working dataset - only players with positive wages
df = fifa[fifa['wage_eur'] > 0][feature_cols + ['wage_eur']].copy()
print(f"Working dataset shape: {df.shape}")
print(f"Players with positive wages: {len(df):,}")

# Create wage categories for stratified sampling (using log scale due to skewness)
df['log_wage'] = np.log1p(df['wage_eur'])
df['wage_cat'] = pd.qcut(df['log_wage'], q=5, labels=[1, 2, 3, 4, 5])

print("Wage category distribution (quintiles):")
print(df['wage_cat'].value_counts().sort_index())

print("\nWage ranges per category:")
for cat in range(1, 6):
    cat_data = df[df['wage_cat'] == cat]['wage_eur']
    print(f"  Category {cat}: EUR {cat_data.min():,.0f} - EUR {cat_data.max():,.0f}")

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df['wage_cat']):
    strat_train_set = df.iloc[train_index].copy()
    strat_test_set = df.iloc[test_index].copy()

print(f"Training set size: {len(strat_train_set):,}")
print(f"Test set size: {len(strat_test_set):,}")

# Verify stratification
def wage_cat_proportions(data):
    return data['wage_cat'].value_counts().sort_index() / len(data)

compare_props = pd.DataFrame({
    'Overall': wage_cat_proportions(df),
    'Train': wage_cat_proportions(strat_train_set),
    'Test': wage_cat_proportions(strat_test_set)
})
print("Stratification verification:")
compare_props

# Remove temporary columns
strat_train_set = strat_train_set.drop(['wage_cat', 'log_wage'], axis=1)
strat_test_set = strat_test_set.drop(['wage_cat', 'log_wage'], axis=1)

# Create copies for exploration
fifa_train = strat_train_set.copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Raw distribution (highly skewed)
axes[0].hist(fifa_train['wage_eur'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Weekly Wage (EUR)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Player Wages (Raw)')

# Log-transformed distribution
axes[1].hist(np.log1p(fifa_train['wage_eur']), bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('Log(Weekly Wage + 1)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Log-Transformed Wages')

# Box plot
axes[2].boxplot(fifa_train['wage_eur'])
axes[2].set_ylabel('Weekly Wage (EUR)')
axes[2].set_title('Box Plot of Wages')

plt.tight_layout()
plt.show()

print(f"Raw wage skewness: {fifa_train['wage_eur'].skew():.3f}")
print(f"Log wage skewness: {np.log1p(fifa_train['wage_eur']).skew():.3f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Overall rating vs Wage
axes[0,0].scatter(fifa_train['overall'], fifa_train['wage_eur'], alpha=0.3, s=10)
axes[0,0].set_xlabel('Overall Rating')
axes[0,0].set_ylabel('Weekly Wage (EUR)')
axes[0,0].set_title('Overall Rating vs Wage')

# Age vs Wage
axes[0,1].scatter(fifa_train['age'], fifa_train['wage_eur'], alpha=0.3, s=10)
axes[0,1].set_xlabel('Age')
axes[0,1].set_ylabel('Weekly Wage (EUR)')
axes[0,1].set_title('Age vs Wage')

# International Reputation vs Wage
fifa_train.boxplot(column='wage_eur', by='international_reputation', ax=axes[0,2])
axes[0,2].set_xlabel('International Reputation')
axes[0,2].set_ylabel('Weekly Wage (EUR)')
axes[0,2].set_title('International Reputation vs Wage')
plt.suptitle('')

# Potential vs Wage
axes[1,0].scatter(fifa_train['potential'], fifa_train['wage_eur'], alpha=0.3, s=10)
axes[1,0].set_xlabel('Potential')
axes[1,0].set_ylabel('Weekly Wage (EUR)')
axes[1,0].set_title('Potential vs Wage')

# Overall vs Log Wage (clearer relationship)
axes[1,1].scatter(fifa_train['overall'], np.log1p(fifa_train['wage_eur']), alpha=0.3, s=10)
axes[1,1].set_xlabel('Overall Rating')
axes[1,1].set_ylabel('Log(Weekly Wage)')
axes[1,1].set_title('Overall Rating vs Log(Wage)')

# Age distribution by wage quartile
fifa_train['wage_quartile'] = pd.qcut(fifa_train['wage_eur'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
fifa_train.boxplot(column='age', by='wage_quartile', ax=axes[1,2])
axes[1,2].set_xlabel('Wage Quartile')
axes[1,2].set_ylabel('Age')
axes[1,2].set_title('Age by Wage Quartile')
plt.suptitle('')

plt.tight_layout()
plt.show()

fifa_train.drop('wage_quartile', axis=1, inplace=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 15 leagues by average wage
avg_wage_league = fifa_train.groupby('league_name')['wage_eur'].agg(['mean', 'count'])
avg_wage_league = avg_wage_league[avg_wage_league['count'] >= 50]
top_leagues = avg_wage_league.sort_values('mean', ascending=False).head(15)

axes[0].barh(top_leagues.index, top_leagues['mean'])
axes[0].set_xlabel('Average Weekly Wage (EUR)')
axes[0].set_title('Top 15 Leagues by Average Wage')
axes[0].invert_yaxis()

# Average wage by primary position
fifa_train['primary_position'] = fifa_train['player_positions'].str.split(',').str[0].str.strip()
avg_wage_pos = fifa_train.groupby('primary_position')['wage_eur'].mean().sort_values(ascending=False)

axes[1].bar(avg_wage_pos.index, avg_wage_pos.values)
axes[1].set_xlabel('Position')
axes[1].set_ylabel('Average Weekly Wage (EUR)')
axes[1].set_title('Average Wage by Position')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

fifa_train.drop('primary_position', axis=1, inplace=True)

# Calculate correlations with wage
numerical_cols = fifa_train.select_dtypes(include=[np.number]).columns.tolist()
correlations = fifa_train[numerical_cols].corr()['wage_eur'].sort_values(ascending=False)

print("Top 20 correlations with 'wage_eur':")
print(correlations.head(21))

print("\nBottom 10 correlations with 'wage_eur':")
print(correlations.tail(10))

# Correlation heatmap for top features
top_features = correlations.head(12).index.tolist()

plt.figure(figsize=(12, 10))
corr_matrix = fifa_train[top_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True)
plt.title('Correlation Heatmap of Top Features')
plt.tight_layout()
plt.show()

# Create engineered features for exploration
fifa_explore = fifa_train.copy()

# BMI
fifa_explore['bmi'] = fifa_explore['weight_kg'] / (fifa_explore['height_cm']/100)**2

# Potential growth
fifa_explore['potential_growth'] = fifa_explore['potential'] - fifa_explore['overall']

# Star power (overall * international reputation)
fifa_explore['star_power'] = fifa_explore['overall'] * fifa_explore['international_reputation']

# Check correlations with log wage
fifa_explore['log_wage'] = np.log1p(fifa_explore['wage_eur'])
new_features = ['bmi', 'potential_growth', 'star_power']

print("Correlations of engineered features with log(wage):")
for feat in new_features:
    corr = fifa_explore[[feat, 'log_wage']].corr().iloc[0, 1]
    print(f"  {feat}: {corr:.4f}")

# Visualize star_power relationship
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(fifa_explore['star_power'], fifa_explore['log_wage'], alpha=0.3, s=10)
axes[0].set_xlabel('Star Power (Overall x Reputation)')
axes[0].set_ylabel('Log(Wage)')
axes[0].set_title('Star Power vs Log(Wage)')

axes[1].scatter(fifa_explore['potential_growth'], fifa_explore['log_wage'], alpha=0.3, s=10)
axes[1].set_xlabel('Potential Growth')
axes[1].set_ylabel('Log(Wage)')
axes[1].set_title('Potential Growth vs Log(Wage)')

plt.tight_layout()
plt.show()

# Separate features and target (log-transformed)
fifa_train_features = strat_train_set.drop('wage_eur', axis=1)
fifa_train_labels = np.log1p(strat_train_set['wage_eur'].copy())

print(f"Training features shape: {fifa_train_features.shape}")
print(f"Training labels shape: {fifa_train_labels.shape}")
print(f"\nTarget (log wage) statistics:")
print(fifa_train_labels.describe())

# Check missing values
missing = fifa_train_features.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"Features with missing values: {len(missing)}")
print(missing.head(15))

# Identify column types
numerical_cols = fifa_train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = fifa_train_features.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"\nCategorical columns: {categorical_cols}")

class WageFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered features for wage prediction"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Star power
        if 'overall' in X.columns and 'international_reputation' in X.columns:
            X['star_power'] = X['overall'] * X['international_reputation']
        
        # Potential growth
        if 'potential' in X.columns and 'overall' in X.columns:
            X['potential_growth'] = X['potential'] - X['overall']
        
        # BMI
        if 'weight_kg' in X.columns and 'height_cm' in X.columns:
            X['bmi'] = X['weight_kg'] / ((X['height_cm']/100)**2 + 0.001)
        
        # Position features
        if 'player_positions' in X.columns:
            X['num_positions'] = X['player_positions'].str.count(',') + 1
            X['primary_position'] = X['player_positions'].str.split(',').str[0].str.strip()
            X['is_goalkeeper'] = (X['primary_position'] == 'GK').astype(int)
            X['is_attacker'] = X['primary_position'].isin(['ST', 'CF', 'LW', 'RW', 'LF', 'RF']).astype(int)
        
        return X

print("Custom transformer defined!")

# Core numerical features
core_numerical = [
    'overall', 'potential', 'age', 'height_cm', 'weight_kg',
    'weak_foot', 'skill_moves', 'international_reputation',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'
]

# Detailed skill attributes
detailed_skills = [
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys',
    'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 
    'movement_reactions', 'movement_balance',
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
    'mentality_vision', 'mentality_penalties', 'mentality_composure',
    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle'
]

# Goalkeeper attributes
gk_attributes = [
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
    'goalkeeping_positioning', 'goalkeeping_reflexes'
]

# Engineered features
engineered = ['star_power', 'potential_growth', 'bmi', 'num_positions', 'is_goalkeeper', 'is_attacker']

# Categorical features
categorical_features = ['preferred_foot', 'work_rate', 'body_type', 'primary_position']

# Filter to existing columns
numerical_to_use = [c for c in core_numerical + detailed_skills + gk_attributes if c in fifa_train_features.columns]
print(f"Numerical features: {len(numerical_to_use)}")

# Apply feature engineering
feature_engineer = WageFeatureEngineer()
fifa_train_engineered = feature_engineer.fit_transform(fifa_train_features)

# Final feature lists
numerical_final = numerical_to_use + [c for c in engineered if c in fifa_train_engineered.columns]
categorical_final = [c for c in categorical_features if c in fifa_train_engineered.columns]

print(f"Final numerical features: {len(numerical_final)}")
print(f"Final categorical features: {len(categorical_final)}")

# Build preprocessing pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_final),
    ('cat', categorical_pipeline, categorical_final)
])

print("Preprocessing pipeline created!")

# Fit and transform
fifa_prepared = preprocessor.fit_transform(fifa_train_engineered)
print(f"Prepared data shape: {fifa_prepared.shape}")

# Get feature names
try:
    cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_final)
    all_feature_names = list(numerical_final) + list(cat_names)
    print(f"Total features: {len(all_feature_names)}")
except:
    all_feature_names = None

def evaluate_model(model, X, y_log, model_name="Model"):
    """Evaluate model with log-transformed target"""
    pred_log = model.predict(X)
    
    # Log scale metrics
    rmse_log = np.sqrt(mean_squared_error(y_log, pred_log))
    mae_log = mean_absolute_error(y_log, pred_log)
    r2 = r2_score(y_log, pred_log)
    
    # Original scale metrics
    y_orig = np.expm1(y_log)
    pred_orig = np.expm1(pred_log)
    rmse_eur = np.sqrt(mean_squared_error(y_orig, pred_orig))
    mae_eur = mean_absolute_error(y_orig, pred_orig)
    
    print(f"{model_name}:")
    print(f"  [Log]  RMSE: {rmse_log:.4f}, MAE: {mae_log:.4f}, R2: {r2:.4f}")
    print(f"  [EUR]  RMSE: EUR {rmse_eur:,.0f}, MAE: EUR {mae_eur:,.0f}")
    
    return {'rmse_log': rmse_log, 'mae_log': mae_log, 'r2': r2}

# Linear Regression
print("="*60)
print("LINEAR REGRESSION")
print("="*60)
lin_reg = LinearRegression()
lin_reg.fit(fifa_prepared, fifa_train_labels)
lin_metrics = evaluate_model(lin_reg, fifa_prepared, fifa_train_labels, "Linear Regression")

# Decision Tree
print("="*60)
print("DECISION TREE")
print("="*60)
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(fifa_prepared, fifa_train_labels)
tree_metrics = evaluate_model(tree_reg, fifa_prepared, fifa_train_labels, "Decision Tree")

# Random Forest
print("="*60)
print("RANDOM FOREST")
print("="*60)
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
forest_reg.fit(fifa_prepared, fifa_train_labels)
forest_metrics = evaluate_model(forest_reg, fifa_prepared, fifa_train_labels, "Random Forest")

def cv_scores(model, X, y, name):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
    rmse = np.sqrt(-scores)
    print(f"{name}: RMSE = {rmse.mean():.4f} (+/- {rmse.std():.4f})")
    return rmse.mean(), rmse.std()

print("="*60)
print("10-FOLD CROSS-VALIDATION (Log Scale RMSE)")
print("="*60)

lin_cv = cv_scores(LinearRegression(), fifa_prepared, fifa_train_labels, "Linear Regression")
ridge_cv = cv_scores(Ridge(alpha=1.0), fifa_prepared, fifa_train_labels, "Ridge Regression")
tree_cv = cv_scores(DecisionTreeRegressor(random_state=42), fifa_prepared, fifa_train_labels, "Decision Tree")
forest_cv = cv_scores(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), fifa_prepared, fifa_train_labels, "Random Forest")
gb_cv = cv_scores(GradientBoostingRegressor(n_estimators=100, random_state=42), fifa_prepared, fifa_train_labels, "Gradient Boosting")

# Visualize CV results
cv_results = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'CV RMSE': [lin_cv[0], ridge_cv[0], tree_cv[0], forest_cv[0], gb_cv[0]],
    'Std': [lin_cv[1], ridge_cv[1], tree_cv[1], forest_cv[1], gb_cv[1]]
}).sort_values('CV RMSE')

plt.figure(figsize=(10, 5))
plt.barh(cv_results['Model'], cv_results['CV RMSE'], xerr=cv_results['Std'], capsize=5)
plt.xlabel('RMSE (Log Scale)')
plt.title('Cross-Validation Results')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nModel Comparison:")
print(cv_results.to_string(index=False))

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid_rf,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1
)

print("Running Grid Search for Random Forest...")
grid_search_rf.fit(fifa_prepared, fifa_train_labels)
print(f"\nBest params: {grid_search_rf.best_params_}")
print(f"Best RMSE: {np.sqrt(-grid_search_rf.best_score_):.4f}")

from scipy.stats import randint, uniform

param_dist_gb = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 12),
    'learning_rate': uniform(0.05, 0.2),
    'min_samples_split': randint(2, 15),
    'subsample': uniform(0.7, 0.3)
}

random_search_gb = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_dist_gb,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42
)

print("Running Randomized Search for Gradient Boosting...")
random_search_gb.fit(fifa_prepared, fifa_train_labels)
print(f"\nBest params: {random_search_gb.best_params_}")
print(f"Best RMSE: {np.sqrt(-random_search_gb.best_score_):.4f}")

best_rf = grid_search_rf.best_estimator_
best_gb = random_search_gb.best_estimator_

# Feature importance
if all_feature_names:
    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Features (Random Forest):")
    print(importance_df.head(15).to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 8))
    top15 = importance_df.head(15)
    plt.barh(top15['feature'], top15['importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Prepare test data
X_test = strat_test_set.drop('wage_eur', axis=1)
y_test_orig = strat_test_set['wage_eur'].copy()
y_test_log = np.log1p(y_test_orig)

# Transform
X_test_eng = feature_engineer.transform(X_test)
X_test_prep = preprocessor.transform(X_test_eng)

print(f"Test set shape: {X_test_prep.shape}")

print("="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

print("\n")
lin_test = evaluate_model(lin_reg, X_test_prep, y_test_log, "Linear Regression")
print()
rf_test = evaluate_model(best_rf, X_test_prep, y_test_log, "Random Forest (Tuned)")
print()
gb_test = evaluate_model(best_gb, X_test_prep, y_test_log, "Gradient Boosting (Tuned)")

# Select best model
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'RMSE (log)': [lin_test['rmse_log'], rf_test['rmse_log'], gb_test['rmse_log']],
    'R2': [lin_test['r2'], rf_test['r2'], gb_test['r2']]
}).sort_values('RMSE (log)')

print("\nFinal Results:")
print(results.to_string(index=False))

best_model_name = results.iloc[0]['Model']
print(f"\nBest Model: {best_model_name}")

final_model = best_rf if 'Random' in best_model_name else best_gb

# Prediction plots
y_pred_log = final_model.predict(X_test_prep)
y_pred_orig = np.expm1(y_pred_log)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log scale
axes[0].scatter(y_test_log, y_pred_log, alpha=0.3, s=10)
axes[0].plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'r--')
axes[0].set_xlabel('Actual Log(Wage)')
axes[0].set_ylabel('Predicted Log(Wage)')
axes[0].set_title('Actual vs Predicted (Log Scale)')

# Residuals
residuals = y_test_log - y_pred_log
axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual Distribution')

plt.tight_layout()
plt.show()

# Additional metrics
mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
rmse_eur = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae_eur = mean_absolute_error(y_test_orig, y_pred_orig)

print(f"Final Model Performance (EUR Scale):")
print(f"  RMSE: EUR {rmse_eur:,.0f}")
print(f"  MAE:  EUR {mae_eur:,.0f}")
print(f"  MAPE: {mape:.1f}%")

# Save models
joblib.dump(final_model, 'fifa_wage_model.pkl')
joblib.dump(feature_engineer, 'fifa_feature_engineer.pkl')
joblib.dump(preprocessor, 'fifa_preprocessor.pkl')

print("Models saved!")

# Test loading
loaded_model = joblib.load('fifa_wage_model.pkl')
loaded_fe = joblib.load('fifa_feature_engineer.pkl')
loaded_prep = joblib.load('fifa_preprocessor.pkl')

# Sample predictions
sample = X_test.head(5)
sample_eng = loaded_fe.transform(sample)
sample_prep = loaded_prep.transform(sample_eng)
sample_pred = np.expm1(loaded_model.predict(sample_prep))

print("Sample Predictions:")
for i, (pred, actual) in enumerate(zip(sample_pred, y_test_orig.head(5)), 1):
    print(f"  Player {i}: Predicted EUR {pred:,.0f}, Actual EUR {actual:,.0f}")

print("="*60)
print("PROJECT SUMMARY: FIFA 22 WAGE PREDICTION")
print("="*60)
print(f"\nDataset: {len(strat_train_set) + len(strat_test_set):,} players")
print(f"Train/Test: {len(strat_train_set):,} / {len(strat_test_set):,}")
print(f"Features: {fifa_prepared.shape[1]}")
print(f"\nBest Model: {best_model_name}")
print(f"R2 Score: {results.iloc[0]['R2']:.4f}")
print(f"RMSE (EUR): EUR {rmse_eur:,.0f}")
print(f"MAPE: {mape:.1f}%")
print("="*60)
