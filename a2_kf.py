import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# load from github
train_data = pd.read_csv('https://raw.githubusercontent.com/cxxclk/ECOM90025/main/Data/train_data.csv')
print(train_data.head())

# Drop id column
train_data = train_data.drop(columns=['ID'])

# correlation as a reference, might save training time
correlation = train_data.corr().iloc[0]
sorted_correlation = correlation.sort_values(ascending=False)
sorted_correlation = sorted_correlation.drop(index=sorted_correlation.index[0])
print("\nTop correlated with Y:")
print(sorted_correlation.head(20))

# Prepare data
X = train_data.drop(columns=['Y'])
y = train_data['Y']

# Linear model with Y~ X_i+X_i^2 for each X, compare RMSE with Y ~ X_i
# Initialize cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results
linear_rmse_scores = []
quadratic_rmse_scores = []

print("\nComparing Linear vs Quadratic models for each feature:")
print("-" * 60)

for feature in X.columns:
    linear_scores = []
    quadratic_scores = []
    
    for train_idx, val_idx in kf.split(X):
        # Split data
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Linear model: Y ~ X_i
        X_linear_train = X_train_fold[[feature]]
        X_linear_val = X_val_fold[[feature]]
        
        linear_pipeline = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, random_state=42))
        linear_pipeline.fit(X_linear_train, y_train_fold)
        y_pred_linear = linear_pipeline.predict(X_linear_val)
        linear_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_linear)))
        
        # Quadratic model: Y ~ X_i + X_i^2
        X_quad_train = pd.DataFrame({
            feature: X_train_fold[feature],
            f'{feature}_squared': X_train_fold[feature] ** 2
        })
        X_quad_val = pd.DataFrame({
            feature: X_val_fold[feature],
            f'{feature}_squared': X_val_fold[feature] ** 2
        })
        
        quad_pipeline = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, random_state=42))
        quad_pipeline.fit(X_quad_train, y_train_fold)
        y_pred_quad = quad_pipeline.predict(X_quad_val)
        quadratic_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_quad)))
    
    # Calculate mean RMSE across folds
    mean_linear_rmse = np.mean(linear_scores)
    mean_quad_rmse = np.mean(quadratic_scores)
    
    linear_rmse_scores.append(mean_linear_rmse)
    quadratic_rmse_scores.append(mean_quad_rmse)
    
    improvement = mean_linear_rmse - mean_quad_rmse
    print(f"{feature:15s} | Linear: {mean_linear_rmse:.4f} | Quadratic: {mean_quad_rmse:.4f} | Improvement: {improvement:+.4f}")

# Select only when RMSE improvement > 0.1
significant_improvements = []
for i, feature in enumerate(X.columns):
    improvement = linear_rmse_scores[i] - quadratic_rmse_scores[i]
    if improvement > 0.01:
        significant_improvements.append((feature, improvement))

print(f"\nFeatures with RMSE improvement > 0.1:")
print("-" * 40)
for feature, improvement in significant_improvements:
    print(f"{feature:15s} | Improvement: {improvement:+.4f}")

# Add selected quadratic term columns
for feature, _ in significant_improvements:
    X[f'{feature}_squared'] = X[feature] ** 2

# Interaction selection
# residuals screening
# Get top 30 correlated features
top_30_features = sorted_correlation.head(30).index.tolist()
print(f"\nTop 30 correlated features: {top_30_features}")

# Train a baseline model to get residuals
X_baseline = X[top_30_features + [f'{feature}_squared' for feature, _ in significant_improvements if feature in top_30_features]]
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_baseline, y, test_size=0.2, random_state=42)

baseline_pipeline = make_pipeline(StandardScaler(), ElasticNetCV(cv=5, random_state=42))
baseline_pipeline.fit(X_train_base, y_train_base)
y_pred_baseline = baseline_pipeline.predict(X_train_base)
residuals = y_train_base - y_pred_baseline

print(f"\nComparing original correlation vs residual correlation for interaction terms:")
print("-" * 80)
print(f"{'Interaction':25s} | {'Original Corr':12s} | {'Residual Corr':12s} | {'Difference':10s}")
print("-" * 80)

interaction_comparisons = []

# Generate interaction terms between top 30 features
for i in range(len(top_30_features)):
    for j in range(i+1, len(top_30_features)):
        feature1 = top_30_features[i]
        feature2 = top_30_features[j]
        
        # Create interaction term
        interaction_term = X_train_base[feature1] * X_train_base[feature2]
        interaction_name = f"{feature1}*{feature2}"
        
        # Original correlation with y
        original_corr = np.corrcoef(interaction_term, y_train_base)[0, 1]
        
        # Residual correlation
        residual_corr = np.corrcoef(interaction_term, residuals)[0, 1]
        
        difference = abs(residual_corr) - abs(original_corr)
        
        interaction_comparisons.append({
            'interaction': interaction_name,
            'original_corr': original_corr,
            'residual_corr': residual_corr,
            'difference': difference
        })
        
        print(f"{interaction_name:25s} | {original_corr:11.4f} | {residual_corr:11.4f} | {difference:9.4f}")

# Sort by residual correlation strength
interaction_comparisons.sort(key=lambda x: abs(x['residual_corr']), reverse=True)

print(f"\nTop 10 interaction terms by residual correlation strength:")
print("-" * 60)
for i in range(min(10, len(interaction_comparisons))):
    comp = interaction_comparisons[i]

# Interactions with residual corr > 0.1 and positive diff
print(f"\nInteractions with Residual Corr > 0.1 and Positive Difference:")
print("-" * 60)
for comp in interaction_comparisons:
    if comp['residual_corr'] > 0.1 and comp['difference'] > 0:
        print(f"{comp['interaction']:25s} |Original Corr:{comp['original_corr']:11.4f} | Residual Corr: {comp['residual_corr']:7.4f} | Difference: {comp['difference']:9.4f}")

# add selected interaction terms to columns
interaction_terms = [comp['interaction'] for comp in interaction_comparisons if comp['residual_corr'] > 0.1 and comp['difference'] > 0]
for term in interaction_terms:
    X[term] = X[term.split('*')[0]] * X[term.split('*')[1]]

# Elastic Net CV
# Elastic Net CV for feature selection
print(f"\nElastic Net CV for feature selection:")
print("-" * 50)

# Prepare final feature set
X_final = X.copy()
y_final = y.copy()

# Split data
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)

# Create Elastic Net pipeline with cross-validation
elastic_net_pipeline = make_pipeline(
    StandardScaler(), 
    ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-4, 1, 50),
        cv=5,
        random_state=42,
        max_iter=2000
    )
)

# Fit the model
elastic_net_pipeline.fit(X_train_final, y_train_final)

# Get the trained ElasticNetCV model
elastic_net_model = elastic_net_pipeline.named_steps['elasticnetcv']

# Print optimal parameters
print(f"Best alpha: {elastic_net_model.alpha_:.6f}")
print(f"Best l1_ratio: {elastic_net_model.l1_ratio_:.3f}")

# Get feature coefficients
feature_names = X_final.columns
coefficients = elastic_net_model.coef_

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

# Display selected features (non-zero coefficients)
selected_features = feature_importance[feature_importance['abs_coefficient'] > 0]
print(f"\nSelected features ({len(selected_features)} out of {len(feature_names)}):")
print("-" * 60)
for _, row in selected_features.iterrows():
    print(f"{row['feature']:25s} | Coefficient: {row['coefficient']:8.4f}")

# Model performance
y_pred_train = elastic_net_pipeline.predict(X_train_final)
y_pred_test = elastic_net_pipeline.predict(X_test_final)

train_rmse = np.sqrt(mean_squared_error(y_train_final, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_test))

print(f"\nModel Performance:")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")
print(f"Number of selected features: {len(selected_features)}")
print(f"Final alpha: {elastic_net_model.alpha_:.6f}")
print(f"Final l1_ratio: {elastic_net_model.l1_ratio_:.3f}")
# Calculate R^2 for training and testing sets
train_r2 = r2_score(y_train_final, y_pred_train)
test_r2 = r2_score(y_test_final, y_pred_test)

print(f"\nModel R^2 Scores:")
print(f"Training R^2: {train_r2:.4f}")
print(f"Testing R^2: {test_r2:.4f}")

# ============================================================
# Elastic Net Forward Search (with outer K-fold CV)
# ============================================================
print(f"\nElastic Net Forward Search:")
print("=" * 60)

# --- Top10 squared terms (prioritize squared terms with large improvement; skip if already in X_final) ---
top_sqr_terms = []
original_features = train_data.drop(columns=['Y']).columns
sqr_improvements = [(feature, linear_rmse_scores[i] - quadratic_rmse_scores[i])
                    for i, feature in enumerate(original_features)
                    if f'{feature}_squared' not in X_final.columns]
sqr_improvements.sort(key=lambda x: x[1], reverse=True)
for feature, _ in sqr_improvements[:10]:
    term = f'{feature}_squared'
    if term not in X_final.columns:
        top_sqr_terms.append(term)

print(f"Top 10 squared terms to consider: {top_sqr_terms}")

# --- Top10 interaction terms (prioritize difference>0; if not enough 10, supplement from others by |residual_corr|) ---
already_in = set(X_final.columns)
pos_diff = [c for c in interaction_comparisons if c['difference'] > 0 and c['interaction'] not in already_in]
pos_diff.sort(key=lambda x: abs(x['residual_corr']), reverse=True)

top_interaction_terms = [c['interaction'] for c in pos_diff[:10]]

if len(top_interaction_terms) < 10:
    need = 10 - len(top_interaction_terms)
    rest = [c for c in interaction_comparisons
            if c['interaction'] not in already_in and c['interaction'] not in top_interaction_terms]
    rest.sort(key=lambda x: abs(x['residual_corr']), reverse=True)
    top_interaction_terms.extend([c['interaction'] for c in rest[:need]])

print(f"Top 10 interaction terms to consider: {top_interaction_terms}")

# --- Consolidate candidates ---
candidate_terms = top_sqr_terms + top_interaction_terms
print(f"\nTotal candidate terms: {len(candidate_terms)}")

# --- Prepare columns for candidates (only create in candidate set; inclusion determined by forward search) ---
X_candidates = X_final.copy()

# Squared terms
for term in top_sqr_terms:
    base = term.replace('_squared', '')
    if term not in X_candidates.columns and base in X.columns:
        X_candidates[term] = X[base] ** 2

# Interaction terms
for term in top_interaction_terms:
    f = term.split('*')
    if len(f) == 2 and all(ff in X.columns for ff in f):
        if term not in X_candidates.columns:
            X_candidates[term] = X[f[0]] * X[f[1]]

# --- Outer K-fold evaluation function (inner ElasticNetCV tuning remains unchanged) ---
def outer_cv_rmse(Xdf, y_series, feat_list, random_state=42, n_splits=5):
    kf_outer = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses = []
    for tr_idx, va_idx in kf_outer.split(Xdf):
        X_tr, X_va = Xdf.iloc[tr_idx][feat_list], Xdf.iloc[va_idx][feat_list]
        y_tr, y_va = y_series.iloc[tr_idx], y_series.iloc[va_idx]
        pipe = make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                alphas=np.logspace(-4, 1, 50),
                cv=5,
                random_state=42,
                max_iter=2000
            )
        )
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        rmses.append(np.sqrt(mean_squared_error(y_va, pred)))
    return float(np.mean(rmses))

# Starting point = your current ElasticNet "non-zero features" set
selected_terms = list(selected_features['feature'])
print(f"\nStarting forward search with {len(selected_terms)} initial features")

# Evaluate starting point score with outer CV
current_cv_rmse = outer_cv_rmse(X_candidates, y_final, selected_terms, random_state=42, n_splits=5)
print(f"Initial 5-fold CV RMSE: {current_cv_rmse:.4f}")
print("-" * 60)

# Forward search configuration
improve_tol = 1e-3  # More robust: require at least 0.001 average RMSE improvement
search_results = []

for iteration in range(len(candidate_terms)):
    remaining = [t for t in candidate_terms if t not in selected_terms]
    if not remaining:
        print("No more candidates to try")
        break

    best_term, best_cv_rmse = None, current_cv_rmse
    for cand in remaining:
        trial_feats = selected_terms + [cand]
        cv_rmse = outer_cv_rmse(X_candidates, y_final, trial_feats, random_state=42, n_splits=5)
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_term = cand

    improvement = current_cv_rmse - best_cv_rmse
    if best_term is not None and improvement > improve_tol:
        selected_terms.append(best_term)
        current_cv_rmse = best_cv_rmse
        search_results.append({
            'iteration': iteration + 1,
            'added_term': best_term,
            'cv_rmse': best_cv_rmse,
            'improvement': improvement,
            'total_features': len(selected_terms)
        })
        print(f"Iter {iteration+1:2d}: +{best_term:25s} | 5-fold CV RMSE: {best_cv_rmse:.4f} | "
              f"Improvement: {improvement:+.4f} | Total features: {len(selected_terms)}")
    else:
        print(f"Iter {iteration+1:2d}: No improvement > {improve_tol}, stopping")
        break

print(f"\nForward search completed!")
print(f"Final number of features: {len(selected_terms)}")
print(f"Final 5-fold CV RMSE: {current_cv_rmse:.4f}")
if len(search_results) > 0:
    total_improve = search_results[0]['cv_rmse'] - search_results[-1]['cv_rmse'] \
        if len(search_results) > 1 else search_results[0]['improvement']
    print(f"Total improvement (CV): {total_improve:+.4f}")

# ============================================================
# Train final model with "final forward-selected features" (full training set), and export submission
# ============================================================
X_final_selected = X_candidates[selected_terms].copy()

final_full_pipeline = make_pipeline(
    StandardScaler(),
    ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-4, 1, 50),
        cv=5,
        random_state=42,
        max_iter=2000
    )
)
final_full_pipeline.fit(X_final_selected, y_final)

# (Optional) Report outer CV performance once more (closer to generalization)
final_cv_rmse = outer_cv_rmse(X_candidates, y_final, selected_terms, random_state=7, n_splits=5)
print(f"\n[Final] 5-fold CV RMSE (re-eval with seed=7): {final_cv_rmse:.4f}")

# ----------------- Generate test features and predict -----------------
test_df = pd.read_csv('https://raw.githubusercontent.com/cxxclk/ECOM90025/main/Data/test_data.csv')
test_ID = test_df['ID']
X_test_raw = test_df.drop(columns=['ID']).copy()

# First copy original columns
X_test_processed = X_test_raw.copy()

# Ensure all selected_terms columns are constructed
for term in selected_terms:
    if term in X_test_processed.columns:
        continue
    if term.endswith('_squared'):
        base = term.replace('_squared', '')
        if base in X_test_processed.columns:
            X_test_processed[term] = X_test_processed[base] ** 2
    elif '*' in term:
        f1, f2 = term.split('*')
        if f1 in X_test_processed.columns and f2 in X_test_processed.columns:
            X_test_processed[term] = X_test_processed[f1] * X_test_processed[f2]

# Only take final features
X_test_final = X_test_processed[selected_terms]
y_test_pred = final_full_pipeline.predict(X_test_final)

submission = pd.DataFrame({'ID': test_ID, 'Y': y_test_pred})
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission file created!")
print(f"Predictions made for {len(submission)} test samples")
print(f"Submission saved as 'submission.csv'")
print("\nFirst 5 predictions:")
print(submission.head())
