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

# ===============================
# Orthogonal polynomial utilities
# ===============================
def _fit_orthopoly_params(x: pd.Series):
    x = x.values.astype(float)
    m1, s1 = np.mean(x), np.std(x)
    s1 = s1 if s1 > 0 else 1.0
    z1 = (x - m1) / s1

    m2 = np.mean(x**2)
    p2_raw = (x**2 - m2)
    denom_z1 = np.dot(z1, z1)
    coef21 = np.dot(p2_raw, z1) / denom_z1 if denom_z1 > 0 else 0.0
    p2 = p2_raw - coef21 * z1
    s2 = np.std(p2)
    s2 = s2 if s2 > 0 else 1.0
    p2_norm = p2 / s2

    m3 = np.mean(x**3)
    p3_raw = (x**3 - m3)
    denom_p2 = np.dot(p2_norm, p2_norm)
    coef31 = np.dot(p3_raw, z1) / denom_z1 if denom_z1 > 0 else 0.0
    coef32 = np.dot(p3_raw, p2_norm) / denom_p2 if denom_p2 > 0 else 0.0
    p3 = p3_raw - coef31 * z1 - coef32 * p2_norm
    s3 = np.std(p3)
    s3 = s3 if s3 > 0 else 1.0

    return {"m1": m1, "s1": s1, "m2": m2, "coef21": coef21, "s2": s2,
            "m3": m3, "coef31": coef31, "coef32": coef32, "s3": s3}

def _transform_op2(series: pd.Series, p):
    z1 = (series - p["m1"]) / p["s1"]
    p2_raw = (series**2 - p["m2"])
    p2 = (p2_raw - p["coef21"] * z1) / p["s2"]
    return p2

def _transform_op3(series: pd.Series, p):
    z1 = (series - p["m1"]) / p["s1"]
    p2 = _transform_op2(series, p)
    p3_raw = (series**3 - p["m3"])
    p3 = (p3_raw - p["coef31"] * z1 - p["coef32"] * p2) / p["s3"]
    return p3

def hinge_pos(x):
    return np.maximum(x, 0.0)

def hinge_neg(x):
    return np.minimum(x, 0.0)

# ===============================
# Load data
# ===============================
train_data = pd.read_csv('https://raw.githubusercontent.com/cxxclk/ECOM90025/main/Data/train_data.csv')
print(train_data.head())

train_data = train_data.drop(columns=['ID'])

correlation = train_data.corr().iloc[0]
sorted_correlation = correlation.sort_values(ascending=False)
sorted_correlation = sorted_correlation.drop(index=sorted_correlation.index[0])
print("\nTop correlated with Y:")
print(sorted_correlation.head(20))

X = train_data.drop(columns=['Y'])
y = train_data['Y']

op_params = {col: _fit_orthopoly_params(X[col]) for col in X.columns}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

linear_rmse_scores = []
quadratic_rmse_scores = []

print("\nComparing Linear vs Quadratic (orthogonal) models for each feature:")
print("-" * 70)

for feature in X.columns:
    linear_scores = []
    quadratic_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        X_linear_train = X_train_fold[[feature]]
        X_linear_val = X_val_fold[[feature]]

        linear_pipeline = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, random_state=42))
        linear_pipeline.fit(X_linear_train, y_train_fold)
        y_pred_linear = linear_pipeline.predict(X_linear_val)
        linear_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_linear)))

        X_quad_train = pd.DataFrame({
            feature: X_train_fold[feature],
            f'{feature}_op2': _transform_op2(X_train_fold[feature], op_params[feature]),
        })
        X_quad_val = pd.DataFrame({
            feature: X_val_fold[feature],
            f'{feature}_op2': _transform_op2(X_val_fold[feature], op_params[feature]),
        })

        quad_pipeline = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, random_state=42))
        quad_pipeline.fit(X_quad_train, y_train_fold)
        y_pred_quad = quad_pipeline.predict(X_quad_val)
        quadratic_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_quad)))

    mean_linear_rmse = np.mean(linear_scores)
    mean_quad_rmse = np.mean(quadratic_scores)

    linear_rmse_scores.append(mean_linear_rmse)
    quadratic_rmse_scores.append(mean_quad_rmse)

    improvement = mean_linear_rmse - mean_quad_rmse


significant_improvements = []
for i, feature in enumerate(X.columns):
    improvement = linear_rmse_scores[i] - quadratic_rmse_scores[i]
    if improvement > 0.01:
        significant_improvements.append((feature, improvement))

print(f"\nFeatures with RMSE improvement > 0.01:")
print("-" * 40)
for feature, improvement in significant_improvements:
    print(f"{feature:15s} | Improvement: {improvement:+.4f}")

for feature, _ in significant_improvements:
    X[f'{feature}_op2'] = _transform_op2(X[feature], op_params[feature])

top_30_features = sorted_correlation.head(30).index.tolist()
print(f"\nTop 30 correlated features: {top_30_features}")

X_baseline_cols = top_30_features + [f'{feature}_op2' for feature, _ in significant_improvements if feature in top_30_features]
X_baseline = X[X_baseline_cols]
# 保留你原来的第一次固定切分
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

for i in range(len(top_30_features)):
    for j in range(i+1, len(top_30_features)):
        feature1 = top_30_features[i]
        feature2 = top_30_features[j]

        interaction_term = X_train_base[feature1] * X_train_base[feature2]
        interaction_name = f"{feature1}*{feature2}"

        original_corr = np.corrcoef(interaction_term, y_train_base)[0, 1]
        residual_corr = np.corrcoef(interaction_term, residuals)[0, 1]

        difference = abs(residual_corr) - abs(original_corr)

        interaction_comparisons.append({
            'interaction': interaction_name,
            'original_corr': original_corr,
            'residual_corr': residual_corr,
            'difference': difference
        })

interaction_comparisons.sort(key=lambda x: abs(x['residual_corr']), reverse=True)

print(f"\nTop 10 interaction terms by residual correlation strength:")
print("-" * 60)
for i in range(min(10, len(interaction_comparisons))):
    comp = interaction_comparisons[i]
    print(f"{comp['interaction']:25s} | Original Corr: {comp['original_corr']:7.4f} | Residual Corr: {comp['residual_corr']:7.4f} | Difference: {comp['difference']:9.4f}")

print(f"\nInteractions with Residual Corr > 0.1 and Positive Difference:")
print("-" * 60)
for comp in interaction_comparisons:
    if comp['residual_corr'] > 0.1 and comp['difference'] > 0:
        print(f"{comp['interaction']:25s} |Original Corr:{comp['original_corr']:11.4f} | Residual Corr: {comp['residual_corr']:7.4f} | Difference: {comp['difference']:9.4f}")

interaction_terms = [comp['interaction'] for comp in interaction_comparisons if comp['residual_corr'] > 0.1 and comp['difference'] > 0]
for term in interaction_terms:
    X[term] = X[term.split('*')[0]] * X[term.split('*')[1]]

# Create X34 over X26 as a new feature
X['X34_over_X26'] = X['X34'] / (X['X26'] + 1e-8)

print(f"\nElastic Net CV for feature selection:")
print("-" * 50)

X_final = X.copy()
y_final = y.copy()

# >>> KFold CV (no holdout) start
kf_final = KFold(n_splits=10, shuffle=True, random_state=42)
oof_pred_final = np.zeros(len(y_final), dtype=float)

def build_enet_pipeline():
    return make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            l1_ratio=[0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
            alphas=np.logspace(-4, 1, 50),
            cv=5,
            random_state=42,
            max_iter=2000
        )
    )

for tr_idx, va_idx in kf_final.split(X_final):
    X_tr, X_va = X_final.iloc[tr_idx], X_final.iloc[va_idx]
    y_tr = y_final.iloc[tr_idx]
    pipe_cv = build_enet_pipeline()
    pipe_cv.fit(X_tr, y_tr)
    oof_pred_final[va_idx] = pipe_cv.predict(X_va)

cv_rmse_final = float(np.sqrt(mean_squared_error(y_final, oof_pred_final)))
cv_r2_final = float(r2_score(y_final, oof_pred_final))
print(f"5-fold OOF RMSE: {cv_rmse_final:.4f}")
print(f"5-fold OOF R^2 : {cv_r2_final:.4f}")

elastic_net_pipeline = build_enet_pipeline()
elastic_net_pipeline.fit(X_final, y_final)
elastic_net_model = elastic_net_pipeline.named_steps['elasticnetcv']
# >>> KFold CV (no holdout) end

print(f"Best alpha: {elastic_net_model.alpha_:.6f}")
print(f"Best l1_ratio: {elastic_net_model.l1_ratio_:.3f}")

feature_names = X_final.columns
coefficients = elastic_net_model.coef_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

selected_features = feature_importance[feature_importance['abs_coefficient'] > 0]
print(f"\nSelected features ({len(selected_features)} out of {len(feature_names)}):")
print("-" * 60)
for _, row in selected_features.iterrows():
    print(f"{row['feature']:25s} | Coefficient: {row['coefficient']:8.4f}")

print(f"\nModel Performance (from OOF):")
print(f"CV RMSE: {cv_rmse_final:.4f}")
print(f"CV R^2 : {cv_r2_final:.4f}")
print(f"Number of selected features: {len(selected_features)}")
print(f"Final alpha: {elastic_net_model.alpha_:.6f}")
print(f"Final l1_ratio: {elastic_net_model.l1_ratio_:.3f}")

print(f"\nElastic Net Forward Search:")
print("=" * 60)

top_op2_terms = []
original_features = train_data.drop(columns=['Y']).columns
sqr_improvements = [(feature, linear_rmse_scores[i] - quadratic_rmse_scores[i])
                    for i, feature in enumerate(original_features)
                    if f'{feature}_op2' not in X_final.columns]
sqr_improvements.sort(key=lambda x: x[1], reverse=True)
for feature, _ in sqr_improvements[:10]:
    term = f'{feature}_op2'
    if term not in X_final.columns:
        top_op2_terms.append(term)

print(f"Top 10 OP2 terms to consider: {top_op2_terms}")

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

candidate_terms = top_op2_terms + top_interaction_terms
print(f"\nTotal candidate terms: {len(candidate_terms)}")

# Force-add orthogonal cubic terms and hinge candidates
x34_op3_name = "X34_op3"
x26_op3_name = "X26_op3"
hinge_terms = ["X34_pos", "X34_neg", "X26_pos", "X26_neg"]
for nm in (x34_op3_name, x26_op3_name) + tuple(hinge_terms):
    if nm not in candidate_terms:
        candidate_terms.append(nm)
print(f"Added candidates: {x34_op3_name}, {x26_op3_name}, {', '.join(hinge_terms)}")

# === NEW: add interaction hinge candidates for X15*X12 and X16*X30 ===
interaction_hinge_candidates = []
if "X15" in X.columns and "X12" in X.columns:
    inter_1512 = X["X15"] * X["X12"]
    X["X15*X12_pos"] = hinge_pos(inter_1512)
    X["X15*X12_neg"] = hinge_neg(inter_1512)
    interaction_hinge_candidates += ["X15*X12_pos", "X15*X12_neg"]
if "X16" in X.columns and "X30" in X.columns:
    inter_1630 = X["X16"] * X["X30"]
    X["X16*X30_pos"] = hinge_pos(inter_1630)
    X["X16*X30_neg"] = hinge_neg(inter_1630)
    interaction_hinge_candidates += ["X16*X30_pos", "X16*X30_neg"]

for nm in interaction_hinge_candidates:
    if nm not in candidate_terms:
        candidate_terms.append(nm)

# de-duplicate while preserving order
candidate_terms = list(dict.fromkeys(candidate_terms))
print(f"Also added interaction hinge candidates: {interaction_hinge_candidates}")

X_candidates = X.copy()

# OP2 terms
for term in top_op2_terms:
    base = term.replace('_op2', '')
    if term not in X_candidates.columns and base in X.columns:
        X_candidates[term] = _transform_op2(X[base], op_params[base])

# Interaction terms
for term in top_interaction_terms:
    f = term.split('*')
    if len(f) == 2 and all(ff in X.columns for ff in f):
        if term not in X_candidates.columns:
            X_candidates[term] = X[f[0]] * X[f[1]]

# OP3 terms
if "X34" in X.columns and x34_op3_name not in X_candidates.columns:
    X_candidates[x34_op3_name] = _transform_op3(X["X34"], op_params["X34"])
if "X26" in X.columns and x26_op3_name not in X_candidates.columns:
    X_candidates[x26_op3_name] = _transform_op3(X["X26"], op_params["X26"])

# Hinge terms for raw features
if "X34" in X.columns:
    X_candidates["X34_pos"] = hinge_pos(X["X34"])
    X_candidates["X34_neg"] = hinge_neg(X["X34"])
if "X26" in X.columns:
    X_candidates["X26_pos"] = hinge_pos(X["X26"])
    X_candidates["X26_neg"] = hinge_neg(X["X26"])

# Hinge terms for INTERACTIONS (X15*X12, X16*X30)
if "X15" in X.columns and "X12" in X.columns:
    _i = X["X15"] * X["X12"]
    X_candidates["X15*X12_pos"] = hinge_pos(_i)
    X_candidates["X15*X12_neg"] = hinge_neg(_i)
if "X16" in X.columns and "X30" in X.columns:
    _i = X["X16"] * X["X30"]
    X_candidates["X16*X30_pos"] = hinge_pos(_i)
    X_candidates["X16*X30_neg"] = hinge_neg(_i)


def outer_cv_rmse(Xdf, y_series, feat_list, random_state=42, n_splits=5):
    kf_outer = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses = []
    for tr_idx, va_idx in kf_outer.split(Xdf):
        X_tr, X_va = Xdf.iloc[tr_idx][feat_list], Xdf.iloc[va_idx][feat_list]
        y_tr, y_va = y_series.iloc[tr_idx], y_series.iloc[va_idx]
        pipe = build_enet_pipeline()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        rmses.append(np.sqrt(mean_squared_error(y_va, pred)))
    return float(np.mean(rmses))

selected_terms = list(selected_features['feature'])
print(f"\nStarting forward search with {len(selected_terms)} initial features")

current_cv_rmse = outer_cv_rmse(X_candidates, y_final, selected_terms, random_state=42, n_splits=5)
print(f"Initial 5-fold CV RMSE: {current_cv_rmse:.4f}")
print("-" * 60)

improve_tol = 1e-3
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

# ----------------- Final model & submission -----------------
X_final_selected = X_candidates[selected_terms].copy()

final_full_pipeline = build_enet_pipeline()
final_full_pipeline.fit(X_final_selected, y_final)

final_cv_rmse = outer_cv_rmse(X_candidates, y_final, selected_terms, random_state=7, n_splits=5)
print(f"\n[Final] 5-fold CV RMSE (re-eval with seed=7): {final_cv_rmse:.4f}")

# ====== Test-time processing ======

test_df = pd.read_csv('https://raw.githubusercontent.com/cxxclk/ECOM90025/main/Data/test_data.csv')
test_ID = test_df['ID']
X_test_raw = test_df.drop(columns=['ID']).copy()
X_test_processed = X_test_raw.copy()

# Add the X34_over_X26 feature to test data
if 'X34' in X_test_processed.columns and 'X26' in X_test_processed.columns:
    X_test_processed['X34_over_X26'] = X_test_processed['X34'] / (X_test_processed['X26'] + 1e-8)

# Materialize selected hinge/op2/op3/interaction terms on test
for term in selected_terms:
    if term in X_test_processed.columns:
        continue
    if term.endswith('_op2'):
        base = term.replace('_op2', '')
        if base in X_test_processed.columns and base in op_params:
            X_test_processed[term] = _transform_op2(X_test_processed[base], op_params[base])
    elif term.endswith('_op3'):
        base = term.replace('_op3', '')
        if base in X_test_processed.columns and base in op_params:
            X_test_processed[term] = _transform_op3(X_test_processed[base], op_params[base])
    elif term.endswith('_pos') or term.endswith('_neg'):
        # handle both raw and interaction hinge terms
        is_pos = term.endswith('_pos')
        base = term[:-4]  # strip _pos/_neg
        if '*' in base:
            f1, f2 = base.split('*')
            if f1 in X_test_processed.columns and f2 in X_test_processed.columns:
                inter = X_test_processed[f1] * X_test_processed[f2]
                X_test_processed[term] = hinge_pos(inter) if is_pos else hinge_neg(inter)
        else:
            if base in X_test_processed.columns:
                X_test_processed[term] = hinge_pos(X_test_processed[base]) if is_pos else hinge_neg(X_test_processed[base])
    elif '*' in term:
        f1, f2 = term.split('*')
        if f1 in X_test_processed.columns and f2 in X_test_processed.columns:
            X_test_processed[term] = X_test_processed[f1] * X_test_processed[f2]

# Only select terms that actually exist in the processed test data
available_terms = [term for term in selected_terms if term in X_test_processed.columns]
X_test_final = X_test_processed[available_terms]

# Generate predictions using the trained model
y_test_pred = final_full_pipeline.predict(X_test_final)

submission = pd.DataFrame({'ID': test_ID, 'Y': y_test_pred})
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission file created!")
print(f"Predictions made for {len(submission)} test samples")
print(f"Submission saved as 'submission.csv'")
print("\nFirst 5 predictions:")
print(submission.head())
