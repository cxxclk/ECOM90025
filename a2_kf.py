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
print(sorted_correlation.head())

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
    print(f"{comp['interaction']:25s} |Original Corr:{comp['original_corr']:11.4f} | Residual Corr: {comp['residual_corr']:7.4f} | Difference: {comp['difference']:9.4f}")

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

# Test on testset
test_df = pd.read_csv('https://raw.githubusercontent.com/cxxclk/ECOM90025/main/Data/test_data.csv')
test_ID = test_df['ID']

X_test = test_df.drop(columns=['ID'])
# Add quadratic terms that were selected during training
for feature, _ in significant_improvements:
    X_test[f'{feature}_squared'] = X_test[feature] ** 2

# Add interaction terms that were selected during training
for term in interaction_terms:
    X_test[term] = X_test[term.split('*')[0]] * X_test[term.split('*')[1]]

# Make predictions using the trained model
y_test_pred = elastic_net_pipeline.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame({
    'ID': test_ID,
    'Y': y_test_pred
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print(f"\nPredictions saved to submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"First 5 predictions:")
print(submission.head())
