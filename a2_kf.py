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

# Get top 5 correlated features (excluding Y)
top5_features = sorted_correlation.head(5).index.tolist()
print(f"\nTop 5 correlated features: {top5_features}")

# Add interaction terms for top 5 correlated features
print("\nAdding interaction terms for top 5 correlated features...")
interaction_count = 0
for i in range(len(top5_features)):
    for j in range(i+1, len(top5_features)):
        feature1 = top5_features[i]
        feature2 = top5_features[j]
        interaction_name = f'{feature1}_{feature2}_interaction'
        X[interaction_name] = X[feature1] * X[feature2]
        interaction_count += 1
        print(f"Added: {interaction_name}")

print(f"\nTotal interaction terms added: {interaction_count}")
print(f"Final feature count: {X.shape[1]}")

# Split train and validation sets after feature engineering
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Elastic Net CV with forward selection approach on split data
print("\n" + "="*60)
print("ELASTIC NET CV WITH FORWARD SELECTION")
print("="*60)

# Initialize variables for forward selection
selected_features = []
remaining_features = list(X.columns)
best_overall_score = float('inf')
feature_scores = []

print(f"Starting forward selection with {len(remaining_features)} features...")

# Forward selection loop
for step in range(min(30, len(remaining_features))):  # Limit to 30 features max
    best_feature = None
    best_score = float('inf')
    
    print(f"\nStep {step + 1}: Testing {len(remaining_features)} remaining features...")
    
    # Test each remaining feature
    for feature in remaining_features:
        current_features = selected_features + [feature]
        X_current = X_train[current_features]
        
        # Cross-validation with current feature set
        cv_scores = []
        for train_idx, val_idx in kf.split(X_current):
            X_train_fold = X_current.iloc[train_idx]
            X_val_fold = X_current.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # Fit ElasticNet with CV
            pipeline = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, random_state=42))
            pipeline.fit(X_train_fold, y_train_fold)
            y_pred = pipeline.predict(X_val_fold)
            cv_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
        
        mean_score = np.mean(cv_scores)
        
        if mean_score < best_score:
            best_score = mean_score
            best_feature = feature
    
    # Check if adding this feature improves the model
    if best_score < best_overall_score:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_overall_score = best_score
        feature_scores.append((best_feature, best_score))
        print(f"  → Added '{best_feature}' | RMSE: {best_score:.4f}")
    else:
        print(f"  → No improvement found. Stopping selection.")
        break

print(f"\nForward Selection Complete!")
print(f"Selected {len(selected_features)} features")
print(f"Final RMSE: {best_overall_score:.4f}")

print("\nSelected features in order:")
for i, (feature, score) in enumerate(feature_scores):
    print(f"{i+1:2d}. {feature:30s} | RMSE: {score:.4f}")

# Final model with selected features - Train on full dataset to get final parameters
X_final = X[selected_features]
final_pipeline = make_pipeline(StandardScaler(), ElasticNetCV(cv=5, random_state=42))
final_pipeline.fit(X_final, y)

# Extract the ElasticNetCV model to get parameters
elastic_net_model = final_pipeline.named_steps['elasticnetcv']
final_alpha = elastic_net_model.alpha_
final_l1_ratio = elastic_net_model.l1_ratio_

print(f"Final Model Parameters:")
print(f"Alpha: {final_alpha:.6f}")
print(f"L1 Ratio: {final_l1_ratio:.6f}")
print(f"\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

X_final = X[selected_features]
final_cv_scores = []

for train_idx, val_idx in kf.split(X_final):
    X_train_fold = X_final.iloc[train_idx]
    X_val_fold = X_final.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]
    
    final_pipeline = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, random_state=42))
    final_pipeline.fit(X_train_fold, y_train_fold)
    y_pred = final_pipeline.predict(X_val_fold)
    final_cv_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))

final_mean_rmse = np.mean(final_cv_scores)
final_std_rmse = np.std(final_cv_scores)

print(f"Final Model Performance:")
print(f"Mean RMSE: {final_mean_rmse:.4f} ± {final_std_rmse:.4f}")
print(f"Selected Features: {selected_features}")

# Validate on validation set
X_val_selected = X_val[selected_features]
y_pred_val = final_pipeline.predict(X_val_selected)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {val_rmse:.4f}")

# Test on test set
test_df = pd.read_csv('https://raw.githubusercontent.com/cxxclk/ECOM90025/main/Data/test_data.csv')
test_ID = test_df['ID']

# Drop ID column and recreate engineered features for test data
X_test = test_df.drop(columns=['ID'])

# Add quadratic terms for features with significant improvements
for feature, _ in significant_improvements:
    X_test[f'{feature}_squared'] = X_test[feature] ** 2

# Add interaction terms for top 5 correlated features
for i in range(len(top5_features)):
    for j in range(i+1, len(top5_features)):
        feature1 = top5_features[i]
        feature2 = top5_features[j]
        interaction_name = f'{feature1}_{feature2}_interaction'
        X_test[interaction_name] = X_test[feature1] * X_test[feature2]

# Select only the features used in the final model
X_test = X_test[selected_features]
y_pred_test = final_pipeline.predict(X_test)

# Submission
submission = pd.DataFrame({'ID': test_ID, 'Y': y_pred_test})
submission.to_csv('submission.csv', index=False)
print("\nSaved: submission.csv")