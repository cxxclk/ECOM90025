import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

#load https://github.com/cxxclk/ECOM90025/blob/main/Data/train_data.csv
train_data = pd.read_csv('https://raw.githubusercontent.com/cxxclk/ECOM90025/main/Data/train_data.csv')

# Display the first few rows of the training data
print(train_data.head())

# Drop 'ID' column
train_data = train_data.drop(columns=['ID'])

# Multi-Variable Linear Regression with LASSO
# Calculate correlation between column 1 and others
correlation = train_data.corr()
correlation = correlation.iloc[0]


# sorting the correlation values
sorted_correlation = correlation.sort_values(ascending=False)

# Drop Y from the sorted correlation
sorted_correlation = sorted_correlation.drop(index=sorted_correlation.index[0])
print("\nSorted correlation values:")
print(sorted_correlation.head())

# Forward search, start from most correlated one
selected_features = []
remaining_features = sorted_correlation.index.tolist()
best_rsquared = 0

while remaining_features:
    best_feature = None
    best_improvement = 0
    
    # Try adding each remaining feature and see which gives best improvement
    for feature in remaining_features:
        test_features = selected_features + [feature]
        X = train_data[test_features]
        y = train_data['Y']
        model = sm.OLS(y, sm.add_constant(X)).fit()
        
        if model.rsquared > best_rsquared + best_improvement:
            best_feature = feature
            best_improvement = model.rsquared - best_rsquared
    
    # If we found an improvement, add the feature
    if best_feature is not None and best_improvement > 0.01:  # minimum improvement threshold
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_rsquared += best_improvement
        print(f"Added {best_feature}, R-squared: {best_rsquared:.4f}")
    else:
        break

print("\nSelected features after forward search:")
print(selected_features)

# CV with SKlearn
kfolds = 30
# Prepare data for cross-validation
X = train_data[selected_features]
y = train_data['Y']

# Create linear regression model
lr_model = LinearRegression()

# Perform k-fold cross-validation
cv_scores = cross_val_score(lr_model, X, y, cv=kfolds, scoring='r2')

print(f"\nCross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# Predict on test set
test_data = pd.read_csv('https://raw.githubusercontent.com/cxxclk/ECOM90025/main/Data/test_data.csv')
test_ID = test_data['ID'].copy() 
test_data = test_data.drop(columns=['ID'])
test_X = test_data[selected_features]
# Fit the model on the full training set
lr_model.fit(X, y)

# Make predictions on test set
test_predictions = lr_model.predict(test_X)

# Create submission dataframe
submission = pd.DataFrame({
    'ID': test_ID,
    'Y': test_predictions
})

# save submission file
submission.to_csv('submission.csv', index=False)