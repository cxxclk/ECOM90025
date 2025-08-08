import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#load \data\train.csv
train_data = pd.read_csv('Data/train_data.csv')

# Display the first few rows of the training data
print(train_data.head())

# Drop 'ID' column
train_data = train_data.drop(columns=['ID'])



# Calculate correlation between column 1 and others
correlation = train_data.corr()
correlation = correlation.iloc[0]


# sorting the correlation values
sorted_correlation = correlation.sort_values(ascending=False)

# Drop Y from the sorted correlation
sorted_correlation = sorted_correlation.drop(index=sorted_correlation.index[0])
print("\nSorted correlation values:")
print(sorted_correlation.head())

Y = train_data.iloc[:, 0]
top_features = sorted_correlation[0:1].index.tolist()
X = train_data[top_features]
X = sm.add_constant(X)  # Add a constant term for the intercept
model = sm.OLS(Y, X).fit()


# Print the summary of the regression model
print("\nRegression model summary:")
print(model.summary())

# K-fold CV
from sklearn.model_selection import KFold
kf = KFold(n_splits=30, shuffle=True, random_state=42)
rmses = []
for train_index, test_index in kf.split(train_data):
    train_fold = train_data.iloc[train_index]
    test_fold = train_data.iloc[test_index]
    
    Y_train = train_fold.iloc[:, 0]
    X_train = train_fold[top_features]
    X_train = sm.add_constant(X_train)
    
    Y_test = test_fold.iloc[:, 0]
    X_test = test_fold[top_features]
    X_test = sm.add_constant(X_test)
    
    model = sm.OLS(Y_train, X_train).fit()
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(np.mean((predictions - Y_test) ** 2))
    rmses.append(rmse)
print("\nRMSE for each fold:")
print(rmses)
print("\nAverage RMSE across folds:", np.mean(rmses))


# Predict on the test set
test_data = pd.read_csv('Data/test_data.csv')
test_ID = test_data['ID'].copy()  # Save ID column before dropping
test_data = test_data.drop(columns=['ID'])
test_X = test_data[top_features]
test_X = sm.add_constant(test_X)  # Add a constant term for the intercept
test_predictions = model.predict(test_X)

# Save predictions to the current submission file
submission = pd.DataFrame({
    'ID': test_ID,
    'Y': test_predictions
})
submission.to_csv('Data/submission.csv', index=False)