import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load \data\train.csv
train_data = pd.read_csv('Data/train_data.csv')

# Display the first few rows of the training data
print(train_data.head())

# Drop 'ID' column
train_data = train_data.drop(columns=['ID'])

# Run linear regression with column 1 as target, and only one top feature
# Split the train data into train and validation sets

def split_dataset(dataset, test_ratio = 0.1):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(train_data)

# Calculate correlation between column 1 and others
correlation = train_ds_pd.corr()
correlation = correlation.iloc[0]


# sorting the correlation values
sorted_correlation = correlation.sort_values(ascending=False)
print("\nSorted correlation values:")
print(sorted_correlation)

Y = train_ds_pd.iloc[:, 0]
top_features = sorted_correlation[1:2].index.tolist()
plt.figure(figsize=(10, 6))
plt.scatter(train_ds_pd[top_features], Y, alpha=0.5)
plt.title("Scatter Plot of Top Feature vs Target Variable")
plt.show()
X = train_ds_pd[top_features]
X = sm.add_constant(X)  # Add a constant term for the intercept
model = sm.OLS(Y, X).fit()

# plot residuals vs yhat
plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, model.resid, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()



# Print the summary of the regression model
print("\nRegression model summary:")
print(model.summary())

# Validate the model on the validation set
valid_X = valid_ds_pd[top_features]
valid_X = sm.add_constant(valid_X)  # Add a constant term for the intercept
valid_Y = valid_ds_pd.iloc[:, 0]
predictions = model.predict(valid_X)

# Calculate RMSE on the validation set
rmse = np.sqrt(np.mean((predictions - valid_Y) ** 2))
print("\nRMSE on validation set:", rmse)

# double confirm with other top features' rmse
for col in sorted_correlation[1:6].index:
    X = sm.add_constant(train_ds_pd[[col]])
    y = train_ds_pd.iloc[:, 0]
    model = sm.OLS(y, X).fit()
    val_X = sm.add_constant(valid_ds_pd[[col]])
    val_y = valid_ds_pd.iloc[:, 0]
    preds = model.predict(val_X)
    rmse = np.sqrt(np.mean((preds - val_y)**2))
    print(f"{col} → RMSE: {rmse:.4f}")


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