import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json  # Import JSON module

# Load the data from the Excel sheet
file_path = 'output/robo/final.xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Drop the 'Image Name' column
df.drop(columns=['Image Name'], inplace=True)

# Split features and target variable
X = df.drop(columns=['Age'])  # Features
y = df['Age']                 # Target variable (Age)

# Define age bins
bins = [30, 40, 50, 60]  # Define bins that cover this range
labels = ['30-40', '41-50', '51-60']  # Corresponding labels

y_binned = pd.cut(y, bins=bins, labels=labels, right=False)  # Binning the ages

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, y_binned_train, y_binned_test = train_test_split(X, y, y_binned, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'src/randomforest/randomforestscaler.pkl')

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train_scaled, y_train)

# Best model from Grid Search
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluate the regression model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate RMSE
rmse = np.sqrt(mse)

# Convert predictions to bins for confusion matrix
y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels, right=False)

# Calculate confusion matrix and accuracy score
conf_matrix = confusion_matrix(y_binned_test, y_pred_binned)
accuracy = accuracy_score(y_binned_test, y_pred_binned)

# Prepare results for JSON output
results = {
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae,
    'Root Mean Squared Error': rmse,  # Include RMSE in the results
    'R-squared': r2,
    'Confusion Matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
    'Accuracy': accuracy * 100
}

# Print results
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')  # Print RMSE
print(f'R-squared: {r2}')
print("Confusion Matrix:")
print(conf_matrix)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save results to JSON file
results_file_path = 'src/randomforest/randomforest_model_results.json'  # Update with your desired path
with open(results_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)  # Write results to JSON

# Optional: Plotting
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs Predicted Ages')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')  # Diagonal line
plt.show()

# Save the model for future use
joblib.dump(best_model, 'src/randomforest/randomforestregressor.pkl')
