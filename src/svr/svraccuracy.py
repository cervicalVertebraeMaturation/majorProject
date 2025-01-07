import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json

# Load the Excel sheet
file_path = 'output/robo/final.xlsx' 
df = pd.read_excel(file_path)

# Drop the 'Image Name' column as it's not needed for training
df = df.drop(columns=['Image Name'])

# Check for missing values and print the count
df = df.dropna()

# Separate features (X) and target variable (y)
X = df.drop(columns=['Age'])
y = df['Age']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize/scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVR model (you can adjust parameters like kernel='linear' or kernel='poly')
svr = SVR(kernel='rbf')

# Train the model
svr.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svr.predict(X_test_scaled)

# Calculate model performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate RMSE
r2 = r2_score(y_test, y_pred)

# Create bins for age ranges
bins = [30, 40, 50, 60]  # Adjust these bins as needed
labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]

# Bin actual ages and predicted ages
y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels)

# Calculate confusion matrix and accuracy
conf_matrix = confusion_matrix(y_test_binned, y_pred_binned, labels=labels)
accuracy = accuracy_score(y_test_binned, y_pred_binned)

# Prepare results for JSON output
results = {
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae,
    'Root Mean Squared Error': rmse,
    'R-squared': r2,
    'Confusion Matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
    'Accuracy': accuracy * 100  # Convert accuracy to percentage
}

# Print results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R² Score): {r2}")
print("Confusion Matrix:")
print(conf_matrix)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save results to JSON file
results_file_path = 'src/svr/svr_model_results.json'  # Update with your desired path
with open(results_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4) 

# Optional: Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Age Range')
plt.ylabel('Actual Age Range')
plt.title('Confusion Matrix for Age Prediction')
plt.show()

# Save the trained model and scaler (if you want to use it later)
joblib.dump(svr, 'src/svr/svr_model.pkl')
joblib.dump(scaler, 'src/svr/scaler.pkl')
