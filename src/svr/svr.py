import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer  # Import imputer for handling NaN values
import joblib

# Load the Excel sheet
file_path = 'output/robo/Copy of final_excel(1).xlsx'  # Update this path
df = pd.read_excel(file_path)

# Drop the 'Image Name' column as it's not needed for training
df = df.drop(columns=['Image Name'])

# Separate features (X) and target variable (y)
X = df.drop(columns=['Age'])
y = df['Age']

# Check for missing values (NaNs)
print(f"Number of missing values in the dataset:\n{df.isna().sum()}")

# Handle missing values by imputing with the mean (you can change this to median or mode if preferred)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # Impute NaN values in the feature set

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

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
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")

# Optional: Print the predictions vs actual values
comparison_df = pd.DataFrame({'Actual Age': y_test, 'Predicted Age': y_pred})
print(comparison_df)

# Save the trained model and scaler (if you want to use it later)
joblib.dump(svr, 'svr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
