import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the data from the Excel sheet
file_path = 'output/robo/Copy of final_excel(1).xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Drop the 'Image Name' column
df.drop(columns=['Image Name'], inplace=True)

# Split features and target variable
X = df.drop(columns=['Age'])  # Features
y = df['Age']                 # Target variable (Age)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
r2 = r2_score(y_test, y_pred)               # Calculate R-squared

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')  # Print MAE
print(f'R-squared: {r2}')              # Print R-squared

# Display actual vs. predicted ages
predictions_df = pd.DataFrame({'Actual Age': y_test, 'Predicted Age': y_pred})
print(predictions_df.head(10))  # Display the first 10 predictions

# Optional: Plotting
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs Predicted Ages')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')  # Diagonal line
plt.show()

# Optionally, you can save the model for future use
joblib.dump(model, 'age_prediction_model.pkl')
