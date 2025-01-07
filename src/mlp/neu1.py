import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np  # Import NumPy for RMSE calculation
import json

# Load the data from the Excel sheet
file_path = 'output/robo/final.xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Drop the 'Image Name' column
df.drop(columns=['Image Name'], inplace=True)

# Split features and target variable
X = df.drop(columns=['Age'])  # Features
y = df['Age']                 # Target variable (Age)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'src/mlp/mlp_scaler.pkl')  # Save the scaler

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(64, activation='relu'))  # Second hidden layer
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(32, activation='relu'))  # Third hidden layer
model.add(Dense(1))  # Output layer (for regression)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,  # Use 20% of the training data for validation
    epochs=100,  # Increased epochs
    batch_size=32,  # Use a batch size of 32
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)  # Calculate RMSE

# Custom accuracy calculation
def calculate_custom_accuracy(y_true, y_pred, tolerance=5):
    y_true = y_true.values.flatten()  # Ensure y_true is a 1D numpy array
    correct_predictions = sum(abs(y_true - y_pred.flatten()) <= tolerance)
    accuracy = correct_predictions / len(y_true) * 100  # Convert to percentage
    return accuracy

# Calculate custom accuracy
custom_accuracy = calculate_custom_accuracy(y_test, y_pred)

# Prepare results dictionary with additional performance metrics
results = {
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae,
    'Root Mean Squared Error': rmse,  # Add RMSE
    'Accuracy': custom_accuracy
}

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')  # Print RMSE
print(f'Accuracy: {custom_accuracy:.2f}%')

# Save results to a JSON file
results_file_path = 'src/mlp/mlp_model_results.json'  # Update with your desired path
with open(results_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4) 

# Save the model
model.save('src/mlp/mlp_model.keras')  # Save in native Keras format

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predicted vs actual ages
plt.scatter(y_test, y_pred, label='Predicted vs Actual', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Prediction')
plt.title('Actual vs Predicted Ages')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.legend()
plt.grid()
plt.show()
