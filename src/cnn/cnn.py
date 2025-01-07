import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json

# Load the dataset from the Excel file
file_path = 'output/robo/img35/1vertebra_features.xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Define the directory where the segmented images are stored
segmented_images_directory = 'output/robo/img35/'  # Update with your directory path

# Function to create the image path
def create_image_path(image_name):
    # Ensure that image_name is a string
    if not isinstance(image_name, str):
        raise ValueError(f"Unexpected data type in 'Image Name': {type(image_name)}")
    
    return os.path.join(segmented_images_directory, image_name)  # Directly use image name

# Create a new column 'Image Path' by applying the function to the 'Image Name' column
df['Image Path'] = df['Image Name'].apply(create_image_path)

# Drop the 'Image Name' column now that the 'Image Path' column is created
df.drop(columns=['Image Name'], inplace=True)

# Split features and target variable
X = df['Image Path']  # Features (image paths)
y = df['Age']         # Target variable (Age)

# Split the dataset into training and testing sets
X_train_paths, X_test_paths, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))  # Adjust size as needed
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Load and preprocess the images
X_train = np.array([load_and_preprocess_image(path) for path in X_train_paths])
X_test = np.array([load_and_preprocess_image(path) for path in X_test_paths])

# Reshape the data if necessary
X_train = X_train.reshape(-1, 150, 150, 3)  # Shape should be (num_samples, height, width, channels)
X_test = X_test.reshape(-1, 150, 150, 3)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # First convolutional layer
model.add(MaxPooling2D((2, 2)))  # Max pooling layer
model.add(Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer
model.add(MaxPooling2D((2, 2)))  # Max pooling layer
model.add(Conv2D(128, (3, 3), activation='relu'))  # Third convolutional layer
model.add(MaxPooling2D((2, 2)))  # Max pooling layer
model.add(Flatten())  # Flatten the output
model.add(Dense(128, activation='relu'))  # Fully connected layer
model.add(Dropout(0.5))  # Dropout layer
model.add(Dense(1))  # Output layer (for regression)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,  # Use 20% of the training data for validation
    epochs=50,  # Set epochs as needed
    batch_size=32,  # Use a batch size of 32
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate RMSE

# Custom accuracy calculation
def calculate_custom_accuracy(y_true, y_pred, tolerance=5):
    y_true = y_true.values.flatten()  # Ensure y_true is a 1D numpy array
    correct_predictions = sum(abs(y_true - y_pred.flatten()) <= tolerance)
    accuracy = correct_predictions / len(y_true) * 100  # Convert to percentage
    return accuracy

# Calculate custom accuracy
custom_accuracy = calculate_custom_accuracy(y_test, y_pred)

# Store results
results = {
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae,
    'Root Mean Squared Error': rmse,
    'R-squared': abs(r_squared),
    'Accuracy': custom_accuracy
}

# Print results
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {abs(r_squared):.2f}')
print(f'Accuracy: {custom_accuracy:.2f}%')

# Save results to JSON file
results_file_path = 'src/cnn/cnn_model_results.json'  # Update with your desired path
with open(results_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4) 

# Save the model
model.save('src/cnn/cnn_model.keras')  # Save in native Keras format

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
