import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    
    return os.path.join(segmented_images_directory, image_name)

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

# Create an instance of ImageDataGenerator for data augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the ResNet model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential()
model.add(base_model)
model.add(Flatten())  # Flatten the output
model.add(Dense(128, activation='relu'))  # Fully connected layer
model.add(Dropout(0.5))  # Dropout layer
model.add(Dense(1))  # Output layer (for regression)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model using data augmentation
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate R² score
r2 = r2_score(y_test, y_pred)

# Calculate RMSE
rmse = np.sqrt(mse)

# Custom accuracy calculation
def calculate_custom_accuracy(y_true, y_pred, tolerance=5):
    y_true = y_true.values.flatten()  # Ensure y_true is a 1D numpy array
    correct_predictions = sum(abs(y_true - y_pred.flatten()) <= tolerance)
    accuracy = correct_predictions / len(y_true) * 100  # Convert to percentage
    return accuracy

# Calculate custom accuracy
custom_accuracy = calculate_custom_accuracy(y_test, y_pred)
results = {
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae,
    'Root Mean Squared Error': rmse,
    'R-squared': abs(r2),
    'Accuracy': custom_accuracy
}

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

print(f'R² Score: {abs(r2)}')
print(f'Accuracy: {custom_accuracy:.2f}%')

# Save results to JSON file
results_file_path = 'src/resnet/resnet_model_results.json'  # Update with your desired path
with open(results_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4) 

# Save the model
model.save('src/resnet/resnet_model.keras')  # Save in native Keras format

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.scatter(y_test, y_pred, label='Predicted vs Actual', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Prediction')
plt.title('Actual vs Predicted Ages')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.legend()
plt.grid()
plt.show()