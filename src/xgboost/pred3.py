import os
import sys
import joblib
import numpy as np
import pandas as pd
import cv2  # Import OpenCV for displaying images

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from segmentation import extract_features  # Import the extract_features function

# Check if an image path argument was provided
if len(sys.argv) != 2:
    print("Usage: python pred3.py <image_path>")
    sys.exit(1)

# Get the image path from the command line arguments
new_image_path = sys.argv[1]

# Extract features from the new image (this will also segment the image)
new_features_df, segmented_image_path = extract_features(new_image_path)  # Unpack the DataFrame and path

# Extract actual age from the image name (assuming last two digits represent age)
actual_age = int(new_image_path[-6:-4])  # Adjust based on naming format

# Load the trained model and scaler
try:
    scaler = joblib.load('src/xgboost/xgboost_scaler.pkl')
    model = joblib.load('src/xgboost/xgboost_regressor.pkl')
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Scale the features
new_features_scaled = scaler.transform(new_features_df)

# Make predictions
predicted_age = model.predict(new_features_scaled)
floored_predicted_age = np.floor(predicted_age).astype(int)

# Add image name, actual age, and floored predicted age to the DataFrame
new_features_df.insert(0, 'Model Name', "XGboost")
new_features_df.insert(1, 'Image Name', os.path.basename(new_image_path))
new_features_df.insert(2, 'Actual Age', actual_age)
new_features_df.insert(3, 'Predicted Age', floored_predicted_age)

# Output the actual and floored predicted age
print(f'Actual Age: {actual_age}')
print(f'Predicted Age: {floored_predicted_age[0]}')

# Save the extracted features to an Excel file
features_excel_path = "output/robo/extracted_features1.xlsx"  # Update as needed

# Check if the Excel file already exists
if os.path.exists(features_excel_path):
    # Load the existing data
    existing_data = pd.read_excel(features_excel_path)

    # Append the new features
    combined_data = pd.concat([existing_data, new_features_df], ignore_index=True)
else:
    # If the file doesn't exist, just use the new features
    combined_data = new_features_df

# Save the combined DataFrame to the Excel file
combined_data.to_excel(features_excel_path, index=False)

# Display the segmented image
segmented_image = cv2.imread(segmented_image_path)  # Load the segmented image
cv2.imshow('Segmented Image', segmented_image)  # Display the image
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows

# Return the results to the Flask app
print(f'Segmented Image Path: {segmented_image_path}')
print(f'Predicted Age: {floored_predicted_age[0]}')
print(f'Actual Age: {actual_age}')