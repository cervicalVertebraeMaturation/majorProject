import cv2
import numpy as np
import os
import joblib
import pandas as pd
from tensorflow import keras
from inference_sdk import InferenceHTTPClient
from skimage.measure import label, regionprops

# Initialize the client for Roboflow inference
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key= ""#use_the_api_key"
)


def preprocess_image(image_path):
    """Load and preprocess the image for the CNN."""
    image = cv2.imread(image_path)
    # Resize the image to the shape expected by the model
    image_resized = cv2.resize(image, (150, 150))  # Resize to 150x150
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_normalized, axis=0)  # Add batch dimension

def extract_features(image_path):
    """Extract features from a single image."""
    # Infer on a local image
    result = CLIENT.infer(image_path, model_id="segmemtation/1")

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Draw segmentation masks
    for prediction in result['predictions']:
        points = prediction.get('points', [])
        if points:
            points_array = np.array([(p['x'], p['y']) for p in points], dtype=np.int32)
            points_array = points_array.reshape((-1, 1, 2))
            cv2.polylines(image, [points_array], isClosed=True, color=(255, 0, 0), thickness=2)

    # Threshold the image to binary
    segmented_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(segmented_image, 128, 255, cv2.THRESH_BINARY)

    # Find contours of the segmented region
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract features for C2, C3, C4 vertebrae
    features = {}
    contour_areas = [(cv2.contourArea(contour), contour) for contour in contours]
    sorted_contours = sorted(contour_areas, key=lambda x: -x[0])[:3]  # Top 3 contours

    for i, (area, contour) in enumerate(sorted_contours):
        vertebra_label = f"C{i + 2}"  # C2, C3, C4
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)

        features[f'{vertebra_label} Area'] = area
        features[f'{vertebra_label} Perimeter'] = perimeter
        features[f'{vertebra_label} Aspect Ratio'] = float(w) / h if h != 0 else 0
        features[f'{vertebra_label} Circularity'] = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
        features[f'{vertebra_label} Vertebral Height'] = h
        features[f'{vertebra_label} Vertebral Width'] = w

       
    label_image = label(binary_image)
    regions = regionprops(label_image)

    for i, (_, contour) in enumerate(sorted_contours):
        if i < len(regions):  # Ensure region exists
            vertebra_label = f"C{i + 2}"
            features[f'{vertebra_label} Eccentricity'] = regions[i].eccentricity
            features[f'{vertebra_label} Solidity'] = regions[i].solidity
            features[f'{vertebra_label} Extent'] = regions[i].extent

    # Calculate cervical lordosis angle
    if len(sorted_contours) >= 3:
        centroids = [cv2.moments(contour) for _, contour in sorted_contours]
        lordosis_angle = np.arctan2(
            centroids[2]['m01'] / centroids[2]['m00'] - centroids[0]['m01'] / centroids[0]['m00'],
            centroids[2]['m00'] / centroids[2]['m00'] - centroids[0]['m00'] / centroids[0]['m00']
        ) * (180 / np.pi)

        for i in range(len(sorted_contours)):
            vertebra_label = f"C{i + 2}"
            features[f'{vertebra_label} Cervical Lordosis Angle'] = lordosis_angle

    return features

     # Return the features as a dictionary

# Load the trained CNN model
model = keras.models.load_model('src/resnet/resnet_model.keras')

# Path to the new image
new_image_path = "boneimg/0402135.png"

# Preprocess the image for prediction
input_image = preprocess_image(new_image_path)

# Make predictions using the CNN model
predicted_age = model.predict(input_image)  # Pass the preprocessed image
predicted_age = predicted_age[0][0]  # Adjust based on your model's output shape

# Extract actual age from the image name (assuming last two digits represent age)
actual_age = int(new_image_path[-6:-4])  # Adjust based on naming format

# Make predictions
floored_predicted_age = np.floor(predicted_age).astype(int)

# Extract features from the image
features = extract_features(new_image_path)

# Create a DataFrame to store the results
new_features_df = pd.DataFrame({
    'Model Name': ["ResNet"],
    'Image Name': [os.path.basename(new_image_path)],
    'Actual Age': [actual_age],
    'Predicted Age': [floored_predicted_age],
    **features  # Unpack the features dictionary into the DataFrame
})

print(f'Actual Age: {actual_age}')
print(f'Predicted Age: {floored_predicted_age}')

# Path to save the features
features_excel_path = "output/robo/extracted_features1.xlsx"  # Update the file path as needed

# Check if the Excel file already exists and append or create it accordingly
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
print(f"Extracted features saved to Excel at: {features_excel_path}")
