import cv2
import os
import numpy as np
import pandas as pd
from inference_sdk import InferenceHTTPClient
from skimage.measure import label, regionprops

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key= ""#use_the_api_key"
)

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

    # Optional: Use skimage to get more shape descriptors
    label_image = label(binary_image)
    regions = regionprops(label_image)

    for i, (_, contour) in enumerate(sorted_contours):
        if i < len(regions):  # Ensure region exists
            vertebra_label = f"C{i + 2}"
            features[f'{vertebra_label} Eccentricity'] = regions[i].eccentricity
            features[f'{vertebra_label} Solidity'] = regions[i].solidity
            features[f'{vertebra_label} Extent'] = regions[i].extent

    # Calculate cervical lordosis angle if there are at least 3 contours
    if len(sorted_contours) >= 3:
        centroids = [cv2.moments(contour) for _, contour in sorted_contours]
        lordosis_angle = np.arctan2(
            centroids[2]['m01'] / centroids[2]['m00'] - centroids[0]['m01'] / centroids[0]['m00'],
            centroids[2]['m00'] / centroids[2]['m00'] - centroids[0]['m00'] / centroids[0]['m00']
        ) * (180 / np.pi)

        for i in range(len(sorted_contours)):
            vertebra_label = f"C{i + 2}"
            features[f'{vertebra_label} Cervical Lordosis Angle'] = lordosis_angle

    # Save the segmented image
    # Save the segmented image in a static directory
    # Save the segmented image in a static directory
    segmented_image_path = os.path.join("static/segmented_images", os.path.basename(image_path))
    cv2.imwrite(segmented_image_path, segmented_image)


    # Create the DataFrame and set the index to the segmented image path
    features_df = pd.DataFrame([features])  # Create DataFrame from features dictionary
    features_df.index = [segmented_image_path]  # Set the index to the path

    return features_df, segmented_image_path # Return the DataFrame
