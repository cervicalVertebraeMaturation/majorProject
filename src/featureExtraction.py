import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import pandas as pd

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key= ""#use_the_api_key"
)


# Infer on a local image
# "C:\Users\manas\OneDrive\Desktop\major project\exp\newtry\boneimg\img500\0677150.png0715153 0728153"
image_path = "boneimg//datasets-PNG//0728153.png"
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

# Save and display the annotated image
annotated_image_path = "output/robo/1annotated_image.png"
cv2.imwrite(annotated_image_path, image)

# Load the segmented image
segmented_image = cv2.imread(annotated_image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary
_, binary_image = cv2.threshold(segmented_image, 128, 255, cv2.THRESH_BINARY)

# Find contours of the segmented region
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the topmost, second, and third largest areas correspond to C2, C3, and C4
contour_areas = [(cv2.contourArea(contour), contour) for contour in contours]
sorted_contours = sorted(contour_areas, key=lambda x: -x[0])[:3]

# Feature extraction for C2, C3, C4 vertebrae
features = {}
for i, (area, contour) in enumerate(sorted_contours):
    vertebra_label = f"C{i + 2}"  # C2, C3, C4
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)

    features[f'{vertebra_label} Area'] = area
    features[f'{vertebra_label} Perimeter'] = perimeter
    features[f'{vertebra_label} Aspect Ratio'] = float(w) / h
    features[f'{vertebra_label} Circularity'] = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
    features[f'{vertebra_label} Vertebral Height'] = h
    features[f'{vertebra_label} Vertebral Width'] = w

# Optional: Use skimage to get more shape descriptors
label_image = label(binary_image)
regions = regionprops(label_image)

for i, (_, contour) in enumerate(sorted_contours):
    vertebra_label = f"C{i + 2}"
    features[f'{vertebra_label} Eccentricity'] = regions[i].eccentricity
    features[f'{vertebra_label} Solidity'] = regions[i].solidity
    features[f'{vertebra_label} Extent'] = regions[i].extent

# Calculate cervical lordosis angle (assuming vertical alignment for simplicity)
if len(sorted_contours) >= 3:
    centroids = [cv2.moments(contour) for _, contour in sorted_contours]
    lordosis_angle = np.arctan2(
        centroids[2]['m01'] / centroids[2]['m00'] - centroids[0]['m01'] / centroids[0]['m00'],
        centroids[2]['m00'] / centroids[2]['m00'] - centroids[0]['m00'] / centroids[0]['m00']
    ) * (180 / np.pi)

    for i in range(len(sorted_contours)):
        vertebra_label = f"C{i + 2}"
        features[f'{vertebra_label} Cervical Lordosis Angle'] = lordosis_angle

# Extract image name and age
image_name = image_path.split('/')[-1]  # Get the image name from the path
age = image_name[-6:-4]  # Extract last two digits for age (assuming the format is consistent)

# Add the image name and age to the features dictionary
features['Image Name'] = image_name
features['Age'] = int(age)  # Convert age to integer if needed

# Convert features to a DataFrame
features_df = pd.DataFrame([features])  # Create DataFrame with a single row

# Save features to an Excel file
excel_path = "output/robo/vertebra_features.xlsx"
features_df.to_excel(excel_path, index=False)

# Print features for C2, C3, and C4 vertebrae
for key, value in features.items():
    print(f"{key}: {value}")

# Display the annotated image
annotated_image = Image.open(annotated_image_path)
plt.imshow(annotated_image)
plt.axis('off')  # Hide axes
plt.show()

print(f"Features saved to Excel at: {excel_path}")
