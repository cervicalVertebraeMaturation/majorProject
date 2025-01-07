import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import matplotlib.pyplot as plt

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key= ""#use_the_api_key"
)


# Infer on a local image
result = CLIENT.infer("boneimg//datasets-PNG//0006035.png", model_id="segmemtation/1")

# Load the image using OpenCV
image_path = "boneimg//datasets-PNG//0006035.png"
image = cv2.imread(image_path)

# Draw segmentation masks
for prediction in result['predictions']:
    # Segmentation Points
    points = prediction.get('points', [])
    if points:
        # Convert points to a format OpenCV can use
        points_array = np.array([(p['x'], p['y']) for p in points], dtype=np.int32)
        points_array = points_array.reshape((-1, 1, 2))
        cv2.polylines(image, [points_array], isClosed=True, color=(255, 0, 0), thickness=2)  # Draw mask in red

# Save and display the annotated image
annotated_image_path = "output//robo//annotated_image12.png"
cv2.imwrite(annotated_image_path, image)

# Display the annotated image
annotated_image = Image.open(annotated_image_path)
plt.imshow(annotated_image)
plt.axis('off')  # Hide axes
plt.show()
