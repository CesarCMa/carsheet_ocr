# %%
import pickle
import cv2
import numpy as np
from src.app.services.image_detection import detect_image
import matplotlib.pyplot as plt

# %%
with open("model_output.pkl", "rb") as f:
    predictions, descriptions = pickle.load(f)

# Print the descriptions
print("\nDetected Descriptions:")
print("---------------------")
for code, desc in descriptions.items():
    print(f"{code}: {desc}")

# %%
# Plot bounding boxes and indices

image = cv2.imread("desc_pic.jpg")
img_with_boxes = image.copy()

# Draw each bounding box and its index
for idx, (box_coords, _) in enumerate(predictions):
    # Convert coordinates to integers
    box_coords = np.array(box_coords).astype(np.int32)

    # Draw the bounding box
    cv2.polylines(img_with_boxes, [box_coords], True, (0, 255, 0), 2)

    # Calculate center point for text placement
    center_x = int(np.mean(box_coords[:, 0]))
    center_y = int(np.mean(box_coords[:, 1]))

    # Put the index number
    cv2.putText(
        img_with_boxes,
        str(idx),
        (center_x, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )

# Display the image with boxes using matplotlib

# Convert BGR to RGB since matplotlib expects RGB
img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(img_with_boxes_rgb)
plt.title("Detected Text Boxes")
plt.axis("off")
plt.show()

# %%
predictions[62]
# %%
# Search for 'PLI' in predictions
for idx, (box_coords, text_info) in enumerate(predictions):
    text = text_info[0]  # Get the text from the prediction
    if "PLI" in text:
        print(f"Found 'PLI' in prediction {idx}:")
        print(f"Full text: {text}")
        print(f"Box coordinates: {box_coords}")
        print(f"Confidence: {text_info[1]}")

# %%
# Calculate average height of bounding boxes
heights = []
for box_coords, _ in predictions:
    # Convert coordinates to numpy array
    box = np.array(box_coords)

    # Calculate height as average of left and right side heights
    left_height = np.linalg.norm(box[0] - box[3])  # Top left to bottom left
    right_height = np.linalg.norm(box[1] - box[2])  # Top right to bottom right
    avg_height = (left_height + right_height) / 2

    heights.append(avg_height)

average_box_height = np.mean(heights)
print(f"Average bounding box height: {average_box_height:.2f} pixels")

# %%
