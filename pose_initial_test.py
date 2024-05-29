import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# List to store the points
points = []

# Mouse callback function to capture points
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Image', img)

# Specify the directory and file name of the image
image_path = r'/home/niraj/FoundationPose/demo_data/test2/rgb/1716823454413.png'

# Read the image
img = cv2.imread(image_path)
if img is None:
    print("Error: Could not read image.")
    exit()

clone = img.copy()

cv2.imshow('Image', img)
cv2.setMouseCallback('Image', select_points)

print("Click to select points. Press 'q' when done.")

# Wait until 'q' is pressed to finish selecting points
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Create a mask with the same dimensions as the image
mask = np.zeros(clone.shape[:2], dtype=np.uint8)

# Create a polygon from the selected points
if len(points) > 0:
    points = np.array(points)
    cv2.fillPoly(mask, [points], 255)

# Display the mask using matplotlib
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Mask')
plt.imshow(mask, cmap='gray')

plt.show()

# Create a directory for masks if it doesn't exist
if not os.path.exists('masks'):
    os.makedirs('masks')

# Save the mask
mask_path = r'demo_data/test2/masks/1716823454413.png'  # Path to save the mask
cv2.imwrite(mask_path, mask)
print(f"Mask saved to {mask_path}")
