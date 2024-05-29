import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Mouse callback function to capture points
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('Image', img)

def create_mask():
    global points
    global img
    # Create blank array of points
    points = []

    # Open video capture
    cap = cv2.VideoCapture(4)  # Use 0 for default camera

    # Set video capture properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Skip the first 24 frames
    for _ in range(24):
        cap.read()

    # Capture the 25th frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        exit()

    img = frame.copy()
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
    mask_path = r'demo_data/MB_Box/masks/mask.png'  # Path to save the mask
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved to {mask_path}")

    # Release video capture
    cap.release()
