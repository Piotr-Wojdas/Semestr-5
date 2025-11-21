import cv2
import numpy as np


def update(val):
    # Get current trackbar positions
    try:
        scale_percent = cv2.getTrackbarPos('Scale', 'Controls')
        interpolation_idx = cv2.getTrackbarPos('Interpolation', 'Controls')
    except cv2.error:
        scale_percent = 100  # Default to 100% scale
        interpolation_idx = 1  # Default to INTER_LINEAR

    # Scale factor from 0.1 to 3.0
    scale_factor = max(0.1, scale_percent / 100.0) 

    interpolation_methods = {
        0: ('INTER_NEAREST', cv2.INTER_NEAREST),
        1: ('INTER_LINEAR', cv2.INTER_LINEAR),
        2: ('INTER_CUBIC', cv2.INTER_CUBIC),
        3: ('INTER_LANCZOS4', cv2.INTER_LANCZOS4)
    }
    
    method_name, interpolation = interpolation_methods[interpolation_idx]

    # Get original image dimensions
    h, w = img.shape[:2]

    # Calculate new dimensions
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Resize the image using the selected interpolation method
    resized_cv = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # Display the resized image
    cv2.imshow(f'Resized - {method_name}', resized_cv)
    
    # --- Difference Calculation ---
    back_resized = cv2.resize(resized_cv, (w, h), interpolation=interpolation)
    diff = cv2.absdiff(img, back_resized)
    cv2.imshow('Difference', diff)


# Load the image

img = cv2.imread('Computer vision/lista 5/zdj.jpg')
if img is None:
    raise FileNotFoundError("Image not found. Please check the path.")
img = cv2.resize(img, (259, 194))
h, w = img.shape[:2]


# Create a window for controls
cv2.namedWindow('Controls')
cv2.resizeWindow('Controls', 500, 100)

# Scale trackbar: 0-100 (maps to 0.1-3.0)
cv2.createTrackbar('Scale', 'Controls', 100, 290, update) 

# Interpolation trackbar: 0-3
cv2.createTrackbar('Interpolation', 'Controls', 1, 3, update)

# Display original image
cv2.imshow('Original', img)

update(0)


while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()



