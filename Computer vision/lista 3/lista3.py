import cv2
import numpy as np

def empty(a):
    pass

window_name = "Result"
cv2.namedWindow(window_name)
cv2.createTrackbar("Threshold", window_name, 128, 255, empty)


img_path = 'Computer vision/lista 3/banana.jfif'
img_original = cv2.imread(img_path)



height, width, _ = img_original.shape
new_width = 1024
new_height = int(new_width * (height / width))

cv2.resizeWindow(window_name, new_width, new_height)

img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

while True:
    # Zmienna pod suwakiem
    threshold_value = cv2.getTrackbarPos("Threshold", window_name)

    # Binaryzacja
    _, img_thresh = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Morfologia
    kernel = np.ones((5, 5), np.uint8)
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

    # Kontury
    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    img_result = img_original.copy()

    if contours:
        
        largest_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        cv2.drawContours(img_result, [largest_contour], -1, (0, 255, 0), 3)

        text_area = f"Pole: {int(area)} px"
        text_perimeter = f"Obwod: {int(perimeter)} px"

        cv2.putText(img_result, text_area, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_result, text_perimeter, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    img_display = cv2.resize(img_result, (new_width, new_height))

    cv2.imshow(window_name, img_display)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



