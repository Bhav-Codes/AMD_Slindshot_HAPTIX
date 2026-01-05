import cv2
import cvzone
import os

from detector import (
    read_frame,
    find_color,
    find_contours,
    generate_yolo_format,
    save_detections
)

# ======================
# SETUP
# ======================
cap = cv2.VideoCapture("./video/-2.mp4")
cap.set(3, 640)
cap.set(4, 480)

output_folder = "detected_objects"
os.makedirs(output_folder, exist_ok=True)

# ======================
# MAIN LOOP
# ======================
while True:
    img = read_frame(cap)
    if img is None:
        break

    imgOrange, mask = find_color(img)
    _, contours = find_contours(img, mask)

    if contours:
        annotations = generate_yolo_format(contours, img.shape)
        save_detections(img, annotations, output_folder)

    imgStack = cvzone.stackImages([img, imgOrange, mask], 2, 0.5)
    cv2.imshow("Image Stack", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
